"""
Evaluate a fine-tuned DOLCE model on the DBT test set.

Runs the full DDPM/DDIM reverse diffusion with one of three data-consistency
arms and reports the masked metric panel + data residual for each test slice:

  --dc none    pure prior (no data consistency)
  --dc prox    historical baseline: rho-weighted prox on the post-update
               iterate (kept as-published)
  --dc exact   exact projection of pred_x0 onto {x : Ax = y} through the
               materialized operator (requires runs/operator/A_25deg_512.npz
               from scripts/materialize_operator.py)

Usage

    python evaluate.py \
        --config  configs/dbt_25deg.yaml \
        --ckpt    runs/dbt_25deg_full/checkpoint_best.pt \
        --sampler ddpm \
        --dc      exact

Legacy flags --prox/--no_prox still work and map onto --dc prox/none.
"""

import os
import sys
import json
import math
import time
import argparse
import logging
from pathlib import Path

import yaml
import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "external" / "DOLCE"))
from guided_diffusion import dist_util
from guided_diffusion.script_util import condtion_create_model_and_diffusion

from data.dataset import DBTSliceDataset
from data.preprocess import patient_id_from_filename
from model.finetune import load_weights
from physics.dbt_projector import build_projector, MaterializedOperator
import eval_metrics as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Helpers

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.squeeze().cpu().float().numpy()


def compute_metrics(gt, pred, mask, labels, pixel_mm) -> dict:
    """Reliable masked panel for limited-angle DBT (see eval_metrics.py)."""
    res_x, res_z = M.directional_frc(gt, pred, pixel_mm=pixel_mm)
    out = {
        "psnr":        M.psnr_masked(gt, pred, mask),
        "ssim":        M.ssim_masked(gt, pred, mask),
        "nrmse":       M.nrmse_masked(gt, pred, mask),
        "frc_res_mm":  M.frc_resolution(gt, pred, pixel_mm=pixel_mm),
        "frc_res_x_mm": res_x,
        "frc_res_z_mm": res_z,
        "frc_anisotropy": res_z / (res_x + 1e-8),   # >1 means worse depth res
    }
    for name, mae in M.per_tissue_mae(gt, pred, labels, mask).items():
        out[f"mae_{name}"] = mae
    return out


def _save_comparison_fig(path, gt, sirt, pred, mask, metrics):
    """GT | SIRT conditioning | reconstruction | |error| panel for one slice."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log.warning("matplotlib unavailable; skipping figure (%s)", e)
        return
    err = np.abs(gt - pred) * mask
    # interpolation="nearest": show the true stored pixels. Without it matplotlib
    # antialias-downsamples the 512x512 panels to the figure size and fabricates
    # moire that looks like aliasing but is not in the data.
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    for a, img, title in [
        (ax[0], gt,   "GT"),
        (ax[1], sirt, "SIRT cond"),
        (ax[2], pred, f"recon (PSNR {metrics.get('psnr', float('nan')):.1f}, "
                      f"SSIM {metrics.get('ssim', float('nan')):.3f})"),
    ]:
        a.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        a.set_title(title); a.axis("off")
    im = ax[3].imshow(err, cmap="magma", interpolation="nearest")
    ax[3].set_title(f"|error|  (eps-MSE {metrics.get('eps_mse_loss', float('nan')):.4f})")
    ax[3].axis("off")
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_provenance(args, op_path: str = "") -> dict:
    """
    Reproducibility fingerprint recorded into every metrics payload: which
    machine, code revision, torch/CUDA build, checkpoint, and operator
    produced these numbers. The two-machine split made this non-optional.
    """
    import socket
    import subprocess
    prov = {
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.ckpt,
    }
    try:
        prov["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent), text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        prov["git_commit"] = "unknown"
    try:
        if args.ckpt and os.path.isfile(args.ckpt):
            prov["checkpoint_mtime"] = time.ctime(
                os.path.getmtime(args.ckpt))
    except OSError:
        pass
    if op_path:
        prov["operator_npz"] = op_path
    return prov


def rho_schedule(t: int, T: int, rho_start: float, rho_end: float) -> float:
    """Linear schedule for the proximal weight rho_t."""
    alpha = t / max(T - 1, 1)
    return rho_start + alpha * (rho_end - rho_start)


def gamma_schedule(t: int, T: int, g_start: float, g_end: float) -> float:
    """Linear schedule for the projection damping gamma_t (exact arm).
    Default g_start = g_end = 1 (full projection every step, DDNM default);
    any gamma in [0,2] is per-application no-harm."""
    alpha = t / max(T - 1, 1)
    return g_start + alpha * (g_end - g_start)


def dolce_rho_schedule(i: int, T: int) -> float:
    """DOLCE's data-consistency trust-region weight schedule.

    DOLCE (guided_diffusion/gaussian_diffusion.py) sets
    rho_t = 1 - exp(-(T - t) / (8T)) over descending timesteps t = T-1 .. 0.
    With i the reverse-step position (0 = first / noisiest step, T-1 = last /
    clean step) this is 1 - exp(-(i + 1) / (8T)): ~0 at the noisiest step,
    ~0.118 at the final step. The weight anchors the APGM iterate to the
    diffusion sample, so DC starts almost off and ramps up as noise falls."""
    return 1.0 - math.exp(-(i + 1) / (8.0 * max(T, 1)))


# Sampling with data-consistency

@torch.no_grad()
def sample_slice(
    model,
    diffusion,
    model_kwargs: dict,          # {"condition_rls": (1,1,H,W), "condition_fbp": (1,1,H,W)}
    sino: torch.Tensor,          # (1, n_ang, W)
    projector,
    sampler: str = "ddpm",
    ddim_steps: int = 100,
    eta: float = 1.0,
    prox_solver: str = "cgrad",
    use_prox: bool = True,
    rho_start: float = 5e-6,
    rho_end: float = 1e-4,
    clip_denoised: bool = True,
    device=None,
    data_range: str = "-11",
    dc_mode: str = None,
    dc_operator: MaterializedOperator = None,
    gamma_start: float = 1.0,
    gamma_end: float = 1.0,
    dc_delta: float = 0.0,
    dc_eps: float = 0.0,
    dolce_step: float = 5e-6,
    dolce_iters: int = 30,
) -> np.ndarray:
    """
    Reverse diffusion + data consistency for one slice.

    x0 is taken from diffusion.p_mean_variance's pred_xstart -- DOLCE's own,
    correct conversion of the model output (verified via diagnose.py) -- so this
    does NOT depend on assuming the epsilon parameterisation. DDIM/DDPM then step
    on that x0.

    dc_mode: "none" | "prox" | "exact". None (default) derives it from the
    legacy use_prox flag, so existing callers (scripts/sweep_rho.py) are
    unchanged. "prox" keeps the historical behaviour exactly: rho-prox applied
    to the post-update iterate. "exact" projects pred_x0 (clamped first)
    onto {x : Ax = y} through dc_operator
    (a MaterializedOperator), gamma-damped per step with gamma forced to 1 at
    the final step so the returned image satisfies Ax = y to fp precision.

    data_range: DOLCE's native image range. "-11" maps the conditioning to
    [-1,1], samples there (clamping x0 to [-1,1]), and maps the result back to
    [0,1] for metrics. "01" keeps everything in [0,1].
    Returns the reconstructed image as a numpy array (H, W) in [0,1].
    """
    if dc_mode is None:
        dc_mode = "prox" if use_prox else "none"
    if dc_mode == "exact" and dc_operator is None:
        raise ValueError("dc_mode='exact' requires dc_operator "
                         "(MaterializedOperator.load of the .npz from "
                         "scripts/materialize_operator.py).")
    cond = model_kwargs["condition_rls"]
    B, _, H, W = cond.shape
    C = 1

    if data_range == "-11":
        model_kwargs = {k: v * 2 - 1 for k, v in model_kwargs.items()}
        lo, hi = -1.0, 1.0
    else:
        lo, hi = 0.0, 1.0

    acp = getattr(diffusion, "alphas_cumprod", None)
    if acp is None:
        raise AttributeError("diffusion has no alphas_cumprod; cannot sample.")
    acp = torch.as_tensor(np.asarray(acp), dtype=torch.float32, device=device)
    Tn = int(diffusion.num_timesteps)

    # Timestep sub-sequence (descending) and the previous step for each.
    if sampler == "ddpm":
        seq = list(range(Tn))
    else:
        step = max(1, Tn // max(1, ddim_steps))
        seq = list(range(0, Tn, step))
    seq = seq[::-1]
    seq_prev = seq[1:] + [-1]          # -1 marks the final x0 step
    T = len(seq)

    x = torch.randn(B, C, H, W, device=device)
    max_range_violation = 0.0                 # exact arm: post-projection range,
    final_range_violation = 0.0               # over the trajectory / at the final
                                              # projection (== the returned image)

    for i, (t_cur, t_prev) in enumerate(zip(seq, seq_prev)):
        t = torch.full((B,), t_cur, device=device, dtype=torch.long)
        # DOLCE's own conversion of the model output -> x0 (parameterisation-safe).
        pmv = diffusion.p_mean_variance(model, x, t, clip_denoised=False,
                                        model_kwargs=model_kwargs)
        pred_x0 = pmv["pred_xstart"]
        if clip_denoised:
            pred_x0 = pred_x0.clamp(lo, hi)

        # Exact data consistency: project the (clamped) pred_x0 onto
        # {x : Ax = y} BEFORE the sampler update, so the transition kernel
        # stays intact. No re-clamp after
        # the projection -- that would break exactness; range
        # violations are tracked and reported instead.
        if dc_mode == "exact":
            x0_full = ((pred_x0 + 1) / 2 if data_range == "-11" else pred_x0)
            g = 1.0 if t_prev < 0 else gamma_schedule(i, T, gamma_start, gamma_end)
            # Per-sample projection (loop over the batch B). B=1 is the default
            # single-sample path (bit-identical to the pre-batch code); B=K is
            # the --batch_samples fast path, where the K posterior samples share
            # the slice's sinogram y (sino broadcast over the batch). The dual
            # projection is ~ms, so looping it over K is negligible next to the
            # batched UNet forward. Noise-aware relaxations:
            # with noisy y the unregularized projection is the WRONG default (it
            # fits A^+ eta into the range component); dc_delta = discrepancy
            # ball, dc_eps = eps-MAP, both -> exact projection as sigma -> 0.
            projected, viol = [], 0.0
            for b in range(B):
                x0_img = x0_full[b, 0]                        # (H, W)
                yb = sino[b] if sino.shape[0] == B else sino[0]
                if dc_delta > 0:
                    x0_img, _, _ = dc_operator.project_ball(
                        x0_img, yb, delta=dc_delta, mode="gamma")
                elif dc_eps > 0:
                    x0_img, _, _ = dc_operator.project(
                        x0_img, yb, gamma=g, eps=dc_eps)
                else:
                    x0_img, _, _ = dc_operator.project(x0_img, yb, gamma=g)
                viol = max(viol, float((-x0_img.min()).clamp(min=0)),
                           float((x0_img.max() - 1).clamp(min=0)))
                projected.append(x0_img)
            max_range_violation = max(max_range_violation, viol)
            if t_prev < 0:                    # x = pred_x0 below: this IS the output
                final_range_violation = viol
            x0_p = torch.stack(projected, 0).unsqueeze(1)     # (B, 1, H, W)
            pred_x0 = x0_p * 2 - 1 if data_range == "-11" else x0_p

        a_t = acp[t_cur]
        eps = (x - torch.sqrt(a_t) * pred_x0) / torch.sqrt(1.0 - a_t).clamp(min=1e-6)

        if t_prev < 0:
            x = pred_x0
        else:
            a_prev = acp[t_prev]
            if sampler == "ddim":
                sigma = eta * torch.sqrt(
                    (1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)
                )
                dir_xt = torch.sqrt((1 - a_prev - sigma ** 2).clamp(min=0)) * eps
                noise  = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
                x = torch.sqrt(a_prev) * pred_x0 + dir_xt + sigma * noise
            else:  # DDPM ancestral (consecutive timesteps)
                beta_t   = 1 - a_t / a_prev
                post_var = (beta_t * (1 - a_prev) / (1 - a_t)).clamp(min=0)
                coef_x0  = torch.sqrt(a_prev) * beta_t / (1 - a_t)
                coef_xt  = torch.sqrt(a_t / a_prev) * (1 - a_prev) / (1 - a_t)
                mean = coef_x0 * pred_x0 + coef_xt * x
                x = mean + torch.sqrt(post_var) * torch.randn_like(x)

        # Proximal data-consistency (historical baseline arm). The projector/
        # sinogram live in [0,1] attenuation space, so map x to [0,1], apply
        # prox, map back. Note: applied to the POST-UPDATE iterate -- kept
        # as-published for the comparison study.
        #
        # The rho-prox argmin ||Ax-y||^2 + rho||x-x_hat||^2 has the closed form
        # x = x_hat - A^T (A A^T + rho I)^{-1}(A x_hat - y)  (push-through
        # identity (A^T A + rho I)^{-1} A^T = A^T (A A^T + rho I)^{-1}). When the
        # trusted materialized operator is available we solve it THROUGH THAT
        # OPERATOR (dc_operator.project(eps=rho)): the *same* prox, but on the
        # exact literal transpose and the SPD cached dual instead of the primal
        # CG on the live LEAP pair. On a mismatched-adjoint build the primal CG
        # is non-SPD -> p^T A^T A p can go negative -> the step explodes to NaN
        # (sanitised to a black image); the dual route removes that failure mode
        # AND puts prox on the same operator of record as the exact arm, so the
        # comparison is fair. --prox_live forces the legacy
        # live-pair primal path for the "naive prox" ablation.
        if dc_mode == "prox":
            rho = rho_schedule(i, T, rho_start, rho_end)
            x_dtype = x.dtype
            x01 = (x + 1) / 2 if data_range == "-11" else x   # (B, 1, H, W)
            outs = []
            for b in range(B):
                x_img = x01[b, 0]                             # (H, W)
                yb = sino[b] if sino.shape[0] == B else sino[0]
                if dc_operator is not None:
                    x_img, _, _ = dc_operator.project(
                        x_img, yb, gamma=1.0, eps=rho)
                else:
                    x_img = projector.prox_solver(
                        x_img.float(), yb.float(), rho=rho, method=prox_solver)
                outs.append(x_img.to(x_dtype))
            x01 = torch.stack(outs, 0).unsqueeze(1)           # (B, 1, H, W)
            x = x01 * 2 - 1 if data_range == "-11" else x01

        # Faithful DOLCE data consistency (dataFidelities/CTClass.py:apgm +
        # gaussian_diffusion q_sample(y, t)). Distinct from the `prox` arm,
        # which applies an EXACT projection (residual ~0 per step) at a swept
        # rho. DOLCE instead applies its DEPLOYED mechanism: a GENTLE,
        # early-stopped APGM correction (small fixed step, ~0.16% residual
        # removed per step) toward the measurement NOISED to the iterate's
        # level, anchored to the post-update iterate. This is the fair
        # reproduction of DOLCE's published behaviour.
        if dc_mode == "dolce":
            a_t = acp[t_cur]
            sqrt_a   = float(torch.sqrt(a_t))
            sqrt_1ma = float(torch.sqrt((1.0 - a_t).clamp(min=0.0)))
            rho_d    = dolce_rho_schedule(i, T)
            x_dtype  = x.dtype
            x01 = (x + 1) / 2 if data_range == "-11" else x   # (B, 1, H, W)
            outs = []
            for b in range(B):
                x_img = x01[b, 0]                             # (H, W)
                yb = sino[b] if sino.shape[0] == B else sino[0]
                # Noise-matched target: the sinogram of the iterate at noise
                # level t is sqrt(a_t) y + sqrt(1-a_t) A eps. We push image-
                # space noise through A (the scale-correct equivalent of
                # DOLCE's q_sample on the sinogram, which implicitly assumes an
                # O(1)-normalised sinogram -- our raw-LEAP sinogram is not).
                eps_img = torch.randn(x_img.shape, device=x_img.device,
                                      dtype=x_img.dtype)
                b_t = sqrt_a * yb + sqrt_1ma * dc_operator.forward(eps_img).to(yb.dtype)
                x_img = dc_operator.dolce_apgm(
                    x_img, b_t, rho=rho_d, step=dolce_step, n_iters=dolce_iters)
                outs.append(x_img.to(x_dtype))
            x01 = torch.stack(outs, 0).unsqueeze(1)           # (B, 1, H, W)
            x = x01 * 2 - 1 if data_range == "-11" else x01

    if data_range == "-11":
        x = (x + 1) / 2                     # back to [0,1] for metrics
    if dc_mode == "exact" and max_range_violation > 0.05:
        # The final-image number is the one that decides the Dykstra escalation;
        # trajectory spikes at noisy early steps are expected and
        # harmless -- they are absorbed by later denoising steps.
        log.info("exact-DC: post-projection range violation: final image %.4f, "
                 "trajectory max %.3f (logged, not clamped; Dykstra "
                 "onto S n [0,1]^n is the designed escalation if the final-image "
                 "violation is material)",
                 final_range_violation, max_range_violation)
    return to_numpy(x)


# Model building (shared with scripts/sweep_rho.py)

def build_eval_model(cfg: dict, ckpt_path: str = "", device="cuda:0",
                     fp16: bool = False):
    """
    Build DOLCE and load the fine-tuned weights for
    evaluation. Built in the dtype eval actually runs in: fp32 unless fp16 is
    requested (the config's use_fp16 only governs the *training* autocast, and a
    UNet built fp16 with float32 weights crashes -- see the --fp16 help text).
    """
    from train import dolce_model_args
    model_args = dolce_model_args(cfg)
    model_args["use_fp16"] = fp16
    model, diffusion = condtion_create_model_and_diffusion(**model_args)

    base = cfg["model"]["pretrained_ckpt"]
    log.info("Loading base DOLCE weights from %s", base)
    state = dist_util.load_state_dict(base, map_location="cpu")
    model.load_state_dict(state, strict=False)

    if ckpt_path:
        # Full fine-tune checkpoint: loads all weights (EMA preferred).
        load_weights(model, ckpt_path)
        log.info("Loaded fine-tuned checkpoint from %s", ckpt_path)
    else:
        log.warning("No checkpoint provided - evaluating base DOLCE model only.")

    model.eval().to(torch.device(device))
    if fp16:
        model.convert_to_fp16()
        log.warning("Running model in native fp16.")
    return model, diffusion


# Main evaluation loop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/dbt_25deg.yaml")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--sampler",   default="", help="Override sampler (ddpm/ddim)")
    parser.add_argument("--prox",      default="", help="Override prox solver (apgm/cgrad)")
    parser.add_argument("--prox_live", action="store_true",
                        help="Prox arm: force the legacy live-pair primal solve "
                             "(prox_cgrad/apgm on the LEAP forward/backward) "
                             "instead of the trusted-operator dual route. This "
                             "is the 'naive prox' ablation and can diverge to a "
                             "black image on a mismatched-adjoint build.")
    parser.add_argument("--no_prox",   action="store_true",
                        help="Legacy: same as --dc none")
    parser.add_argument("--dc", default="",
                        choices=["", "none", "prox", "exact", "dolce"],
                        help="Data-consistency arm: none (pure prior), prox "
                             "(exact rho-prox on the post-update iterate), "
                             "dolce (FAITHFUL DOLCE data consistency -- gentle "
                             "noise-matched APGM, its deployed mechanism), "
                             "exact (projection onto {x: Ax=y} via the "
                             "materialized operator). Overrides --no_prox and "
                             "the config's dc_mode.")
    parser.add_argument("--operator", default="",
                        help="Materialized operator .npz for --dc exact "
                             "(default: config sampling.operator_npz)")
    parser.add_argument("--dolce_step", type=float, default=5e-6,
                        help="DOLCE arm: fixed APGM data-term step (DOLCE's "
                             "literal 5e-6 in raw-LEAP units). Per-step "
                             "residual contraction ~ step*sigma1^2; sweep it on "
                             "a new geometry (sigma1^2 printed by factorize).")
    parser.add_argument("--dolce_iters", type=int, default=30,
                        help="DOLCE arm: APGM inner iterations (DOLCE: 30).")
    parser.add_argument("--noise_sigma", type=float, default=0.0,
                        help="Std of Gaussian noise added to the simulated "
                             "sinogram (seeded per slice). 0 = the noiseless "
                             "inverse-crime regime. With noise, the exact "
                             "arm automatically relaxes per --noise_mode.")
    parser.add_argument("--noise_mode", default="ball", choices=["ball", "eps"],
                        help="Exact-arm relaxation under --noise_sigma>0: "
                             "'ball' = discrepancy principle, delta = "
                             "sigma*sqrt(m), parameter-free (default); "
                             "'eps' = MAP step with eps = sigma^2/tau^2 "
                             "(set --tau).")
    parser.add_argument("--tau", type=float, default=0.1,
                        help="Prior std of range components for --noise_mode "
                             "eps (eps = sigma^2/tau^2).")
    parser.add_argument("--rho_start", type=float, default=None,
                        help="Override config sampling.rho_start (prox arm; "
                             "set start == end for a constant rho, e.g. the "
                             "best value from scripts/sweep_rho.py)")
    parser.add_argument("--rho_end", type=float, default=None,
                        help="Override config sampling.rho_end (prox arm)")
    parser.add_argument("--device",    default="cuda:0")
    parser.add_argument("--max_slices",type=int, default=-1,
                        help="Limit number of slices (for a quick look)")
    parser.add_argument("--slice_stride", type=int, default=1,
                        help="Evaluate every Nth slice of the split. Use with "
                             "--max_slices to spread a small budget across the "
                             "whole split (and hence across patients) instead "
                             "of taking the first slices of one patient, e.g. "
                             "--max_slices 30 --slice_stride 480 on a ~14k "
                             "split. Default 1 = consecutive (pilot behaviour).")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Posterior samples per slice (>1 enables the "
                             "Stage-1 ensemble: mean + uncertainty map + "
                             "calibration panel; K=8-16 suggested).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base RNG seed. Each (slice, sample) gets a "
                             "deterministic seed derived from it, so arms run "
                             "with the same --seed share noise realisations "
                             "and differ only by data consistency (same "
                             "checkpoint, same seeds).")
    parser.add_argument("--batch_samples", action="store_true",
                        help="Fast path: run the K posterior samples as ONE "
                             "batch through the UNet (large speedup at batch=1 "
                             "GPU under-utilisation). Draws the K noise "
                             "realisations batched, so NOT bit-identical to the "
                             "per-sample-seeded default -- use uniformly across "
                             "arms in a campaign, never mixed with unbatched "
                             "runs.")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"],
                        help="Which file list to evaluate on.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run: slices already in "
                             "partial_{arm}_{split}.jsonl are reused instead "
                             "of recomputed. Per-(slice,sample) seeding makes "
                             "the resumed run identical to an uninterrupted "
                             "one. Without this flag an existing partial file "
                             "is moved to a timestamped .bak.")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Run the model in native fp16 (convert_to_fp16). "
                             "Off by default: DOLCE's conversion does not cover "
                             "every module, so fp16 can raise a Half/float "
                             "dtype error. fp32 is the safe choice.")
    parser.add_argument("--n_loss_t", type=int, default=8,
                        help="Timesteps sampled per slice to estimate the "
                             "epsilon-MSE training loss (comparable to the "
                             "training/val loss curve).")
    parser.add_argument("--figs", type=int, default=10,
                        help="Save a GT/SIRT/recon/error comparison PNG for the "
                             "first N evaluated slices (0 to disable). Arrays "
                             "and figures go to slices_{arm}_{split}/ so arms "
                             "don't overwrite each other.")
    args   = parser.parse_args()
    cfg    = load_config(args.config)
    device = torch.device(args.device)

    # Effective settings (CLI overrides YAML)
    sc    = cfg["sampling"]
    sampler     = args.sampler    or sc["sampler"]
    prox_solver = args.prox       or sc["prox_solver"]
    dc_mode     = args.dc or ("none" if args.no_prox else sc.get("dc_mode", "prox"))
    ddim_steps  = sc["ddim_steps"]
    eta         = sc["eta"]
    rho_start   = sc["rho_start"] if args.rho_start is None else args.rho_start
    rho_end     = sc["rho_end"]   if args.rho_end   is None else args.rho_end
    gamma_start = float(sc.get("gamma_start", 1.0))
    gamma_end   = float(sc.get("gamma_end", 1.0))
    results_dir = cfg["evaluation"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Crash-safe incremental persistence: one JSON line per completed slice.
    # A 30-slice K=8 ensemble run is ~16 h; without this, a crash at slice 29
    # loses everything. --resume reuses completed slices (identical to an
    # uninterrupted run thanks to per-(slice,sample) seeding).
    partial_path = os.path.join(results_dir,
                                f"partial_{dc_mode}_{args.split}.jsonl")
    done_slices = {}
    if os.path.isfile(partial_path):
        if args.resume:
            with open(partial_path) as pf:
                for line in pf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue          # torn final line from a crash
                    if "slice_index" in row:
                        done_slices[int(row["slice_index"])] = row
            log.info("Resume: reusing %d completed slices from %s",
                     len(done_slices), partial_path)
        else:
            bak = f"{partial_path}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
            os.replace(partial_path, bak)
            log.warning("Existing partial results moved to %s "
                        "(pass --resume to continue them instead)", bak)

    # Importing train installs its checkpoint monkey-patch and exposes the
    # epsilon-MSE loss used during training.
    from train import local_training_losses
    model, diffusion = build_eval_model(cfg, ckpt_path=args.ckpt,
                                        device=device, fp16=args.fp16)

    # Exact arm: load the materialized operator (T1 measured that the live
    # LEAP pair has no valid adjoint). The exact
    # arm both simulates y and projects through this SAME operator, so
    # y in range(A) holds exactly and the final projection is exact.
    # The exact arm requires the operator; the prox arm uses it too (the
    # trusted dual route for the rho-prox, unless --prox_live is set), so it is
    # solved on the same operator of record and can't diverge to a black image
    # on a mismatched-adjoint build.
    dc_operator = None
    op_path = ""
    prox_needs_op = (dc_mode == "prox" and not args.prox_live)
    # The faithful DOLCE arm runs its APGM through the trusted materialized
    # (A, A^T) pair too (same operator of record as exact/prox).
    dolce_needs_op = (dc_mode == "dolce")
    if dc_mode == "exact" or prox_needs_op or dolce_needs_op:
        op_path = (args.operator
                   or os.environ.get("DBT_OPERATOR_NPZ", "")
                   or sc.get("operator_npz", "runs/operator/A_25deg_512.npz"))
        if not os.path.isfile(op_path):
            # Both the exact arm and the (default) trusted-dual prox arm REQUIRE
            # the operator. Never silently fall back to the diverging live-pair
            # prox (a buffered nohup log hides a warning) -- fail loudly;
            # --prox_live is the explicit opt-in to the naive path.
            if prox_needs_op:
                hint = ("Pass --operator <path> or set DBT_OPERATOR_NPZ (e.g. "
                        "the /net share); or --prox_live for the naive "
                        "live-pair prox.")
            elif dolce_needs_op:
                hint = ("The faithful DOLCE arm needs the materialized "
                        "operator; pass --operator <path> or set "
                        "DBT_OPERATOR_NPZ (e.g. the /net share).")
            else:
                hint = "Build it once with scripts/materialize_operator.py."
            raise FileNotFoundError(f"{op_path} not found -- {hint}")
        dc_operator = MaterializedOperator.load(op_path, device=device)
    if dc_operator is not None:
        if dc_operator.img_size != int(cfg["geometry"]["image_size"]):
            raise ValueError(
                f"operator geometry ({dc_operator.img_size}^2) does not match "
                f"config ({cfg['geometry']['image_size']}^2); re-materialize.")
        # View-count guard: a wrong-geometry operator would NOT trip T8 (y is
        # simulated through the same operator), so a leaked DBT_OPERATOR_NPZ
        # export pointing at another geometry's npz would silently run the
        # wrong physics. Fail loudly instead (multi-geometry campaigns).
        if dc_operator.num_angles != int(cfg["geometry"]["num_projections"]):
            raise ValueError(
                f"operator has {dc_operator.num_angles} views but config "
                f"geometry.num_projections={cfg['geometry']['num_projections']} "
                f"({op_path}). Check --operator / DBT_OPERATOR_NPZ / "
                f"sampling.operator_npz precedence.")
        log.info("DC operator (%s arm): %s (nnz=%d); factorizing dual Gram ...",
                 dc_mode, op_path, dc_operator.A_sp.nnz)
        dc_operator.factorize(rcond=float(sc.get("pinv_rcond", 1e-12)))

    # Build the LIVE LEAP projector -- only when an arm actually needs it. The
    # exact / dolce / prox(trusted-operator) arms run ENTIRELY through the
    # materialized operator (y-simulation, projection, and residual all use
    # dc_operator), so they must NOT require the geometry .cfg: a missing or
    # behind live cfg killed the 25v50 Phase-A exact arm at build_projector even
    # though it had its operator. The live pair is used only where
    # dc_operator is None -- the pure-prior (none) arm and the --prox_live
    # ablation (y-simulation, data_residual, and the naive primal prox).
    projector = None
    if dc_operator is None:
        projector = build_projector(cfg["geometry"]["leap_cfg"], device=device)

    # Dataset (chosen split)
    pixel_mm = float(cfg["geometry"]["pixel_size_mm"])
    file_list = os.path.join(cfg["data"]["processed_dir"], f"{args.split}_files.txt")
    if not os.path.isfile(file_list):
        raise FileNotFoundError(
            f"{file_list} not found. Pick a --split whose *_files.txt exists "
            f"(preprocessing writes these at the end of a full run)."
        )
    # Keep the eval dataset in [0,1] so GT and the SIRT baseline are scored in
    # [0,1]; sample_slice maps to the model's [-1,1] range internally.
    dataset = DBTSliceDataset(file_list, augment=False, deterministic=True,
                              data_range="01")
    log.info("%s set: %d slices", args.split, len(dataset))

    # Per-run output dir for arrays + figures, tagged like metrics_{arm}_{split}
    # so multi-arm campaigns don't overwrite each other's images. With the same
    # --seed/--slice_stride, the same absolute slice indices are saved in every
    # arm -> directly comparable panels.
    slices_dir = os.path.join(results_dir, f"slices_{dc_mode}_{args.split}")
    os.makedirs(slices_dir, exist_ok=True)

    # Noise-aware relaxation parameters for the exact arm (0 = noiseless
    # regime, plain exact projection). delta from the discrepancy principle
    # is parameter-free; eps needs --tau.
    dc_delta = dc_eps = 0.0
    if args.noise_sigma > 0 and dc_mode == "exact":
        m_dim = dc_operator.m
        if args.noise_mode == "ball":
            dc_delta = args.noise_sigma * float(np.sqrt(m_dim))
        else:
            dc_eps = (args.noise_sigma / args.tau) ** 2
        log.info("Noise mode '%s': sigma=%g -> delta=%g eps=%g",
                 args.noise_mode, args.noise_sigma, dc_delta, dc_eps)

    # Evaluation loop. i = absolute slice index in the split (names, seeds);
    # j = position in this run (first-N array/figure saving).
    all_metrics = []
    indices = list(range(0, len(dataset), max(1, args.slice_stride)))
    if args.max_slices >= 0:
        indices = indices[:args.max_slices]

    for j, i in enumerate(tqdm(indices, desc="Evaluating")):
        if i in done_slices:                    # --resume: already computed
            all_metrics.append(done_slices[i])
            continue
        gt_t, mkw = dataset[i]                          # (1,H,W), named conds (1,H,W)
        gt_t  = gt_t.unsqueeze(0).to(device)           # (1,1,H,W)
        model_kwargs = {
            k: v.unsqueeze(0).to(device) for k, v in mkw.items()
        }                                               # each (1,1,H,W)

        # Mask + tissue labels (aligned to dataset's vertical flip)
        path = dataset.files[i]
        with h5py.File(path, "r") as hf:
            mask   = np.flipud(hf["arr_mask"][0]).copy().astype(bool)
            labels = np.flipud(hf["arr_labels"][0]).copy().astype(np.int32)
        patient = patient_id_from_filename(Path(path).parent.name)

        # Re-simulate sinogram from GT. Any arm holding the materialized
        # operator (exact always; prox unless --prox_live) simulates y through
        # that SAME operator, so y in range(A) holds exactly and prox/exact are
        # compared on identical measurements; the pure-prior and legacy-prox
        # paths keep the live LEAP forward.
        gt_img = gt_t.squeeze()                        # (H,W)
        if dc_operator is not None:
            sino = dc_operator.forward(gt_img).unsqueeze(0)   # (1,n_ang,W) fp64
        else:
            sino = projector.forward(gt_img).unsqueeze(0)     # (1,n_ang,W)
        if args.noise_sigma > 0:
            # Seeded per slice: identical noise realisation across arms/runs.
            g_noise = torch.Generator().manual_seed(args.seed * 999_983 + i)
            sino_noise = args.noise_sigma * torch.randn(
                sino.shape, generator=g_noise, dtype=torch.float64)
            sino = sino + sino_noise.to(sino.device, sino.dtype)

        # Posterior sampling (optionally averaged over n_samples).
        K = max(1, args.n_samples)
        common = dict(
            model=model, diffusion=diffusion, projector=projector,
            sampler=sampler, ddim_steps=ddim_steps, eta=eta,
            prox_solver=prox_solver, rho_start=rho_start, rho_end=rho_end,
            dc_mode=dc_mode, dc_operator=dc_operator,
            gamma_start=gamma_start, gamma_end=gamma_end,
            dc_delta=dc_delta, dc_eps=dc_eps, device=device,
            dolce_step=args.dolce_step, dolce_iters=args.dolce_iters,
            data_range=cfg["model"].get("data_range", "-11"),
        )
        if args.batch_samples and K > 1:
            # Fast path: run all K posterior samples as one batch through the
            # UNet (the GPU is far under-utilised at batch=1). One seed per
            # slice; the K samples share the slice's sinogram y (broadcast over
            # the batch). NOTE: this draws the K noise realisations as a single
            # batched tensor, so results are NOT bit-identical to the
            # per-sample-seeded default -- use it uniformly across arms in a
            # campaign, not mixed with unbatched runs.
            torch.manual_seed(args.seed * 1_000_003 + i)
            mkw_b = {k: v.expand(K, *v.shape[1:]).contiguous()
                     for k, v in model_kwargs.items()}
            sino_b = sino.expand(K, *sino.shape[1:]).contiguous()
            out = sample_slice(model_kwargs=mkw_b, sino=sino_b, **common)
            samples = out if out.ndim == 3 else out[None]   # (K, H, W)
        else:
            samples = []
            for k in range(K):
                # Deterministic per-(slice, sample) seed: arms with the same
                # --seed see identical noise and differ only by the DC mechanism.
                torch.manual_seed(args.seed * 1_000_003 + i * 1_009 + k)
                samples.append(sample_slice(
                    model_kwargs=model_kwargs, sino=sino, **common))
            samples = np.stack(samples, axis=0)            # (K, H, W)
        pred = samples.mean(axis=0)

        # Guard against a diverged sample so one bad slice can't crash the run
        # (and so it is visible if the reconstruction itself is unstable).
        if not np.isfinite(pred).all():
            log.warning("slice %d: %d non-finite recon values; sanitising.",
                        i, int((~np.isfinite(pred)).sum()))
            pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)

        gt_np   = to_numpy(gt_t)
        cond_np = to_numpy(model_kwargs["condition_rls"])   # SIRT conditioning

        metrics = compute_metrics(gt_np, pred, mask, labels, pixel_mm)
        if dc_mode == "exact":
            # Residual w.r.t. the operator of record (the materialized A).
            pred_t = torch.from_numpy(pred).to(device)
            y = sino[0]
            res_pct = float((dc_operator.forward(pred_t) - y).norm()
                            / y.norm().clamp(min=1e-30)) * 100.0
            metrics["data_residual"] = res_pct
            # Range-violation extent of the delivered samples:
            # the max alone can't tell "one hot pixel" from "a region"; the
            # fraction is the other number the Dykstra decision needs.
            metrics["range_violation_frac"] = float(
                np.mean((samples < -1e-3) | (samples > 1 + 1e-3)))
            metrics["range_violation_max"] = float(
                max(samples.max() - 1.0, -samples.min(), 0.0))
            # T8 runtime assert: exactness is an invariant, not a metric. The
            # ensemble mean of exactly-consistent samples is consistent too
            # (S is affine), so this holds for n_samples > 1 as well. In the
            # noisy regime exactness is deliberately NOT delivered: the ball
            # mode guarantees residual <= delta instead (asserted); the eps
            # mode has only a soft bound (logged, not gated).
            if args.noise_sigma == 0:
                assert res_pct <= 0.1, (
                    f"slice {i}: exact-DC data residual {res_pct:.4f}% > 0.1% "
                    f"-- the exactness invariant is violated; investigate "
                    f"before trusting any output (T8).")
            elif dc_delta > 0:
                bound_pct = dc_delta / float(y.norm().clamp(min=1e-30)) * 100.0
                assert res_pct <= bound_pct * 1.05 + 1e-6, (
                    f"slice {i}: ball-mode residual {res_pct:.4f}% exceeds "
                    f"the discrepancy bound {bound_pct:.4f}% (T8, noisy).")
        elif dc_operator is not None:
            # Prox arm on the trusted operator: measure the residual against the
            # SAME operator that produced y (the prox route uses it too), so the
            # number is self-consistent and directly comparable to the exact arm.
            pred_t = torch.from_numpy(pred).to(device)
            y = sino[0]
            metrics["data_residual"] = float(
                (dc_operator.forward(pred_t) - y).norm()
                / y.norm().clamp(min=1e-30)) * 100.0
        else:
            metrics["data_residual"] = projector.data_residual(
                torch.from_numpy(pred).to(device), sino[0]
            )
        # SIRT baseline (masked, for reference)
        metrics["sirt_psnr"] = M.psnr_masked(gt_np, cond_np, mask)
        metrics["sirt_ssim"] = M.ssim_masked(gt_np, cond_np, mask)

        # Epsilon-MSE training loss for these weights (averaged over random t),
        # directly comparable to the training/val loss curve.
        with torch.no_grad():
            eps_losses = []
            for _ in range(max(1, args.n_loss_t)):
                t_ = torch.randint(0, diffusion.num_timesteps, (1,), device=device)
                l = local_training_losses(diffusion, model, gt_t, t_,
                                          model_kwargs=model_kwargs)
                eps_losses.append(float(l["loss"].mean().item()))
        metrics["eps_mse_loss"] = float(np.mean(eps_losses))

        # Stage-1 ensemble outputs (only meaningful with >1 sample): Bessel-
        # corrected uncertainty map + calibration panel (ECE,
        # coverage, AUSE, z-stats), plus the perception-distortion pair --
        # single posterior sample AND ensemble mean, reported separately,
        # never mixed in one table column.
        unc = None
        if args.n_samples > 1:
            K = samples.shape[0]
            unc = samples.std(axis=0, ddof=1)
            metrics.update(M.calibration_panel(gt_np, pred, unc, mask, K=K))
            metrics["sample_psnr"]  = M.psnr_masked(gt_np, samples[0], mask)
            metrics["sample_ssim"]  = M.ssim_masked(gt_np, samples[0], mask)
            metrics["sample_nrmse"] = M.nrmse_masked(gt_np, samples[0], mask)
            if dc_mode == "exact":
                # Ensemble consistency check: EVERY sample must lie in S, not just the
                # mean; the worst per-sample residual is the evidence.
                y = sino[0]
                res = [float((dc_operator.forward(
                            torch.from_numpy(s).to(device)) - y).norm()
                            / y.norm().clamp(min=1e-30)) * 100.0
                       for s in samples]
                metrics["data_residual_max_sample"] = max(res)
            # Persist masked (error, sigma_eff) pairs for the pooled
            # recalibration protocol -- fit scalar/isotonic on VAL, report on
            # TEST (scripts/analyze_results.py). Subsampled to bound disk use.
            calib_dir = os.path.join(results_dir,
                                     f"calib_{dc_mode}_{args.split}")
            os.makedirs(calib_dir, exist_ok=True)
            m = mask.astype(bool)
            err_m = (gt_np[m] - pred[m]).astype(np.float32)
            sig_m = M.sigma_effective(unc[m], K).astype(np.float32)
            rng = np.random.default_rng(args.seed * 1_000_003 + i)
            if err_m.size > 50_000:
                sel = rng.choice(err_m.size, 50_000, replace=False)
                err_m, sig_m = err_m[sel], sig_m[sel]
            np.savez_compressed(
                os.path.join(calib_dir, f"calib_{i:04d}.npz"),
                err=err_m, sigma=sig_m, K=K, patient=patient)

        metrics["patient"] = patient
        metrics["slice_index"] = int(i)
        all_metrics.append(metrics)
        with open(partial_path, "a") as pf:     # crash-safe checkpoint
            pf.write(json.dumps(metrics) + "\n")

        # Save arrays + a comparison figure for the first few slices.
        if j < max(10, args.figs):
            extra = {} if unc is None else {"std": unc.astype(np.float32),
                                            "sample0": samples[0]}
            np.savez_compressed(
                os.path.join(slices_dir, f"slice_{i:04d}.npz"),
                gt=gt_np, pred=pred, sirt=cond_np, mask=mask, **extra,
            )
        if j < args.figs:
            _save_comparison_fig(
                os.path.join(slices_dir, f"compare_{i:04d}.png"),
                gt_np, cond_np, pred, mask, metrics,
            )

    # Per-patient aggregation, then bootstrap CI across patients
    metric_keys = sorted({k for m in all_metrics
                          for k, v in m.items() if isinstance(v, float)})
    by_patient = {}
    for m in all_metrics:
        by_patient.setdefault(m["patient"], []).append(m)

    summary = {}
    for k in metric_keys:
        patient_means = [
            float(np.nanmean([mm[k] for mm in slices if k in mm]))
            for slices in by_patient.values()
            if any(k in mm for mm in slices)
        ]
        summary[k] = M.bootstrap_ci(patient_means)

    prox_backend = ("live-pair primal" if (dc_operator is None or args.prox_live)
                    else "trusted dual")
    dc_desc = {"none": "none (pure prior)",
               "prox": f"prox ({prox_backend}, rho {rho_start}->{rho_end})",
               "dolce": f"dolce (faithful APGM, step {args.dolce_step:g}, "
                        f"{args.dolce_iters} it, noise-matched)",
               "exact": f"exact (gamma {gamma_start}->{gamma_end})"}[dc_mode]
    print("\n" + "=" * 72)
    print(f"  Sampler: {sampler}  |  DC: {dc_desc}  |  "
          f"n_samples: {args.n_samples}  |  patients: {len(by_patient)}")
    print("=" * 72)
    for k in metric_keys:
        s = summary[k]
        print(f"  {k:<22s}  median={s['median']:.4f}  "
              f"IQR={s['iqr']:.4f}  95%CI=[{s['ci_low']:.4f}, {s['ci_high']:.4f}]")
    print("=" * 72)

    payload = {"config": {"sampler": sampler, "dc_mode": dc_mode,
                          "prox": prox_solver,
                          "prox_backend": prox_backend if dc_mode == "prox" else None,
                          "dolce_step": args.dolce_step if dc_mode == "dolce" else None,
                          "dolce_iters": args.dolce_iters if dc_mode == "dolce" else None,
                          "use_prox": dc_mode == "prox",  # legacy field
                          "gamma_start": gamma_start, "gamma_end": gamma_end,
                          "n_samples": args.n_samples, "seed": args.seed,
                          "batch_samples": args.batch_samples,
                          "fp16": args.fp16,
                          "split": args.split,
                          "noise_sigma": args.noise_sigma,
                          "noise_mode": args.noise_mode if args.noise_sigma > 0
                                        else None,
                          "dc_delta": dc_delta, "dc_eps": dc_eps,
                          "provenance": run_provenance(args, op_path)},
               "summary_per_patient": summary,
               "per_slice": all_metrics}
    # metrics.json keeps the legacy name; the arm/split-tagged copy survives
    # multi-arm campaigns without manual renaming (the pilot's papercut).
    tagged = os.path.join(results_dir, f"metrics_{dc_mode}_{args.split}.json")
    for out_json in (os.path.join(results_dir, "metrics.json"), tagged):
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
    log.info("Results saved to metrics.json and %s", tagged)


if __name__ == "__main__":
    main()
