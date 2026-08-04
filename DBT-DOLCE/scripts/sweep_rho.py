"""
Sweep the proximal data-consistency weight rho against a trained checkpoint.

The prox solves  argmin_x ||Ax - b||^2 + rho * ||x - x_hat||^2  at each reverse
step. For the 9p25d geometry A^T A is heavily rank-deficient,
so rho controls how much the diffusion prior is trusted in the unmeasured
directions -- the single most sensitive inference knob. This sweeps constant
values and start:end schedules on a handful of val slices and reports masked
PSNR/SSIM vs the no-prox recon and the SIRT baseline.

    python scripts/sweep_rho.py \
        --config configs/dbt_25deg.yaml \
        --ckpt   runs/dbt_25deg_full/checkpoint_best.pt \
        --rhos   no_prox 0.03 0.1 0.3 1 3 10 1:0.1 3:0.3 \
        --n 4 --ddim_steps 50 --device cuda:1

Each --rhos entry is "no_prox", a constant ("0.3"), or a linear schedule
"start:end" over the reverse trajectory. Writes results/rho_sweep/sweep.json
and sweep.png (SSIM/PSNR vs setting), and prints a ranked table.
"""

import os
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import h5py
import torch

from evaluate import build_eval_model, sample_slice, load_config
from data.dataset import DBTSliceDataset
from physics.dbt_projector import build_projector, MaterializedOperator
import eval_metrics as M


def parse_rho(spec: str):
    """'no_prox' -> None; '0.3' -> (0.3, 0.3); '1:0.1' -> (1.0, 0.1)."""
    if spec == "no_prox":
        return None
    if ":" in spec:
        s, e = spec.split(":")
        return float(s), float(e)
    v = float(spec)
    return v, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dbt_25deg.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=4, help="slices (spread over the split)")
    ap.add_argument("--rhos", nargs="+",
                    default=["no_prox", "0.03", "0.1", "0.3", "1", "3", "10", "1:0.1"])
    ap.add_argument("--ddim_steps", type=int, default=50)
    ap.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"],
                    help="Sweep under the sampler the comparison will use. "
                         "MEASURED (2026-07-03): rho does NOT transfer across "
                         "samplers (rho 1->0.1 finite under DDIM, divergent "
                         "under DDPM-1800). Use ddpm before trusting a rho "
                         "for a DDPM study; slower (full timestep chain).")
    ap.add_argument("--prox_solver", default="cgrad")
    ap.add_argument("--prox_live", action="store_true",
                    help="Sweep the legacy live-pair primal prox instead of the "
                         "trusted-operator dual route. The live path can diverge "
                         "to black on a mismatched-adjoint build -- a broken "
                         "solver, not evidence about rho. Default: trusted dual.")
    ap.add_argument("--operator", default="",
                    help="Materialized operator .npz (default: env DBT_OPERATOR_NPZ "
                         "or config sampling.operator_npz). Used for the trusted "
                         "prox route so the sweep matches the exact arm's operator.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/rho_sweep")
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device(a.device)
    torch.manual_seed(0)

    model, diffusion = build_eval_model(cfg, ckpt_path=a.ckpt, device=device)
    projector = build_projector(cfg["geometry"]["leap_cfg"], device=device)
    data_range = cfg["model"].get("data_range", "-11")

    # Trusted operator for the prox solve (the same one the exact arm uses):
    # the rho-prox through the SPD cached dual, not the primal CG on the live
    # LEAP pair. Without it a mismatched-adjoint build makes prox diverge to a
    # black image, so the sweep would "measure" no_prox winning for the wrong
    # reason. --prox_live opts back into the naive path.
    sc = cfg["sampling"]
    dc_operator = None
    if not a.prox_live:
        op_path = (a.operator or os.environ.get("DBT_OPERATOR_NPZ", "")
                   or sc.get("operator_npz", "runs/operator/A_25deg_512.npz"))
        if not os.path.isfile(op_path):
            # Do NOT silently fall back to the diverging live-pair path (that is
            # the bug this sweep exists to avoid, and a buffered nohup log hides
            # the warning). Fail loudly; --prox_live is the explicit opt-in.
            raise FileNotFoundError(
                f"operator npz not found: {op_path}\n"
                f"  Pass --operator <path> (e.g. the /net share) or set "
                f"DBT_OPERATOR_NPZ, so the prox solve uses the trusted dual.\n"
                f"  To deliberately sweep the naive live-pair prox instead, "
                f"pass --prox_live.")
        dc_operator = MaterializedOperator.load(op_path, device=device)
        dc_operator.factorize(rcond=float(sc.get("pinv_rcond", 1e-12)))
        print(f"  prox route: trusted dual via {op_path} (nnz={dc_operator.A_sp.nnz})")
    else:
        print("  prox route: legacy live-pair primal (--prox_live)")

    ds = DBTSliceDataset(os.path.join(cfg["data"]["processed_dir"],
                                      f"{a.split}_files.txt"),
                         augment=False, deterministic=True, data_range="01")

    # Slice selection: seeded-random from the INTERIOR of the split, keeping
    # only slices with substantial breast content. The old linspace picked
    # the split's endpoints -- boundary slices (mostly skin, tiny mask) that
    # make the swept rho unrepresentative. Interior 70%, mask fraction >= 5%
    # of FOV (healthy slices measure ~18%); seeded rng for reproducibility.
    rng = np.random.default_rng(0)
    lo, hi = int(0.15 * len(ds)), int(0.85 * len(ds))
    candidates = rng.permutation(np.arange(lo, max(hi, lo + 1)))
    idx, slices = [], []
    for i in candidates:
        with h5py.File(ds.files[i], "r") as hf:
            mask = np.flipud(hf["arr_mask"][0]).copy().astype(bool)
        if mask.mean() < 0.05:
            continue                              # edge/near-empty slice
        gt_t, mkw = ds[i]
        gt_t = gt_t.unsqueeze(0).to(device)
        mk = {k: v.unsqueeze(0).to(device) for k, v in mkw.items()}
        # Match the prox solver's operator: trusted forward when available so
        # y in range(A) holds exactly, else the live LEAP forward.
        if dc_operator is not None:
            sino = dc_operator.forward(gt_t.squeeze()).unsqueeze(0)
        else:
            sino = projector.forward(gt_t.squeeze()).unsqueeze(0)
        gt_np = gt_t.squeeze().cpu().numpy()
        sirt_np = mk["condition_rls"].squeeze().cpu().numpy()
        slices.append(dict(mk=mk, sino=sino, mask=mask, gt=gt_np, sirt=sirt_np))
        idx.append(int(i))
        if len(slices) == a.n:
            break
    if len(slices) < a.n:
        raise RuntimeError(f"only {len(slices)}/{a.n} usable slices found "
                           f"(mask fraction >= 5%) -- check the dataset.")
    print(f"  sweep slices (interior, mask>=5%): {idx}")

    sirt_ss = float(np.mean([M.ssim_masked(s["gt"], s["sirt"], s["mask"]) for s in slices]))
    sirt_ps = float(np.mean([M.psnr_masked(s["gt"], s["sirt"], s["mask"]) for s in slices]))
    print(f"\nSIRT baseline over {a.n} slices: PSNR={sirt_ps:.2f}  SSIM={sirt_ss:.3f}\n")

    results = []
    for spec in a.rhos:
        rho = parse_rho(spec)
        ps, ss = [], []
        for s in slices:
            # Fixed noise seed per slice so settings differ only by the prox.
            torch.manual_seed(1234)
            pred = sample_slice(
                model=model, diffusion=diffusion, model_kwargs=s["mk"],
                sino=s["sino"], projector=projector,
                sampler=a.sampler, ddim_steps=a.ddim_steps,
                eta=(0.0 if a.sampler == "ddim" else 1.0),
                prox_solver=a.prox_solver,
                dc_mode=("prox" if rho is not None else "none"),
                dc_operator=dc_operator,
                rho_start=(rho[0] if rho else 0.0),
                rho_end=(rho[1] if rho else 0.0),
                device=device, data_range=data_range,
            )
            pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
            ps.append(M.psnr_masked(s["gt"], pred, s["mask"]))
            ss.append(M.ssim_masked(s["gt"], pred, s["mask"]))
        r = dict(rho=spec, psnr=float(np.mean(ps)), ssim=float(np.mean(ss)))
        results.append(r)
        print(f"  rho={spec:>8s}   PSNR={r['psnr']:6.2f}   SSIM={r['ssim']:.3f}")

    ranked = sorted(results, key=lambda r: -r["ssim"])
    print("\n=== ranked by SSIM ===")
    for r in ranked:
        beat = "BEATS SIRT" if r["ssim"] > sirt_ss else ""
        print(f"  {r['rho']:>8s}   SSIM={r['ssim']:.3f}  PSNR={r['psnr']:6.2f}  {beat}")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "sweep.json", "w") as f:
        json.dump({"ckpt": a.ckpt, "n": a.n, "ddim_steps": a.ddim_steps,
                   "sampler": a.sampler, "slices": idx,
                   "sirt": {"psnr": sirt_ps, "ssim": sirt_ss},
                   "results": results}, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [r["rho"] for r in results]
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        xs = range(len(labels))
        ax1.plot(xs, [r["ssim"] for r in results], "o-", color="tab:orange", label="SSIM")
        ax1.axhline(sirt_ss, ls="--", color="gray", label=f"SIRT SSIM {sirt_ss:.3f}")
        ax1.set_xticks(list(xs)); ax1.set_xticklabels(labels, rotation=30)
        ax1.set_ylabel("SSIM"); ax1.set_xlabel("rho setting")
        ax2 = ax1.twinx()
        ax2.plot(xs, [r["psnr"] for r in results], "s-", color="tab:blue", alpha=0.6)
        ax2.set_ylabel("PSNR (dB)", color="tab:blue")
        ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / "sweep.png", dpi=120)
        plt.close(fig)
        print(f"\nSaved {out/'sweep.json'} and {out/'sweep.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
