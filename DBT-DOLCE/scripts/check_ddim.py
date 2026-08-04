"""
DDIM-vs-DDPM quality/speed check for the exact-DC arm (the big-multiplier
decision from the performance diagnosis).

The exact projection is sampler-agnostic (residual -> 0 by construction under
any sampler), so the only question is whether the FEWER-STEP DDIM trajectory
preserves per-sample reconstruction quality (PSNR/SSIM) versus the 1800-step
DDPM chain the current campaign uses. This runs a handful of interior slices
through DDPM-1800 and a set of DDIM step budgets, one posterior sample each
(K=1: we are testing the trajectory, not the ensemble), and prints
PSNR/SSIM/residual and wall-clock per slice with the speedup vs DDPM.

    python scripts/check_ddim.py --config configs/dbt_25deg.yaml \
        --ckpt <ckpt> --dc exact --n 4 --ddim_steps 50 100 250 \
        --device cuda:1

Use it BEFORE adopting DDIM for the noise campaign / Stage-3: if SSIM holds
within a small tolerance, DDIM is a ~20-40x speedup at no quality cost.
"""

import os
import sys
import time
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


def pick_slices(ds, n, seed=0):
    """Seeded-random interior slices with real breast content (mask >= 5%),
    matching scripts/sweep_rho.py so the two tools sample comparably."""
    rng = np.random.default_rng(seed)
    lo, hi = int(0.15 * len(ds)), int(0.85 * len(ds))
    cand = rng.permutation(np.arange(lo, max(hi, lo + 1)))
    idx = []
    for i in cand:
        with h5py.File(ds.files[i], "r") as hf:
            mask = np.flipud(hf["arr_mask"][0]).copy().astype(bool)
        if mask.mean() >= 0.05:
            idx.append(int(i))
        if len(idx) == n:
            break
    if len(idx) < n:
        raise RuntimeError(f"only {len(idx)}/{n} usable slices found.")
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dbt_25deg.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=4, help="slices to test")
    ap.add_argument("--dc", default="exact", choices=["none", "exact"],
                    help="Arm to test (exact is the one that matters; none is "
                         "the pure-prior sanity check).")
    ap.add_argument("--ddim_steps", nargs="+", type=int, default=[50, 100, 250])
    ap.add_argument("--ddim_eta", type=float, default=0.0)
    ap.add_argument("--operator", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    device = torch.device(a.device)
    data_range = cfg["model"].get("data_range", "-11")
    sc = cfg["sampling"]

    model, diffusion = build_eval_model(cfg, ckpt_path=a.ckpt, device=device,
                                        fp16=a.fp16)
    projector = build_projector(cfg["geometry"]["leap_cfg"], device=device)

    dc_operator = None
    if a.dc == "exact":
        op_path = (a.operator or os.environ.get("DBT_OPERATOR_NPZ", "")
                   or sc.get("operator_npz", "runs/operator/A_25deg_512.npz"))
        dc_operator = MaterializedOperator.load(op_path, device=device)
        dc_operator.factorize(rcond=float(sc.get("pinv_rcond", 1e-12)))
        print(f"  exact-DC operator: {op_path} (nnz={dc_operator.A_sp.nnz})")

    ds = DBTSliceDataset(os.path.join(cfg["data"]["processed_dir"],
                                      f"{a.split}_files.txt"),
                         augment=False, deterministic=True, data_range="01")
    idx = pick_slices(ds, a.n)
    print(f"  test slices (interior, mask>=5%): {idx}\n")

    def run(i, sampler, ddim_steps):
        gt_t, mkw = ds[i]
        gt_t = gt_t.unsqueeze(0).to(device)
        mk = {k: v.unsqueeze(0).to(device) for k, v in mkw.items()}
        gt_img = gt_t.squeeze()
        if dc_operator is not None:
            sino = dc_operator.forward(gt_img).unsqueeze(0)
        else:
            sino = projector.forward(gt_img).unsqueeze(0)
        torch.manual_seed(1234)                       # same seed across samplers
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()
        pred = sample_slice(
            model=model, diffusion=diffusion, model_kwargs=mk, sino=sino,
            projector=projector, sampler=sampler, ddim_steps=ddim_steps,
            eta=(0.0 if sampler == "ddim" else 1.0),
            dc_mode=a.dc, dc_operator=dc_operator,
            device=device, data_range=data_range)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dt = time.time() - t0
        pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        gt_np = gt_img.cpu().numpy()
        with h5py.File(ds.files[i], "r") as hf:
            mask = np.flipud(hf["arr_mask"][0]).copy().astype(bool)
        ps = M.psnr_masked(gt_np, pred, mask)
        ss = M.ssim_masked(gt_np, pred, mask)
        pt = torch.from_numpy(pred).to(device)
        if dc_operator is not None:
            y = sino.squeeze(0)
            res = float((dc_operator.forward(pt) - y).norm()
                        / y.norm().clamp(min=1e-30)) * 100.0
        else:
            res = projector.data_residual(pt, sino.squeeze(0))
        return ps, ss, res, dt

    configs = [("ddpm", diffusion.num_timesteps)] + \
              [("ddim", s) for s in a.ddim_steps]
    rows = []
    for sampler, steps in configs:
        ps, ss, rs, dt = [], [], [], []
        for i in idx:
            p, s, r, t = run(i, sampler, steps)
            ps.append(p); ss.append(s); rs.append(r); dt.append(t)
        rows.append(dict(sampler=sampler, steps=steps,
                         psnr=float(np.mean(ps)), ssim=float(np.mean(ss)),
                         resid=float(np.mean(rs)), sec=float(np.mean(dt))))
        print(f"  {sampler}-{steps:<4d}  PSNR={rows[-1]['psnr']:6.2f}  "
              f"SSIM={rows[-1]['ssim']:.3f}  resid={rows[-1]['resid']:.3f}%  "
              f"{rows[-1]['sec']:.1f} s/slice")

    base = rows[0]["sec"]
    print("\n=== summary (vs DDPM-{}) ===".format(configs[0][1]))
    print(f"  {'sampler':<12s} {'PSNR':>6s} {'SSIM':>6s} {'resid%':>7s} "
          f"{'s/slice':>8s} {'speedup':>8s} {'dSSIM':>7s}")
    for r in rows:
        spd = base / r["sec"] if r["sec"] > 0 else float("nan")
        dss = r["ssim"] - rows[0]["ssim"]
        print(f"  {r['sampler']+'-'+str(r['steps']):<12s} {r['psnr']:6.2f} "
              f"{r['ssim']:6.3f} {r['resid']:7.3f} {r['sec']:8.1f} "
              f"{spd:7.1f}x {dss:+7.3f}")
    print("\n  Decision: if dSSIM is within ~0.01-0.02 of DDPM at a DDIM budget, "
          "adopt that budget for the campaign -- the speedup is free quality.")


if __name__ == "__main__":
    main()
