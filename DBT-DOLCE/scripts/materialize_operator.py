"""
One-time materialization of the production forward operator A
(the adjoint hard-failure fix).

T1 measured on the dbt-dolce LEAP build that neither `backward` nor the
autograd VJP is the adjoint of `forward` (they are the same operator, ~20%
away from A^T), so the exact-consistency projection must run through an
explicit sparse A whose literal transpose IS the adjoint. This script builds
it by batched pixel-basis forward projection, verifies it against the live
LEAP forward on random images, saves it, and factorizes the dual Gram AA^T
(printing the spectrum -- this also feeds T7 and fixes the pinv cutoff).

    conda activate dbt-dolce
    python scripts/materialize_operator.py --device cuda:1
        [--cfg configs/leap_dbt_25deg.cfg]
        [--out runs/operator/A_25deg_512.npz]
        [--batch 256] [--rcond 1e-12]

Runtime: a few minutes at 512^2 (262144 basis projections); the result is
reused forever after via MaterializedOperator.load(path, device).
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from physics.dbt_projector import build_projector, MaterializedOperator


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default=str(ROOT / "configs" / "leap_dbt_25deg.cfg"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "operator" / "A_25deg_512.npz"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--rcond", type=float, default=1e-12)
    args = ap.parse_args()

    proj = build_projector(args.cfg, device=args.device)
    print(f"Geometry: {proj.num_angles} views, {proj.img_size}^2 image "
          f"-> A is ({proj.num_angles * proj.img_size}, {proj.img_size ** 2})")

    t0 = time.time()
    M = MaterializedOperator.from_projector(proj, batch_size=args.batch)
    print(f"Materialization took {time.time() - t0:.1f}s")

    # Verify the explicit A reproduces the live forward on random images
    # (validates both linearity of the live forward and faithful assembly).
    torch.manual_seed(0)
    worst = 0.0
    for _ in range(5):
        x = torch.rand(proj.img_size, proj.img_size, device=proj.device)
        ref = proj.forward(x)
        worst = max(worst, float((M.forward(x) - ref).norm() / ref.norm()))
    print(f"Materialized-vs-live forward: max rel_err={worst:.3e} over 5 images")
    if worst > 1e-5:
        print("WARNING: mismatch above fp32 tolerance -- do not trust this "
              "operator; investigate before use.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    M.save(str(out))
    print(f"Saved sparse A to {out}  (nnz={M.A_sp.nnz})")

    t0 = time.time()
    info = M.factorize(rcond=args.rcond, verbose=True)
    print(f"Gram eigendecomposition took {time.time() - t0:.1f}s "
          f"(m={info['m']}; done once, reused every step)")

    # Round-trip smoke: exact projection of a random image onto {Ax = y}.
    x_star = torch.rand(proj.img_size, proj.img_size, device=proj.device)
    y = M.forward(x_star)
    x_hat = torch.rand(proj.img_size, proj.img_size, device=proj.device)
    _, _, pinfo = M.project(x_hat, y)
    print(f"Projection smoke: data_rel_residual={pinfo['data_rel_residual']:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
