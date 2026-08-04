"""
Smoke test for the LEAP-backed DBT projector.

Run this on CPU BEFORE launching the full GPU pipeline:

    conda activate dbt-dolce
    python tests/test_projector.py            # standalone (prints PASS/FAIL)
    pytest tests/test_projector.py            # or via pytest

It validates the three things that must hold for the rest of the pipeline
to be correct:
  1. Shape round-trip      : forward (H,W)->(A,W),  backward (A,W)->(H,W)
  2. Adjoint consistency   : <A x, y> ~= <x, A^T y>   (matched projector pair)
  3. SIRT sanity           : SIRT of a disk's sinogram correlates with the disk

If LEAP is not installed the test is skipped (it does not fail), so it is
safe to run in any environment.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CFG = str(ROOT / "configs" / "leap_dbt_25deg.cfg")

try:
    from physics.dbt_projector import build_projector, _LEAP_AVAILABLE
except Exception as e:  # pragma: no cover
    print(f"[SKIP] could not import projector: {e}")
    _LEAP_AVAILABLE = False


def _disk(size: int, radius_frac: float = 0.3) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size),
        torch.linspace(-1, 1, size),
        indexing="ij",
    )
    r = torch.sqrt(xx ** 2 + yy ** 2)
    return (r < radius_frac).float()


def _build(device="cpu"):
    return build_projector(CFG, device=device)


# Individual checks (also importable by pytest)

def check_shapes(device="cpu"):
    proj = _build(device)
    S, A = proj.img_size, proj.num_angles
    img = _disk(S).to(device)
    sino = proj.forward(img)
    assert sino.shape == (A, S), f"forward shape {tuple(sino.shape)} != {(A, S)}"
    rec = proj.backward(sino)
    assert rec.shape == (S, S), f"backward shape {tuple(rec.shape)} != {(S, S)}"
    # batched
    imgb = img[None].repeat(2, 1, 1)
    sinob = proj.forward(imgb)
    assert sinob.shape == (2, A, S), f"batched forward shape {tuple(sinob.shape)}"
    return True


def check_adjoint(device="cpu", tol=0.05):
    """<A x, y> should equal <x, A^T y> for a matched forward/backprojector."""
    proj = _build(device)
    S, A = proj.img_size, proj.num_angles
    torch.manual_seed(0)
    x = torch.rand(S, S, device=device)
    y = torch.rand(A, S, device=device)
    lhs = (proj.forward(x) * y).sum().item()      # <A x, y>
    rhs = (x * proj.backward(y)).sum().item()      # <x, A^T y>
    denom = max(abs(lhs), abs(rhs), 1e-8)
    rel = abs(lhs - rhs) / denom
    print(f"  adjoint: <Ax,y>={lhs:.4e}  <x,A^Ty>={rhs:.4e}  rel_err={rel:.3e}")
    assert rel < tol, f"adjoint mismatch rel_err={rel:.3e} > {tol}"
    return True


def check_sirt(device="cpu", iters=30):
    """SIRT of a disk's sinogram should positively correlate with the disk."""
    proj = _build(device)
    S = proj.img_size
    gt = _disk(S).to(device)
    sino = proj.forward(gt)
    rec = proj.sirt(sino, num_iters=iters)
    a = gt.flatten().cpu().numpy()
    b = rec.flatten().cpu().numpy()
    corr = float(np.corrcoef(a, b)[0, 1])
    print(f"  sirt: correlation(gt, recon) after {iters} iters = {corr:.3f}")
    assert corr > 0.5, f"SIRT correlation too low: {corr:.3f}"
    return True


# pytest entry points (pytest is optional; standalone runner works without it)

try:
    import pytest
    leap_required = pytest.mark.skipif(
        not _LEAP_AVAILABLE, reason="LEAP_torch not installed"
    )

    @leap_required
    def test_shapes():
        assert check_shapes()

    @leap_required
    def test_adjoint():
        assert check_adjoint()

    @leap_required
    def test_sirt():
        assert check_sirt()
except ImportError:
    pass


# Standalone runner

def main():
    if not _LEAP_AVAILABLE:
        print("[SKIP] LEAP_torch not installed - run scripts/setup.sh first.")
        return 0
    print("Running projector smoke test on CPU ...")
    ok = True
    for name, fn in [("shapes", check_shapes), ("adjoint", check_adjoint),
                     ("sirt", check_sirt)]:
        try:
            fn()
            print(f"[PASS] {name}")
        except AssertionError as e:
            ok = False
            print(f"[FAIL] {name}: {e}")
    print("All projector checks PASSED." if ok else "Some checks FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
