"""
Unit tests for eval_metrics.py.  Pure numpy/skimage - runs on CPU, no LEAP.

    python tests/test_metrics.py      # standalone
    pytest tests/test_metrics.py      # or via pytest
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import eval_metrics as M


def _phantom(size=128, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size))
    gt = np.zeros((size, size), np.float32)
    r = np.sqrt((yy - size/2)**2 + (xx - size/2)**2)
    gt[r < size*0.4] = 0.4              # adipose
    gt[r < size*0.2] = 0.78             # fibroglandular
    mask = r < size*0.4
    labels = np.zeros((size, size), np.int32)
    labels[r < size*0.4] = 1
    labels[r < size*0.2] = 2
    return gt, mask, labels


def check_identity():
    gt, mask, labels = _phantom()
    assert M.psnr_masked(gt, gt, mask) >= 90
    assert M.ssim_masked(gt, gt, mask) > 0.99
    assert M.nrmse_masked(gt, gt, mask) < 1e-4
    maes = M.per_tissue_mae(gt, gt, labels, mask)
    assert all(v < 1e-4 for v in maes.values())
    return True


def check_affine_invariance():
    gt, mask, labels = _phantom()
    biased = 0.5 * gt + 0.1            # scale + offset
    # affine-matched PSNR should be ~perfect despite the bias
    assert M.psnr_masked(gt, biased, mask, match=True) >= 40
    # without matching it should be much worse
    assert M.psnr_masked(gt, biased, mask, match=False) < 30
    return True


def check_frc_directional():
    gt, mask, labels = _phantom()
    # Blur only along Z (axis 0) -> depth resolution should be worse than X
    from scipy.ndimage import gaussian_filter1d
    blurred = gaussian_filter1d(gt, sigma=3, axis=0)
    rx, rz = M.directional_frc(gt, blurred, pixel_mm=0.273)
    assert rz > rx, f"expected worse Z res: rx={rx:.3f} rz={rz:.3f}"
    return True


def check_bootstrap():
    vals = list(np.linspace(0, 1, 20))
    ci = M.bootstrap_ci(vals)
    assert ci["ci_low"] <= ci["median"] <= ci["ci_high"]
    assert ci["n"] == 20
    return True


try:
    import pytest

    def test_identity():            assert check_identity()
    def test_affine_invariance():   assert check_affine_invariance()
    def test_bootstrap():           assert check_bootstrap()

    @pytest.mark.skipif(__import__("importlib").util.find_spec("scipy") is None,
                        reason="scipy not installed")
    def test_frc_directional():     assert check_frc_directional()
except ImportError:
    pass


def main():
    ok = True
    checks = [("identity", check_identity),
              ("affine_invariance", check_affine_invariance),
              ("bootstrap", check_bootstrap)]
    try:
        import scipy  # noqa: F401
        checks.append(("frc_directional", check_frc_directional))
    except ImportError:
        print("[skip] frc_directional (scipy not installed)")
    for name, fn in checks:
        try:
            fn(); print(f"[PASS] {name}")
        except AssertionError as e:
            ok = False; print(f"[FAIL] {name}: {e}")
    print("All metric checks PASSED." if ok else "Some checks FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
