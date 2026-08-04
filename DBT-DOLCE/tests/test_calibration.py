"""
Unit tests for the Stage-1 calibration metrics in eval_metrics.py.
Pure numpy/scipy - CPU, no LEAP.

The synthetic ground truth: a heteroscedastic Gaussian ensemble where the
reported sigma either IS the generating sigma (calibrated), is half of it
(over-confident), or is uninformative. Each metric must (a) certify the
calibrated case, (b) flag the miscalibrated one, and (c) the scalar
recalibration must recover the miscalibration factor.

    python tests/test_calibration.py      # standalone
    pytest tests/test_calibration.py      # or via pytest
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_metrics as M

N = 200_000


def _synthetic(seed=0, n=N):
    """(err, sigma_true): heteroscedastic, err ~ N(0, sigma_true^2)."""
    rng = np.random.default_rng(seed)
    sigma = 0.01 + 0.04 * rng.random(n)            # [0.01, 0.05]
    err = sigma * rng.standard_normal(n)
    return err, sigma


def check_sigma_effective():
    std = np.array([1.0, 2.0])
    np.testing.assert_allclose(M.sigma_effective(std, 8),
                               std * np.sqrt(1 + 1 / 8))
    return True


def check_standardized_errors():
    err, sigma = _synthetic()
    z = M.standardized_error_stats(err, sigma)
    assert abs(z["z_mean"]) < 0.01, z
    assert abs(z["z_std"] - 1.0) < 0.01, z
    assert abs(z["z_frac_gt3"] - 0.0027) < 0.002, z      # 2*Phi(-3)
    assert z["sigma_floor_frac"] == 0.0
    # Degenerate: all-zero sigma must not crash and must report the floor.
    z0 = M.standardized_error_stats(err, np.zeros_like(sigma))
    assert z0["sigma_floor_frac"] == 1.0 and np.isnan(z0["z_std"])
    return True


def check_reliability_ece():
    err, sigma = _synthetic()
    cal = M.reliability(err, sigma)
    # Calibrated: per-bin RMSE tracks per-bin sigma -> tiny ECE.
    assert cal["ece"] < 2e-3, cal["ece"]
    assert len(cal["bin_sigma"]) == 15 == len(cal["bin_rmse"])
    # Curve is on the identity within sampling noise, and monotone in sigma.
    np.testing.assert_allclose(cal["bin_sigma"], cal["bin_rmse"], rtol=0.05)
    # Over-confident by 2x: ECE ~ mean(sigma_true)/2, far above the gate.
    half = M.reliability(err, sigma / 2)
    assert half["ece"] > 10 * cal["ece"], (cal["ece"], half["ece"])
    return True


def check_coverage():
    err, sigma = _synthetic()
    cov = M.coverage(err, sigma)
    for q, c in cov.items():
        assert abs(c - q) < 0.01, (q, c)
    # Over-confident sigma/2: |err| <= z_.9 * sigma/2 <=> |z| <= 0.822,
    # analytic coverage 2*Phi(0.822)-1 = 0.589.
    c90 = M.coverage(err, sigma / 2, quantiles=(0.9,))[0.9]
    assert abs(c90 - 0.589) < 0.01, c90
    return True


def check_ause():
    err, sigma = _synthetic()
    # Oracle sigma = |err| ranks removal exactly like the oracle: AUSE ~ 0.
    a_oracle = M.ause(err, np.abs(err))
    assert abs(a_oracle["ause"]) < 1e-9, a_oracle["ause"]
    # Informative sigma beats an uninformative (constant-ish random) one.
    rng = np.random.default_rng(1)
    a_info = M.ause(err, sigma)["ause"]
    a_rand = M.ause(err, rng.random(err.size))["ause"]
    assert 0 <= a_info < a_rand, (a_info, a_rand)
    # Curves normalized: both start at 1.
    assert abs(a_oracle["curve_pred"][0] - 1.0) < 1e-12
    return True


def check_scalar_recalibration():
    err, sigma = _synthetic()
    # Reported sigma is half the generating one -> s should recover ~2,
    # and applying it should restore nominal coverage and shrink ECE.
    reported = sigma / 2
    s = M.fit_scalar_recalibration(err, reported)
    assert abs(s - 2.0) < 0.02, s
    c90 = M.coverage(err, s * reported, quantiles=(0.9,))[0.9]
    assert abs(c90 - 0.9) < 0.01, c90
    assert (M.reliability(err, s * reported)["ece"]
            < M.reliability(err, reported)["ece"] / 5)
    return True


def check_pava():
    np.testing.assert_allclose(M._pava(np.array([3.0, 1.0, 2.0]),
                                       np.ones(3)), [2.0, 2.0, 2.0])
    y = np.array([1.0, 2.0, 5.0])
    np.testing.assert_allclose(M._pava(y, np.ones(3)), y)  # monotone: untouched
    # weights matter: heavy first element dominates the pooled value
    np.testing.assert_allclose(M._pava(np.array([2.0, 0.0]),
                                       np.array([3.0, 1.0])), [1.5, 1.5])
    return True


def check_isotonic_recalibration():
    """
    SHAPE miscalibration a scalar cannot fix: reported sigma is a monotone
    NONLINEAR distortion of the generating sigma (compresses the dynamic
    range), so per-bin RMSE vs reported sigma is off the identity by a
    varying factor. The scalar can only rescale globally; the isotonic map
    must recover per-bin calibration and nominal coverage.
    """
    err, sigma_true = _synthetic(seed=3)
    reported = np.sqrt(0.03 * sigma_true)          # monotone, nonlinear

    s = M.fit_scalar_recalibration(err, reported)
    iso = M.fit_isotonic_recalibration(err, reported)
    assert all(iso["y"][i + 1] >= iso["y"][i] - 1e-12
               for i in range(len(iso["y"]) - 1)), "isotonic map not monotone"

    sig_scalar = s * reported
    sig_iso = M.apply_recalibration(reported, iso)
    ece_scalar = M.reliability(err, sig_scalar)["ece"]
    ece_iso = M.reliability(err, sig_iso)["ece"]
    assert ece_iso < ece_scalar / 3, (ece_scalar, ece_iso)
    cov = M.coverage(err, sig_iso)
    for q, c in cov.items():
        assert abs(c - q) < 0.02, (q, c)
    # and on an already-calibrated input the map is ~identity
    iso0 = M.fit_isotonic_recalibration(err, sigma_true)
    sig0 = M.apply_recalibration(sigma_true, iso0)
    assert M.reliability(err, sig0)["ece"] < 2e-3
    return True


def check_panel():
    """calibration_panel end-to-end on 2D maps, calibrated by construction."""
    rng = np.random.default_rng(0)
    H = W = 128
    K = 8
    gt = rng.random((H, W)).astype(np.float32)
    std_true = (0.01 + 0.04 * rng.random((H, W))).astype(np.float32)
    # gt = mean + sigma_eff * z: the panel's target is exactly calibrated.
    z = rng.standard_normal((H, W)).astype(np.float32)
    mean = gt - M.sigma_effective(std_true, K) * z
    yy, xx = np.indices((H, W))
    mask = (yy - H / 2) ** 2 + (xx - W / 2) ** 2 < (0.45 * H) ** 2
    p = M.calibration_panel(gt, mean, std_true, mask, K=K)
    for key in ("uncertainty_mean", "uncertainty_calib_corr", "cal_ece",
                "cal_coverage_50", "cal_coverage_90", "cal_coverage_95",
                "cal_ause", "cal_z_std", "cal_z_frac_gt3",
                "cal_sigma_floor_frac", "cal_recal_scale"):
        assert key in p and np.isfinite(p[key]), (key, p.get(key))
    assert abs(p["cal_coverage_90"] - 0.9) < 0.03, p["cal_coverage_90"]
    assert abs(p["cal_z_std"] - 1.0) < 0.05, p["cal_z_std"]
    assert abs(p["cal_recal_scale"] - 1.0) < 0.05, p["cal_recal_scale"]
    assert p["cal_ece"] < 5e-3, p["cal_ece"]
    return True


def check_analyze_roundtrip():
    """calib npz dirs -> analyze_results.write_calibration_report -> report."""
    import analyze_results as AR
    err, sigma = _synthetic(seed=2, n=40_000)
    reported = sigma / 2                       # over-confident on purpose
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for split in ("val", "test"):
            d = td / f"calib_exact_{split}"
            d.mkdir()
            for i, pat in enumerate(("pat_a", "pat_b")):
                sel = slice(i * 20_000, (i + 1) * 20_000)
                np.savez_compressed(d / f"calib_{i:04d}.npz",
                                    err=err[sel].astype(np.float32),
                                    sigma=reported[sel].astype(np.float32),
                                    K=8, patient=pat)
        AR.write_calibration_report(td / "calib_exact_val",
                                    td / "calib_exact_test",
                                    td / "calibration.md",
                                    td / "calibration.png")
        text = (td / "calibration.md").read_text()
    # The fitted scalar must be reported and ~2 (the injected miscalibration).
    import re
    s = float(re.search(r"s = \*?\*?([0-9.]+)", text).group(1))
    assert abs(s - 2.0) < 0.05, s
    assert "coverage_90" in text and "ece" in text and "ause" in text
    return True


CHECKS = [
    ("sigma_effective", check_sigma_effective),
    ("standardized_errors", check_standardized_errors),
    ("reliability_ece", check_reliability_ece),
    ("coverage", check_coverage),
    ("ause", check_ause),
    ("scalar_recalibration", check_scalar_recalibration),
    ("pava", check_pava),
    ("isotonic_recalibration", check_isotonic_recalibration),
    ("calibration_panel", check_panel),
    ("analyze_roundtrip", check_analyze_roundtrip),
]

try:
    import pytest  # noqa: F401

    def test_sigma_effective():        assert check_sigma_effective()
    def test_standardized_errors():    assert check_standardized_errors()
    def test_reliability_ece():        assert check_reliability_ece()
    def test_coverage():               assert check_coverage()
    def test_ause():                   assert check_ause()
    def test_scalar_recalibration():   assert check_scalar_recalibration()
    def test_pava():                   assert check_pava()
    def test_isotonic_recalibration(): assert check_isotonic_recalibration()
    def test_panel():                  assert check_panel()
    def test_analyze_roundtrip():      assert check_analyze_roundtrip()
except ImportError:
    pass


def main():
    ok = True
    for name, fn in CHECKS:
        try:
            fn(); print(f"[PASS] {name}")
        except AssertionError as e:
            ok = False; print(f"[FAIL] {name}: {e}")
    print("All calibration checks PASSED." if ok else "Some checks FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
