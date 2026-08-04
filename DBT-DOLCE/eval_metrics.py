"""
Reliable evaluation metrics for limited-angle DBT reconstruction.

All full-reference metrics are computed INSIDE the breast mask (the air
background otherwise dominates and inflates the numbers).  Intensity-sensitive
metrics (PSNR / NRMSE) are computed after an affine intensity match, because
limited-angle reconstructions carry a low-frequency bias that otherwise
dominates the error without reflecting structural quality.

Functions
  affine_match(pred, gt, mask)            -> intensity-matched pred
  psnr_masked / nrmse_masked / ssim_masked
  per_tissue_mae(gt, pred, labels, mask)  -> {tissue: MAE in [0,1] units}
  frc_resolution(gt, pred, pixel_mm)      -> global resolution (mm)
  directional_frc(gt, pred, pixel_mm)     -> (res_x_mm, res_z_mm)
  bootstrap_ci(values)                    -> dict(median, iqr, ci_low, ci_high)

Stage-1 uncertainty calibration (all take
flattened masked 1D arrays of signed error and predictive sigma):
  sigma_effective(std, K)                 -> std * sqrt(1 + 1/K)
  standardized_error_stats(err, sigma)    -> z-score mean/std/tail stats
  reliability(err, sigma, n_bins)         -> per-bin curve + ECE
  coverage(err, sigma, quantiles)         -> empirical vs nominal coverage
  ause(err, sigma, n_steps)               -> sparsification error (AUSE) + curves
  fit_scalar_recalibration(err, sigma)    -> scalar s for sigma -> s*sigma
  calibration_panel(gt, mean, std, mask, K) -> flat dict of the above for one slice
"""

import numpy as np
from skimage.metrics import structural_similarity as _ssim

# Tissue label -> name (must match data/preprocess.py ATTENUATION ordering)
TISSUE_NAMES = {1: "adipose", 2: "fibroglandular", 3: "skin"}


# Intensity matching

def affine_match(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Affine fit a*pred + b to gt over the masked region (removes the global
    scale/offset bias typical of limited-angle recon).

    Uses the closed-form least-squares solution for a line (slope + intercept)
    rather than np.linalg.lstsq: it avoids the LAPACK SVD path (which can fail
    under MKL / on non-finite input) and degrades gracefully when pred is
    constant. Non-finite values are sanitised before fitting.
    """
    m = mask.astype(bool)
    if m.sum() < 16:
        return pred
    p = np.nan_to_num(pred[m].astype(np.float64))
    g = np.nan_to_num(gt[m].astype(np.float64))
    pm, gm = p.mean(), g.mean()
    var = np.mean((p - pm) ** 2)
    if var < 1e-12:                     # constant prediction -> nothing to scale
        return pred
    a = np.mean((p - pm) * (g - gm)) / var
    b = gm - a * pm
    return np.nan_to_num(a * pred + b).astype(np.float32)


# Full-reference, masked

def _mse_masked(gt, pred, mask):
    m = mask.astype(bool)
    return float(np.mean((gt[m] - pred[m]) ** 2))


def psnr_masked(gt, pred, mask, data_range=1.0, match=True) -> float:
    if match:
        pred = affine_match(pred, gt, mask)
    mse = _mse_masked(gt, pred, mask)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def nrmse_masked(gt, pred, mask, match=True) -> float:
    if match:
        pred = affine_match(pred, gt, mask)
    m = mask.astype(bool)
    rmse = np.sqrt(_mse_masked(gt, pred, mask))
    denom = np.sqrt(np.mean(gt[m] ** 2)) + 1e-8
    return float(rmse / denom)


def ssim_masked(gt, pred, mask, data_range=1.0) -> float:
    """SSIM averaged over the masked region (uses the full SSIM map)."""
    _, smap = _ssim(gt, pred, data_range=data_range, full=True)
    m = mask.astype(bool)
    if m.sum() == 0:
        return 0.0
    return float(smap[m].mean())


# Tissue-aware

def per_tissue_mae(gt, pred, labels, mask, match=True) -> dict:
    """Mean absolute error within each tissue class (in normalised [0,1])."""
    if match:
        pred = affine_match(pred, gt, mask)
    out = {}
    for lab, name in TISSUE_NAMES.items():
        sel = (labels == lab) & mask.astype(bool)
        if sel.sum() == 0:
            continue
        out[name] = float(np.mean(np.abs(gt[sel] - pred[sel])))
    return out


# Fourier Ring Correlation (resolution)

def _frc_curve(gt, pred):
    """
    Fourier Ring Correlation as a function of radius (cycles).  Returns
    (radii_pixels, frc_values).  Inputs are 2D arrays of equal shape.
    """
    F1 = np.fft.fftshift(np.fft.fft2(gt))
    F2 = np.fft.fftshift(np.fft.fft2(pred))
    H, W = gt.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.indices((H, W))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_int = r.astype(int)
    n_bins = min(cy, cx)
    num = np.zeros(n_bins)
    d1 = np.zeros(n_bins)
    d2 = np.zeros(n_bins)
    cross = (F1 * np.conj(F2))
    for b in range(n_bins):
        ring = (r_int == b)
        if not ring.any():
            continue
        num[b] = np.real(cross[ring].sum())
        d1[b] = (np.abs(F1[ring]) ** 2).sum()
        d2[b] = (np.abs(F2[ring]) ** 2).sum()
    denom = np.sqrt(d1 * d2) + 1e-12
    frc = num / denom
    return np.arange(n_bins), frc


def _resolution_from_frc(radii, frc, shape_n, pixel_mm, threshold=0.5):
    """
    Convert the first FRC threshold crossing into a resolution in mm.
    resolution = 1 / cutoff_frequency, cutoff in cycles/mm.
    """
    below = np.where(frc < threshold)[0]
    if len(below) == 0:
        r_c = radii[-1]            # resolution limited by pixel grid
    else:
        r_c = max(below[0], 1)
    freq_cyc_per_px = r_c / shape_n          # cycles / pixel
    freq_cyc_per_mm = freq_cyc_per_px / pixel_mm
    return float(1.0 / (freq_cyc_per_mm + 1e-12))


def frc_resolution(gt, pred, pixel_mm=0.273, threshold=0.5) -> float:
    radii, frc = _frc_curve(gt, pred)
    return _resolution_from_frc(radii, frc, gt.shape[0], pixel_mm, threshold)


def directional_frc(gt, pred, pixel_mm=0.273, threshold=0.5, wedge_deg=30.0):
    """
    FRC restricted to angular wedges around the kx axis (X resolution) and the
    ky axis (Z / depth resolution).  For limited-angle DBT the Z resolution is
    the degraded one; the X-vs-Z gap is the key quality signal.
    Returns (res_x_mm, res_z_mm).
    """
    F1 = np.fft.fftshift(np.fft.fft2(gt))
    F2 = np.fft.fftshift(np.fft.fft2(pred))
    H, W = gt.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.indices((H, W))
    dy, dx = (yy - cy), (xx - cx)
    r = np.sqrt(dy ** 2 + dx ** 2)
    r_int = r.astype(int)
    theta = np.degrees(np.arctan2(dy, dx))     # -180..180
    cross = np.real(F1 * np.conj(F2))
    p1 = np.abs(F1) ** 2
    p2 = np.abs(F2) ** 2
    n_bins = min(cy, cx)

    def wedge_res(angle_mask):
        num = np.zeros(n_bins); d1 = np.zeros(n_bins); d2 = np.zeros(n_bins)
        for b in range(n_bins):
            ring = (r_int == b) & angle_mask
            if not ring.any():
                continue
            num[b] = cross[ring].sum()
            d1[b] = p1[ring].sum()
            d2[b] = p2[ring].sum()
        frc = num / (np.sqrt(d1 * d2) + 1e-12)
        return _resolution_from_frc(np.arange(n_bins), frc, H, pixel_mm, threshold)

    a = np.abs(theta)
    x_wedge = (a < wedge_deg) | (a > 180 - wedge_deg)        # near kx axis
    z_wedge = (np.abs(a - 90) < wedge_deg)                   # near ky axis
    return wedge_res(x_wedge), wedge_res(z_wedge)


# Stage-1 uncertainty calibration.
#
# Convention: `err` is the SIGNED per-pixel error (gt - mean) and `sigma` the
# predictive std, both flattened 1D arrays over the breast mask. The
# calibration target is |err| explained by sigma_effective = sigma*sqrt(1+1/K)
# (the 1/K term accounts for the ensemble mean being estimated from the same
# K samples). Callers pass sigma_effective to these functions; the
# calibration_panel wrapper does this for you.

_SIGMA_FLOOR = 1e-8      # guard for z-scores on (near-)zero-spread pixels


def sigma_effective(std: np.ndarray, K: int) -> np.ndarray:
    """Predictive std of gt around the K-sample ensemble mean: std*sqrt(1+1/K)."""
    return std * np.sqrt(1.0 + 1.0 / max(int(K), 1))


def standardized_error_stats(err: np.ndarray, sigma: np.ndarray) -> dict:
    """
    z = err / sigma statistics. For a perfectly calibrated Gaussian ensemble
    z ~ N(0,1): mean 0, std 1, ~0.3% beyond |z|=3. Also reports the fraction
    of pixels at the sigma floor (degenerate spread), which are excluded from
    the z statistics rather than allowed to blow them up.
    """
    err = np.asarray(err, np.float64).ravel()
    sigma = np.asarray(sigma, np.float64).ravel()
    floor = sigma <= _SIGMA_FLOOR
    keep = ~floor
    if keep.sum() < 2:
        return {"z_mean": float("nan"), "z_std": float("nan"),
                "z_frac_gt3": float("nan"),
                "sigma_floor_frac": float(floor.mean())}
    z = err[keep] / sigma[keep]
    return {
        "z_mean": float(z.mean()),
        "z_std": float(z.std()),
        "z_frac_gt3": float((np.abs(z) > 3).mean()),
        "sigma_floor_frac": float(floor.mean()),
    }


def reliability(err: np.ndarray, sigma: np.ndarray, n_bins: int = 15) -> dict:
    """
    Reliability curve + expected calibration error for regression uncertainty.
    Pixels are binned by predicted sigma into equal-count (quantile) bins; per
    bin the empirical RMSE of err is compared to the mean predicted sigma.
    ECE = count-weighted mean |RMSE_b - mean_sigma_b| (deviation of the curve
    from the identity line, in [0,1] intensity units).

    Returns {"ece", "bin_sigma", "bin_rmse", "bin_count"} (arrays as lists so
    the result is JSON-serializable).
    """
    err = np.asarray(err, np.float64).ravel()
    sigma = np.asarray(sigma, np.float64).ravel()
    n = err.size
    if n < n_bins:
        return {"ece": float("nan"), "bin_sigma": [], "bin_rmse": [],
                "bin_count": []}
    order = np.argsort(sigma, kind="stable")
    e2 = err[order] ** 2
    s = sigma[order]
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    bin_sigma, bin_rmse, bin_count = [], [], []
    for b in range(n_bins):
        sel = slice(edges[b], edges[b + 1])
        cnt = edges[b + 1] - edges[b]
        if cnt == 0:
            continue
        bin_sigma.append(float(s[sel].mean()))
        bin_rmse.append(float(np.sqrt(e2[sel].mean())))
        bin_count.append(int(cnt))
    w = np.asarray(bin_count, np.float64) / n
    ece = float(np.sum(w * np.abs(np.asarray(bin_rmse) - np.asarray(bin_sigma))))
    return {"ece": ece, "bin_sigma": bin_sigma, "bin_rmse": bin_rmse,
            "bin_count": bin_count}


def coverage(err: np.ndarray, sigma: np.ndarray,
             quantiles=(0.5, 0.9, 0.95)) -> dict:
    """
    Empirical two-sided coverage: fraction of pixels with |err| <= z_q * sigma,
    where z_q is the Gaussian two-sided quantile (z_q = sqrt(2)*erfinv(q)).
    For calibrated uncertainty the empirical value matches the nominal q.
    Returns {q: empirical_coverage}. The Gaussianity assumption itself is
    reported separately via standardized_error_stats.
    """
    from scipy.special import erfinv
    err = np.abs(np.asarray(err, np.float64).ravel())
    sigma = np.asarray(sigma, np.float64).ravel()
    out = {}
    for q in quantiles:
        z_q = float(np.sqrt(2.0) * erfinv(q))
        out[float(q)] = float((err <= z_q * sigma).mean())
    return out


def ause(err: np.ndarray, sigma: np.ndarray, n_steps: int = 20) -> dict:
    """
    Sparsification / AUSE (Ilg et al., ECCV 2018) -- rank-based, hence
    distribution-free. Remove pixels in order of decreasing predicted sigma
    and track the RMSE of the remainder; the oracle removes by decreasing
    true |err|. Both curves are normalized by the full-set RMSE; AUSE is the
    mean gap between them over removed fractions f in [0, 1). 0 = the
    predicted sigma ranks errors exactly like the oracle.

    Returns {"ause", "fractions", "curve_pred", "curve_oracle"}.
    """
    err = np.asarray(err, np.float64).ravel()
    sigma = np.asarray(sigma, np.float64).ravel()
    n = err.size
    if n < max(n_steps, 4):
        return {"ause": float("nan"), "fractions": [], "curve_pred": [],
                "curve_oracle": []}
    e2 = err ** 2
    rmse_all = np.sqrt(e2.mean())
    if rmse_all <= 1e-30:
        return {"ause": 0.0, "fractions": [], "curve_pred": [],
                "curve_oracle": []}

    def curve(order):
        # order: removal order (first removed first). RMSE of the kept suffix.
        e2o = e2[order]
        suffix = np.cumsum(e2o[::-1])[::-1]           # suffix sums of e2
        fracs = np.linspace(0.0, 1.0, n_steps, endpoint=False)
        idx = (fracs * n).astype(int)                  # first kept index
        kept = n - idx
        rmse = np.sqrt(suffix[idx] / kept)
        return fracs, rmse / rmse_all

    fr, c_pred = curve(np.argsort(-sigma, kind="stable"))
    _, c_orac = curve(np.argsort(-e2, kind="stable"))
    return {
        "ause": float(np.mean(c_pred - c_orac)),
        "fractions": fr.tolist(),
        "curve_pred": c_pred.tolist(),
        "curve_oracle": c_orac.tolist(),
    }


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: weighted isotonic (nondecreasing) fit of y."""
    y = y.astype(np.float64).copy()
    w = w.astype(np.float64).copy()
    n = len(y)
    # blocks as (value, weight, count), merged while decreasing
    vals, wts, cnts = [], [], []
    for i in range(n):
        vals.append(y[i]); wts.append(w[i]); cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wts[-2] += wts[-1]; cnts[-2] += cnts[-1]
            vals[-2] = v
            vals.pop(); wts.pop(); cnts.pop()
    out = np.empty(n)
    i = 0
    for v, c in zip(vals, cnts):
        out[i:i + c] = v
        i += c
    return out


def fit_isotonic_recalibration(err: np.ndarray, sigma: np.ndarray,
                               n_bins: int = 25) -> dict:
    """
    Monotone recalibration map sigma -> sigma_cal ('isotonic' option):
    quantile-bin the predicted sigma, target each bin's
    empirical RMSE, enforce monotonicity with PAVA (count-weighted), and
    interpolate piecewise-linearly between bin mean sigmas. Fixes SHAPE
    miscalibration (e.g. heavy tails / nonlinear distortion) that a single
    scalar cannot -- fit on VAL, apply on TEST, same protocol as the scalar.
    Returns {"x": [...], "y": [...]} (JSON-serializable knots) for
    apply_recalibration.
    """
    err = np.asarray(err, np.float64).ravel()
    sigma = np.asarray(sigma, np.float64).ravel()
    n = err.size
    order = np.argsort(sigma, kind="stable")
    e2 = err[order] ** 2
    s = sigma[order]
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    bx, by, bw = [], [], []
    for b in range(n_bins):
        sel = slice(edges[b], edges[b + 1])
        cnt = edges[b + 1] - edges[b]
        if cnt == 0:
            continue
        bx.append(float(s[sel].mean()))
        by.append(float(np.sqrt(e2[sel].mean())))
        bw.append(float(cnt))
    y_iso = _pava(np.asarray(by), np.asarray(bw))
    return {"x": [float(v) for v in bx], "y": [float(v) for v in y_iso]}


def apply_recalibration(sigma: np.ndarray, iso: dict) -> np.ndarray:
    """Apply a fit_isotonic_recalibration map (linear interp, flat ends)."""
    return np.interp(np.asarray(sigma, np.float64).ravel(),
                     np.asarray(iso["x"]), np.asarray(iso["y"]))


def fit_scalar_recalibration(err: np.ndarray, sigma: np.ndarray) -> float:
    """
    Moment-matching scalar s for sigma -> s*sigma: s = sqrt(E[err^2]/E[sigma^2]).
    Fit on the VAL split, apply on TEST (calibration claims only
    post-recalibration-on-val). s > 1 means the
    ensemble is over-confident (spread too small), s < 1 under-confident.
    """
    err = np.asarray(err, np.float64).ravel()
    sigma = np.asarray(sigma, np.float64).ravel()
    ms2 = float(np.mean(sigma ** 2))
    if ms2 <= 1e-30:
        return float("nan")
    return float(np.sqrt(np.mean(err ** 2) / ms2))


def calibration_panel(gt: np.ndarray, mean: np.ndarray, std: np.ndarray,
                      mask: np.ndarray, K: int, n_bins: int = 15,
                      quantiles=(0.5, 0.9, 0.95), ause_steps: int = 20) -> dict:
    """
    One-slice calibration summary for evaluate.py: flat {metric: float} dict.
    gt/mean/std are (H, W) in [0,1]; std is the Bessel-corrected ensemble std
    of K samples; everything is computed inside the breast mask with
    sigma_effective = std*sqrt(1+1/K) as the predictive std.
    """
    m = mask.astype(bool)
    err = (gt[m] - mean[m]).astype(np.float64)
    sig = sigma_effective(std[m].astype(np.float64), K)

    out = {"uncertainty_mean": float(std[m].mean())}
    ae = np.abs(err)
    if sig.std() > 1e-8 and ae.std() > 1e-8:
        out["uncertainty_calib_corr"] = float(np.corrcoef(sig, ae)[0, 1])
    out["cal_ece"] = reliability(err, sig, n_bins=n_bins)["ece"]
    for q, c in coverage(err, sig, quantiles=quantiles).items():
        out[f"cal_coverage_{int(round(q * 100))}"] = c
    out["cal_ause"] = ause(err, sig, n_steps=ause_steps)["ause"]
    z = standardized_error_stats(err, sig)
    out["cal_z_std"] = z["z_std"]
    out["cal_z_frac_gt3"] = z["z_frac_gt3"]
    out["cal_sigma_floor_frac"] = z["sigma_floor_frac"]
    # Per-slice moment-matching scalar (diagnostic; the *protocol* scalar is
    # fitted on the pooled val split by scripts/analyze_results.py).
    out["cal_recal_scale"] = fit_scalar_recalibration(err, sig)
    return out


# Aggregation

def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0) -> dict:
    """Median, IQR and bootstrap CI of the median across patient-level values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"median": float("nan"), "iqr": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    meds = [np.median(rng.choice(v, size=len(v), replace=True))
            for _ in range(n_boot)]
    return {
        "median": float(np.median(v)),
        "iqr": float(np.subtract(*np.percentile(v, [75, 25]))),
        "ci_low": float(np.percentile(meds, 100 * alpha / 2)),
        "ci_high": float(np.percentile(meds, 100 * (1 - alpha / 2))),
        "n": int(len(v)),
    }
