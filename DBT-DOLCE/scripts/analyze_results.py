"""
Turn evaluate.py's metrics.json (and optionally sweep_rho.py's sweep.json) into
a results table + figures, so the end-of-run analysis is one command.

    python scripts/analyze_results.py --metrics results/dbt_25deg/metrics.json
    python scripts/analyze_results.py --metrics ... --sweep results/rho_sweep/sweep.json

Writes, next to the metrics file:
  summary.md   -- markdown table (median [95% CI]) for every metric, with the
                  recon-vs-SIRT delta on the key ones.
  summary.png  -- recon vs SIRT bar chart (PSNR/SSIM) with CI error bars, and
                  the FRC depth-anisotropy and per-tissue MAE panels.
  sweep.png    -- (if --sweep given) SSIM/PSNR vs rho with the SIRT line.

Multi-arm comparison + Stage-1 calibration panel:

    python scripts/analyze_results.py \
        --arm none=results/dbt_25deg/metrics_none_test.json \
        --arm prox=results/dbt_25deg/metrics_prox_test.json \
        --arm exact=results/dbt_25deg/metrics_exact_test.json \
        --calib_val  results/dbt_25deg/calib_exact_val \
        --calib_test results/dbt_25deg/calib_exact_test

  arms.md          -- side-by-side median [95% CI] table across the arms
                      (the H1/H2/H3 evidence table).
  calibration.md   -- ECE / coverage / AUSE on the test pairs, before and
                      after the scalar recalibration fitted on the VAL pairs
                      (the honest protocol: fit on val, report on test).
                      Per-patient aggregation with bootstrap CIs, like
                      everything else.
  calibration.png  -- reliability diagram + sparsification curves (test,
                      before/after recalibration).

The calib_* directories are written by `evaluate.py --n_samples K` (K > 1):
one npz of masked (error, sigma_effective) pairs per slice.
"""

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import eval_metrics as M


def load(p):
    with open(p) as f:
        return json.load(f)


def fmt(d):
    if not isinstance(d, dict):
        return str(d)
    return f"{d.get('median', float('nan')):.4f} [{d.get('ci_low', float('nan')):.4f}, {d.get('ci_high', float('nan')):.4f}]"


def write_summary_md(summary, path, cfg):
    lines = ["# Evaluation summary", ""]
    lines.append(f"Config: `{cfg}`")
    lines.append("")
    lines.append("| Metric | median [95% CI] |")
    lines.append("|--------|-----------------|")
    for k in sorted(summary):
        lines.append(f"| {k} | {fmt(summary[k])} |")
    # Key deltas vs SIRT baseline.
    lines.append("")
    lines.append("## Recon vs SIRT (the headline)")
    for m in ("psnr", "ssim"):
        r = summary.get(m, {}).get("median")
        s = summary.get(f"sirt_{m}", {}).get("median")
        if r is not None and s is not None:
            delta = r - s
            verdict = "recon WINS" if delta > 0 else "SIRT wins"
            lines.append(f"- **{m.upper()}**: recon {r:.4f} vs SIRT {s:.4f}  "
                         f"(delta {delta:+.4f}, {verdict})")
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def make_figure(summary, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(figure skipped: {e})")
        return

    def med_ci(k):
        d = summary.get(k, {})
        m = d.get("median", float("nan"))
        lo = d.get("ci_low", m); hi = d.get("ci_high", m)
        return m, [[m - lo], [hi - m]]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel 1: recon vs SIRT for PSNR and SSIM (twin axis).
    labels = ["recon", "SIRT"]
    p_r, p_re = med_ci("psnr"); p_s, p_se = med_ci("sirt_psnr")
    s_r, s_re = med_ci("ssim"); s_s, s_se = med_ci("sirt_ssim")
    a0 = ax[0]; a0b = a0.twinx()
    a0.bar([0, 1], [p_r, p_s], width=0.35, color="tab:blue", alpha=0.7,
           yerr=[[e[0] for e in (p_re, p_se)], [e[1] for e in (p_re, p_se)]], capsize=4)
    a0b.bar([0.4, 1.4], [s_r, s_s], width=0.35, color="tab:orange", alpha=0.7,
            yerr=[[e[0] for e in (s_re, s_se)], [e[1] for e in (s_re, s_se)]], capsize=4)
    a0.set_xticks([0.2, 1.2]); a0.set_xticklabels(labels)
    a0.set_ylabel("PSNR (dB)", color="tab:blue")
    a0b.set_ylabel("SSIM", color="tab:orange")
    a0.set_title("recon vs SIRT")

    # Panel 2: FRC directional resolution (x vs z = depth).
    rx, rxe = med_ci("frc_res_x_mm"); rz, rze = med_ci("frc_res_z_mm")
    ax[1].bar([0, 1], [rx, rz], color=["tab:green", "tab:red"], alpha=0.7,
              yerr=[[rxe[0][0], rze[0][0]], [rxe[1][0], rze[1][0]]], capsize=4)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["x (in-plane)", "z (depth)"])
    ax[1].set_ylabel("FRC resolution (mm, lower=better)")
    ax[1].set_title("directional resolution (depth blur = limited-angle wall)")

    # Panel 3: per-tissue MAE.
    tissues = [("mae_adipose", "adipose"), ("mae_fibroglandular", "fibro"),
               ("mae_skin", "skin")]
    vals, errs, names = [], [[], []], []
    for k, nm in tissues:
        if k in summary:
            m, e = med_ci(k); vals.append(m)
            errs[0].append(e[0][0]); errs[1].append(e[1][0]); names.append(nm)
    if vals:
        ax[2].bar(range(len(vals)), vals, color="tab:purple", alpha=0.7,
                  yerr=errs, capsize=4)
        ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names)
        ax[2].set_ylabel("MAE (normalised)")
        ax[2].set_title("per-tissue attenuation error")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def make_sweep_fig(sweep, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(sweep figure skipped: {e})")
        return
    res = sweep["results"]; sirt = sweep.get("sirt", {})
    labels = [r["rho"] for r in res]; xs = range(len(labels))
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(xs, [r["ssim"] for r in res], "o-", color="tab:orange", label="SSIM")
    if sirt:
        ax1.axhline(sirt["ssim"], ls="--", color="gray",
                    label=f"SIRT SSIM {sirt['ssim']:.3f}")
    ax1.set_xticks(list(xs)); ax1.set_xticklabels(labels, rotation=30)
    ax1.set_ylabel("SSIM"); ax1.set_xlabel("rho setting")
    ax2 = ax1.twinx()
    ax2.plot(xs, [r["psnr"] for r in res], "s-", color="tab:blue", alpha=0.6)
    ax2.set_ylabel("PSNR (dB)", color="tab:blue")
    ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"wrote {path}")


# Multi-arm comparison

def write_arms_md(arms: dict, path):
    """arms: {label: loaded metrics.json}. Side-by-side median [95% CI]."""
    labels = list(arms)
    keys = sorted({k for m in arms.values()
                   for k in m.get("summary_per_patient", {})})
    lines = ["# Arms comparison", ""]
    for lab, m in arms.items():
        lines.append(f"- **{lab}**: `{m.get('config', {})}`")
    lines += ["", "| Metric | " + " | ".join(labels) + " |",
              "|--------|" + "|".join(["-" * 8] * len(labels)) + "|"]
    for k in keys:
        row = [fmt(arms[lab].get("summary_per_patient", {}).get(k, float("nan")))
               for lab in labels]
        lines.append(f"| {k} | " + " | ".join(row) + " |")
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


# Stage-1 calibration panel

def load_calib_dir(d):
    """
    Load every calib_*.npz written by evaluate.py --n_samples K.
    Returns (err, sigma, patients) with per-pixel arrays concatenated and a
    parallel per-pixel patient id array (for per-patient aggregation).
    """
    files = sorted(Path(d).glob("calib_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no calib_*.npz in {d} -- run evaluate.py with --n_samples > 1.")
    errs, sigs, pats = [], [], []
    for f in files:
        z = np.load(f)
        errs.append(z["err"].astype(np.float64))
        sigs.append(z["sigma"].astype(np.float64))
        pats.append(np.full(z["err"].shape[0], str(z["patient"])))
    return np.concatenate(errs), np.concatenate(sigs), np.concatenate(pats)


def _panel_per_patient(err, sigma, patients, transform=None):
    """{metric: bootstrap_ci over patients} for ECE/coverage/AUSE; transform
    is an optional sigma -> sigma_cal map (identity when None)."""
    per = {}
    for p in np.unique(patients):
        s = patients == p
        e = err[s]
        g = transform(sigma[s]) if transform is not None else sigma[s]
        row = {"ece": M.reliability(e, g)["ece"],
               "ause": M.ause(e, g)["ause"]}
        for q, c in M.coverage(e, g).items():
            row[f"coverage_{int(round(q * 100))}"] = c
        per[p] = row
    keys = sorted(next(iter(per.values())))
    return {k: M.bootstrap_ci([per[p][k] for p in per]) for k in keys}


def write_calibration_report(val_dir, test_dir, out_md, out_png):
    """
    The calibration protocol: fit the recalibration maps on the pooled VAL
    pairs, report ECE/coverage/AUSE on TEST raw / scalar / isotonic.
    Calibration claims are only made post-recalibration-on-val.
    The scalar fixes the second moment; the isotonic map additionally fixes
    SHAPE miscalibration (the heavy-tail pattern the pilot measured:
    coverage@90 near nominal but coverage@50 far below).
    """
    e_val, s_val, _ = load_calib_dir(val_dir)
    e_tst, s_tst, p_tst = load_calib_dir(test_dir)
    scale = M.fit_scalar_recalibration(e_val, s_val)
    iso = M.fit_isotonic_recalibration(e_val, s_val)

    variants = [
        ("raw", None),
        (f"scalar (s={scale:.3f})", lambda g: scale * g),
        ("isotonic", lambda g: M.apply_recalibration(g, iso)),
    ]
    panels = {lab: _panel_per_patient(e_tst, s_tst, p_tst, transform=tr)
              for lab, tr in variants}
    z_stats = {lab: M.standardized_error_stats(
                   e_tst, tr(s_tst) if tr is not None else s_tst)
               for lab, tr in variants}

    labels = [lab for lab, _ in variants]
    lines = ["# Stage-1 calibration panel (test split)", "",
             f"- val pairs:  `{val_dir}`  ({e_val.size} px)",
             f"- test pairs: `{test_dir}`  ({e_tst.size} px, "
             f"{len(np.unique(p_tst))} patients)",
             f"- recalibration fitted on val: scalar **s = {scale:.4f}** "
             f"(s > 1: over-confident); isotonic map with "
             f"{len(iso['x'])} knots", "",
             "| Metric | " + " | ".join(labels) + " |",
             "|--------|" + "|".join(["-" * 8] * len(labels)) + "|"]
    for k in sorted(panels[labels[0]]):
        lines.append("| " + k + " | "
                     + " | ".join(fmt(panels[lab][k]) for lab in labels) + " |")
    lines += ["", "Standardized errors z = err/sigma (Gaussianity check): "
              + ";  ".join(f"{lab}: std={z_stats[lab]['z_std']:.3f}, "
                           f"|z|>3={z_stats[lab]['z_frac_gt3']:.4f}"
                           for lab in labels)
              + f".  sigma-floor frac={z_stats['raw']['sigma_floor_frac']:.4f}."]
    Path(out_md).write_text("\n".join(lines) + "\n")
    print(f"wrote {out_md}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(calibration figure skipped: {e})")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for (lab, tr), col in zip(variants, ("tab:red", "tab:blue", "tab:green")):
        g = tr(s_tst) if tr is not None else s_tst
        r = M.reliability(e_tst, g)
        ax[0].plot(r["bin_sigma"], r["bin_rmse"], "o-", color=col,
                   label=f"{lab}  ECE={r['ece']:.4f}")
    lim = max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])
    ax[0].plot([0, lim], [0, lim], "k--", alpha=0.5, label="identity")
    ax[0].set_xlabel("predicted sigma (binned)")
    ax[0].set_ylabel("empirical RMSE")
    ax[0].set_title("reliability diagram (test)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    a_res = M.ause(e_tst, s_tst)   # rank-based: scale-invariant, one curve
    ax[1].plot(a_res["fractions"], a_res["curve_pred"], "-",
               color="tab:blue", label=f"by predicted sigma (AUSE={a_res['ause']:.4f})")
    ax[1].plot(a_res["fractions"], a_res["curve_oracle"], "--",
               color="k", alpha=0.6, label="oracle (by |err|)")
    ax[1].set_xlabel("fraction of pixels removed")
    ax[1].set_ylabel("RMSE of remainder (normalized)")
    ax[1].set_title("sparsification (test)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="", help="results/.../metrics.json")
    ap.add_argument("--sweep", default="", help="results/rho_sweep/sweep.json (optional)")
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=PATH",
                    help="repeatable: metrics.json per arm for the "
                         "side-by-side table, e.g. exact=.../metrics_exact_test.json")
    ap.add_argument("--calib_val", default="",
                    help="calib_* dir (val split) to FIT the recalibration on")
    ap.add_argument("--calib_test", default="",
                    help="calib_* dir (test split) to REPORT calibration on")
    ap.add_argument("--out_dir", default="",
                    help="output dir (default: next to the first input)")
    a = ap.parse_args()
    if not (a.metrics or a.arm or a.calib_test):
        ap.error("nothing to do: pass --metrics, --arm, and/or "
                 "--calib_val/--calib_test")

    if a.metrics:
        m = load(a.metrics)
        summary = m.get("summary_per_patient", {})
        out = Path(a.out_dir or Path(a.metrics).parent)
        write_summary_md(summary, out / "summary.md", m.get("config", {}))
        make_figure(summary, out / "summary.png")
        if a.sweep:
            make_sweep_fig(load(a.sweep), out / "sweep.png")

    if a.arm:
        arms = {}
        for spec in a.arm:
            lab, _, p = spec.partition("=")
            if not p:
                ap.error(f"--arm expects LABEL=PATH, got {spec!r}")
            arms[lab] = load(p)
        out = Path(a.out_dir or Path(a.arm[0].partition("=")[2]).parent)
        write_arms_md(arms, out / "arms.md")

    if a.calib_test:
        if not a.calib_val:
            ap.error("--calib_test needs --calib_val (the protocol fits the "
                     "recalibration on val and reports on test; use the same "
                     "dir for both ONLY for a self-check, never for a claim)")
        out = Path(a.out_dir or Path(a.calib_test).parent)
        write_calibration_report(a.calib_val, a.calib_test,
                                 out / "calibration.md", out / "calibration.png")


if __name__ == "__main__":
    main()
