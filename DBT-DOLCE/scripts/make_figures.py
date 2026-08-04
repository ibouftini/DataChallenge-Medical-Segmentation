"""
Paper figures from saved evaluation artifacts (CPU-only; no model, no LEAP).

Two COMPOSITE figures are built for the paper (few, dense, page-width):

  Figure "qualitative"  -> fig_qualitative.{pdf,png}
    (a) per-arm reconstructions + |error| grid (GT / SIRT / pure prior /
        exact ... same slice, identical anatomy by the seeded strided runs)
    (b) trust map: ensemble mean | recalibrated sigma (isotonic) | true |error|
    one representative slice (median exact PSNR by default; --select worst for
    the failure-case panel the discussion needs).

  Figure "calibration" -> fig_calibration.pdf
    (a) reliability diagram, exact arm raw / scalar / isotonic (fit on val)
    (b) sparsification (AUSE), exact vs pure prior + oracle
    (c) rho sweep (optional): SSIM vs rho, every setting vs the no-DC / SIRT
        references -- the proximal-fragility panel

Examples:
  python scripts/make_figures.py --figure qualitative \
      --compare none=results/dbt_25deg/slices_none_test \
                exact=results/dbt_25deg/slices_exact_test \
      --calib_val results/dbt_25deg/calib_exact_val \
      --out results/dbt_25deg/figures

  python scripts/make_figures.py --figure calibration \
      --exact_val  results/dbt_25deg/calib_exact_val \
      --exact_test results/dbt_25deg/calib_exact_test \
      --none_val   results/dbt_25deg/calib_none_val \
      --none_test  results/dbt_25deg/calib_none_test \
      --sweep results/rho_sweep/sweep.json \
      --out results/dbt_25deg/figures

The legacy per-slice helpers (--slices, --compare alone) are kept below.
"""

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import eval_metrics as M
from analyze_results import load_calib_dir


# CVD-validated categorical palette (dataviz skill, light mode). Colour follows
# the ENTITY, fixed, never cycled. green/orange kept non-adjacent (their pair is
# only in the 8-12 CVD floor band) and every series is also legend/label-backed.
ARM_COLOR = {"exact": "#2a78d6", "none": "#eb6834", "prox": "#008300",
             "sirt": "#7a7a76", "gt": "#0b0b0b"}
RECAL_COLOR = {"raw": "#e34948", "scalar": "#2a78d6", "isotonic": "#008300"}
INK = "#0b0b0b"
MUTED = "#52514e"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Publication style: recessive axes/grid, thin marks, no chartjunk.
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.labelcolor": MUTED,
        "legend.fontsize": 7.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": "#c9c8c3",     # recessive spines
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#c9c8c3",
        "grid.alpha": 0.35,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.8,
        "image.interpolation": "nearest",
    })
    return plt


def _panel_label(fig_or_ax, s):
    # bold panel tag at the top-left of an axis (in axis fraction coords)
    fig_or_ax.text(0.0, 1.02, s, transform=fig_or_ax.transAxes, fontsize=11,
                   fontweight="bold", va="bottom", ha="left", color=INK)


def _load_arm_slices(compare):
    """compare: ['label=dir', ...] -> {label: {stem: path}} preserving order."""
    arms = {}
    for spec in compare:
        lab, _, d = spec.partition("=")
        arms[lab] = {p.stem: p for p in sorted(Path(d).glob("slice_*.npz"))}
    return arms


def _norm_stem(s):
    """'3' / '0003' / 'slice_0003' -> 'slice_0003'."""
    s = str(s)
    if s.startswith("slice_"):
        return s
    return f"slice_{int(s):04d}"


def _scored_stems(arms, ref_arm):
    """[(psnr, stem)] over slices common to all arms, sorted by PSNR asc."""
    common = sorted(set.intersection(*[set(v) for v in arms.values()]))
    ref = arms.get(ref_arm) or next(iter(arms.values()))
    scored = []
    for stem in common:
        z = np.load(ref[stem])
        m = z["mask"].astype(bool)
        scored.append((M.psnr_masked(z["gt"], z["pred"], m), stem))
    scored.sort()
    return scored


def _pick_stem(arms, select, ref_arm, explicit=""):
    """Common slice stem: explicit override, else worst / median PSNR."""
    scored = _scored_stems(arms, ref_arm)
    if not scored:
        return None
    if explicit:
        want = _norm_stem(explicit)
        stems = {s for _, s in scored}
        if want not in stems:
            raise SystemExit(f"slice {want} not common to all arms; "
                             f"available: {sorted(stems)}")
        return want
    if select == "worst":
        return scored[0][1]                       # lowest PSNR = failure case
    return scored[len(scored) // 2][1]            # median = representative


# ---------------------------------------------------------------------
# Figure: qualitative (reconstructions + trust map), one page-width block
# ---------------------------------------------------------------------
def list_slices(compare, ref_arm="exact"):
    """Print the slices common to all arms with the ref-arm PSNR, to choose."""
    arms = _load_arm_slices(compare)
    if not arms:
        print("(no --compare arm dirs given)")
        return
    scored = _scored_stems(arms, ref_arm)
    if not scored:
        print("(no common slice index across arms)")
        return
    print(f"# available slices (common to {list(arms)}), {ref_arm} PSNR:")
    for psnr, stem in sorted(scored, key=lambda t: t[1]):
        print(f"  {stem}   PSNR={psnr:6.2f} dB")
    med = scored[len(scored) // 2][1]
    print(f"# default (--select median) -> {med};  "
          f"worst -> {scored[0][1]};  pick with --slice N")


def _bare(ax):
    """Turn an axis into a clean image tile: no ticks, no spines."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def figure_qualitative(compare, out_dir, calib_val="", K=8, select="median",
                       ref_arm="exact", slice_id="", fmt=("pdf", "png")):
    plt = _mpl()
    arms = _load_arm_slices(compare)
    if not arms:
        print("(no --compare arm dirs given)")
        return
    stem = _pick_stem(arms, select, ref_arm, explicit=slice_id)
    if stem is None:
        print("(no common slice index across arms)")
        return

    ref = np.load(next(iter(arms.values()))[stem])
    gt, mask = ref["gt"], ref["mask"].astype(bool)
    cols = [("GT", gt), ("SIRT", ref["sirt"])]
    for lab, d in arms.items():
        cols.append((lab, np.load(d[stem])["pred"]))
    ncol = len(cols)

    err_imgs = {t: np.abs(gt - im) * mask for t, im in cols if t != "GT"}
    err_vmax = float(np.percentile(np.concatenate(
        [e[mask] for e in err_imgs.values()]), 99)) or 1.0

    ex = np.load((arms.get(ref_arm) or next(iter(arms.values())))[stem])
    have_unc = "std" in ex
    if have_unc:
        sigma = M.sigma_effective(ex["std"], K) * mask
        if calib_val:
            e_v, s_v, _ = load_calib_dir(calib_val)
            iso = M.fit_isotonic_recalibration(e_v, s_v)
            sigma = M.apply_recalibration(sigma.ravel(), iso).reshape(sigma.shape) * mask
        terr = np.abs(gt - ex["pred"]) * mask
        tv = float(np.percentile(np.concatenate([sigma[mask], terr[mask]]), 99)) or 1.0

    # ImageGrid packs equal image tiles tightly with a shared colorbar -- the
    # right tool for image panels (constrained_layout spreads fixed-aspect
    # images apart). Two blocks: (a) recon+error grid, (b) trust map.
    from mpl_toolkits.axes_grid1 import ImageGrid
    tile = 1.32
    nrow = 2 + (1 if have_unc else 0)
    fig = plt.figure(figsize=(ncol * tile + 0.6, nrow * tile + 0.7))
    gs = fig.add_gridspec(2 if have_unc else 1, 1,
                          height_ratios=[2, 1.02] if have_unc else [1],
                          hspace=0.16)

    ga = ImageGrid(fig, gs[0], nrows_ncols=(2, ncol), axes_pad=0.05,
                   cbar_mode="edge", cbar_location="right",
                   cbar_size="4%", cbar_pad=0.06)
    for k, (title, img) in enumerate(cols):
        a0 = ga[k]                                   # row 0: reconstructions
        a0.imshow(img, cmap="gray", vmin=0, vmax=1)
        p = None if title == "GT" else M.psnr_masked(gt, img, mask)
        t = title if p is None else f"{title}  {p:.1f} dB"
        a0.set_title(("(a)   " + t) if k == 0 else t, fontsize=8.5)
        _bare(a0)
        a1 = ga[ncol + k]                            # row 1: |error| / mask
        if title == "GT":
            a1.imshow(mask, cmap="gray")
            a1.set_ylabel("mask $\\;$/$\\;$ $|$error$|$", color=MUTED, fontsize=8)
        else:
            im = a1.imshow(err_imgs[title], cmap="magma", vmin=0, vmax=err_vmax)
        _bare(a1)
    ga.cbar_axes[0].set_visible(False)               # no bar for the gray row
    cb = ga.cbar_axes[1].colorbar(im)
    cb.set_label("abs. error", fontsize=8)

    if have_unc:
        gb = ImageGrid(fig, gs[1], nrows_ncols=(1, 3), axes_pad=0.05,
                       cbar_mode="single", cbar_location="right",
                       cbar_size="4%", cbar_pad=0.06)
        tpanels = [(ex["pred"], "ensemble mean", "gray", (0, 1)),
                   (sigma, "recalibrated $\\sigma$", "magma", (0, tv)),
                   (terr, "true $|$error$|$", "magma", (0, tv))]
        for k, (img, title, cmap, rng) in enumerate(tpanels):
            a = gb[k]
            im2 = a.imshow(img, cmap=cmap, vmin=rng[0], vmax=rng[1])
            a.set_title(("(b)   " + title) if k == 0 else title, fontsize=8.5)
            _bare(a)
        gb.cbar_axes[0].colorbar(im2)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tag = f"_{select}" if select != "median" else ""
    for ext in fmt:
        p = Path(out_dir) / f"fig_qualitative{tag}.{ext}"
        fig.savefig(p)
        print(f"wrote {p}  (slice {stem}, arms={list(arms)})")
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure: calibration (reliability + sparsification + rho sweep)
# ---------------------------------------------------------------------
def figure_calibration(exact_val, exact_test, none_val="", none_test="",
                       sweep="", out_dir=".", fmt=("pdf",)):
    plt = _mpl()
    e_te, s_te, _ = load_calib_dir(exact_test)
    e_va, s_va, _ = load_calib_dir(exact_val)
    scale = M.fit_scalar_recalibration(e_va, s_va)
    iso = M.fit_isotonic_recalibration(e_va, s_va)

    npanel = 2 + (1 if sweep else 0)
    fig, ax = plt.subplots(1, npanel, figsize=(3.5 * npanel, 3.4),
                           layout="constrained")
    if npanel == 1:
        ax = [ax]

    def below_legend(a, ncol):
        a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol,
                 frameon=False, columnspacing=1.1, handlelength=1.5,
                 handletextpad=0.4)

    # (a) reliability: raw / scalar / isotonic on the exact arm (fit on val)
    variants = [("raw", s_te, RECAL_COLOR["raw"]),
                (f"scalar (s={scale:.2f})", scale * s_te, RECAL_COLOR["scalar"]),
                ("isotonic", M.apply_recalibration(s_te, iso), RECAL_COLOR["isotonic"])]
    a = ax[0]
    for name, s, c in variants:
        r = M.reliability(e_te, s)
        a.plot(r["bin_sigma"], r["bin_rmse"], "o-", ms=3.5, color=c,
               label=f"{name}  ECE={r['ece']:.3f}")
    lim = max(a.get_xlim()[1], a.get_ylim()[1])
    a.plot([0, lim], [0, lim], "--", color=MUTED, lw=0.9, label="identity")
    a.set_xlim(0, lim); a.set_ylim(0, lim)
    a.set_xlabel("predicted $\\sigma$ (binned)")
    a.set_ylabel("empirical RMSE")
    a.set_title("reliability (test)")
    a.set_aspect("equal", "box")
    below_legend(a, 2)
    _panel_label(a, "(a)")

    # (b) sparsification: exact vs none + oracle (colour follows the arm)
    a = ax[1]
    rp = M.ause(e_te, s_te)
    a.plot(rp["fractions"], rp["curve_pred"], color=ARM_COLOR["exact"],
           label=f"exact  (AUSE={rp['ause']:.3f})")
    a.plot(rp["fractions"], rp["curve_oracle"], "--", color=ARM_COLOR["exact"],
           lw=1, alpha=0.55, label="exact oracle")
    if none_test:
        en, sn, _ = load_calib_dir(none_test)
        rn = M.ause(en, sn)
        a.plot(rn["fractions"], rn["curve_pred"], color=ARM_COLOR["none"],
               label=f"pure prior  (AUSE={rn['ause']:.3f})")
    a.set_xlim(0, 1); a.set_ylim(0, 1.02)
    a.set_xlabel("fraction of pixels removed")
    a.set_ylabel("RMSE of remainder (norm.)")
    a.set_title("sparsification (test)")
    below_legend(a, 1)
    _panel_label(a, "(b)")

    # (c) rho sweep (optional): SSIM vs rho, with no-DC / SIRT references
    if sweep:
        a = ax[2]
        sw = json.load(open(sweep))
        res = sw["results"]
        rho_res = [r for r in res if r["rho"] != "no_prox"]  # curve = rho only
        labels = [r["rho"] for r in rho_res]
        ssim = [r["ssim"] for r in rho_res]
        x = list(range(len(labels)))
        a.plot(x, ssim, "o-", ms=3.5, color=ARM_COLOR["prox"], label="tuned prox")
        noprox = next((r["ssim"] for r in res if r["rho"] == "no_prox"), None)
        if noprox is not None:
            a.axhline(noprox, ls="--", color=ARM_COLOR["none"], lw=1, label="no DC")
        if "sirt" in sw:
            a.axhline(sw["sirt"]["ssim"], ls=":", color=INK, lw=1, label="SIRT")
        a.set_xlim(-0.4, (x[-1] if x else 0) + 0.4)
        a.set_xticks(x)
        a.set_xticklabels(labels, rotation=45, ha="right")
        a.set_xlabel("$\\rho$ schedule")
        a.set_ylabel("SSIM")
        a.set_title("proximal $\\rho$ sweep")
        below_legend(a, 3)
        _panel_label(a, "(c)")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for ext in fmt:
        p = Path(out_dir) / f"fig_calibration.{ext}"
        fig.savefig(p)
        print(f"wrote {p}")
    plt.close(fig)


# ---------------------------------------------------------------------
# Legacy per-slice helpers (kept)
# ---------------------------------------------------------------------
def trust_map_figs(slices_dir, out_dir, calib_val="", K=8):
    plt = _mpl()
    files = sorted(Path(slices_dir).glob("slice_*.npz"))
    iso = None
    if calib_val:
        e_val, s_val, _ = load_calib_dir(calib_val)
        iso = M.fit_isotonic_recalibration(e_val, s_val)
    made = 0
    for f in files:
        z = np.load(f)
        if "std" not in z:
            continue
        gt, pred, std, mask = z["gt"], z["pred"], z["std"], z["mask"].astype(bool)
        err = np.abs(gt - pred) * mask
        sigma = M.sigma_effective(std, K) * mask
        panels = [(pred, "ensemble mean", "gray", (0, 1)),
                  (sigma, "raw trust map $\\sigma$", "magma", None),
                  (err, "true $|$error$|$", "magma", None)]
        if iso is not None:
            sig_iso = M.apply_recalibration(sigma.ravel(), iso).reshape(sigma.shape) * mask
            panels.insert(2, (sig_iso, "recalibrated $\\sigma$ (isotonic)", "magma", None))
        vmax = max(float(p[0].max()) for p in panels[1:])
        fig, ax = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
        for a, (img, title, cmap, rng) in zip(ax, panels):
            vmin, vm = rng if rng else (0, vmax)
            im = a.imshow(img, cmap=cmap, vmin=vmin, vmax=vm)
            a.set_title(title); a.axis("off")
            if rng is None:
                fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = Path(out_dir) / f"trustmap_{f.stem.split('_')[-1]}.png"
        fig.savefig(out, dpi=150); plt.close(fig); made += 1
        print(f"wrote {out}")
    if made == 0:
        print(f"(no npz with an ensemble 'std' in {slices_dir})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", choices=["qualitative", "calibration"],
                    default="", help="build a composite paper figure")
    ap.add_argument("--compare", nargs="+", default=[], metavar="LABEL=DIR",
                    help="arm slice dirs (qualitative figure / legacy)")
    ap.add_argument("--select", choices=["median", "worst"], default="median",
                    help="representative (median PSNR) or failure-case slice")
    ap.add_argument("--slice", dest="slice_id", default="",
                    help="pick a specific slice (e.g. 3, 0003, slice_0003); "
                         "overrides --select")
    ap.add_argument("--list", action="store_true",
                    help="list slices common to the --compare arms (with PSNR) "
                         "and exit -- use to choose --slice")
    ap.add_argument("--ref_arm", default="exact",
                    help="arm used for the trust-map row + slice selection")
    ap.add_argument("--calib_val", default="",
                    help="calib VAL dir to fit the isotonic map (trust map)")
    ap.add_argument("--exact_val", default="")
    ap.add_argument("--exact_test", default="")
    ap.add_argument("--none_val", default="")
    ap.add_argument("--none_test", default="")
    ap.add_argument("--sweep", default="", help="results/rho_sweep/sweep.json")
    ap.add_argument("--slices", default="", help="legacy per-slice trust maps")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--out", default="", help="output dir")
    a = ap.parse_args()

    if a.figure == "qualitative":
        if a.list:
            list_slices(a.compare, ref_arm=a.ref_arm)
            return
        out = a.out or (Path(a.compare[0].partition("=")[2]).parent / "figures")
        figure_qualitative(a.compare, out, calib_val=a.calib_val, K=a.K,
                            select=a.select, ref_arm=a.ref_arm,
                            slice_id=a.slice_id)
        return
    if a.figure == "calibration":
        out = a.out or "figures"
        figure_calibration(a.exact_val, a.exact_test, a.none_val, a.none_test,
                           sweep=a.sweep, out_dir=out)
        return

    if a.slices:
        out = Path(a.out or Path(a.slices).parent / "figures")
        trust_map_figs(a.slices, out, calib_val=a.calib_val, K=a.K)
    elif a.compare:
        # legacy side-by-side (kept for back-compat)
        figure_qualitative(a.compare, a.out or "figures", calib_val=a.calib_val,
                           K=a.K, select=a.select, ref_arm=a.ref_arm)
    else:
        ap.error("pass --figure {qualitative,calibration}, or --slices")


if __name__ == "__main__":
    main()
