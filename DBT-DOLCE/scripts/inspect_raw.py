"""
Diagnose the raw phantom data and the resize step.

Answers two open questions before committing to a full re-preprocess + train:

  1. LABEL MAPPING -- are the raw .nrrd values the assumed integer labels
     (0=air, 1=adipose, 2=fibroglandular, 3=skin), and is the breast mostly
     adipose (expected) or mostly fibroglandular (suspicious / likely a
     swapped mapping)?

  2. ALIASING -- does downsampling a native slice to 512 with INTER_LINEAR
     produce the moire / concentric-ring texture we saw, and does INTER_AREA
     (the fix in data/preprocess.py) remove it?

For each of the first --n phantoms it prints the dtype/shape and the value
histogram, reports the in-breast tissue composition, and writes a comparison
figure: label slice | attenuation (native) | INTER_LINEAR 512 | INTER_AREA 512.

    python scripts/inspect_raw.py --raw_dir dataset/compressed --n 3
"""

import sys
import argparse
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import nrrd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.preprocess import (
    canonicalise_axes, labels_to_attenuation, normalise_gt, pad_to_square,
    ATTENUATION, ATTEN_MAX,
)

TISSUE = {0: "air", 1: "adipose", 2: "fibroglandular", 3: "skin"}


def inspect_one(path: str, out_dir: Path, y_frac: float):
    data, header = nrrd.read(path)
    name = Path(path).stem
    print(f"\n=== {name} ===")
    print(f"  dtype={data.dtype}  shape={data.shape}  "
          f"min={data.min()}  max={data.max()}")

    # Value histogram -- are these clean integer labels?
    u, c = np.unique(data, return_counts=True)
    n_unique = u.size
    print(f"  unique values: {n_unique}")
    order = np.argsort(c)[::-1]
    for idx in order[:8]:
        val = u[idx]
        pct = 100.0 * c[idx] / data.size
        tag = f"  <- {TISSUE.get(int(val), '?')}" if float(val).is_integer() else ""
        print(f"    value {val}: {pct:5.1f}%{tag}")
    looks_label = n_unique <= 8 and np.allclose(u, np.round(u))
    print(f"  looks like an integer label map: {looks_label}")

    # In-breast tissue composition (labels > 0)
    if looks_label:
        breast = data[data > 0]
        if breast.size:
            print("  in-breast composition:")
            for lab in (1, 2, 3):
                frac = 100.0 * (breast == lab).mean()
                print(f"    {TISSUE[lab]:<15s}: {frac:5.1f}%")
            adipose = 100.0 * (breast == 1).mean()
            fibro   = 100.0 * (breast == 2).mean()
            if fibro > adipose:
                print("  >> NOTE: breast is mostly FIBROGLANDULAR. Real breasts "
                      "are mostly adipose -- the label mapping may be swapped.")

        # Objective swap test: subcutaneous adipose hugs the breast boundary,
        # fibroglandular is central. Compare each label's mean depth (distance
        # from the breast boundary); the more PERIPHERAL label is the real
        # adipose. Uses the whole volume so it is robust.
        try:
            from scipy.ndimage import distance_transform_edt
            breast_mask = data > 0
            depth = distance_transform_edt(breast_mask)      # 0 at boundary, large in centre
            d1 = depth[data == 1].mean() if (data == 1).any() else float("nan")
            d2 = depth[data == 2].mean() if (data == 2).any() else float("nan")
            print(f"  mean depth-from-boundary: label1={d1:.2f}  label2={d2:.2f} "
                  f"(larger = more central)")
            if np.isfinite(d1) and np.isfinite(d2):
                peripheral = 1 if d1 < d2 else 2
                central    = 2 if peripheral == 1 else 1
                print(f"  >> spatially, label {peripheral} is peripheral (=adipose) "
                      f"and label {central} is central (=fibroglandular).")
                if peripheral == 2:
                    print("  >> This CONTRADICTS the assumed mapping (1=adipose, "
                          "2=fibroglandular): the labels appear SWAPPED.")
                else:
                    print("  >> Consistent with the assumed mapping "
                          "(1=adipose, 2=fibroglandular).")
        except Exception as e:
            print(f"  (spatial swap test skipped: {e})")

    # Canonicalise and take a representative sagittal slice.
    vol = canonicalise_axes(data, header)
    labels = vol.astype(np.uint8)
    atten  = labels_to_attenuation(labels)
    y = int(np.clip(y_frac, 0, 1) * (labels.shape[1] - 1))
    lab_sl = labels[:, y, :]
    at_sl  = atten[:, y, :]

    # Fraction of tissue that saturates after [0,1] normalisation.
    gt_native = normalise_gt(at_sl)
    m = lab_sl > 0
    if m.any():
        print(f"  slice y={y}: tissue frac saturated (>0.98) = "
              f"{(gt_native[m] > 0.98).mean():.2f}")

    # Aliasing comparison: pad to square, then 512 with LINEAR vs AREA.
    sq = pad_to_square(at_sl)
    lin  = cv2.resize(sq, (512, 512), interpolation=cv2.INTER_LINEAR)
    area = cv2.resize(sq, (512, 512), interpolation=cv2.INTER_AREA)
    lin_n, area_n, sq_n = normalise_gt(lin), normalise_gt(area), normalise_gt(sq)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    for a, img, title in [
        (ax[0], lab_sl,  f"labels (native {lab_sl.shape[0]}x{lab_sl.shape[1]})"),
        (ax[1], sq_n,    "attenuation (native, padded)"),
        (ax[2], lin_n,   "512 INTER_LINEAR (old)"),
        (ax[3], area_n,  "512 INTER_AREA (fix)"),
    ]:
        cmap = "viridis" if title.startswith("labels") else "gray"
        vmax = 3 if title.startswith("labels") else 1.0
        im = a.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
        a.set_title(title, fontsize=10); a.axis("off")
    fig.suptitle(name, fontsize=11)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"inspect_{name}.png"
    fig.savefig(fpath, dpi=110)
    plt.close(fig)
    print(f"  figure -> {fpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="dataset/compressed")
    ap.add_argument("--n", type=int, default=3, help="how many phantoms to inspect")
    ap.add_argument("--y_frac", type=float, default=0.5,
                    help="sagittal slice position along Y (0..1)")
    ap.add_argument("--out", default="results/raw_inspect")
    args = ap.parse_args()

    files = sorted(glob(str(Path(args.raw_dir) / "*.nrrd")))
    if not files:
        raise FileNotFoundError(f"No .nrrd files in {args.raw_dir}")
    print(f"Found {len(files)} phantoms; inspecting first {min(args.n, len(files))}.")
    out_dir = Path(args.out)
    for f in files[:args.n]:
        inspect_one(f, out_dir, args.y_frac)

    print("\nDone. Open the PNGs in", out_dir,
          "\n  - Compare 'INTER_LINEAR (old)' vs 'INTER_AREA (fix)' for moire.",
          "\n  - Check the in-breast composition printed above (adipose vs "
          "fibroglandular).")


if __name__ == "__main__":
    main()
