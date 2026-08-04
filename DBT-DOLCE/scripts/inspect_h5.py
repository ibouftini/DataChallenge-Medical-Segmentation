"""
Inspect what is ACTUALLY stored in a preprocessed .h5 slice.

Unlike scripts/inspect_raw.py (which recomputes the resize from the raw nrrd and
lets matplotlib resample for display), this reads the real saved arr_img512 /
arr_la_rls and renders them WITHOUT display resampling (interpolation="nearest")
so screen-moire cannot be mistaken for data. It also saves a zoomed 1:1 crop and
the log-FFT magnitude -- genuine periodic ring/aliasing artefacts show up as
rings or discrete spots in the FFT; tissue texture does not.

    python scripts/inspect_h5.py --processed_dir dataset/processed_25deg
    python scripts/inspect_h5.py --h5 dataset/processed_25deg/KBCT.../y0089.h5
"""

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_one(processed_dir: str) -> str:
    hits = glob(str(Path(processed_dir) / "**" / "y*.h5"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"No y*.h5 under {processed_dir}")
    # pick a mid-file (more likely to contain breast, not an edge slice)
    return sorted(hits)[len(hits) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="dataset/processed_25deg")
    ap.add_argument("--h5", default="", help="specific .h5 (overrides --processed_dir)")
    ap.add_argument("--out", default="results/h5_inspect")
    args = ap.parse_args()

    path = args.h5 or find_one(args.processed_dir)
    with h5py.File(path, "r") as hf:
        gt  = np.asarray(hf["arr_img512"][0], dtype=np.float32)
        rls = np.asarray(hf["arr_la_rls"][0], dtype=np.float32)
        dtypes = {k: hf[k].dtype for k in hf.keys()}

    print(f"file: {path}")
    print(f"stored dtypes: {dtypes}")
    print(f"arr_img512: shape={gt.shape} min={gt.min():.4f} max={gt.max():.4f} "
          f"unique={np.unique(gt).size}")

    # log-FFT magnitude of the GT (breast region carries the structure).
    F = np.fft.fftshift(np.abs(np.fft.fft2(gt)))
    logF = np.log1p(F)

    cy, cx = [s // 2 for s in gt.shape]
    crop = gt[cy - 64:cy + 64, cx - 64:cx + 64]   # 128x128 centre, true pixels

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    ax[0].imshow(gt,   cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax[0].set_title("saved arr_img512 (nearest)")
    ax[1].imshow(crop, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax[1].set_title("128x128 centre crop (1:1 pixels)")
    ax[2].imshow(rls,  cmap="gray", interpolation="nearest")
    ax[2].set_title("saved arr_la_rls (SIRT)")
    ax[3].imshow(logF, cmap="magma", interpolation="nearest")
    ax[3].set_title("log|FFT| of GT (rings here = real aliasing)")
    for a in ax:
        a.axis("off")
    fig.suptitle(Path(path).name)
    fig.tight_layout()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fpath = out / f"h5_{Path(path).parent.name}_{Path(path).stem}.png"
    fig.savefig(fpath, dpi=130)
    plt.close(fig)
    print(f"figure -> {fpath}")
    print("Read it: if 'nearest' GT is clean but inspect_raw looked ringed, the "
          "rings were a matplotlib display artefact. Concentric rings in the "
          "log|FFT| panel would indicate genuine periodic aliasing in the data.")


if __name__ == "__main__":
    main()
