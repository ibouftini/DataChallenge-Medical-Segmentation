"""
Preprocessing pipeline: raw NRRD breast phantoms -> HDF5 slice pairs.

Pipeline per phantom

1. Load .nrrd segmentation volume (labels 0-3).
   - Validate and canonicalise axis order to (X, Y, Z) using the NRRD header.
2. Convert tissue labels -> linear attenuation map (cm^-1 at ~20 keV).
3. Extract sagittal slices along the Y axis.
4. Pad each sagittal slice to a square (preserving pixel isotropy), then resize to
   target_size x target_size.
5. Normalise GT to [0, 1] using known physical attenuation bounds.
6. For each slice:
     a. Simulate a 9p25d parallel-beam sinogram via LEAP.
     b. Reconstruct a SIRT conditioning image (200 iterations).
7. Clip SIRT conditioning negatives to 0 (keep full positive range); the
   dataset loader applies DOLCE's per-image min-max normalisation at load time.
8. Optionally apply CLAHE to the SIRT conditioning (off by default to match
   DOLCE, which conditions on plain min-max reconstructions).
9. Save (gt_slice, sirt_slice) as an HDF5 file.

Usage

    python data/preprocess.py \
        --raw_dir    dataset/compressed \
        --out_dir    dataset/processed_25deg \
        --cfg        configs/leap_dbt_25deg.cfg \
        --split_file dataset/split_25deg.json \
        --device     cuda:0

Resuming an interrupted run
---------------------------
If a run is interrupted (e.g. the disk fills up), move the partial output to a
larger location and re-run with --resume pointing --out_dir at it.  Each slice
output is recomputed deterministically from the source phantom, so completed,
fully-written slices are detected and skipped; only missing or truncated slices
are (re)processed.  The interrupted phantom is repaired automatically.

    python data/preprocess.py \
        --raw_dir    dataset/compressed \
        --out_dir    /bigdisk/processed_25deg \
        --cfg        configs/leap_dbt_25deg.cfg \
        --split_file dataset/split_25deg.json \
        --device     cuda:0 \
        --resume
"""

import os
import re
import sys
import json
import argparse
import logging
from pathlib import Path

# Allow `python data/preprocess.py` to find the sibling `physics` package by
# putting the repo root (parent of this file's directory) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import h5py
import nrrd
import cv2
from skimage.exposure import equalize_adapthist

from physics.dbt_projector import build_projector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Tissue attenuation constants  (cm^-1 at effective energy ~ 20 keV)
# Label: 0=air, 1=adipose, 2=fibroglandular, 3=skin
ATTENUATION = np.array([0.000, 0.410, 0.780, 0.780], dtype=np.float32)
ATTEN_MAX   = 0.780   # upper bound for [0,1] normalisation of the GT slice


# Helpers

def labels_to_attenuation(labels: np.ndarray) -> np.ndarray:
    return ATTENUATION[np.clip(labels, 0, 3)].astype(np.float32)


def normalise_gt(img: np.ndarray) -> np.ndarray:
    """Map [0, ATTEN_MAX] -> [0, 1] using fixed physical bounds."""
    return np.clip(img / ATTEN_MAX, 0.0, 1.0).astype(np.float32)


def pad_to_square(s: np.ndarray) -> np.ndarray:
    """
    Zero-pad the shorter axis of a 2D array to make it square.
    Preserves pixel isotropy -- no stretching.
    """
    h, w = s.shape
    if h == w:
        return s
    size = max(h, w)
    padded = np.zeros((size, size), dtype=s.dtype)
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    padded[y0:y0 + h, x0:x0 + w] = s
    return padded


def resize_to(s: np.ndarray, size: int) -> np.ndarray:
    if s.shape[0] == size and s.shape[1] == size:
        return s
    # INTER_AREA area-averages when DOWNSAMPLING, which anti-aliases fine tissue
    # structure. INTER_LINEAR (a 2x2 sample) does not, so downscaling a native
    # breast-CT slice to 512 with it aliases the glandular pattern into moire /
    # concentric-ring artefacts. Use LINEAR only when upsampling.
    interp = cv2.INTER_AREA if size < s.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(s, (size, size), interpolation=interp)


def apply_clahe(img: np.ndarray, clip_limit: float = 0.02, grid: int = 8) -> np.ndarray:
    """
    CLAHE on a [0,1] float image.  Returns [0,1] float.
    Enhances local contrast in the SIRT conditioning (limited-angle artefacts
    suppress low-contrast structures; CLAHE partially compensates).
    """
    img_u16 = (img * 65535).astype(np.uint16)
    enhanced = equalize_adapthist(img_u16, clip_limit=clip_limit,
                                  nbins=256, kernel_size=grid)
    return enhanced.astype(np.float32)


def breast_mask_2d(labels_sag: np.ndarray) -> np.ndarray:
    """Boolean mask: True where label > 0 (not air)."""
    return labels_sag > 0


# Resume support

# Core datasets are the GT slice and the *expensive* SIRT reconstruction; their
# presence means the costly work for this slice is already done. (The DOLCE
# "fbp" conditioning slot is identical to "rls" and is no longer stored
# separately, so it is not part of the core set.) The aux datasets (mask +
# tissue labels) are cheap eval extras added later (commit deb5fb4), so older
# preprocessed data may predate them. Resume must NOT force a full SIRT
# recompute just because the cheap aux arrays are absent.
CORE_DATASETS = ("arr_img512", "arr_la_rls")
AUX_DATASETS  = ("arr_mask", "arr_labels")
ALL_DATASETS  = CORE_DATASETS + AUX_DATASETS
# Back-compat alias for callers/tests that referenced the old full list.
REQUIRED_DATASETS = ALL_DATASETS


def slice_status(fpath: str) -> str:
    """
    Classify an expected slice output for resume decisions. Returns one of:

      "missing"     - no file on disk
      "incomplete"  - file exists but a CORE (SIRT) dataset is absent; the
                      expensive reconstruction is not present, recompute it
      "unreadable"  - file exists but cannot be opened / decompressed (e.g. a
                      truncated gzip block from a disk-full interruption)
      "ok_no_aux"   - all CORE datasets present and readable, but one or both
                      cheap aux datasets (mask/labels) are missing
      "ok"          - all CORE and aux datasets present and readable

    This only classifies a file; the keep-vs-recompute policy lives in the
    caller. `process_phantom` keeps only "ok" slices and recomputes the rest
    (including "ok_no_aux"), so a resumed dataset is uniform. The CORE/aux
    split is kept so the recompute reason can be reported. To catch a truncated
    gzip block we force a full read of the LAST dataset actually present (they
    are written in `ALL_DATASETS` order). The `phantom`/`y_idx` attrs are pure
    metadata the loader never reads and are not required.
    """
    if not os.path.isfile(fpath):
        return "missing"
    try:
        with h5py.File(fpath, "r") as hf:
            for key in CORE_DATASETS:
                if key not in hf:
                    return "incomplete"
            present = [k for k in ALL_DATASETS if k in hf]
            # Force decompression of the last-written dataset present to catch
            # a truncated gzip block from a half-finished write.
            _ = hf[present[-1]][:]
            has_aux = all(k in hf for k in AUX_DATASETS)
    except (OSError, KeyError):
        return "unreadable"
    return "ok" if has_aux else "ok_no_aux"


def slice_is_complete(fpath: str) -> bool:
    """True only if `fpath` has ALL datasets (core + aux) present and readable."""
    return slice_status(fpath) == "ok"


# NRRD axis canonicalisation

def canonicalise_axes(data: np.ndarray, header: dict) -> np.ndarray:
    """
    Ensure the volume is ordered (X, Y, Z) where:
      X = anterior-posterior,  Y = lateral,  Z = cranio-caudal (compressed).

    The Zenodo dataset is derived from ITK-based bCT reconstructions.
    ITK/SimpleITK NRRD files typically store as (X, Y, Z) with the 'space'
    field set to 'left-posterior-superior' or similar.

    We inspect the 'space directions' matrix to detect transpositions.
    If the volume appears to be stored as (Z, Y, X) we transpose it.
    """
    space = header.get("space", "").lower()
    dirs  = header.get("space directions", None)

    # If space directions are available, use the dominant axis per dimension
    # to determine ordering.  A diagonal matrix -> already canonical.
    if dirs is not None and len(dirs) == 3:
        dirs = np.array(dirs, dtype=float)
        # dominant axis index for each dimension
        dominant = [int(np.argmax(np.abs(dirs[i]))) for i in range(3)]
        if dominant == [0, 1, 2]:
            return data          # already (X, Y, Z)
        elif dominant == [2, 1, 0]:
            return data.transpose(2, 1, 0)   # (Z, Y, X) -> (X, Y, Z)
        # Other orderings are rare; log a warning and return as-is
        log.warning("Unexpected space directions dominant axes %s; using data as-is", dominant)
        return data

    # Heuristic fallback: for compressed breasts the cranio-caudal (Z) axis is
    # the shortest.  If axis 0 is shortest, assume (Z, Y, X) and transpose.
    shape = data.shape
    if shape[0] < shape[1] and shape[0] < shape[2]:
        log.debug("Heuristic: transposing (Z,Y,X) -> (X,Y,Z)")
        return data.transpose(2, 1, 0)
    return data


# Patient-wise split

def patient_id_from_filename(fname: str) -> str:
    """KBCT010001R_compressed_39_0 -> KBCT010001"""
    m = re.match(r"(KBCT\d+)[LR]?_", fname)
    return m.group(1) if m else fname


def build_patient_split(nrrd_files: list, train_r=0.80, val_r=0.10, seed=42) -> dict:
    from collections import defaultdict
    patient_map = defaultdict(list)
    for f in nrrd_files:
        pid = patient_id_from_filename(Path(f).stem)
        patient_map[pid].append(f)

    patients = sorted(patient_map.keys())
    n        = len(patients)
    n_test   = max(1, int(n * (1 - train_r - val_r)))
    n_val    = max(1, int(n * val_r))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    test_pids  = {patients[i] for i in idx[:n_test]}
    val_pids   = {patients[i] for i in idx[n_test:n_test + n_val]}

    split = {"train": [], "val": [], "test": []}
    for pid, files in patient_map.items():
        if pid in test_pids:
            split["test"].extend(files)
        elif pid in val_pids:
            split["val"].extend(files)
        else:
            split["train"].extend(files)

    log.info(
        "Split: %d train / %d val / %d test phantoms  "
        "(%d / %d / %d patients)",
        len(split["train"]), len(split["val"]), len(split["test"]),
        n - len(test_pids) - len(val_pids), len(val_pids), len(test_pids),
    )
    return split


# Per-phantom processing

def process_phantom(
    nrrd_path: str,
    out_dir: str,
    projector,
    target_size: int = 512,
    sirt_iters: int = 200,
    do_clahe: bool = True,
    device="cuda:0",
    min_tissue_fraction: float = 0.05,
    resume: bool = False,
) -> list:
    """
    Process one compressed breast phantom -> list of saved HDF5 paths.

    When `resume` is True, any slice whose output already exists and passes
    `slice_is_complete` is kept as-is and skipped, so an interrupted run can be
    continued without recomputing finished work.  The returned list always
    covers every expected slice (kept + freshly written) so downstream file
    lists stay complete.
    """
    import torch

    stem       = Path(nrrd_path).stem
    out_subdir = os.path.join(out_dir, stem)
    os.makedirs(out_subdir, exist_ok=True)
    n_skipped  = 0
    n_redone   = 0   # existed but not fully complete -> recomputed from scratch

    # Load and canonicalise
    data, header = nrrd.read(nrrd_path)
    data  = canonicalise_axes(data, header)     # -> (X, Y, Z)
    labels = data.astype(np.uint8)

    # Volume -> attenuation
    atten = labels_to_attenuation(labels)       # (X, Y, Z)  cm^-1

    # Iterate over Y slices (sagittal planes)
    n_y = labels.shape[1]
    saved_paths = []

    for y_idx in range(n_y):
        sag_atten  = atten[:, y_idx, :]          # (X, Z)
        sag_labels = labels[:, y_idx, :]         # (X, Z)

        # Skip mostly-air slices
        mask_2d = breast_mask_2d(sag_labels)
        if mask_2d.mean() < min_tissue_fraction:
            continue

        # This slice is expected output.  On resume, keep it if already done.
        fpath = os.path.join(out_subdir, f"y{y_idx:04d}.h5")
        if resume:
            status = slice_status(fpath)
            if status == "ok":
                # Fully complete (all datasets) -> keep, skip recompute.
                saved_paths.append(fpath)
                n_skipped += 1
                continue
            if status != "missing":
                # Present but not fully complete -> recompute from scratch.
                reason = {
                    "ok_no_aux":  "missing mask/labels",
                    "incomplete": "missing core dataset",
                    "unreadable": "corrupt/truncated",
                }.get(status, status)
                log.warning("  recomputing %s (%s)", fpath, reason)
                n_redone += 1

        # Pad to square then resize  (preserves pixel isotropy)
        sag_atten  = pad_to_square(sag_atten)    # (S, S) where S = max(X, Z)
        sag_atten  = resize_to(sag_atten, target_size)   # (512, 512)
        mask_2d    = pad_to_square(mask_2d.astype(np.float32))
        mask_2d    = resize_to(mask_2d, target_size) > 0.5   # bool (512, 512)

        # Tissue label map for per-tissue evaluation metrics. Nearest-neighbour
        # resize to keep integer labels intact.
        sag_lab_sq = pad_to_square(sag_labels.astype(np.float32))
        sag_lab_r  = cv2.resize(sag_lab_sq, (target_size, target_size),
                                interpolation=cv2.INTER_NEAREST)
        labels_512 = np.rint(sag_lab_r).astype(np.uint8)
        labels_512[~mask_2d] = 0

        # Normalise GT to [0, 1] using physical bounds
        gt = normalise_gt(sag_atten)             # (512, 512) in [0, 1]
        gt[~mask_2d] = 0.0

        # Forward project normalised GT -> sinogram
        gt_t = torch.from_numpy(gt).float().to(device)   # (H, W)
        sino = projector.forward(gt_t)                    # (num_angles, W)

        # SIRT conditioning
        with torch.no_grad():
            sirt_t = projector.sirt(sino, num_iters=sirt_iters)  # (H, W)

        # Store SIRT with negatives removed but WITHOUT upper-clipping, so the
        # full dynamic range is preserved.  The dataset loader applies DOLCE's
        # per-image min-max normalisation at load time, matching how
        # model512_all.pt was trained.
        sirt = sirt_t.cpu().numpy()
        sirt = np.clip(sirt, 0.0, None).astype(np.float32)
        sirt[~mask_2d] = 0.0

        # Optional CLAHE (off by default; DOLCE conditions on plain min-max
        # reconstructions).  CLAHE needs a [0,1] input, so normalise first.
        if do_clahe:
            lo, hi = sirt.min(), sirt.max()
            sirt = (sirt - lo) / (hi - lo + 1e-8)
            sirt = apply_clahe(sirt)
            sirt[~mask_2d] = 0.0   # re-zero background after CLAHE

        # Save as HDF5. Store with channel dim (1, H, W) to match DOLCE.
        # gt/sirt are [0,1]-scale image data that the loader min-max renormalises
        # at load time, so float16 is plenty and halves their on-disk size.
        # The DOLCE "fbp" conditioning slot is identical to "rls", so it is NOT
        # duplicated on disk -- the loader reuses arr_la_rls for it.
        with h5py.File(fpath, "w") as hf:
            hf.create_dataset("arr_img512", data=gt[None].astype(np.float16),
                              compression="gzip")
            hf.create_dataset("arr_la_rls", data=sirt[None].astype(np.float16),
                              compression="gzip")
            # Evaluation auxiliaries: breast mask + tissue labels (0..3)
            hf.create_dataset("arr_mask",   data=mask_2d[None].astype(np.uint8),
                              compression="gzip")
            hf.create_dataset("arr_labels", data=labels_512[None], compression="gzip")
            hf.attrs["phantom"] = stem
            hf.attrs["y_idx"]   = y_idx

        saved_paths.append(fpath)

    if resume:
        n_new = len(saved_paths) - n_skipped
        log.info("  %s -> %d slices (%d kept, %d recomputed, %d new)",
                 stem, len(saved_paths), n_skipped, n_redone, n_new - n_redone)
    else:
        log.info("  %s -> %d slices saved", stem, len(saved_paths))
    return saved_paths


# Entry point

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",    required=True)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--cfg",        required=True)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--target_size", type=int,   default=512)
    parser.add_argument("--sirt_iters",  type=int,   default=200)
    # CLAHE is OFF by default to match DOLCE (which conditions on plain
    # min-max normalised reconstructions). Enable only to experiment.
    parser.add_argument("--clahe",       action="store_true", default=False,
                        help="Apply CLAHE to SIRT conditioning (off by default).")
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--resume",      action="store_true", default=False,
                        help="Skip slices already written as complete HDF5 "
                             "files in --out_dir and process only what is "
                             "missing (e.g. after a disk-full interruption). "
                             "Reuses an existing --split_file if present.")
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio",   type=float, default=0.10)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.resume:
        existing = sum(1 for _ in Path(args.out_dir).rglob("y*.h5"))
        log.info("RESUME enabled. Output dir %s (abs: %s) already holds %d .h5 "
                 "slice files; complete ones will be kept.",
                 args.out_dir, os.path.abspath(args.out_dir), existing)

    nrrd_files = sorted(Path(args.raw_dir).glob("*.nrrd"))
    if not nrrd_files:
        raise FileNotFoundError(f"No .nrrd files in {args.raw_dir}")
    log.info("Found %d phantoms", len(nrrd_files))

    # On resume, reuse the existing split so the train/val/test partition is
    # byte-for-byte identical to the interrupted run.
    if args.resume and os.path.isfile(args.split_file):
        with open(args.split_file) as f:
            split = json.load(f)
        log.info("Resuming with existing split <- %s", args.split_file)
    else:
        split = build_patient_split(
            [str(f) for f in nrrd_files],
            train_r=args.train_ratio,
            val_r=args.val_ratio,
            seed=args.seed,
        )
        with open(args.split_file, "w") as f:
            json.dump(split, f, indent=2)
        log.info("Split saved -> %s", args.split_file)

    projector = build_projector(args.cfg, device=args.device)

    all_h5 = {"train": [], "val": [], "test": []}
    for subset, phantom_list in split.items():
        for nrrd_path in phantom_list:
            paths = process_phantom(
                nrrd_path=nrrd_path,
                out_dir=args.out_dir,
                projector=projector,
                target_size=args.target_size,
                sirt_iters=args.sirt_iters,
                do_clahe=args.clahe,
                device=args.device,
                resume=args.resume,
            )
            all_h5[subset].extend(paths)

    for subset, paths in all_h5.items():
        list_file = os.path.join(args.out_dir, f"{subset}_files.txt")
        with open(list_file, "w") as f:
            f.write("\n".join(paths))
        log.info("%s: %d slices -> %s", subset, len(paths), list_file)

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
