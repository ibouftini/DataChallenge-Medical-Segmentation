#!/usr/bin/env bash
# Run the full preprocessing pipeline:
#   .nrrd compressed breast phantoms -> HDF5 (gt, sirt) slice pairs
#
# Before running:
#   1. Place all .nrrd files in dataset/compressed/
#   2. Activate the conda environment: conda activate dbt-dolce
#
# Usage:
#   scripts/prepare_data.sh [DEVICE] [--resume]
#
#   DEVICE    LEAP/torch device (default: cuda:0)
#   --resume  Continue an interrupted run: skip slices already written as
#             complete HDF5 files in OUT_DIR and process only what is missing
#             (e.g. after a disk-full interruption). Reuses the existing split.
#
# Resuming onto a larger disk:
#   Move the partial output to the bigger location, point OUT_DIR at it
#   (override via the OUT_DIR env var), and pass --resume. Example:
#       OUT_DIR=/bigdisk/processed_25deg scripts/prepare_data.sh cuda:0 --resume

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Parse args: positional DEVICE and/or the --resume flag, in any order.
DEVICE="cuda:0"
RESUME=""
for arg in "$@"; do
    case "$arg" in
        --resume) RESUME="--resume" ;;
        *)        DEVICE="$arg" ;;
    esac
done

RAW_DIR="${RAW_DIR:-${ROOT}/dataset/compressed}"
OUT_DIR="${OUT_DIR:-${ROOT}/dataset/processed_25deg}"
CFG="${CFG:-${ROOT}/configs/leap_dbt_25deg.cfg}"
SPLIT_FILE="${SPLIT_FILE:-${ROOT}/dataset/split_25deg.json}"

echo "[prepare_data] Raw NRRD dir : $RAW_DIR"
echo "[prepare_data] Output dir   : $OUT_DIR"
echo "[prepare_data] LEAP config  : $CFG"
echo "[prepare_data] Device       : $DEVICE"
echo "[prepare_data] Resume       : ${RESUME:-no}"

python "${ROOT}/data/preprocess.py" \
    --raw_dir    "$RAW_DIR" \
    --out_dir    "$OUT_DIR" \
    --cfg        "$CFG" \
    --split_file "$SPLIT_FILE" \
    --sirt_iters 200 \
    --device     "$DEVICE" \
    $RESUME

echo "[prepare_data] Done."
