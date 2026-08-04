#!/usr/bin/env bash
#
# Finish the Phase-A geometry probe: run ONLY the arms that failed in the
# overnight batch, split across two GPUs. The three arms that already
# completed (05v15 exact, 05v15 none, 13v35 exact) are NOT rerun.
#
# Failed arms handled here:
#   09v25 exact, 09v25 none   (dataset load: anchor test_files.txt was missing)
#   13v35 none                (build_projector: leap_dbt_phaseA_13v35.cfg missing)
#   25v50 exact, 25v50 none   (build_projector: leap_dbt_phaseA_25v50.cfg missing)
#
# Usage (from the repo root, after `git pull`):
#   nohup bash scripts/run_phaseA_finish.sh > phaseA_finish.log 2>&1 &
#   tail -f phaseA_finish.log
#
# Required env: DBT_DATA (dataset root). Overridable: DBT_STRIDE, DBT_LOGDIR, DBT_GPUS ("0 1").

DATA=${DBT_DATA:?set DBT_DATA to the dataset root}
STRIDE=${DBT_STRIDE:-22}          # MUST match the stride used for 05v15/13v35
LOGDIR=${DBT_LOGDIR:-phaseA_logs}
read -r GPU_A GPU_B <<<"${DBT_GPUS:-0 1}"

CKPT=$DATA/runs/dbt_25deg_full/checkpoint_best.pt
OPDIR=$DATA/runs/operator
declare -A OP=( [09v25]=A_25deg_512.npz \
                [13v35]=A_phaseA_13v35_512.npz \
                [25v50]=A_phaseA_25v50_512.npz )

mkdir -p "$LOGDIR"

# ---- Prerequisite checks (fail fast with a clear message) --------------------
[[ -f "$CKPT" ]] || { echo "MISSING checkpoint: $CKPT"; exit 1; }
# The none arms use the live LEAP projector, so they need the geometry .cfg.
for c in leap_dbt_25deg leap_dbt_phaseA_13v35 leap_dbt_phaseA_25v50; do
  [[ -f "configs/${c}.cfg" ]] || {
    echo "MISSING: configs/${c}.cfg -- run: git pull origin claude/read-memory-46a4nl"
    exit 1; }
done

# ---- Step 1: write the 09v25 anchor file list if it does not exist -----------
# Reuse the EXISTING processed_25deg files for the 4 Phase-A phantoms, mapped
# from the already-working 05v15 list so the anchor evaluates the identical
# slice set (only the conditioning geometry differs).
if [[ ! -f "$DATA/processed_phaseA_09v25/test_files.txt" ]]; then
  echo "[step1] writing 09v25 anchor test_files.txt ..."
  DATA="$DATA" python3 - <<'PYEOF' || { echo "[step1] FAILED -- see message above"; exit 1; }
import os
from pathlib import Path
DATA = os.environ["DATA"]
SRC = f"{DATA}/processed_phaseA_05v15/test_files.txt"
DST = f"{DATA}/processed_phaseA_09v25"
paths, missing = [], []
for line in (l.strip() for l in open(SRC) if l.strip()):
    stem, fname = Path(line).parts[-2], Path(line).parts[-1]   # <phantom>/y####.h5
    q = os.path.join(DATA, "processed_25deg", stem, fname)
    (paths if os.path.isfile(q) else missing).append(q)
assert not missing, (f"{len(missing)} 25deg files absent, e.g. {missing[:3]} -- "
                     "processed_25deg has a different slice set; glob it instead")
import h5py
with h5py.File(paths[0]) as hf:
    assert "arr_mask" in hf and "arr_labels" in hf, "old-format files; re-preprocess anchor"
os.makedirs(DST, exist_ok=True)
open(f"{DST}/test_files.txt", "w").write("\n".join(paths))
print(f"[step1] {len(paths)} anchor slices -> {DST}/test_files.txt")
PYEOF
else
  echo "[step1] 09v25 test_files.txt already present -- skipping"
fi

# ---- Preflight: verify EVERY dir/file each arm needs, before any GPU work ----
# Each geometry needs: its config yaml, its operator .npz, and its processed
# data dir with test_files.txt. 09v25's list was just written by Step 1; the
# others' dirs already exist on the share. Fail here (seconds) rather than ~1
# min into a 13 h run.
echo "[check] verifying all inputs for: 09v25 13v35 25v50 ..."
preflight_ok=1
for tag in 09v25 13v35 25v50; do
  cfg="configs/dbt_phaseA_${tag}.yaml"
  pdir="$DATA/processed_phaseA_${tag}"
  tf="$pdir/test_files.txt"
  op="$OPDIR/${OP[$tag]}"
  miss=""
  [[ -f "$cfg" ]] || miss+=" config($cfg)"
  [[ -d "$pdir" ]] || miss+=" datadir($pdir)"
  [[ -f "$tf"  ]] || miss+=" test_files($tf)"
  [[ -f "$op"  ]] || miss+=" operator($op)"
  if [[ -n "$miss" ]]; then echo "  $tag MISSING:$miss"; preflight_ok=0
  else echo "  $tag ok"; fi
done
[[ $preflight_ok -eq 1 ]] || {
  echo "[check] FAILED -- fix the missing paths above before rerunning."; exit 1; }
echo "[check] all inputs present; starting GPU runs."

# ---- Step 2: run the 5 failed arms, split across the two GPUs -----------------
run_arm () {   # $1 = tag:arm   $2 = gpu
  local tag=${1%:*} arm=${1#*:} gpu=$2
  local log="$LOGDIR/${tag}_${arm}.log"
  echo "[gpu$gpu][start $(date +%H:%M)] $tag $arm -> $log"
  python evaluate.py \
      --config    "configs/dbt_phaseA_${tag}.yaml" \
      --ckpt      "$CKPT" --dc "$arm" \
      --operator  "$OPDIR/${OP[$tag]}" \
      --split test --max_slices 30 --slice_stride "$STRIDE" \
      --n_samples 8 --seed 0 --batch_samples --fp16 \
      --resume --device "cuda:$gpu" >>"$log" 2>&1 \
    && echo "[gpu$gpu][done  $(date +%H:%M)] $tag $arm" \
    || echo "[gpu$gpu][FAIL  $(date +%H:%M)] $tag $arm (see $log)"
}

# GPU A: 3 arms   |   GPU B: 2 arms   (each GPU runs its list sequentially)
( for j in 09v25:exact 13v35:none 25v50:none; do run_arm "$j" "$GPU_A"; done ) &
( for j in 09v25:none  25v50:exact;           do run_arm "$j" "$GPU_B"; done ) &
wait
echo "[all] Phase-A finish complete -- gather results/dbt_phaseA_* for analyze_results"
