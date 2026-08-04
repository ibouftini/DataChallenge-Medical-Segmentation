#!/usr/bin/env bash
# Run evaluation on the test set.
# Usage:
#   bash scripts/evaluate.sh [checkpoint.pt]

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${ROOT}/configs/dbt_25deg.yaml"
CKPT="${1:-}"

echo "[evaluate] Config     : $CONFIG"
echo "[evaluate] Checkpoint : ${CKPT:-<none - base DOLCE only>}"

# DDPM with proximal step (CG: parameter-free, robust default)
python "${ROOT}/evaluate.py" \
    --config    "$CONFIG" \
    --ckpt      "$CKPT" \
    --sampler   ddpm \
    --prox      cgrad

# DDIM-100 with proximal step
python "${ROOT}/evaluate.py" \
    --config    "$CONFIG" \
    --ckpt      "$CKPT" \
    --sampler   ddim \
    --prox      cgrad

# DDPM without proximal step
python "${ROOT}/evaluate.py" \
    --config    "$CONFIG" \
    --ckpt      "$CKPT" \
    --sampler   ddpm \
    --no_prox
