#!/usr/bin/env bash
# Launch training. Every weight of the DOLCE UNet is fine-tuned; the
# hyperparameters come from configs/dbt_25deg.yaml.
#
#   bash scripts/train.sh
#
# Hangup-proof background launch (survives SSH/terminal close):
#   nohup bash scripts/train.sh >> scripts/train.log 2>&1 & disown
#
# Auto-resume (training.auto_resume) continues from
# <output_dir>/checkpoint_latest.pt, so you can safely restart.

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${ROOT}/configs/dbt_25deg.yaml"

# Reduce allocator fragmentation (helps when memory is tight near capacity).
# max_split_size_mb is supported on PyTorch >= 1.10; expandable_segments would
# need >= 2.1 (this env is 2.0.1) and errors out, so it is not used here.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

echo "[train] Config: $CONFIG   (full fine-tune)"
echo "[train] GPU   : A100 80GB VRAM"
echo "[train] PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"

python "${ROOT}/train.py" --config "$CONFIG"
