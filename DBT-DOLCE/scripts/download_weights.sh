#!/usr/bin/env bash
# Downloads pretrained DOLCE model512_all.pt from the official Google Drive release.
# Requires: pip install gdown==4.7.1

set -e

pip install gdown==4.7.1 --quiet

mkdir -p model_zoo

echo "[1/1] Downloading model512_all.pt ..."
gdown --id 1BYBZhd4IKR1cGM0qRNN125ey4Vfcd38U -O model_zoo/model512_all.pt

echo "Done. Checkpoint saved to model_zoo/model512_all.pt"
