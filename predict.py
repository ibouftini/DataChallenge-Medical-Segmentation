"""
CT Multi-Organ Segmentation — Prediction & Submission Script
=============================================================

Usage:
    python predict.py \
        --checkpoint  /kaggle/working/checkpoints/phase2_best.pth \
        --test_img_dir /kaggle/input/.../test-images \
        --output_dir  /kaggle/working/submission

Optional:
    --tta        Average original + horizontal-flip predictions
    --batch_size 32  (default)
"""

import argparse
import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from segmentation.config import Config
from segmentation.model import SegModel


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TestDataset(Dataset):
    def __init__(self, image_dir: str, cfg: Config):
        self.image_dir = image_dir
        self.cfg = cfg
        self.indices = sorted([
            int(f[:-4]) for f in os.listdir(image_dir)
            if f.endswith(".png") and f[:-4].isdigit()
        ])
        print(f"[Predict] {len(self.indices)} test images found")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img = cv2.imread(
            os.path.join(self.image_dir, f"{idx}.png"), cv2.IMREAD_GRAYSCALE
        )
        if img is None:
            img = np.zeros((self.cfg.img_size, self.cfg.img_size), dtype=np.uint8)
        img = img.astype(np.float32) / 255.0
        img = (img - self.cfg.img_mean) / self.cfg.img_std
        return {
            "image": torch.from_numpy(img[None]),   # (1, H, W)
            "idx":   idx,
        }


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def save_submission(predictions: dict, path: str) -> None:
    """
    predictions: {image_index (int) → (H, W) int32 array}
    CSV format: rows = pixels (65 536), columns = "0.png", "1.png", ...
    """
    sorted_idx = sorted(predictions.keys())
    data = np.stack([predictions[i].flatten() for i in sorted_idx], axis=1)
    df = pd.DataFrame(
        data,
        columns=[f"{i}.png" for i in sorted_idx],
        index=[f"Pixel {i}" for i in range(data.shape[0])],
    )
    df.to_csv(path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"[Predict] Submission saved → {path}  ({df.shape}, {size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   type=str, required=True,
                   help="Path to phase2_best.pth (or phase1_best.pth)")
    p.add_argument("--test_img_dir", type=str, required=True)
    p.add_argument("--output_dir",   type=str, default="./submission")
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--tta", action="store_true",
                   help="Average original + horizontal-flip predictions")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(test_img_dir=args.test_img_dir)

    # Load model
    print(f"[Predict] Loading: {args.checkpoint}")
    model = SegModel(cfg)
    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    val_dice = ckpt.get("val_dice", float("nan"))
    print(f"[Predict] Checkpoint val_dice = {val_dice:.4f}  device={device}  TTA={args.tta}")

    loader = DataLoader(
        TestDataset(args.test_img_dir, cfg),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    predictions = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            imgs = batch["image"].to(device, non_blocking=True)

            with torch.amp.autocast(device.type):
                logits = model(imgs)
                if args.tta:
                    logits_flip = model(torch.flip(imgs, dims=[-1]))
                    logits = (logits + torch.flip(logits_flip, dims=[-1])) * 0.5

            preds = logits.argmax(dim=1).cpu().numpy()  # (B, H, W)
            for pred, idx in zip(preds, batch["idx"].tolist()):
                predictions[idx] = pred.astype(np.int32)

    # Summary
    all_classes = sorted({c for p in predictions.values() for c in np.unique(p)})
    print(f"[Predict] {len(predictions)} images  |  {len(all_classes)} unique classes  "
          f"|  range {all_classes[0]}–{all_classes[-1]}")

    save_submission(predictions, os.path.join(args.output_dir, "submission.csv"))


if __name__ == "__main__":
    main()
