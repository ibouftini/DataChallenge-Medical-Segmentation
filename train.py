"""
CT Multi-Organ Segmentation — Training Script
==============================================

Phase 1: Supervised training on 800 annotated slices.
    python train.py --phase 1

Phase 2: Pseudo-label fine-tuning on Phase 1 checkpoint + 1700 unlabeled images.
    python train.py --phase 2 --phase1_ckpt ./checkpoints/best_model.pth

Path overrides (all optional, defaults in Config):
    --train_img_dir  --test_img_dir  --label_csv  --annotated_json
    --radimagenet_weights  --output_dir  --pseudo_dir
"""

import argparse
import json
import os
import glob
import torch
from torch.utils.data import ConcatDataset


def _check_gpu():
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    if major < 7:
        name = torch.cuda.get_device_name()
        raise RuntimeError(
            f"\nGPU '{name}' has compute capability sm_{major}{minor}, "
            f"but PyTorch 2.x requires sm_70 or newer.\n"
            f"On Kaggle: Settings → Accelerator → GPU T4 x1, then restart."
        )

from segmentation.config import Config
from segmentation.dataset import (
    build_datasets, PseudoDataset,
)
from segmentation.model import SegModel
from segmentation.augmentation import train_transform, val_transform
from segmentation.trainer import Trainer
from segmentation.pseudo_label import generate_pseudo_labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, choices=[1, 2], required=True,
                   help="1 = supervised, 2 = pseudo-label fine-tuning")
    p.add_argument("--phase1_ckpt", type=str, default=None,
                   help="Path to Phase 1 checkpoint (required for --phase 2)")

    # Optional path overrides
    p.add_argument("--train_img_dir",      type=str, default=None)
    p.add_argument("--test_img_dir",       type=str, default=None)
    p.add_argument("--label_csv",          type=str, default=None)
    p.add_argument("--annotated_json",     type=str, default=None)
    p.add_argument("--radimagenet_weights",type=str, default=None)
    p.add_argument("--output_dir",         type=str, default=None)
    p.add_argument("--pseudo_dir",         type=str, default=None)
    return p.parse_args()


def apply_overrides(cfg: Config, args) -> Config:
    for field in ("train_img_dir", "test_img_dir", "label_csv",
                  "annotated_json", "radimagenet_weights",
                  "output_dir", "pseudo_dir"):
        val = getattr(args, field)
        if val is not None:
            setattr(cfg, field, val)
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _png_files_in(directory: str) -> list:
    files = sorted(glob.glob(os.path.join(directory, "*.png")))
    return [os.path.basename(f) for f in files]


def _unannotated_train_files(cfg: Config) -> list:
    """PNG files for training indices 800-1999 (unannotated)."""
    all_files = _png_files_in(cfg.train_img_dir)
    # Files are named "{index}.png"; keep those with index >= n_annotated
    result = []
    for fn in all_files:
        try:
            idx = int(os.path.splitext(fn)[0])
            if idx >= cfg.n_annotated:
                result.append(fn)
        except ValueError:
            pass
    return result


def build_model(cfg: Config) -> SegModel:
    model = SegModel(cfg)
    if os.path.isfile(cfg.radimagenet_weights):
        model.load_radimagenet_weights(cfg.radimagenet_weights)
    else:
        print(f"[WARNING] RadioImageNet weights not found at "
              f"{cfg.radimagenet_weights} — using random init for encoder.")
    return model


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def run_phase1(cfg: Config) -> str:
    print("\n" + "=" * 60)
    print("PHASE 1 — Supervised training")
    print("=" * 60)

    train_ds, val_ds, sampler = build_datasets(cfg, train_transform, val_transform)
    print(f"[Data] train={len(train_ds)}  val={len(val_ds)}")

    model   = build_model(cfg)
    trainer = Trainer(
        cfg            = cfg,
        model          = model,
        train_dataset  = train_ds,
        val_dataset    = val_ds,
        encoder_lr     = cfg.encoder_lr,
        decoder_lr     = cfg.decoder_lr,
        max_epochs     = cfg.phase1_epochs,
        early_stop_patience = cfg.early_stop_patience,
        checkpoint_name = "phase1_best.pth",
        sampler        = sampler,
    )
    _, per_class = trainer.run()

    # Persist per-class dice so Phase 2 can filter pseudo-labels
    per_class_path = os.path.join(cfg.output_dir, "phase1_per_class_dice.json")
    with open(per_class_path, "w") as f:
        json.dump({str(k): v for k, v in per_class.items()}, f)
    print(f"[Phase1] Per-class dice saved → {per_class_path}")

    return os.path.join(cfg.output_dir, "phase1_best.pth")


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def run_phase2(cfg: Config, phase1_ckpt: str) -> None:
    print("\n" + "=" * 60)
    print("PHASE 2 — Pseudo-label fine-tuning")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Phase 1 model for pseudo-label generation ---
    print(f"[Phase2] Loading Phase 1 checkpoint: {phase1_ckpt}")
    model = SegModel(cfg)
    ckpt  = torch.load(phase1_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    # --- Load Phase 1 per-class dice for pseudo-label quality gate ---
    per_class_path = os.path.join(cfg.output_dir, "phase1_per_class_dice.json")
    phase1_per_class_dice = None
    if os.path.isfile(per_class_path):
        with open(per_class_path) as f:
            data = json.load(f)
        phase1_per_class_dice = {int(k): v for k, v in data.items()}
        print(f"[Phase2] Loaded Phase 1 per-class dice ({len(phase1_per_class_dice)} classes)")
    else:
        print(f"[Phase2] No per-class dice found at {per_class_path} — skipping quality gate")

    # --- Collect unlabeled images ---
    unannotated_files = _unannotated_train_files(cfg)
    test_files        = _png_files_in(cfg.test_img_dir)

    print(f"[Phase2] Unlabeled pool: {len(unannotated_files)} train + "
          f"{len(test_files)} test = "
          f"{len(unannotated_files) + len(test_files)} images")

    unlabeled_dirs = [
        (cfg.train_img_dir, unannotated_files),
        (cfg.test_img_dir,  test_files),
    ]

    # --- Generate pseudo-labels ---
    os.makedirs(cfg.pseudo_dir, exist_ok=True)
    pseudo_labels, class_counts = generate_pseudo_labels(
        model, cfg, unlabeled_dirs, device,
        phase1_class_dice=phase1_per_class_dice,
    )

    # --- Build combined training dataset ---
    train_ds, val_ds, _ = build_datasets(cfg, train_transform, val_transform)

    # Split pseudo-labels by source directory
    pseudo_train_files = [fn for fn in unannotated_files if fn in pseudo_labels]
    pseudo_test_files  = [fn for fn in test_files        if fn in pseudo_labels]

    pseudo_datasets = []
    if pseudo_train_files:
        pseudo_datasets.append(PseudoDataset(
            cfg, cfg.train_img_dir, pseudo_train_files, pseudo_labels, train_transform,
        ))
    if pseudo_test_files:
        pseudo_datasets.append(PseudoDataset(
            cfg, cfg.test_img_dir, pseudo_test_files, pseudo_labels, train_transform,
        ))

    combined_train = ConcatDataset([train_ds] + pseudo_datasets)
    print(f"[Phase2] Combined train size: {len(combined_train)} "
          f"({len(train_ds)} annotated + "
          f"{sum(len(d) for d in pseudo_datasets)} pseudo-labeled)")

    # --- Continue training from Phase 1 checkpoint ---
    trainer = Trainer(
        cfg            = cfg,
        model          = model,
        train_dataset  = combined_train,
        val_dataset    = val_ds,
        encoder_lr     = cfg.phase2_encoder_lr,
        decoder_lr     = cfg.phase2_decoder_lr,
        max_epochs     = cfg.phase2_epochs,
        early_stop_patience = cfg.phase2_early_stop_patience,
        checkpoint_name = "phase2_best.pth",
    )
    trainer.run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _check_gpu()
    args = parse_args()
    cfg  = apply_overrides(Config(), args)

    os.makedirs(cfg.output_dir, exist_ok=True)

    if args.phase == 1:
        run_phase1(cfg)

    elif args.phase == 2:
        ckpt = args.phase1_ckpt or os.path.join(cfg.output_dir, "phase1_best.pth")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"Phase 1 checkpoint not found: {ckpt}\n"
                f"Run --phase 1 first, or pass --phase1_ckpt <path>."
            )
        run_phase2(cfg, ckpt)


if __name__ == "__main__":
    main()
