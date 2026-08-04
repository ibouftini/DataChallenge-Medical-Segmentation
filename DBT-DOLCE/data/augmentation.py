"""
Online augmentations for DBT breast phantom slices.

Augmentation strategy:
  - Horizontal flip (p=0.5)
  - Vertical flip   (p=0.5)
  - Random zoom +/-20% (p=0.3)

All transforms are applied identically to both the GT image and the
SIRT conditioning image to preserve alignment.
"""

import random
import torch


# Torch transforms (used online during training)

class RandomFlipPair:
    """Randomly flip both gt and cond tensors with the same flip."""
    def __init__(self, p_h=0.5, p_v=0.5):
        self.p_h = p_h
        self.p_v = p_v

    def __call__(self, gt: torch.Tensor, cond: torch.Tensor):
        if random.random() < self.p_h:
            gt   = torch.flip(gt,   dims=[-1])
            cond = torch.flip(cond, dims=[-1])
        if random.random() < self.p_v:
            gt   = torch.flip(gt,   dims=[-2])
            cond = torch.flip(cond, dims=[-2])
        return gt, cond


class RandomZoomPair:
    """Random central zoom applied identically to both tensors."""
    def __init__(self, zoom_range=(0.80, 1.20), p=0.5):
        self.zoom_range = zoom_range
        self.p = p

    def __call__(self, gt: torch.Tensor, cond: torch.Tensor):
        if random.random() > self.p:
            return gt, cond
        scale = random.uniform(*self.zoom_range)
        gt   = self._zoom_tensor(gt,   scale)
        cond = self._zoom_tensor(cond, scale)
        return gt, cond

    @staticmethod
    def _zoom_tensor(t: torch.Tensor, scale: float) -> torch.Tensor:
        # t: (C, H, W)
        _, H, W = t.shape
        new_H = int(H * scale)
        new_W = int(W * scale)
        if scale >= 1.0:
            y0 = (H - new_H) // 2
            x0 = (W - new_W) // 2
            cropped = t[:, y0:y0 + new_H, x0:x0 + new_W]
        else:
            cropped = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(new_H, new_W), mode="bilinear",
                align_corners=False
            ).squeeze(0)
        out = torch.nn.functional.interpolate(
            cropped.unsqueeze(0), size=(H, W), mode="bilinear",
            align_corners=False
        ).squeeze(0)
        return out


class TrainingAugmentation:
    """Composed online augmentation: flips + optional zoom."""
    def __init__(self, p_h=0.5, p_v=0.5, zoom_range=(0.80, 1.20), p_zoom=0.3):
        self.flip = RandomFlipPair(p_h=p_h, p_v=p_v)
        self.zoom = RandomZoomPair(zoom_range=zoom_range, p=p_zoom)

    def __call__(self, gt: torch.Tensor, cond: torch.Tensor):
        gt, cond = self.flip(gt, cond)
        gt, cond = self.zoom(gt, cond)
        return gt, cond
