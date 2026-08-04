"""
Full fine-tuning helpers for the DOLCE guided-diffusion UNet.

Every pretrained weight is trainable: the medical-CT -> breast domain shift is
large enough that the frozen base collapses to its prior mean, so partial
adaptation is not enough.  Checkpoints therefore carry the complete state dict
(~273M params) for both the training model and its EMA copy.

Usage

    from model.finetune import set_full_finetune, trainable_params, print_param_summary

    model = load_dolce_model(...)
    set_full_finetune(model)
    opt   = torch.optim.AdamW(trainable_params(model), lr=2e-5)
    print_param_summary(model)
"""

from typing import List

import torch
import torch.nn as nn


def set_full_finetune(model: nn.Module) -> None:
    """Mark every parameter of `model` as trainable.  Modifies in-place."""
    for p in model.parameters():
        p.requires_grad_(True)


def trainable_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def print_param_summary(model: nn.Module) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(
        f"[finetune] Parameters -- "
        f"total: {total / 1e6:.1f}M | "
        f"trainable: {trainable / 1e6:.2f}M ({100 * trainable / total:.2f}%) | "
        f"frozen: {frozen / 1e6:.1f}M"
    )


def save_weights(model: nn.Module, path: str) -> None:
    """Save the model's full state dict."""
    state = model.state_dict()
    torch.save(state, path)
    print(f"[finetune] Saved weights ({len(state)} tensors) -> {path}")


def load_weights(model: nn.Module, path: str, strict: bool = False,
                 prefer: str = "ema") -> None:
    """
    Load fine-tuned weights into `model`.

    Accepts either format:
      * a bare state dict (save_weights output), or
      * a training checkpoint dict {"ema", "model", ...} from train.py, in
        which case `prefer` selects "ema" (deployed weights, default) or
        "model".
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and ("ema" in obj or "model" in obj):
        state = obj.get(prefer) or obj.get("ema") or obj.get("model")
        print(f"[finetune] Loading '{prefer}' weights from training checkpoint "
              f"(step {obj.get('step', '?')})")
    else:
        state = obj
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        print(f"[finetune] Missing keys ({len(missing)}): {list(missing)[:5]} ...")
    if unexpected:
        print(f"[finetune] Unexpected keys ({len(unexpected)}): {list(unexpected)[:5]} ...")
    print(f"[finetune] Loaded weights from {path}")
