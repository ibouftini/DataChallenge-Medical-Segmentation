"""
Full fine-tuning of DOLCE for DBT 9p25d reconstruction.

Steps performed

1. Load pretrained DOLCE model (model512_all.pt).
2. Unfreeze every weight (full fine-tune).
3. Fine-tune on preprocessed DBT breast-phantom slice pairs.
4. Validate periodically (PSNR / SSIM on val set, sampling without prox).
5. Save checkpoint at regular intervals.

Usage

Single GPU:
    python train.py --config configs/dbt_25deg.yaml

Multi-GPU (e.g. 4 GPUs with MPI):
    mpiexec -n 4 python train.py --config configs/dbt_25deg.yaml
"""

import os
import sys
import copy
import json
import functools
import argparse
import logging
from pathlib import Path

import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# DOLCE imports (installed from external/DOLCE via setup.sh)
sys.path.insert(0, str(Path(__file__).parent / "external" / "DOLCE"))
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    condtion_model_and_diffusion_defaults,
    condtion_create_model_and_diffusion,
)
try:
    from guided_diffusion.resample import create_named_schedule_sampler
except ModuleNotFoundError as _e:
    # Only fall back when resample.py itself is absent; if it exists but pulls
    # in some other missing dependency, surface that instead of masking it.
    if _e.name != "guided_diffusion.resample":
        raise
    # Some DOLCE checkouts ship a trimmed guided_diffusion without resample.py.
    # We only ever use the "uniform" schedule sampler (see ForwardTrainer), so
    # provide a minimal local equivalent: uniform timestep sampling with unit
    # loss weights (exactly what UniformSampler does for the "uniform" case).
    class _UniformSampler:
        def __init__(self, diffusion):
            self._num_timesteps = int(diffusion.num_timesteps)

        def sample(self, batch_size, device):
            t = torch.randint(0, self._num_timesteps, (batch_size,),
                              device=device).long()
            weights = torch.ones(batch_size, device=device, dtype=torch.float32)
            return t, weights

    def create_named_schedule_sampler(name, diffusion):
        if name != "uniform":
            raise ValueError(
                f"guided_diffusion.resample is unavailable and the local "
                f"fallback only supports the 'uniform' schedule sampler, "
                f"got {name!r}."
            )
        return _UniformSampler(diffusion)

# Neutralise DOLCE's gradient checkpointing. Its UNet hardcodes checkpointing in
# the attention/residual blocks (the per-block use_checkpoint flag is ignored),
# and its CheckpointFunction is incompatible with this setup: it recomputes the
# forward without the autocast context (fp16 activations vs float32 weights ->
# "Input type Half and weight type Float"). Replacing the checkpoint helper with
# a direct call makes every block run normally (standard autograd, autocast-
# consistent) at the cost of extra activation memory. Patch both the definition
# module and the unet module namespace, since unet imports the name directly.
import guided_diffusion.nn as _gd_nn
import guided_diffusion.unet as _gd_unet


def _no_checkpoint(func, inputs, params, flag):
    return func(*inputs)


_gd_nn.checkpoint = _no_checkpoint
_gd_unet.checkpoint = _no_checkpoint

from data.dataset import build_loader, infinite_loader
from model.finetune import (
    set_full_finetune, trainable_params, print_param_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Config loading

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def dolce_model_args(cfg: dict) -> dict:
    """
    Build the kwargs dict expected by condtion_create_model_and_diffusion.

    We start from DOLCE's own defaults (which already carry the correct
    out_channels=2 / learned-variance diffusion configuration) and override
    only the architecture flags that must match model512_all.pt.  Keys not in
    the factory's signature (e.g. channel_mult, predict_xstart) are NOT set --
    channel_mult is hardcoded inside condtion_create_model for 512.
    """
    m = cfg["model"]
    defaults = condtion_model_and_diffusion_defaults()
    defaults.update({
        "image_size":            m["image_size"],
        "num_channels":          m["num_channels"],
        "num_res_blocks":        m["num_res_blocks"],
        "num_heads":             m["num_heads"],
        "num_head_channels":     m["num_head_channels"],
        "attention_resolutions": m["attention_resolutions"],
        "resblock_updown":       m["resblock_updown"],
        "dropout":               m["dropout"],
        "use_fp16":              m["use_fp16"],
        "diffusion_steps":       m["diffusion_steps"],
        "noise_schedule":        m["noise_schedule"],
        "weighted_condition":    m["weighted_condition"],
        "use_condtion":          m["use_condition"],
    })
    # Note: gradient checkpointing is neutralised globally by the _no_checkpoint
    # monkey-patch at import time (DOLCE hardcodes it and it clashes with
    # autocast), so no use_checkpoint arg is set here.
    return defaults


# Model loading

def build_model(cfg: dict, device):
    model_args = dolce_model_args(cfg)
    model, diffusion = condtion_create_model_and_diffusion(**model_args)

    # Load pretrained DOLCE checkpoint
    ckpt_path = cfg["model"]["pretrained_ckpt"]
    log.info("Loading pretrained DOLCE weights from %s", ckpt_path)
    state = dist_util.load_state_dict(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    # Full fine-tune: every weight is trainable (best domain adaptation for the
    # medical-CT -> breast shift, needs a low LR to protect the pretrained init).
    set_full_finetune(model)
    print_param_summary(model)

    model.to(device)
    return model, diffusion


# DOLCE ships an inference-only guided_diffusion: its SpacedDiffusion has the
# sampling primitives (q_sample, p_mean_variance, ...) but NO training_losses
# (and no resample.py / _vb_terms_bpd). So we supply the loss ourselves.
_LOSS_METHOD_CANDIDATES = (
    "training_losses",            # stock guided_diffusion
    "condtion_training_losses",   # DOLCE conditional fork (sic: "condtion")
    "conditional_training_losses",
    "training_losses_conditional",
)


def _mean_flat(x):
    """Mean over all dimensions except the batch dimension."""
    return x.reshape(x.shape[0], -1).mean(dim=1)


def local_training_losses(diffusion, model, x_start, t, model_kwargs=None,
                          noise=None):
    """
    Minimal epsilon-prediction training loss for DOLCE's inference-only
    diffusion (which ships no training_losses).

    model512_all.pt predicts noise with a learned-variance head
    (out_channels = 2 * image_channels).  We optimise the standard *simple*
    objective -- MSE between the predicted noise (first C channels) and the true
    noise -- which is the correct fine-tuning target for the diffusion mean.
    The improved-diffusion VLB term that would train the variance head is
    omitted: the helper (_vb_terms_bpd) is unavailable here, so the variance
    head keeps its pretrained values.

    Assumes identity timestep spacing (num_timesteps == diffusion_steps), which
    holds for this config (1800 == 1800), so t is passed to the model directly.
    """
    if model_kwargs is None:
        model_kwargs = {}
    if noise is None:
        noise = torch.randn_like(x_start)
    x_t = diffusion.q_sample(x_start, t, noise=noise)
    model_output = model(x_t, t, **model_kwargs)
    C = x_start.shape[1]
    if model_output.shape[1] > C:
        model_output = model_output[:, :C]   # drop learned-variance channels
    loss = _mean_flat((noise - model_output) ** 2)
    return {"loss": loss}


def resolve_loss_fn(diffusion):
    """
    Return a callable loss_fn(model, x_start, t, model_kwargs=...) -> {"loss"}.

    Prefer a built-in training_losses method if this guided_diffusion build has
    one; otherwise fall back to the local epsilon-prediction implementation.
    """
    for name in _LOSS_METHOD_CANDIDATES:
        fn = getattr(diffusion, name, None)
        if callable(fn):
            log.info("Using diffusion.%s for the training loss.", name)
            return lambda model, x_start, t, model_kwargs=None, _fn=fn: _fn(
                model, x_start, t, model_kwargs=model_kwargs)
    log.warning("Diffusion has no training_losses; using a local "
                "epsilon-prediction MSE loss (q_sample + noise-MSE).")
    return functools.partial(local_training_losses, diffusion)


# Simple training loop (step-based)

class DBTTrainer:
    def __init__(self, cfg: dict, rank: int = 0, local_rank: int = 0, world_size: int = 1):
        self.cfg        = cfg
        self.rank       = rank
        self.world_size = world_size
        self.device     = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        tc              = cfg["training"]

        # Model + diffusion
        self.model, self.diffusion = build_model(cfg, self.device)
        # Training-loss callable (built-in if available, else a local
        # epsilon-prediction loss for DOLCE's inference-only diffusion).
        self.loss_fn = resolve_loss_fn(self.diffusion)

        # EMA model (for sampling)
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_rate = tc["ema_rate"]

        # Wrap in DDP. Some DOLCE conditional-branch params (and the variance
        # head, which the epsilon loss never touches) do not receive gradients
        # every step, so find_unused_parameters avoids a DDP "unused parameter"
        # crash.
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[local_rank],
                             find_unused_parameters=True)

        # Optimizer
        self.opt = torch.optim.AdamW(
            trainable_params(self.model),
            lr=tc["lr"],
            weight_decay=tc["weight_decay"],
        )

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg["model"]["use_fp16"])

        # Diffusion timestep sampler
        self.schedule_sampler = create_named_schedule_sampler("uniform", self.diffusion)

        # Data loaders
        data_cfg  = cfg["data"]
        processed = data_cfg["processed_dir"]
        drange    = cfg["model"].get("data_range", "-11")
        self.train_loader = build_loader(
            os.path.join(processed, "train_files.txt"),
            batch_size=tc["batch_size"],
            num_workers=tc["num_workers"],
            augment=True,
            rank=rank, world_size=world_size,
            data_range=drange,
        )
        self.val_loader = build_loader(
            os.path.join(processed, "val_files.txt"),
            batch_size=tc.get("val_batch_size", 8),
            num_workers=tc["num_workers"],
            augment=False,
            deterministic=True,
            data_range=drange,
        )
        # Cap validation to a fixed number of batches (0 = full set).
        self.max_val_batches = tc.get("max_val_batches", 0)
        self.train_iter   = infinite_loader(self.train_loader)

        # Output
        self.out_dir       = tc["output_dir"]
        self.max_steps     = tc["max_steps"]
        self.log_interval  = tc["log_interval"]
        self.val_interval  = tc.get("val_interval", tc.get("save_interval", 5000))
        self.step          = 0

        # nnU-Net-style tracking: best/latest checkpoints, metric history, plot.
        self.best_val         = float("inf")
        self.last_train_loss  = None
        self.history          = []   # list of {step, train_loss, val_loss}
        self.history_path     = os.path.join(self.out_dir, "progress.json")
        self.plot_path        = os.path.join(self.out_dir, "progress.png")

        if rank == 0:
            os.makedirs(self.out_dir, exist_ok=True)

        # Resume. An explicit resume_checkpoint wins; otherwise, if auto_resume
        # is on (default), pick up checkpoint_latest.pt from out_dir so a
        # crashed / preempted run continues instead of restarting from scratch.
        resume = tc.get("resume_checkpoint", "")
        if not resume and tc.get("auto_resume", True):
            cand = os.path.join(self.out_dir, "checkpoint_latest.pt")
            if os.path.isfile(cand):
                resume = cand
                log.info("auto_resume: found %s", cand)
        if resume:
            self._resume(resume)

    # EMA update

    def _update_ema(self):
        src = self.model.module if hasattr(self.model, "module") else self.model
        for ema_p, p in zip(self.ema_model.parameters(), src.parameters()):
            if p.requires_grad:
                ema_p.data.mul_(self.ema_rate).add_(p.data, alpha=1 - self.ema_rate)

    # Single training step

    def _train_step(self, batch) -> dict:
        gt, model_kwargs = batch
        gt = gt.to(self.device)
        model_kwargs = {k: v.to(self.device) for k, v in model_kwargs.items()}

        t, weights = self.schedule_sampler.sample(gt.shape[0], self.device)

        self.opt.zero_grad()
        # autocast is gated on model.use_fp16. Safe because gradient
        # checkpointing is disabled in code (AMP must not be combined with
        # DOLCE's CheckpointFunction -- see the _no_checkpoint patch above).
        with torch.cuda.amp.autocast(enabled=self.cfg["model"]["use_fp16"]):
            losses = self.loss_fn(self.model, gt, t, model_kwargs=model_kwargs)
            loss = (losses["loss"] * weights).mean()

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(
            trainable_params(self.model), max_norm=1.0
        )
        self.scaler.step(self.opt)
        self.scaler.update()
        self._update_ema()

        return {"loss": loss.item()}

    # Main loop

    def train(self):
        log.info("Starting training for %d steps", self.max_steps)
        self.model.train()

        running_loss = 0.0
        while self.step < self.max_steps:
            batch = next(self.train_iter)
            metrics = self._train_step(batch)
            running_loss += metrics["loss"]
            self.step += 1

            if self.step % self.log_interval == 0 and self.rank == 0:
                avg_loss = running_loss / self.log_interval
                self.last_train_loss = avg_loss
                log.info("step %d | loss: %.5f", self.step, avg_loss)
                running_loss = 0.0

            if self.step % self.val_interval == 0 and self.rank == 0:
                val_loss = self._validate()
                self._record(val_loss)          # append to history + JSON + plot
                self._save("latest")            # always overwrite latest
                if val_loss < self.best_val:    # keep the single best (EMA sense)
                    self.best_val = val_loss
                    self._save("best")
                    log.info("New best val_loss %.5f at step %d",
                             val_loss, self.step)

        if self.rank == 0:
            self._save("latest")
        log.info("Training complete.")

    # Validation

    @torch.no_grad()
    def _validate(self):
        # Validate the EMA weights -- that is what we deploy, so best-checkpoint
        # selection stays consistent with the saved model.
        model = self.ema_model
        model.eval()

        total_loss = 0.0
        n = 0
        for batch in self.val_loader:
            if self.max_val_batches and n >= self.max_val_batches:
                break
            gt, model_kwargs = batch
            gt = gt.to(self.device)
            model_kwargs = {k: v.to(self.device) for k, v in model_kwargs.items()}
            t, weights = self.schedule_sampler.sample(gt.shape[0], self.device)
            with torch.cuda.amp.autocast(enabled=self.cfg["model"]["use_fp16"]):
                losses = self.loss_fn(model, gt, t, model_kwargs=model_kwargs)
            total_loss += (losses["loss"] * weights).mean().item()
            n += 1

        avg_val_loss = total_loss / max(n, 1)
        log.info("step %d | val_loss(EMA): %.5f  (%d batches)",
                 self.step, avg_val_loss, n)
        return avg_val_loss

    # Metric history + progress plot (nnU-Net style)

    def _record(self, val_loss: float):
        self.history.append({
            "step":       self.step,
            "train_loss": self.last_train_loss,
            "val_loss":   val_loss,
        })
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        self._plot_progress()

    def _plot_progress(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            log.warning("matplotlib unavailable; skipping progress plot (%s)", e)
            return
        steps = [h["step"] for h in self.history]
        vloss = [h["val_loss"] for h in self.history]
        tloss = [h["train_loss"] for h in self.history]

        fig, ax = plt.subplots(figsize=(8, 5))
        if any(v is not None for v in tloss):
            xs = [s for s, v in zip(steps, tloss) if v is not None]
            ys = [v for v in tloss if v is not None]
            ax.plot(xs, ys, label="train loss", color="tab:blue")
        ax.plot(steps, vloss, label="val loss (EMA)", color="tab:orange")
        # Mark the best val point.
        best_i = min(range(len(vloss)), key=lambda i: vloss[i])
        ax.scatter([steps[best_i]], [vloss[best_i]], color="red", zorder=5,
                   label=f"best {vloss[best_i]:.4f} @ {steps[best_i]}")
        ax.set_xlabel("step")
        ax.set_ylabel("epsilon-MSE loss")
        ax.set_title("Training progress")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=120)
        plt.close(fig)

    # Checkpoint helpers (only "best" and "latest" are kept; both overwritten)

    def _save(self, tag: str):
        src  = self.model.module if hasattr(self.model, "module") else self.model
        ckpt = {
            "step":     self.step,
            "best_val": self.best_val,
            "history":  self.history,
            "model":    src.state_dict(),
            "ema":      self.ema_model.state_dict(),
            "opt":      self.opt.state_dict(),
            "scaler":   self.scaler.state_dict(),
        }
        path = os.path.join(self.out_dir, f"checkpoint_{tag}.pt")
        torch.save(ckpt, path)
        log.info("Saved checkpoint_%s (step %d) -> %s", tag, self.step, path)

    def _resume(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        src  = self.model.module if hasattr(self.model, "module") else self.model

        if isinstance(ckpt, dict) and "model" in ckpt:
            # Training checkpoint: full state restore.
            src.load_state_dict(ckpt["model"], strict=False)
            self.ema_model.load_state_dict(ckpt["ema"], strict=False)
            if "opt" in ckpt:
                self.opt.load_state_dict(ckpt["opt"])
            if "scaler" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler"])
            self.step     = ckpt.get("step", 0)
            self.best_val = ckpt.get("best_val", float("inf"))
            self.history  = ckpt.get("history", [])
            log.info("Resumed from %s (step %d, best_val %.5f)",
                     path, self.step, self.best_val)
        else:
            # Bare weights file (no optimizer/step/history). Warm start only.
            state = ckpt.get("ema") if isinstance(ckpt, dict) and "ema" in ckpt else ckpt
            src.load_state_dict(state, strict=False)
            self.ema_model.load_state_dict(state, strict=False)
            log.warning("Warm-started weights from legacy checkpoint %s; "
                        "optimizer/step/history NOT restored (continuing at "
                        "step %d).", path, self.step)


# Entry point

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dbt_25deg.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)

    # Distributed setup
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    trainer = DBTTrainer(cfg, rank=rank, local_rank=local_rank, world_size=world_size)
    trainer.train()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
