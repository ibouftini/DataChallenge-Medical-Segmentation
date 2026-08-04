"""
Diagnostic: dump the diffusion object's API + parameterisation.

DOLCE ships an inference-only guided_diffusion whose SpacedDiffusion lacks
`training_losses`, so we must supply the loss ourselves. This prints exactly
what building blocks and settings are available, without loading the 1.1 GB
checkpoint (the model is constructed but its weights are not needed here).

    python scripts/inspect_diffusion.py --config configs/dbt_25deg.yaml
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import train  # sets up the external/DOLCE path and guided_diffusion imports


def _enum(v):
    return f"{type(v).__name__}.{v.name}" if hasattr(v, "name") else repr(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "dbt_25deg.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model_args = train.dolce_model_args(cfg)
    model, diffusion = train.condtion_create_model_and_diffusion(**model_args)

    print("=== diffusion type ===")
    print("MRO:", [c.__name__ for c in type(diffusion).__mro__])

    print("\n=== public methods ===")
    methods = sorted(m for m in dir(diffusion)
                     if not m.startswith("__") and callable(getattr(diffusion, m, None)))
    for m in methods:
        print("  ", m)

    print("\n=== key attributes ===")
    for attr in ("num_timesteps", "model_mean_type", "model_var_type",
                 "loss_type", "rescale_timesteps"):
        print(f"  {attr} = {_enum(getattr(diffusion, attr, '<missing>'))}")

    print("\n=== building blocks present? ===")
    for m in ("q_sample", "p_mean_variance", "q_posterior_mean_variance",
              "_predict_xstart_from_eps", "_vb_terms_bpd",
              "_prior_bpd", "_scale_timesteps"):
        print(f"  {m}: {hasattr(diffusion, m)}")

    print("\n=== model channels ===")
    print("  in_channels :", getattr(model, "in_channels", "?"))
    print("  out_channels:", getattr(model, "out_channels", "?"))


if __name__ == "__main__":
    main()
