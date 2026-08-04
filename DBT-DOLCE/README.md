# DBT Learned Reconstruction

Reproduction and improvement of learned Digital Breast Tomosynthesis (DBT)
reconstruction using only public resources.  The method fine-tunes
[DOLCE](https://github.com/wustl-cig/DOLCE) (a conditional diffusion model for
CT reconstruction) on public breast phantom data, targeting a limited-angle DBT
geometry: 9 projections over 25 degrees.

Adaptation is a **full fine-tune** of all 273M weights -- the medical-CT ->
breast domain shift is large enough that nothing less adapts properly.  DOLCE is
trained in the `[-1,1]` image range, so GT and conditioning are mapped to
`[-1,1]` for training/sampling and metrics are computed back in `[0,1]`.

---

## Method overview

```
Raw breast phantom (.nrrd)
        |
        v
  Label -> attenuation map  (cm^-1 at ~20 keV)
        |
        v
  Extract sagittal slices
        |
        v
  Pad to square (INTER_AREA anti-aliased) + resize to 512x512
        |
        v
  LEAP forward projection  (9p25d)
        |
        v
  SIRT reconstruction  (200 iters)  -> conditioning image
        |
        v
  HDF5: (gt_slice, sirt_slice)  [gt/sirt stored float16; mapped to [-1,1] at load]
        |
        v
  Fine-tune DOLCE UNet  (all 273M weights)
        |
        v
  DDIM/DDPM sampling on p_mean_variance pred_xstart + proximal data-consistency
        |
        v
  Evaluation: PSNR / SSIM / FRC resolution / per-tissue MAE
```

**Conditioning input to DOLCE**: two-channel tensor `[cond_rls, cond_fbp]`
where both channels contain the SIRT reconstruction (we use SIRT for both slots
since we do not have a separate FBP baseline).

**Adaptation**: every one of the 273M weights is trained.  Base DOLCE (a
medical-CT prior) collapses to its prior mean on breast data, so the full
fine-tune is what makes the model track breast anatomy.

**Sampling**: the reverse step takes x0 from `diffusion.p_mean_variance`'s
`pred_xstart` (DOLCE's own, parameterisation-correct conversion of the model
output), then applies the DDIM/DDPM update -- not a hand-rolled epsilon formula.

**Proximal data-consistency**: at each reverse step, solve
`argmin_x ||Ax - b||^2 + rho * ||x - x_hat||^2` (CG or APGM).  For the
rank-deficient 9p25d operator, rho must be large enough to trust the
prior in the unmeasured null space (a tiny rho lets it blow up).  Tune rho per
checkpoint with `scripts/sweep_rho.py`; evaluate `--no_prox` first to isolate
the model.

---

## Repository layout

```
.
|-- configs/
|   |-- dbt_25deg.yaml          # master config (paths, geometry, model, training)
|   `-- leap_dbt_25deg.cfg      # LEAP geometry: 9p25d
|-- data/
|   |-- augmentation.py         # online augmentations (flips, zoom)
|   |-- dataset.py              # PyTorch Dataset + DataLoader factories
|   `-- preprocess.py           # NRRD -> HDF5 preprocessing pipeline
|-- model/
|   `-- finetune.py             # full fine-tune setup, save/load helpers
|-- physics/
|   `-- dbt_projector.py        # LEAP wrapper: forward, backward, SIRT, prox
|-- scripts/
|   |-- setup.sh                # environment setup (conda + LEAP + guided_diffusion)
|   |-- download_weights.sh     # download pretrained DOLCE model512_all.pt
|   |-- prepare_data.sh         # run preprocessing pipeline
|   |-- train.sh                # launch training
|   |-- sweep_rho.py            # tune the prox weight rho against a checkpoint
|   |-- analyze_results.py      # metrics.json/sweep.json -> tables + figures
|   |-- inspect_raw.py          # check raw .nrrd labels + resize aliasing
|   `-- inspect_h5.py           # check the actually-saved slices (true pixels)
|-- train.py                    # full fine-tuning loop
|-- evaluate.py                 # evaluation loop + metrics
|-- diagnose.py                 # verify parameterisation / conditioning / recon
|-- eval_metrics.py             # masked PSNR/SSIM, FRC resolution, per-tissue MAE
`-- environment.yml             # conda environment spec
```

---

## Requirements

- Linux with CUDA >= 11.8
- conda (Miniconda or Anaconda)
- 1x NVIDIA A100 80 GB VRAM GPU
- ~150 GB disk for the full Zenodo dataset + processed slices

---

## Step 1 -- Get the dataset

Download the **821 compressed breast phantoms** from Zenodo:

```
https://zenodo.org/records/18617629
```

Place all `.nrrd` files in `dataset/compressed/`:

```
dataset/
`-- compressed/
    |-- KBCT010001L_compressed_39_0.nrrd
    |-- KBCT010001R_compressed_39_0.nrrd
    `-- ...
```

---

## Step 2 -- Environment setup

Clone this repository, then run:

```bash
bash scripts/setup.sh
```

This will:
1. Clone DOLCE into `external/DOLCE/`
2. Create the `dbt-dolce` conda environment from `environment.yml`
3. Build and install LEAP (differentiable projector) into the environment
4. Install `guided_diffusion` from DOLCE in editable mode
5. Verify the LEAP import and run a CPU projector smoke test
   (shape round-trip, forward/backprojector adjoint test, SIRT sanity)

Activate the environment:

```bash
conda activate dbt-dolce
```

To re-run the projector smoke test at any time:

```bash
python tests/test_projector.py     # or: pytest tests/test_projector.py
```

A green smoke test means the physics core (forward / backprojection / SIRT)
works end to end, so the preprocessing and evaluation stages will run.

---

## Step 3 -- Download pretrained weights

```bash
bash scripts/download_weights.sh
```

Downloads `model512_all.pt` (~1.1 GB) from the official DOLCE Google Drive
release into `model_zoo/`.

---

## Step 4 -- Preprocess the data

```bash
bash scripts/prepare_data.sh [cuda:0]
```

The optional argument sets the GPU device (default: `cuda:0`).

This script:
- Converts each NRRD phantom to a set of 512x512 HDF5 slices
- Splits patients 80 / 10 / 10 (train / val / test) with seed 42
- Writes `dataset/split_25deg.json` and per-subset file lists under
  `dataset/processed_25deg/`

Processing 821 phantoms takes roughly 6-10 hours on a single GPU.

Key preprocessing steps per slice:
- Tissue labels (0=air, 1=adipose, 2=fibroglandular, 3=skin) converted to
  linear attenuation at ~20 keV: `[0.000, 0.410, 0.780, 0.780]` cm^-1
- Sagittal slices padded to square then resized to 512x512
- GT normalised to `[0, 1]` by dividing by 0.780 cm^-1 (fibroglandular max)
- SIRT conditioning: 9-angle forward projection then 200 SIRT iterations
- Conditioning stored with negatives removed; the dataset loader applies
  DOLCE's per-image min-max normalisation at load time (matches model512_all.pt)
- CLAHE is OFF by default (DOLCE conditions on plain min-max reconstructions);
  enable with `--clahe` only to experiment
- Air-dominated slices skipped (tissue fraction < 5%)

To customise:

```bash
python data/preprocess.py \
    --raw_dir    dataset/compressed \
    --out_dir    dataset/processed_25deg \
    --cfg        configs/leap_dbt_25deg.cfg \
    --split_file dataset/split_25deg.json \
    --sirt_iters 200 \
    --device     cuda:0
```

---

## Step 5 -- Fine-tuning

Every weight is trained; the hyperparameters come from the config.  Run on a
single A100 80 GB GPU:

```bash
bash scripts/train.sh
```

Hangup-proof (survives SSH/terminal close):

```bash
nohup bash scripts/train.sh >> scripts/train.log 2>&1 & disown
```

Checkpointing is nnU-Net style: only two files are kept in `output_dir`
(default `runs/dbt_25deg_full`), overwritten each validation:
- `checkpoint_best.pt`   -- lowest val loss (EMA weights); use for inference
- `checkpoint_latest.pt` -- most recent; used for auto-resume

Each stores step, best_val, metric history, model + EMA weights, optimizer and
scaler state.  `training.auto_resume: true` (default) continues from
`checkpoint_latest.pt` automatically, so a crash/preemption loses no progress.
Each validation also writes `progress.json` and `progress.png` (train/val loss
curves).

Key training settings (all in `configs/dbt_25deg.yaml`):

| Parameter       | Value           | Notes                                   |
|-----------------|-----------------|-----------------------------------------|
| batch_size      | 2               | single A100 80 GB GPU (full fine-tune)  |
| lr              | 2e-5            | AdamW; low, to protect the pretrained init |
| max_steps       | 100000          |                                         |
| ema_rate        | 0.9999          | val + inference use the EMA weights     |
| data_range      | -1..1           | DOLCE's native image range              |
| diffusion_steps | 1800            | linear noise schedule, epsilon target   |
| use_fp16        | true            | AMP autocast (gradient checkpointing off) |

---

## Step 6 -- Evaluation

```bash
python evaluate.py --config configs/dbt_25deg.yaml \
    --ckpt runs/dbt_25deg_full/checkpoint_best.pt \
    --sampler ddim --split test --device cuda:0
```

`--no_prox` disables data-consistency (isolate the model); omit it to include
the proximal step.  Tune the prox weight first with
`python scripts/sweep_rho.py --ckpt <checkpoint> ...`.

Results are written to `results/dbt_25deg/`:
- `metrics.json` -- per-slice values + per-patient bootstrap summary
- `compare_000X.png` -- GT | SIRT | recon | error panels (true pixels)
- `slice_0000.npz` ... -- `gt`, `pred`, `sirt`, `mask` arrays

Turn results into tables/figures with
`python scripts/analyze_results.py --metrics results/dbt_25deg/metrics.json`.

### Diagnostics

- `python diagnose.py --ckpt <ckpt>` -- verify parameterisation, whether
  conditioning is used, and reconstruction vs SIRT (base and checkpoint).
- `python scripts/inspect_raw.py` / `inspect_h5.py` -- check raw labels and the
  actual saved slices.

### Metrics panel (see `eval_metrics.py`)

All full-reference metrics are computed INSIDE the breast mask, and
intensity-sensitive ones after an affine intensity match (limited-angle
reconstructions carry a low-frequency bias that otherwise dominates).

| Metric | What it measures | Why it is reliable here |
|--------|------------------|-------------------------|
| `ssim`, `psnr`, `nrmse` | masked structural / intensity fidelity | bias-tolerant, breast-only |
| `frc_res_x_mm`, `frc_res_z_mm`, `frc_anisotropy` | directional Fourier Ring Correlation resolution | captures the limited-angle depth (Z) blur directly |
| `mae_adipose`, `mae_fibroglandular`, `mae_skin` | per-tissue attenuation error | clinically meaningful (density) |
| `sirt_psnr`, `sirt_ssim` | SIRT conditioning baseline | quantifies the diffusion improvement |
| `data_residual` | `||Ax-b||/||b||` consistency | sanity check only (NOT a quality metric) |
| `uncertainty_*` | ensemble std vs error (with `--n_samples > 1`) | calibration of the generative posterior |

Reported per metric: per-patient median, IQR, and bootstrap 95% CI
(aggregated by patient to avoid inflating n with correlated slices).

To evaluate base DOLCE (no fine-tuning), omit the checkpoint:

```bash
python evaluate.py --config configs/dbt_25deg.yaml --ckpt "" \
    --sampler ddim --split test --no_prox --device cuda:0
```

To evaluate a single mode manually:

```bash
python evaluate.py \
    --config    configs/dbt_25deg.yaml \
    --ckpt runs/dbt_25deg_full/checkpoint_best.pt \
    --sampler   ddim \
    --split     test \
    --device    cuda:0
```

Optional flags:
- `--no_prox` -- disable proximal data-consistency (isolate the model)
- `--n_samples N` -- average N posterior samples (>1 adds an uncertainty map)
- `--max_slices N` -- limit to N slices for quick debugging
- `--figs N` -- save N GT|SIRT|recon|error comparison PNGs

---

## Configuration reference

All settings live in `configs/dbt_25deg.yaml`.  CLI arguments override YAML
values where both exist.

```yaml
data:
  raw_dir:       dataset/compressed        # input NRRD files
  processed_dir: dataset/processed_25deg   # output HDF5 slices
  split_file:    dataset/split_25deg.json  # patient-wise split

geometry:
  leap_cfg:        configs/leap_dbt_25deg.cfg
  num_projections: 9
  angle_range:     25     # degrees
  sirt_iterations: 200
  image_size:      512
  pixel_size_mm:   0.273

model:
  pretrained_ckpt:      model_zoo/model512_all.pt
  image_size:           512
  num_channels:         128
  num_res_blocks:       2
  num_heads:            4
  num_head_channels:    64        # must match model512_all.pt
  attention_resolutions: "32,16,8"  # must match model512_all.pt
  resblock_updown:      true       # must match model512_all.pt
  dropout:              0.0
  use_fp16:             true
  diffusion_steps:      1800
  noise_schedule:       linear
  weighted_condition:   false
  use_condition:        rls        # conditioning slot used (rls or fbp)
  data_range:           "-11"      # DOLCE's native range; metrics map back to [0,1]

training:                          # all 273M weights are fine-tuned
  batch_size:     2
  num_workers:    4
  lr:             2.0e-5           # low, to protect the pretrained init
  weight_decay:   0.0
  ema_rate:       0.9999
  log_interval:   100
  val_interval:   5000             # validate + write best/latest + progress plot
  max_steps:      100000
  resume_checkpoint: ""            # explicit path overrides auto_resume
  auto_resume:    true             # else load <output_dir>/checkpoint_latest.pt
  output_dir:     runs/dbt_25deg_full

sampling:
  sampler:      ddim      # ddim or ddpm
  ddim_steps:   100
  eta:          0.0       # deterministic DDIM
  prox_solver:  cgrad     # cgrad (parameter-free default) or apgm
  rho_start:    1.0       # trust the prior in the rank-deficient null space;
  rho_end:      0.1       # sweep per checkpoint with scripts/sweep_rho.py

evaluation:
  results_dir: results/dbt_25deg
```

---

## DBT geometry

| Parameter          | Value      |
|--------------------|------------|
| Acquisition arc    | 25 degrees |
| Number of angles   | 9          |
| Detector columns   | 512        |
| Pixel pitch        | 0.273 mm   |
| Reconstruction     | 512x512    |
| Geometry type      | Parallel beam (2D per-slice) |

Each sagittal slice is reconstructed independently as a 2D problem.

---

## Tissue attenuation values

Used to convert integer tissue labels to linear attenuation (cm^-1) at an
effective energy of approximately 20 keV.

| Label | Tissue         | Attenuation (cm^-1) |
|-------|----------------|---------------------|
| 0     | Air            | 0.000               |
| 1     | Adipose        | 0.410               |
| 2     | Fibroglandular | 0.780               |
| 3     | Skin           | 0.780               |

GT slices are normalised to `[0, 1]` by dividing by 0.780 (the maximum).

---

## Credits

- **DOLCE**: Gao et al., ICCV 2023 -- https://github.com/wustl-cig/DOLCE
- **LEAP**: differentiable X-ray CT projector bundled with DOLCE
- **Dataset**:Pacheco, G., Michielsen, K.& Sechopoulos, I. (2026). Patient-derived compressed digital breast phantom dataset for mammography and digital breast tomosynthesis simulations (Versions Version 1.0 — Initial public release (821 phantoms)) Zenodo.
- **Dataset2 (not used)**: Sarno, A., Mettivier, G., di Franco, F., Varallo, A., Bliznakova, K., Hernandez, A. M., Boone, J. M.& Russo, P. (2021). Dataset of patient-derived 3D digital breast phantoms for research in digital breast tomosynthesis and digital mammography. Zenodo. https://doi.org/10.5281/zenodo.4515360