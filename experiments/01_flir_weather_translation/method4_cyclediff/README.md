# Method 4 — CycleDiff for FLIR ADAS v2

## Overview

CycleDiff is a cycle diffusion model for unpaired image-to-image translation (arXiv:2508.06625).
It combines latent diffusion models with cycle consistency for high-quality unpaired translation.

## Training Pipeline

The training follows 3 stages per condition pair (clear → fog, clear → rain, clear → snow):

### Stage 1: Train VAE (AutoencoderKL)
```bash
# Clear (source)
accelerate launch train_vae.py --cfg configs/flir_clear2fog/clear_ae_kl_256x256_d4.yaml
# Fog (target)
accelerate launch train_vae.py --cfg configs/flir_clear2fog/fog_ae_kl_256x256_d4.yaml
```

### Stage 2: Train LDM (Latent Diffusion Model)
```bash
# Clear LDM
accelerate launch train_uncond_ldm.py --cfg configs/flir_clear2fog/clear_ddm_const4_ldm_unet6_114_ode_2.yaml
# Fog LDM
accelerate launch train_uncond_ldm.py --cfg configs/flir_clear2fog/fog_ddm_const4_ldm_unet6_114_ode_2.yaml
```

### Stage 3: Train Cycle Translator
```bash
python train_uncond_ldm_cycle.py --cfg configs/flir_clear2fog/translation_C_disc_timestep_ode_2.yaml
```

Repeat for rain and snow (configs in `flir_clear2rain/` and `flir_clear2snow/`).

## Data Structure Required

```
data/FLIR_ADAS_v2/
  train/
    clear/
      0.png, 1.png, ...
  test/
    clear/
      0.png, 1.png, ...

m3fd_experiments/method3_cyclegan/data/DSERT-RoLL/
  train/
    fog/
      0.png, 1.png, ...
    rain/
      0.png, 1.png, ...
  test/
    fog/
      0.png, 1.png, ...
    rain/
      0.png, 1.png, ...

Heavy_Snow/
  train/
    snow/
      0.png, 1.png, ...
  test/
    snow/
      0.png, 1.png, ...
```

## Pretrained Weights

Download the pretrained VAE weight:
```
https://ommer-lab.com/files/latent-diffusion/kl-f4.zip
```
Place at `pretrain_weight/model.ckpt` (referenced in VAE configs).

## Compute Requirements

- **VAE training**: ~6h per domain (4 domains = clear, fog, rain, snow)
- **LDM training**: ~48h per domain (4 domains)
- **Cycle translator**: ~36h per pair (3 pairs = clear→fog, clear→rain, clear→snow)
- **Total**: ~200-300 GPU hours
- **GPU**: Requires ≥24GB VRAM for batch sizes in configs

## FLIR ADAS v2 Specific Notes

- FLIR ADAS v2 has paired IR+RGB images; we use IR channel for thermal weather translation
- DSERT-RoLL provides real thermal fog/rain (already downloaded for CycleGAN Method 3)
- Heavy_Snow must be acquired separately (not in DSERT-RoLL v1)
- IR images are 3-channel pseudo-color; configs use 3-channel input
- Resolution: 256×256 latent (512×512 input with 286×286 crop during training)

## Testing/Inference

```bash
accelerate launch translation_uncond_ldm_cycle.py --cfg configs/flir_clear2fog/translation_C_disc_timestep_ode_2.yaml
```

## Dependencies

Install in `gateir` conda env:
```bash
pip install accelerate==0.23.0 einops ema-pytorch==0.3.0 pytorch-lightning==1.9.3 scikit-learn scipy thop timm tqdm tensorboard==2.16.2 fvcore albumentations omegaconf ninja imageio matplotlib numpy==1.24.4 transformers==4.34.0 protobuf==4.25.3 kornia webdataset
pip install -e git+https://github.com/toshas/torch-fidelity.git@v0.3.0#egg=torch-fidelity
```

Note: Requires torch 2.14+ (gateir env has 2.14.0+cu126) - CycleDiff was tested on 1.13.1. Some adaptation may be needed.

## Files

- `configs/flir_clear2fog/` — Clear → Fog configs
- `configs/flir_clear2rain/` — Clear → Rain configs
- `configs/flir_clear2snow/` — Clear → Snow configs
- `train_vae.py`, `train_uncond_ldm.py`, `train_uncond_ldm_cycle.py`, `translation_uncond_ldm_cycle.py` — Training scripts (copied from CycleDiff repo)
- `pretrain_weight/model.ckpt` — Pretrained VAE weight (download from ommer-lab.com)
- `results/` — Training outputs and checkpoints
- `sample_results/` — Inference sample grids