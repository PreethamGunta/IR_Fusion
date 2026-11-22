# 01 — FLIR ADAS v2 weather translation and detectors

## Overview

This workstream produced the **synthetic adverse-weather data** used everywhere
else and the original single-detector baselines.

Two questions drove it:

1. Can unpaired translation (CycleGAN, CycleDiff) turn clear thermal FLIR frames
   into convincing fog/rain/snow, using DSERT-RoLL as the target domain?
2. Does a detector trained on clear + synthetic weather close the weather gap,
   and at what cost? The historical `3a_*` runs are mixed clear + synthetic-fog
   runs; `3b` is an evaluation mode for the clear-only model, not a second
   training protocol.

The answer recorded here: CycleGAN and CycleDiff-75 fog translations are usable;
the full CycleDiff blend is not; rain/snow translation exists but the detector
runs were not completed. This is why the project moved to the gate + filter
design in workstream 02.

## Directory contents

```
01_flir_weather_translation/
├── README.md                     this file
├── train_detector.py             YOLO training and 3B evaluation
├── blend_cyclediff_fog.py        blends clear + CycleDiff fog at 75 %
├── convert_video_thermal_test_yolo.py  COCO -> YOLO labels for the test set
├── download.py / inspect_dataset.py / prepare.py   original FLIR prep
├── method1_classical/            classical fog/rain/snow overlays
│   ├── generate.py               main generator
│   ├── apply_weather_classical.py  original method-1 implementation (recovered)
│   └── run_method1.sh            original driver (recovered)
├── method2_depth/                depth-based Koschmieder fog
├── method3_cyclegan/             unpaired CycleGAN translation
│   ├── train_cyclegan.py         resumable training (checkpoint every 100 iters)
│   ├── translate.py              inference
│   ├── models.py                 generator/discriminator
│   └── runs/{fog,rain,snow}/     checkpoints
├── method4_cyclediff/            CycleDiff (latent diffusion + cycle)
│   ├── configs/                  per-condition YAMLs (contain absolute paths)
│   ├── taming/ unet/ ddm/ ...    model code
│   ├── sample_results/           translated samples
│   └── results/                  VAE/LDM/translator checkpoints (~50 GB)
├── runs/                         YOLO run dirs (3a_cyclegan_fog-3, 3a_cyclediff_fog-8, ...)
├── results/                      historical 3A result JSONs (mixed clear + synthetic fog)
├── report/                       report.pdf / report.tex / figures
│   └── build/                    LaTeX build logs (report.aux/.log/.out)
├── figures/sample_images/        30 qualitative CycleGAN/strength samples
├── scripts/                      historical SLURM/srun wrappers, incl.
│                                 run_fog_vae_sbatch.sh (recovered)
└── logs/                         translation and detector logs
```

## Data requirements

| Path | Needed for |
|---|---|
| `data/flir_adas_v2/FLIR_ADAS_v2/` | clear source frames and test-set frames |
| `data/flir_adas_v2/yolo/thermal/` | labels (`labels_all`, `labels_vtt_test`) |
| `data/flir_adas_v2/yolo/train_3a/` | sequence-disjoint train/val splits |
| `data/flir_adas_v2/synthetic/` | generated fog/rain/snow |
| `data/dsert_roll/` | target-domain Fog / Heavy_Rain / Heavy_Snow |
| `models/pretrained/yolo12s.pt` | detector initialisation |

Generated data is written to `data/flir_adas_v2/synthetic/<method>/<condition>/`.

## Execution

All commands run from this directory. Activate the `gateir` environment first.

```bash
cd experiments/01_flir_weather_translation

# CycleGAN translation (fog shown; rain/snow are --config rain|snow)
python method3_cyclegan/train_cyclegan.py --config fog --epochs 50
python method3_cyclegan/translate.py --config fog --split test

# CycleDiff (heavy: 3 stages per condition)
accelerate launch train_vae.py --cfg method4_cyclediff/configs/flir_clear2fog/clear_ae_kl_256x256_d4.yaml
accelerate launch train_uncond_ldm.py --cfg method4_cyclediff/configs/flir_clear2fog/clear_ddm_const4_ldm_unet6_114_ode_2.yaml
python method4_cyclediff/train_uncond_ldm_cycle.py --cfg method4_cyclediff/configs/flir_clear2fog/translation_C_disc_timestep_ode_2.yaml

# Detector: clear-only baseline; 3a = mixed clear + synthetic; 3b = evaluate clear baseline on translated test (no training)
python train_detector.py --mode clear_baseline --epochs 100
python train_detector.py --mode 3a --condition fog --method cyclegan --epochs 100
python train_detector.py --mode 3b --condition fog --method cyclegan --epochs 100
```

CycleDiff training is the expensive part (~30–60 GPU-hours per condition pair).
The historical wrappers in `scripts/` and the top-level `run_*.sh` files are
kept for provenance but were written against the old worktree layout; prefer the
Python entry points above.

## Results

Source: `results/*.json` (historical mixed clear + synthetic-fog runs, YOLOv8s).

| Result | n | mAP50-95 | mAP50 | P | R |
|---|---:|---:|---:|---:|---:|
| CycleGAN fog — clear test | 1,200 | 0.32739 | 0.54193 | 0.65078 | 0.51818 |
| CycleGAN fog — 100 % fog test | 3,749 | 0.14448 | 0.27182 | 0.40460 | 0.28190 |
| CycleDiff-75 — clear test | 1,200 | 0.32702 | 0.54752 | 0.68111 | 0.52710 |
| CycleDiff-75 — 75 % fog test | 3,749 | 0.16221 | 0.30371 | 0.39515 | 0.30286 |
| CycleDiff full — 100 % fog test | 3,749 | 0.00000 | 0.00000 | 0.00031 | 0.00136 |

Matched test-set re-score of the CycleGAN model (same 3,749 frames for clear and
fog): clear **0.16686 / 0.30548**, fog **0.14448 / 0.27182**, gap ≈ **0.022**.

The old clear−fog gap of ≈0.18 was a **protocol artefact**, not weather: clear
was scored on 1,200 in-domain frames and fog on 3,749 test-set frames. This is the
central methodological lesson of the workstream and the reason workstream 02
uses matched frames everywhere.

## Reproducibility notes

- `method4_cyclediff/configs/*.yaml` and the generated YAMLs contain absolute
  paths into this release; run `python3 scripts/relocate_paths.py` after moving
  the folder.
- Checkpoints are large (≈50 GB in `method4_cyclediff/results/`); they are kept
  because they are the only copies.
- Rain detector runs stopped after one epoch and snow has no detector run
  (partial-scope limitation, see `docs/KNOWN_ISSUES.md`).
