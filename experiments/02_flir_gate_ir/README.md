# 02 — GATE-IR on FLIR ADAS v2 synthetic fog

## Overview

This is the main experiment of the project. It asks whether a **binary
clear/fog gate** that routes fog frames through a learned dehazer, leaving clear
frames untouched, beats a detector trained on **clear + unfiltered fog** — and
whether gating is cheaper than filtering every frame.

The comparison is **B1 vs G1**:

| | B1 (baseline) | G1 (gated + filter) |
|---|---|---|
| Clear frames | raw | raw |
| Fog frames | raw synthetic fog | U-Net-dehazed fog |
| Train set | clear + raw fog | clear + filtered fog |
| Inference | detector only | gate routes fog through the U-Net |
| Detector | YOLOv12s | YOLOv12s (same recipe) |

Everything is scored on the **same 3,749 clear and 3,749 fog test-set frames** with
the same 15-class labels.

### Scope caveat

FLIR ADAS v2 has **no real fog**. All fog is CycleGAN / CycleDiff output, so the
gate can learn generator artefacts. The result is a **synthetic-fog proof of
concept**, not a real-weather claim. Cross-generator accuracy is therefore
reported separately (`results/gate_cyclegan.json`).

## Final results (matched test-set frames)

### CycleGAN fog — YOLOv12s, 100-epoch budget

| Arm | Clear mAP50 | Clear mAP50-95 | Fog mAP50 | Fog mAP50-95 | Gap (50-95) |
|---|---:|---:|---:|---:|---:|
| B1 | 0.2849 | 0.1525 | 0.2430 | 0.1266 | 0.0259 |
| **G1 U-Net** | **0.3063** | **0.1683** | **0.2856** | **0.1535** | **0.0148** |
| Δ | +0.0214 | +0.0158 | **+0.0426 (+17.5 %)** | **+0.0269 (+21.3 %)** | **−42.9 %** |

B1 early-stopped at epoch 55 (best 30); G1 at epoch 57 (best 32). Source:
`results/{b1_yolo12s_100,g1_unet_100}_{clear,fog}_vtt.json`.

### CycleDiff-75 fog — YOLOv12s, 100-epoch budget

| Arm | Clear mAP50 | Clear mAP50-95 | Fog mAP50 | Fog mAP50-95 | Gap (50-95) |
|---|---:|---:|---:|---:|---:|
| B1 | 0.2746 | 0.1491 | 0.2747 | 0.1495 | −0.0004 |

B1 stopped at epoch 65 (best 40). **G1 is incomplete**: stopped at epoch 27/100,
resumable (see below). Source: `results/b1_cyclediff75_yolo12s_100_*_vtt.json`.

### Filter screen — YOLOv8s, 60 epochs (CycleGAN)

| Filter | Clear mAP50-95 | Fog mAP50-95 | Gap |
|---|---:|---:|---:|
| B1 (none) | 0.16464 | 0.14134 | 0.02330 |
| gamma (FogEnhancer) | 0.15883 | 0.13694 | 0.02189 |
| CLAHE | 0.16357 | 0.13352 | 0.03005 |
| DCP | 0.16505 | 0.13940 | 0.02565 |
| **U-Net dehazer** | **0.16757** | **0.15241** | **0.01516** |
| gamma + CLAHE | 0.16104 | 0.13838 | 0.02266 |

**Only the learned U-Net helps; all classical filters fail.**

### Preliminary 40-epoch YOLOv12s (CycleGAN)

| Arm | Clear | Fog | Gap |
|---|---:|---:|---:|
| B1 | 0.14639 | 0.12340 | 0.02299 |
| G1 U-Net | 0.15178 | 0.14333 | 0.00845 |

Kept for context; the 100-epoch table supersedes it.

### Cross-generator transfer (negative result)

A U-Net trained on CycleGAN fog **hurts** on CycleDiff-75
(40-epoch YOLOv8s: clear −0.0072, fog −0.0107 mAP50-95). Retraining the U-Net on
CycleDiff-75 pairs improves it but still lands below B1 (0.1638 vs 0.1664 clear,
0.1560 vs 0.1619 fog). The reconstruction objective is misaligned with
detection, not merely generator-specific.

### Gate metrics

`results/gate_cyclegan.json`:

| test | accuracy | F1 | FNR fog→clear | FPR clear→fog |
|---|---:|---:|---:|---:|
| Clear test set | 0.9141 | 0.0000 | 0.0000 | 0.0859 |
| Fog test set (CycleGAN) | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| Fog test set (CycleDiff-75) | 0.6612 | 0.7961 | 0.3388 | 0.0000 |

The gate separates the generator it saw but transfers poorly — the same
generator-specificity seen in the filter.

## Directory contents

```
02_flir_gate_ir/
├── src/
│   ├── paths.py             all paths derived from this file (no hardcoded roots)
│   ├── gate.py              binary Clear/Fog gate (5 features + MLP)
│   ├── filters.py           gamma / CLAHE / gamma+CLAHE / DCP
│   ├── unet_dehazer.py      paired U-Net dehazer
│   ├── train_unet.py        train the dehazer on (clear, fog) pairs
│   ├── apply_unet.py        run the dehazer over a directory
│   ├── build_datasets.py    matched b0/b1/b2/b3/g1 datasets + YAMLs
│   ├── build_direct.py      direct YAMLs for the 100-epoch B1/G1 runs
│   ├── build_screen.py      filter-screen YAMLs
│   ├── build_cyclediff.py   CycleDiff-75 B1/G1 YAMLs
│   ├── build_ablations.py   B0/B2/B3 YAMLs
│   ├── train_detector.py    YOLO training (resume support)
│   ├── eval_detector.py     matched test-set evaluation + provenance JSON
│   ├── route.py             materialise gated_learned / gated_oracle sets
│   ├── summarize*.py        Markdown tables
│   └── make_report.py       builds results/REPORT.md
├── scripts/
│   ├── run_main_100.sh             CycleGAN B1 + G1 (100 epochs)
│   ├── run_main_100_cyclediff.sh   CycleDiff-75 B1 + G1 (100 epochs)
│   ├── run_flir_main_100_all.sh    both chains in one allocation
│   ├── supervise_main100.sh        retry/resume supervisor
│   ├── progress_watch.sh           15-minute status logger
│   ├── run_screen_direct.sh        Phase 1 filter screen
│   ├── run_phase2_direct.sh        Phase 2
│   ├── run_phase3_direct.sh        Phase 3 ablations
│   ├── run_prep.sh / run_training.sh / run_eval.sh
│   └── ...                         historical wrappers
├── configs/                  generated dataset YAMLs (absolute paths)
├── runs/                     57 Ultralytics run dirs, weights included
├── results/                  JSONs, REPORT.md, all_results.md, RUN_STATUS.md,
│                             gate_cyclegan.pkl, unet_dehazer*.pt
├── logs/                     training/eval/orchestration logs (141 MB)
└── data/                     vtt_clear symlink tree + filtered images (1.2 GB)
```

## Data requirements

| Path | Role |
|---|---|
| `data/flir_adas_v2/FLIR_ADAS_v2/test/clear` | 3,749 clear test-set frames |
| `data/flir_adas_v2/FLIR_ADAS_v2/video_thermal_test/data` | 3,749 fog source frames |
| `data/flir_adas_v2/yolo/thermal/labels_vtt_test` | shared test labels |
| `data/flir_adas_v2/yolo/train_3a/` | 9,051 clear train / 1,635 val |
| `data/flir_adas_v2/synthetic/cyclegan/fog/` | CycleGAN fog |
| `data/flir_adas_v2/synthetic/cyclediff_75fog/fog/` | CycleDiff-75 fog |
| `models/pretrained/yolo12s.pt`, `yolov8s.pt` | detector init |

`data/filtered/…` is created at run time and deleted afterwards to save quota.

## Execution

```bash
cd experiments/02_flir_gate_ir

# rebuild generated YAMLs for this checkout
python -m src.build_direct init

# inside a live salloc allocation (never as a direct salloc child)
srun --jobid="$SLURM_JOB_ID" --overlap --cpus-per-task=16 --gres=gpu:1 \
     bash scripts/run_main_100.sh
srun --jobid="$SLURM_JOB_ID" --overlap --cpus-per-task=16 --gres=gpu:1 \
     bash scripts/run_main_100_cyclediff.sh
```

`scripts/supervise_main100.sh` wraps the two chains, retries on failure, and
skips arms whose `runs/<name>/DONE` marker exists. Every arm resumes from
`runs/<name>/weights/last.pt`.

### Resuming the incomplete CycleDiff-75 G1

```bash
python -m src.train_detector --dataset g1_unet_cd75_direct \
    --name g1_unet_cd75_yolo12s_100 --resume
python -m src.eval_detector \
    --weights runs/g1_unet_cd75_yolo12s_100/weights/best.pt \
    --data configs/g1_unet_cd75_direct_eval_clear.yaml \
    --name g1_unet_cd75_100_clear_vtt --force
python -m src.eval_detector \
    --weights runs/g1_unet_cd75_yolo12s_100/weights/best.pt \
    --data configs/g1_unet_cd75_direct_eval_fog.yaml \
    --name g1_unet_cd75_100_fog_vtt --force
```

Exact state and the U-Net regeneration recipe: `results/RUN_STATUS.md`.

## Reproducibility notes

- Training recipe for every arm: `epochs=100`, `batch=16`, `imgsz=640`,
  `cache=ram`, `close_mosaic=10`, `patience=25`, `seed=0`, AdamW auto.
- Training-time validation is **clear-only** (`yolo/train_3a/images/val`, 1,635
  frames) and drives `best.pt` and early stopping. The reported numbers are the
  frozen matched test-set evaluations.
- Single seed; no variance estimate.
- Reported `mAP50-95` = Ultralytics `metrics/mAP50-95(B)`.
- The 100-epoch CycleGAN numbers supersede the 40-epoch preliminary table.
