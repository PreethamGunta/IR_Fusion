# 05 — UNIRI-TID detector bench

## Overview

A shared harness for training and scoring detectors on UNIRI-TID. It exists so
every baseline (YOLOv8/10/11/12, RT-DETR-l) is trained and evaluated with one
recipe and one scorer, with per-condition metrics and COCO-style APs.

The bench is also the scorer used by workstream 04.

## Directory contents

```
05_detector_bench/
├── README.md
├── __init__.py
├── debugmode.py              tiny debug runs (quarantined results)
├── train_detector.py         training entry point (YOLO family)
├── data/                     UNIRI-TID prep: prepare_uniri, make_splits,
│                             contact_sheet, label_conditions, seq_preview,
│                             build_sequences, find_sequences, read_counter
├── eval/detect_eval.py       the shared scorer (test/val, per-condition, APm)
├── runs/                     one run dir per detector (weights included)
├── results/                  one JSON per run per split
├── contact_sheets/           qualitative contact sheets (19 MB)
├── weights/                  local pretrained copies used by the runs
├── scripts/                  overnight.sh, sweep.sbatch, estimate_runtime.py
└── logs/
```

## Data requirements

`data/uniri_tid/` — see `experiments/04_uniri_gate/README.md` for the layout.
The scorer reads `data/uniri_tid/conditions.csv` for per-condition AP.

## Execution

From this directory:

```bash
# train one detector (YOLOv8s shown)
python train_detector.py --model weights/yolov8s.pt --name v8s --epochs 100 --cache ram

# evaluate it on the test split
python eval/detect_eval.py --run v8s --split test
```

`train_detector.py` accepts `--model` (a `.pt` or a model `.yaml`), `--name`,
`--epochs`, `--imgsz`, `--batch`, `--workers`, `--cache`, `--patience`,
`--save-period`, `--seed`, `--weather-aug`, `--data`, `--no-resume`.
`eval/detect_eval.py` accepts `--run` or `--weights`, `--split`, `--imgsz`,
`--conf`, `--iou`, `--batch`.

## Results (test split)

From the `overall` block of the per-model evaluation files. Per-run evaluation JSONs are produced locally by the evaluation scripts and are not versioned; the tables below are the record.

| Model | mAP50-95 | mAP50 | APm clear | APm fog | APm rain | clear−fog (APm) |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8s | 0.6204 | ~0.96 | 0.6623 | 0.5344 | 0.5951 | 0.128 |
| YOLO11s | 0.6194 | ~0.96 | 0.6585 | 0.5264 | 0.6038 | 0.132 |
| **YOLOv12s** | **0.6296** | ~0.97 | **0.6670** | **0.5733** | 0.6025 | **0.094** |
| YOLOv8s + weather aug | 0.6278 | ~0.96 | 0.6652 | 0.5691 | 0.6035 | 0.096 |
| YOLOv10s | 0.6063 | ~0.96 | 0.6459 | 0.5316 | 0.5873 | 0.114 |
| YOLOv8s-P2 | 0.6071 | ~0.97 | 0.6480 | 0.4954 | 0.5909 | 0.153 |
| ⚠️ RT-DETR-l | 0.2778 | 0.5324 | — | — | — | INVALID |

- mAP@50 is saturated; use mAP@50-95 and APm.
- **RT-DETR-l is invalid**: 29 epochs (walltime without requeue). `eval_rtdetrl_*`
  must not be quoted as a baseline until rerun.
- `results/*.json` also contain `overall` (images, boxes, detections, APs,
  AR100), `per_condition`, `size_profile` and a `crosscheck` block.

## Reproducibility notes

- Same recipe for every detector: 100 epochs, AdamW lr0=1e-3, cosine,
  `patience=25`, `close_mosaic=10`, seed 0, `cache=ram`.
- `train_detector.py` sets `YOLO_CONFIG_DIR` inside the experiment so
  Ultralytics does not write to the home config.
- Debug runs are quarantined so they cannot be mistaken for real results.
