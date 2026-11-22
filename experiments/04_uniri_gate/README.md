# 04 — UNIRI-TID de-risking baselines

## Overview

UNIRI-TID is the only **real-weather** thermal dataset in the project. This
workstream asked whether a weather gate/enhancer has value beyond latency on
real fog/rain, before investing in heavy restoration baselines.

**Outcome: no.** Once a detector is trained on mixed real clear+fog data, the
clear/fog gap essentially closes, so a gate has little accuracy headroom left.
The project therefore kept FLIR synthetic fog as the main track.

This directory now contains the variant builder and the logs from those runs;
the detector results themselves are in workstream 05.

## Directory contents

```
04_uniri_gate/
├── README.md
├── src/
│   └── build_variants.py      builds gate/variant feature tables + MLP models
├── scripts/
│   ├── run_shortlist.sh       weather-aug YOLOv12s + RT-DETR-l rerun
│   ├── run_variants.sh        preprocessing variants (always-on / gated / oracle)
│   └── wait_then_run.sh       waits for a free allocation, then launches
└── logs/                      four run logs; no result JSONs were produced
```

## Data requirements

`data/uniri_tid/`:

```
uniri_tid/
├── raw/                       original sequences
├── annotations/               COCO/All-In-One annotations
├── yolo/uniri.yaml            sequence-disjoint, condition-stratified split
├── yolo_debug/                debug subset
├── conditions.csv             frame -> condition (clear/fog/rain)
├── sequences.csv / splits.csv / manifest.csv
└── counter*.csv / counter_templates.json
```

The shared scorer is `experiments/05_detector_bench/eval/detect_eval.py`.

## Execution

```bash
cd experiments/04_uniri_gate

# weather-augmented YOLOv12s + RT-DETR-l rerun
bash scripts/run_shortlist.sh

# preprocessing variants
bash scripts/run_variants.sh
```

Both scripts locate the release root themselves and call
`experiments/05_detector_bench/train_detector.py`. They expect a live SLURM
allocation; `scripts/wait_then_run.sh` polls for one.

## Results

The measured baselines are in `experiments/05_detector_bench/results/` and are
summarised here:

| Model | mAP50-95 | clear−fog APm |
|---|---:|---:|
| YOLOv8s | 0.6204 | 0.128 |
| YOLO11s | 0.6194 | 0.132 |
| **YOLOv12s** | **0.6296** | **0.094** |
| YOLOv8s + weather aug | 0.6278 | 0.096 |
| YOLOv10s | 0.6063 | 0.114 |
| YOLOv8s-P2 | 0.6071 | 0.153 |
| RT-DETR-l | 0.2778 | invalid (29 epochs) |

Bar for a GATE-IR claim on UNIRI-TID: beat **YOLOv8s+aug fog APm = 0.5691**.
mAP@50 is saturated (0.96–0.97); report mAP@50-95 and APm.

## Known blockers (documented, not solved)

- **PromptIR / DEA-Net / NAFNet / Restormer**: official repos are RGB-trained.
  A fair thermal baseline needs retraining on thermal degraded/clean pairs;
  UNIRI-TID has real fog/rain but **no paired clean images**. Zero-shot RGB
  weights are not a valid thermal baseline.
- **IA-YOLO / TogetherNet / RDMNet**: full detection systems with different
  detectors; integration is a separate multi-day project.
- **DTRDNet / BAD-Net**: no verified code.
