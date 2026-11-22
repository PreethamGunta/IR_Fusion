# `experiments/` — workstream index

Each directory is a self-contained experiment/workstream: its own README, code,
configs, run outputs, logs and result files. Nothing here is shared by copying;
shared data lives in `../data/`.

| # | Directory | Status | Headline |
|---|---|---|---|
| 01 | `01_flir_weather_translation/` | complete for fog; rain/snow partial | FLIR weather translation + original YOLOv8s detectors |
| 02 | `02_flir_gate_ir/` | CycleGAN complete; CycleDiff B1 complete, G1 at epoch 27/100 | B1 vs G1 100-epoch comparison |
| 03 | `03_m3fd_weather/` | complete | M3FD methods 1/2/3 + YOLOv12s |
| 04 | `04_uniri_gate/` | scripts + logs only | UNIRI-TID variant/gate builder |
| 05 | `05_detector_bench/` | complete (RT-DETR-l invalid) | UNIRI-TID detector bench |
| 06 | `06_gate_ir_scaffolding/` | scaffolding only | early GATE-IR package |
| 07 | `07_flir_adverse_conditions/` | complete; 5/8 conditions retrained | eight FLIR thermal conditions, screen + retrain protocol |
| 08 | `08_graph_kd/` | running / in progress | graph-structured cross-modal KD into an IR-only student (aligned FLIR pairs, all matchers) |

## Conventions

- `src/` or top-level `*.py` — code for the experiment.
- `scripts/` — shell entry points / SLURM wrappers.
- `configs/` — generated or hand-written dataset YAMLs and configs.
- `runs/` — Ultralytics run directories; each contains `weights/`,
  `results.csv`, `args.yaml`, plots. These are the training artefacts and are
  deliberately kept with the experiment.
- `results/` — evaluation JSONs and Markdown summaries.
- `logs/` — training / evaluation / orchestration logs.
- `data/` (where present) — experiment-local intermediate data (filtered images,
  symlink farms). Regenerable.

## Result JSON shape

Workstream 02 and 03 write the same provenance-rich schema:

```json
{
  "name": "...", "weights": "...", "data": "...",
  "mAP50": 0.2849, "mAP50_95": 0.1525,
  "precision": ..., "recall": ..., "n_images": 3749,
  "weights_sha256": "...", "split_hash": "...", "git_commit": "..."
}
```

Workstream 05 uses `overall` / `per_condition` blocks with COCO-style APs.
