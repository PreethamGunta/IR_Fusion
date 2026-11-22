# `experiments/` — workstream index

Nine self-contained experiment directories. Each holds its own code, configs,
run directories and result files; shared datasets live once under `../data/`.

| # | Directory | Status | Scope |
|---|---|---|---|
| 01 | [`01_flir_weather_translation/`](01_flir_weather_translation/README.md) | fog complete; rain/snow partial | FLIR weather synthesis (CycleGAN, CycleDiff, classical, depth) and the first single-generator YOLOv8s detectors |
| 02 | [`02_flir_gate_ir/`](02_flir_gate_ir/README.md) | CycleGAN complete; CycleDiff filtered arm incomplete | GATE-IR: gate + filter comparison against the unfiltered baseline |
| 03 | [`03_m3fd_weather/`](03_m3fd_weather/README.md) | complete | Synthetic weather on paired M3FD IR/RGB; learnability study |
| 04 | [`04_uniri_gate/`](04_uniri_gate/README.md) | scripts + logs only | UNIRI-TID gate/preprocessing variant builder |
| 05 | [`05_detector_bench/`](05_detector_bench/README.md) | complete (RT-DETR-l invalid) | Detector comparison on UNIRI-TID clear/fog/rain |
| 07 | [`07_flir_adverse_conditions/`](07_flir_adverse_conditions/README.md) | complete | Eight-condition FLIR screen and selective retraining |
| 08 | [`08_graph_kd/`](08_graph_kd/README.md) | **stopped** (checkpointed, resumable) | Graph-structured cross-modal distillation into an IR-only student |
| 09 | [`09_graph_fusion/`](09_graph_fusion/README.md) | **active** | Controlled RGB-IR fusion ladder (references → simple → attention → graph) |

The early GATE-IR scaffolding prototype (experiment 06) is not part of this
repository; the original idea and implementation are at
<https://github.com/zer0-data/GATE-IR>.

## Conventions

- `src/` or top-level `*.py` — experiment code.
- `scripts/` — shell entry points and cluster wrappers.
- `configs/` — generated or hand-written dataset YAMLs and method configs.
- `runs/` — Ultralytics run directories: `args.yaml`, `results.csv`,
  `weights/{best,last}.pt`, training plots. These are the training artefacts
  and stay with the experiment.
- `results/` — evaluation JSONs (and small audit/summary files).
- `logs/` — training, evaluation and orchestration logs.
- `data/` (where present) — experiment-local intermediates (filtered images,
  symlink farms); regenerable.

`runs/<run>/results.csv` is the **training-time validation** curve used for
checkpoint selection. Reported test numbers are the JSON files in `results/`.

## Result JSON shape

Experiments 02 and 03 write a provenance-rich schema:

```json
{
  "name": "...", "weights": "...", "data": "...",
  "mAP50": 0.2849, "mAP50_95": 0.1525,
  "precision": ..., "recall": ..., "n_images": 3749,
  "weights_sha256": "...", "split_hash": "...", "git_commit": "..."
}
```

Experiment 05 uses `overall` / `per_condition` blocks with COCO-style APs.
Experiment 09 uses COCO prediction/GT JSONs scored by its own evaluator, with
`LOCK_<exp>.json` files recording the pre-registered test choices.
