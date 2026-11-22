# `results/metrics/` — metric file index

The metric JSONs are stored next to their experiment. This index says where to
find them and what schema to expect.

| Experiment | Directory | Count |
|---|---|---:|
| 01 FLIR translation | `../../experiments/01_flir_weather_translation/results/` | 5 |
| 02 GATE-IR | `../../experiments/02_flir_gate_ir/results/` | 42 (+ gate `.pkl`) |
| 03 M3FD | `../../experiments/03_m3fd_weather/results/` | 5 |
| 05 detector bench | `../../experiments/05_detector_bench/results/` | 20 |

## Schemas

**Workstreams 01 and 02** (`eval_detector.py`):

```json
{
  "name": "...", "weights": "...", "data": "...", "split": "test",
  "mAP50": 0.2849, "mAP50_95": 0.1525,
  "precision": ..., "recall": ..., "n_images": 3749,
  "weights_sha256": "...", "split_hash": "...", "git_commit": "..."
}
```

**Workstream 03** (`eval_m3fd.py`): the same provenance fields plus a
`metrics: {val: {...}, test: {...}}` block and `n_images` per split.

**Workstream 05** (`detect_eval.py`): `overall` and `per_condition` blocks with
COCO-style `mAP50_95`, `mAP50`, `mAP75`, `AP_small/medium/large`, `AR_100`.

`mAP50-95` is always Ultralytics/COCO `mAP@[.5:.95]`; `mAP50` is `mAP@.5`.
