# `results/metrics/` — metric file index

Metric files are produced by each experiment's evaluation script and are not
versioned. This index records where they are regenerated and which schema to
expect; the reported numbers are in the Markdown reports and experiment
READMEs.

| Experiment | Directory | Count |
|---|---|---:|
| 01 FLIR translation | `../../experiments/01_flir_weather_translation/results/` | 5 |
| 02 GATE-IR | `../../experiments/02_flir_gate_ir/results/` | 42 (+ gate `.pkl`) |
| 03 M3FD | `../../experiments/03_m3fd_weather/results/` | 5 |
| 05 detector bench | `../../experiments/05_detector_bench/results/` | 20 |
| 07 FLIR conditions | `../../experiments/07_flir_adverse_conditions/results/` | screen + retrain JSONs |
| 08 graph KD | `../../experiments/08_graph_kd/results/` | per-arm `*_test.json` |
| 09 graph fusion | `../../experiments/09_graph_fusion/results/` | `*_test_metrics.json`, `LOCK_*.json` |

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

**Workstream 08** (`graphkd.training.evaluate`): AP/AP50/AP75/APs/APm/APl,
per-class AP, ECE/Brier, parameters, latency, weights SHA-256, split hash.

**Workstream 09** (`src/eval/coco_eval.py`): `AP`, `AP50`, `AP75`, `APs/APm/APl`,
per-class AP, prediction counts, plus the lock record that authorised the run.

`mAP50-95` is always COCO `mAP@[.5:.95]`; `mAP50` is `mAP@.5`.
