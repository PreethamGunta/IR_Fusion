# `results/` — cross-experiment results index

Results stay with the experiment that produced them, next to the model, config
and logs. This directory indexes them.

## Authoritative reports

| Report | Path |
|---|---|
| GATE-IR on FLIR (main) | `experiments/02_flir_gate_ir/results/REPORT.md` |
| All GATE-IR result JSONs | `experiments/02_flir_gate_ir/results/all_results.md` |
| FLIR weather-translation report | `experiments/01_flir_weather_translation/report/report.pdf` |
| M3FD write-up | `experiments/03_m3fd_weather/methods_report.tex` |
| FLIR adverse conditions | `experiments/07_flir_adverse_conditions/results/FINAL_REPORT.md` |
| Graph-KD summary (08) | `experiments/08_graph_kd/results/SUMMARY.md` and `PROJECT_REPORT.md` |
| Graph fusion (09) | `experiments/09_graph_fusion/RESULTS.md` |

## Subdirectories

| Directory | Purpose |
|---|---|
| `reports/` | Pointers to the authoritative reports above |
| `metrics/` | Index of metric files, their locations and schemas |
| `figures/` | Index of qualitative figures (contact sheets, sample grids) |

## Where the numbers are

Reported numbers live in the Markdown reports listed above and in the experiment
READMEs. Per-run metric JSONs are produced by each experiment's evaluation
script and are **not versioned**: they are regenerated into the experiment's
`results/` directory (see `metrics/README.md` for the commands and schemas).
Each regenerated file records the weights path and SHA-256, dataset
description, image count and split provenance.
