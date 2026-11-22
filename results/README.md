# `results/` — cross-experiment results index

This directory is the **entry point** for results, not a second copy of them.
The authoritative files stay with their experiment so that a model, its config,
its log and its score remain together.

## Authoritative reports

| Report | Path |
|---|---|
| GATE-IR on FLIR (main) | `experiments/02_flir_gate_ir/results/REPORT.md` |
| All GATE-IR result JSONs as a table | `experiments/02_flir_gate_ir/results/all_results.md` |
| Paused/resume state (CycleDiff G1) | `experiments/02_flir_gate_ir/results/RUN_STATUS.md` |
| FLIR weather-translation report | `experiments/01_flir_weather_translation/report/report.pdf` |
| M3FD write-up | `experiments/03_m3fd_weather/methods_report.tex` |

## Subdirectories

| Directory | Purpose |
|---|---|
| `reports/` | Pointers to the authoritative reports above |
| `metrics/` | Index of the JSON metric files per experiment |
| `figures/` | Index of qualitative figures (contact sheets, sample grids) |

## Where the numbers are

| Experiment | Metrics |
|---|---|
| 01 FLIR translation | `experiments/01_flir_weather_translation/results/*.json` |
| 02 GATE-IR | `experiments/02_flir_gate_ir/results/*.json` |
| 03 M3FD | `experiments/03_m3fd_weather/results/*.json` |
| 04 UNIRI gate | results are in workstream 05 |
| 05 detector bench | `experiments/05_detector_bench/results/*.json` |

Every JSON carries the weights path and SHA, the dataset YAML, `n_images`, and
the git/run provenance. Do not quote a number without opening its JSON.
