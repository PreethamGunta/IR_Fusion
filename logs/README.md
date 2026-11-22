# `logs/` — log index

Logs stay with the experiment, next to the runs they describe. This directory
is an index.

| Experiment | Log directory | Contents |
|---|---|---|
| 01 FLIR translation | `experiments/01_flir_weather_translation/logs/` and the method subdirectories | translation, detector and health-check logs |
| 02 GATE-IR | `experiments/02_flir_gate_ir/logs/` | `train_*.log`, `eval_*.log`, `alloc_*.log`, `supervisor.log`, `progress.log` |
| 03 M3FD | `experiments/03_m3fd_weather/logs/` | pipeline stages, detector training |
| 04 UNIRI gate | `experiments/04_uniri_gate/logs/` | shortlist, variants, wait loop |
| 05 detector bench | `experiments/05_detector_bench/logs/` | sweep and overnight runs |
| 07 FLIR conditions | `experiments/07_flir_adverse_conditions/logs/` | screen and retraining stages |
| 08 graph KD | `experiments/08_graph_kd/logs/` | teacher, cache, student and evaluation phases |
| 09 graph fusion | `experiments/09_graph_fusion/logs/` | reference, ladder, prediction and evaluation phases |

## What to look for

- **02:** `supervisor.log` is the authoritative chain log for the 100-epoch
  runs; `train_<arm>.log` contains the Ultralytics progress bars.
- Training logs contain carriage-return progress bars; use
  `tr '\r' '\n' < file | grep ...` when searching them.
- Experiment 09's training scripts write one log per run
  (`logs/ref_*.log`, `logs/e*.log`); the final evaluations write
  `logs/final_evals.log`.
