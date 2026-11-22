# GATE-IR — Gated Adaptive Thermal Enhancement for Infrared Detection

GATE-IR is a research project on **thermal (infrared) object detection**: how to
keep detection reliable when the thermal signal is degraded by weather, and how
to exploit a second (visible/RGB) modality through cross-modal fusion and
knowledge distillation. The repository contains nine experiment workstreams, the
datasets and label pipelines they use, trained checkpoints, evaluation code and
the resulting metric files.

Two research threads run through the project:

1. **Weather robustness (experiments 01–07).** Synthetic adverse weather is
   generated for thermal imagery, and detectors are trained and evaluated on
   clear vs. degraded conditions. The central question is comparative: does
   adding synthetic weather to training — or filtering/gating it at inference —
   close the clear→adverse performance gap, and at what cost?
2. **Cross-modal graph fusion and distillation (experiments 08–09).** Paired
   RGB/IR data is used either to distil a compact IR-only student from
   cross-modal teachers (08, graph-structured KD) or to build a single fused
   detector that combines both modalities mid-network (09, graph-based fusion).
   Here the question is whether graph-structured interaction beats simple or
   attention-based fusion under matched capacity.

## Project objectives

- Quantify the clear→adverse-weather gap for thermal detectors across several
  datasets and degradation types.
- Establish whether synthetic-weather training and dehazing/gating reduce that
  gap without harming clear-condition accuracy.
- Build a controlled experimental ladder for RGB-IR fusion (references →
  simple fusion → attention → graph) on a common detector, data and recipe.
- Investigate graph-structured knowledge distillation from RGB/IR teachers into
  compact IR-only students, with matched-capacity controls.
- Keep every comparison traceable: fixed splits, locked test sets, provenance
  in each result file.

## Methodology and architecture

Detectors are Ultralytics YOLO models (YOLOv8s/YOLO11s/YOLO12s, and RT-DETR-l in
the detector bench). Two composite pipelines appear:

```
GATE-IR (01–02):  thermal frame ─► gate (clear/fog) ─┬─ clear ──────────────► YOLO
                                                     └─ fog ─► U-Net filter ─► YOLO

Graph fusion (09): RGB ─► backbone ─┐
                                    ├─► Fuse(P3,P4,P5) ─► shared neck/head ─► detections
                   IR  ─► backbone ─┘
```

Experiment 08 instead trains a compact IR-only YOLO student with knowledge
distillation from frozen RGB, IR and (early-fusion) RGB-IR teachers; graph
matching (Hungarian, Sinkhorn, Fused Gromov-Wasserstein, partial OT) aligns
teacher and student detections/embeddings.

## Datasets

All datasets live once under `data/` (see [`data/README.md`](data/README.md)):

| Dataset | Role |
|---|---|
| FLIR ADAS v2 (+ synthetic fog/rain/snow) | Main thermal benchmark for 01, 02, 07 |
| M3FD (paired IR/RGB) | Synthetic-weather study with paired modalities (03) |
| UNIRI-TID | Real adverse-weather thermal detection benchmark (04, 05) |
| DSERT-RoLL | Weather domains used for unpaired translation (01, 03) |
| FLIR-aligned (Zhang et al., RGB/IR pairs) | Paired RGB-IR protocol for 08 |
| DroneVehicle (paired aerial RGB/IR) | Paired RGB-IR protocol with strong RGB (08, 09) |

## Models and teacher/student architecture

- **Detectors:** YOLOv8s in the original FLIR runs, YOLO12s for the controlled
  comparisons (02, 03, 07, 08, 09), YOLOv8s/10s/11s/12s/P2/RT-DETR-l in the
  detector bench (05).
- **Teachers (08):** IR-only, RGB-only and 6-channel early-fusion YOLO12s
  teachers, frozen and cached before student training.
- **Students (08):** YOLO12n, IR-only at inference (2.57 M parameters), with the
  KD loss applied only during training.
- **Fused model (09):** dual-stream YOLO12s backbones with a fusion module at
  P3/P4/P5 and one shared neck and detection head (15–17 M parameters
  depending on the fusion rung).

## Knowledge distillation approaches (08)

Conventional dense prediction KD (A1), graph node KD with Hungarian matching
(A2), node+edge KD (A3), Sinkhorn (A4), entropic Fused Gromov-Wasserstein (A5),
partial-OT (A6), reliability weighting (A7), multi-teacher consensus (B), three-
teacher transport consistency (C), GT-anchored heterogeneous graphs (D),
teacher-graph condensation (E) and IR-preserving residual decomposition (F).
The FLIR run is limited to the five IR-teacher arms because the early-fusion
teacher underperformed there; the full matrix targets DroneVehicle.

## Repository structure

```
gate_ir_release/
├── experiments/          nine workstreams (code, configs, runs, results)
├── data/                 datasets, labels, dataset YAMLs
├── docs/                 consolidated report, baseline survey, references
├── results/              cross-experiment results index
├── models/pretrained/    pretrained initialisation weights
├── configs/              configuration index and regeneration notes
├── environment/          conda/pip environment definitions
├── scripts/              repository validation tools
└── logs/                 cross-experiment log index
```

Every experiment keeps its own `src/` (or scripts), `configs/`, `runs/`,
`results/` and `logs/`. Training runs retain `args.yaml`, `results.csv` and
`weights/`; evaluation numbers live in `results/*.json` next to the model that
produced them.

## Experiments

| # | Workstream | Objective | Status |
|---|---|---|---|
| 01 | [FLIR weather translation](experiments/01_flir_weather_translation/README.md) | Generate FLIR fog/rain/snow; train first single-generator detectors | Fog complete; rain stopped after 1 epoch; snow translation only |
| 02 | [GATE-IR on FLIR](experiments/02_flir_gate_ir/README.md) | Gate + filter comparison against the unfiltered baseline (main result) | Complete for CycleGAN; CycleDiff-75 filtered arm incomplete |
| 03 | [M3FD synthetic weather](experiments/03_m3fd_weather/README.md) | Classical/depth/CycleGAN weather on paired IR/RGB; learnability | Complete |
| 04 | [UNIRI gate variants](experiments/04_uniri_gate/README.md) | Gate/preprocessing variant builder for UNIRI-TID | Scripts and logs only |
| 05 | [Detector bench](experiments/05_detector_bench/README.md) | Detector comparison on UNIRI-TID clear/fog/rain | Complete (RT-DETR-l invalid) |
| 07 | [FLIR adverse conditions](experiments/07_flir_adverse_conditions/README.md) | Eight-condition screen + selective retraining | Complete |
| 08 | [Graph-structured KD](experiments/08_graph_kd/README.md) | Distil cross-modal teachers into a compact IR-only student | **Stopped** (resumable) |
| 09 | [Graph-based RGB-IR fusion](experiments/09_graph_fusion/README.md) | Fusion ladder: references → simple → attention → graph | **Active** |

The early GATE-IR scaffolding prototype is not part of this repository; the
original implementation is at <https://github.com/zer0-data/GATE-IR>.

### Experiment 08 — graph-structured KD (stopped)

The experiment asks whether graph-structured matching between teacher and
student improves cross-modal distillation beyond conventional prediction KD.
Teacher activations (pre-NMS nodes with class probabilities, objectness, boxes
and ROI embeddings, plus dense anchor maps) are matched to student activations
with Hungarian/Sinkhorn/FGW/partial-OT, and the alignment loss is added to the
student's detection loss. On the 609-frame aligned-FLIR test set, GT-only A0
reaches 0.2518 mAP50-95; the best KD arm (A3, node+edge) reaches **0.2586**, a
+0.0068 gain, and results are single-seed. The experiment is **stopped**: the
DroneVehicle matrix was not completed, its test labels were found to be
mis-specified, and the remaining arms would only distil into IR-only students
rather than improve the fused model that the project now needs. All runs are
checkpointed and resumable. Details:
[08 README](experiments/08_graph_kd/README.md),
[project report](experiments/08_graph_kd/PROJECT_REPORT.md).

### Experiment 09 — graph-based RGB-IR fusion (active)

The active direction. It builds a controlled fusion ladder on DroneVehicle v2
(thermal ground truth) to test whether a graph formulation of cross-modal
interaction beats (a) single-modality references, (b) free late fusion,
(c) simple early/fused baselines, (d) attention fusion, under one detector,
recipe and seed. Reference results on the locked 8,980-frame thermal test set:

| Run | Method | Test mAP50-95 | Test mAP50 |
|---|---|---:|---:|
| **E00** | IR-only YOLO12s | **0.5783** | 0.7628 |
| E01 | RGB-only, graded on thermal GT | 0.4126 | 0.6534 |
| E02 | 6-channel early fusion | 0.4762 | 0.6942 |
| E03 | late fusion (weighted box fusion) | **0.5794** | 0.7868 |

E02 shows early fusion *hurts* on this benchmark; free late fusion is
statistically indistinguishable from IR-only (ΔAP50-95 +0.11 pp). A
complementarity oracle shows +3.2 pp of recoverable recall overall and +6.7 pp
on small objects, so fusion headroom exists but simple methods do not capture
it. The dual-stream GF-YOLO testbed and the E10–E35 ladder are in progress;
no ladder rung has a completed evaluation yet. Details:
[09 README](experiments/09_graph_fusion/README.md),
[09 results](experiments/09_graph_fusion/RESULTS.md).

## Important completed results

| Study | Key number | Source |
|---|---|---|
| GATE-IR, CycleGAN fog, 100 ep | B1 fog mAP50-95 0.1266 → G1 **0.1535** (+21.3%); clear also improves | `experiments/02_flir_gate_ir/results/REPORT.md` |
| Detector bench, UNIRI-TID | YOLO12s **0.6296** mAP50-95, smallest clear→fog APm drop (0.094) | `experiments/05_detector_bench/README.md` |
| FLIR adverse conditions | 5/8 conditions crossed the 5% screen; e.g. snow gap reduced 78%, sensor 59% | `experiments/07_flir_adverse_conditions/results/FINAL_REPORT.md` |
| M3FD synthetic weather | Deterministic fog fully learnable (gap −0.001); CycleGAN fog gap 0.049 | `experiments/03_m3fd_weather/README.md` |
| Graph KD, aligned FLIR | A3 node+edge **0.2586** vs A0 0.2518 mAP50-95 (single seed) | `experiments/08_graph_kd/results/SUMMARY.md` |
| RGB-IR fusion references | E00 0.5783, E03 0.5794, E02 0.4762 on locked thermal test | `experiments/09_graph_fusion/RESULTS.md` |

## Reproducing experiments

The environment is `environment/environment.yml` (conda) or
`environment/requirements.txt` (pip); see [environment/README.md](environment/README.md).
Every experiment README has an execution section with the exact commands.
Cluster runs use SLURM: launch every job as an `srun` step inside a live
`salloc` allocation (training scripts assume `--device 0`).

Minimal checks:

```bash
# repository layout and dataset-YAML targets
python3 scripts/verify_release.py

# experiment 09 protocol checks
cd experiments/09_graph_fusion
PYTHONPATH=src python -m pytest tests -q
python scripts/check_leakage.py --policy warp
```

Test sets are locked: experiment 09 scores its test split only through
`scripts/final_eval.py` against pre-registered locks in `results/LOCK_*.json`.

## Documentation

- [Experiments index](experiments/README.md) — one line per workstream.
- [Consolidated report](docs/report/main.pdf) (LaTeX at `docs/report/main.tex`).
- [Baseline survey](docs/baseline_survey_and_experiment_matrix.md) and
  [baseline links](docs/baseline_links.md).
- [Data layouts](data/README.md), [models](models/README.md),
  [configs](configs/README.md), [environment](environment/README.md),
  [logs](logs/README.md), [results index](results/README.md).
- Experiment 08: [README](experiments/08_graph_kd/README.md),
  [method proposal](Graph_Structured_Knowledge_Distillation.md),
  [report](experiments/08_graph_kd/PROJECT_REPORT.md).
- Experiment 09: [README](experiments/09_graph_fusion/README.md),
  [results](experiments/09_graph_fusion/RESULTS.md),
  [plan](experiments/09_graph_fusion/EXPERIMENT_PLAN.md),
  [provenance](experiments/09_graph_fusion/PROVENANCE.md).

## Known limitations

- Results are single-seed unless stated; "100 epochs" means up to 100 with
  early stopping (`patience=25`–`30`). Small differences (≲0.5 pp mAP50-95)
  are within noise.
- Experiment 02's CycleDiff-75 filtered arm is incomplete (epoch 27/100) and
  resumable; experiment 09's fusion ladder is mid-flight.
- RT-DETR-l in experiment 05 is invalid (29 epochs) and must not be quoted.
- Experiment 08's DroneVehicle test evaluation is not valid; only validation
  numbers are reported for that dataset.
- Some generated dataset YAMLs contain absolute paths; regenerate them with the
  builders or point them at the current checkout.
