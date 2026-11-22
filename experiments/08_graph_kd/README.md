# 08 — Graph-Structured Knowledge Distillation for IR detection

Training-only, cross-modal graph distillation into a compact **IR-only** YOLO
student. RGB and fused teachers are never present at inference.

**Status: stopped.** The runs are checkpointed and resumable; results below are
the verified subset. The full DroneVehicle matrix and its test evaluation are
incomplete (see *Results*).

The experiment implements the plan in
[`Graph_Structured_Knowledge_Distillation.md`](../../Graph_Structured_Knowledge_Distillation.md)
(Experiments A–F, all four matchers) on aligned FLIR pairs and DroneVehicle,
plus the evaluation and provenance machinery needed to compare against a
well-tuned conventional-KD baseline.

## Headline protocol

| | |
|---|---|
| Data | Zhang et al. aligned FLIR release (`data/flir_aligned/`), 640×512 RGB/IR pairs, 4 classes: person, bicycle, car, dog |
| Split | paper train/val split, val further split by frame id: **4,129 train / 404 val / 609 test** pairs, sequence-disjoint (train frames 2–8859, val 8864–9500, test 9501–10227) |
| Teachers | YOLO12s **IR**, YOLO12s **RGB**, YOLO12s **early-fusion (6-channel RGB+IR)** |
| Student | YOLO12n, IR only, 2.56 M parameters |
| Training | 100 epochs max, patience 30, batch 16, 640 px, seed 0 |
| Test | locked 609-frame paired test split; COCOeval + ECE/Brier + latency |
| Deployment | student is a plain IR YOLO; no teacher, graph or OT code at inference |

The FLIR train/val images are **not** time-synced in the original release; only
the video set has an RGB↔thermal map. The aligned release solves this. 667 of
5,142 RGB frames are stored letterboxed inside the 640×512 canvas; the builder
crops their valid content and resizes it, recovering registration. A 300-pair
edge-registration audit is saved in `results/dataset_audit.json` (median shift
≈2 px, i.e. object-level not pixel-level registration).

## Methods implemented

| ID | Arm | What it tests |
|---|---|---|
| A0 | `a0_gt` | GT-only IR student (no KD), same no-aug recipe |
| A1 | `a1_pred` | conventional dense/anchor prediction KD (class + box + objectness) |
| A2 | `a2_node` | graph node KD with Hungarian matching |
| A2ir | `a2_ir_control` | IR-teacher graph KD control (capacity vs cross-modal) |
| A3 | `a3_node_edge` | node + edge/relation KD |
| A4 | `a4_sinkhorn` | node+edge with Sinkhorn OT |
| A5 | `a5_fgw` | node+edge + entropic Fused Gromov-Wasserstein structure term |
| A6 | `a6_partial` | FGW-style graph KD with partial OT / dustbin |
| A7 | `a7_reliability` | reliability-weighted node/edge KD |
| B | `b_consensus` | independent RGB + IR teachers, reliability-aware consensus |
| C | `c_three_teacher` | RGB + IR + fused teachers with transport consistency |
| D | `d_gt_anchor` | GT-anchored heterogeneous graph KD |
| E | `e_condense` | teacher-graph condensation (centrality, K=30) |
| F | `f_decomposed` | IR-preserving + residual compensation adapter, instance feature KD |

Matching code: Hungarian (`scipy`), log-domain Sinkhorn, entropic FGW
(proximal/autograd), partial OT with dustbin. Unit tests: `tests/test_matching.py`.

## How KD is integrated

- Teacher graphs are **cached once** per training image: pre-NMS nodes
  (class probs, objectness, box, P3/P4/P5 ROI embedding), dense anchor maps and
  graph edges. The frozen teacher is never run during student training.
- All student KD arms and A0 use **identical deterministic inputs**: mosaic,
  mixup, affine, flips, HSV and the default Albumentations blur/CLAHE are
  disabled. This is required for cache alignment and makes A0 a fair baseline.
- The KD loss is added in `KDDetectionModel.loss()`; teacher tensors are
  detached, graph matching is detached, and only the student path receives
  gradients. Per-term values are written to `runs/<arm>/kd_terms.jsonl`.
- `warmup_epochs=5` ramps the KD weight from 20 % to 100 %.
- GT detection loss is always active.

## Directory layout

```
08_graph_kd/
├── README.md
├── src/graphkd/
│   ├── paths.py                 all paths derived from this file
│   ├── data/build_dataset.py    aligned FLIR -> IR/RGB YOLO datasets + audit
│   ├── data/dataset.py          deterministic datasets, paired 6-channel loader
│   ├── data/build_dronevehicle.py  (secondary dataset, same protocol)
│   ├── graph/candidates.py      pre-NMS decoding + ROI embeddings + dense maps
│   ├── graph/cache.py           sharded teacher cache
│   ├── graph/cache_teacher.py   cache CLI
│   ├── matching/                hungarian, sinkhorn, fused_gw, partial OT
│   ├── distillation/            node/edge/pred/struct/feature KD, reliability,
│   │                            consensus, GT anchoring, condensation
│   ├── models/kd_model.py       KDDetectionModel + KDModule
│   └── training/                trainer, train CLI, COCO evaluation
├── configs/                     generated ir/rgb/paired YAMLs + kd/*.json
├── scripts/                     data, teacher, cache, student, eval, full chain
├── tests/
├── runs/                        Ultralytics runs (weights, results.csv, kd terms)
├── results/                     evaluation JSONs, dataset audit, summaries
├── logs/
└── data/                        teacher caches (regenerable)
```

The aligned source data lives at `data/flir_aligned/` (repository data root),
not inside this experiment, so it can be shared. Everything else is
experiment-local.

## Commands

All training runs as a GPU job (one `srun` step per allocation). From the
experiment root:

```bash
# 1. data (once; ~1 minute)
cd src && PYTHONPATH=. python -m graphkd.data.build_dataset && cd ..

# 2. generate KD configs
bash scripts/make_configs.sh

# 3. full chain inside the allocation
srun --jobid="$SLURM_JOB_ID" --overlap --ntasks=1 --gres=gpu:1 \
     --cpus-per-task=8 bash scripts/run_all.sh

# or individual phases
srun ... bash scripts/train_teachers.sh     # IR + RGB + fused teachers
srun ... bash scripts/cache_teachers.sh     # cache teacher graphs
srun ... bash scripts/train_students.sh     # A0 + all KD arms (priority order)
srun ... bash scripts/evaluate_all.sh       # test-set COCO eval
```

`ARMS="a0_gt a2_node a5_fgw" bash scripts/train_students.sh` runs a subset.
Every phase skips runs with a `DONE` marker and every training run resumes from
`weights/last.pt` when re-launched with `--resume`.

Use `--ntasks=1` on `srun`: the allocation has 16 CPUs, and without it srun can
start two tasks that race on the same run directory and corrupt checkpoints.

## Results

### FLIR aligned protocol — locked 609-frame test

AP is COCO mAP50-95; deltas are relative to the GT-only student `a0_gt`.
Per-run evaluation JSONs are produced locally by the evaluation scripts and are
not versioned; the tables below are the record. Invalid runs are excluded.

| Arm | AP | AP50 | AP75 | ΔAP | ECE | Params |
|---|---:|---:|---:|---:|---:|---:|
| a0_gt (GT only) | 0.2518 | 0.4955 | 0.2206 | — | 0.1257 | 2.57 M |
| a2_ir_control | 0.2498 | 0.4911 | 0.2190 | −0.0020 | 0.1004 | 2.57 M |
| a1_pred_ir (conventional KD) | 0.2409 | 0.4821 | 0.1990 | −0.0110 | 0.1160 | 2.57 M |
| **a3_node_edge_ir** | **0.2586** | **0.5085** | **0.2223** | **+0.0068** | 0.0995 | 2.57 M |
| a5_fgw_ir | 0.2569 | 0.4990 | 0.2228 | +0.0051 | 0.1020 | 2.57 M |

Best arm per class (`a3_node_edge_ir`): car 0.509, person 0.318, bicycle 0.203,
dog 0.005. Single seed: the +0.68 pp gain of A3 over A0 is within seed noise and
should be read as suggestive, not established.

On FLIR the fused RGB-IR teacher failed the plan's go/no-go check (AP50-95 0.206
vs 0.417 for the IR teacher; AP75 0.078), so the FLIR matrix was limited to the
five IR-teacher arms above.

### DroneVehicle protocol (validation only)

The 6-channel early-fusion teacher underperforms both independent teachers; the
student matrix is therefore a distillation baseline, not a fusion study.

| Teacher (validation) | Best epoch | AP50-95 | AP50 |
|---|---:|---:|---:|
| IR-only | 54 | 0.5484 | 0.8000 |
| RGB-only | 84 | 0.5882 | 0.8132 |
| Fused (paired validator) | last | 0.4865 | 0.7134 |

| Student arm (validation) | State | Epochs | Best val AP50-95 |
|---|---|---:|---:|
| A0 GT-only | complete | 58 | 0.4348 |
| A1 prediction KD (fused teacher) | complete | 59 | 0.4407 |
| A2 node KD (IR teacher) | complete | 54 | 0.4383 |
| A2 node KD (fused teacher) | stopped | 55 | 0.4317 |
| A3 node+edge (fused teacher) | stopped | 40 | 0.4414 |
| B consensus (IR+RGB teachers) | stopped | 17 | 0.4194 |

DroneVehicle **test** evaluation is not valid for this experiment: the only
available test JSON (`results/dv_a0_gt_test_WRONG_GT.json`) used mis-specified
IR labels and scores 0.0596; it must not be quoted. The remaining arms
(A4–A7, C–F) were not run.

Each evaluation records AP/AP50/AP75/APs/APm/APl, per-class AP, ECE/Brier,
parameters, latency, weights SHA-256 and split hash; the JSON files are
regenerated by the evaluation CLI and are not versioned.

## Caveats

- Single seed for the main matrix; the reported deltas (ΔAP ≤ 0.011) are within seed noise unless a multi-seed rerun is done.
- The aligned FLIR split is a **new protocol** (4 classes, 609-frame test), so
  its absolute numbers are not comparable to the 15-class 3,749-frame GATE-IR
  tables. Comparisons are internal to this experiment.
- RGB/IR registration is object-level (median ≈2 px); the RGB teacher is
  therefore a privileged but not pixel-perfect signal.
- Fused teacher is an **early-fusion 6-channel** baseline, a deliberate
  simplification of the two-stream architecture in the plan; it is the simplest
  fusion baseline and avoids adding untested attention modules.
- Teacher `best.pt` selection is on the 404-frame val split; the test split is
  never used for model selection.
- The fused teacher has no validation split; its final (`last.pt`) checkpoint
  is the teacher. For DroneVehicle it is trained with paired augmentation so
  the 6-channel model does not overfit: horizontal/vertical flips, 90-degree
  rotations (aerial imagery), random scale/translation, and independent
  photometric jitter on the IR and RGB halves. The cache and all student KD
  inputs remain deterministic; augmentation is only used while fitting the
  teacher.
- DroneVehicle teachers are trained to convergence (100 epochs max, patience
  30). The full DroneVehicle matrix (A0–A7, B/C/D/E/F) was not completed; every
  run is resumable from `weights/last.pt` and phases skip completed arms.
- KD is applied every 4th training batch (`kd_every=4`, FGW every 6th) and the
  loss is scaled by the same factor, so the average KD strength is unchanged
  while the matching cost is amortised. This was necessary to fit the full arm
  matrix into one 24 h allocation.
- An early `a1_pred` run is kept as an artefact but is **invalid**: its dense
  student maps were detached, so the KD loss had no gradient and its curve is
  identical to A0. The reported conventional-KD baseline is `a1_pred_fix`,
  which was run after the dense-map fix.
