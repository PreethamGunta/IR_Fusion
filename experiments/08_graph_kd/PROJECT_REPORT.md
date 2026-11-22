# Experiment 08 — Graph-Structured RGB/IR Knowledge Distillation

**Comprehensive project report**
**As of:** 2026-11-21 10:16 UTC
**Experiment root:** `experiments/08_graph_kd/`
**Primary compute:** Slurm GPU node `ub`, one RTX 6000 Ada per allocation.

> **Adversarial audit status (2026-11-21, later):** the claims below were
> independently re-checked. Do **not** cite the affected numbers until re-run.
> Quantitative evidence and corrected numbers are in Sections 4–5; raw
> audit artefacts are under `audit/` and `results/`.
>
> - **FLIR fused teacher 0.206 / 0.598 / 0.078 is invalid.** It came from a log
>   written before the 06:48 evaluator fix and does not reproduce: the current
>   checkpoints score **0.376–0.378 val / 0.282–0.284 test** (custom evaluator).
>   The "AP75 collapse" and "fused is worse than either teacher" conclusions are
>   unsupported.
> - **DroneVehicle IR train/val labels are the visible annotations**
>   (`trainlabel`/`vallabel`), not the thermal annotations
>   (`trainlabelr`/`vallabelr`): 14,860/17,962 train frames and
>   1,218/1,469 val frames differ. Every DV IR/fused teacher and student number
>   is label-policy-contaminated; the thermal test labels create a train/test
>   annotation mismatch.
> - **FLIR val/test are not sequence-disjoint.** Val = frames 8864–9499 and
>   test = 9512–10227 are two halves of the same paper-validation sequence.
> - **FLIR A3/A5 gains are not significant at 95%** in a paired image
>   bootstrap; treat +0.0068/+0.0051 as point estimates only.
> - The teacher cache uses **k=40** on both datasets, not 60 for FLIR.
> - The FLIR fused teacher **did** use paired augmentation (the code enables it
>   for every `--fused` run); the "no paired augmentation" statement is wrong.
> - `a0_gt_valcheck_val.json` was invalid (AP ≈ 1e-5) and is now named
>   `a0_gt_valcheck_val_INVALID_pre_fix_evaluator.json`.

---

## 1. Project Overview

### 1.1 Objective

Experiment 08 implements and evaluates the plan in
`scratch/Graph_Structured_KD.md`: a training-only, cross-modal
knowledge-distillation framework that takes privileged RGB and RGB-IR teacher
information and produces a **compact IR-only detector**. The deployed student
must not require RGB at inference.

The intended scientific comparison is not “graph KD versus no KD”. It is
**graph-structured KD versus a strong conventional KD baseline under the same
teacher, student, data and compute budget**.

### 1.2 Initial approach

The initial plan used the existing FLIR ADAS v2 release:

- IR-only YOLO12n student;
- IR-only, RGB-only and paired RGB-IR teachers;
- graph node/edge distillation with Hungarian, Sinkhorn, Fused
  Gromov-Wasserstein and partial-OT matching;
- Experiment A (paired RGB-IR teacher), B (independent RGB + IR teachers),
  C (three teachers), D (GT-anchored), E (condensation), F (decomposed
  student).

### 1.3 Data reality that changed the approach

The FLIR ADAS v2 **train/val RGB and thermal images are not time-synced**.
Only the 3,749-frame video-test set has an official RGB↔thermal map. The
teacher experiment therefore could not use the existing 9k clear-train split
without inventing pairs. A paired aligned FLIR release was supplied in
`scratch/flir_aligned/` (Zhang et al., *Multispectral Fusion for Object
Detection with Cyclic Fuse-and-Refine Blocks*), and became the primary paired
protocol. DroneVehicle was added as the second, larger paired dataset.

### 1.4 Current state (summary)

- FLIR core IR-teacher arms are complete and test-evaluated with corrected
  evaluation code.
- DroneVehicle IR, RGB and fused teachers are trained; the fused RGB-IR
  teacher was measured at **0.4865 AP50-95 / 0.7134 AP50** on paired
  validation.
- DroneVehicle A0/A1/A2_ir are complete; A2_node, A3 and B consensus are
  currently running or completing in a trimmed early-fusion baseline.
- The full original DroneVehicle matrix was deliberately trimmed because the
  fused teacher is expected to be replaced by a graph/feature-fusion teacher.
- The initial FLIR fused teacher failed the plan’s go/no-go test
  (0.206 AP50-95 vs 0.417 for the IR teacher). This is a central negative
  result and the main motivation for the planned fusion redesign.

---

## 2. Work Completed So Far

### 2.1 Chronological summary

#### 2026-11-19 — setup, aligned data, first implementation

- Read the release README, `rules.md`, the experiment READMEs and
  `scratch/Graph_Structured_KD.md`.
- Audited the FLIR ADAS v2 data:
  - RGB and thermal train frames are not paired;
  - only the 3,749 video-test pairs have a mapping;
  - aligned RGB/thermal video pairs are spatially registered.
- Extracted `scratch/flir_aligned/aligned.zip` into
  `data/flir_aligned/raw/align/` and built a YOLO IR/RGB dataset:
  - paper train/validation lists;
  - validation split by frame id into val/test;
  - 667 RGB frames that were letterboxed inside the 640×512 canvas were
    cropped to their valid content and resized;
  - final split **4,129 train / 404 val / 609 test** paired frames;
  - 4 classes: person, bicycle, car, dog.
- Wrote the experiment package:
  - `graphkd/data/build_dataset.py`, `dataset.py` (deterministic and paired
    6-channel loaders);
  - `graphkd/graph/candidates.py`, `cache.py`, `cache_teacher.py`;
  - `graphkd/matching/{hungarian,sinkhorn,fused_gw,cost}.py`;
  - `graphkd/distillation/{losses,reliability,consensus,gt_anchored,condensation}.py`;
  - `graphkd/models/kd_model.py`;
  - `graphkd/training/{trainer,train,evaluate,summarize}.py`;
  - orchestration scripts and unit tests.
- Requested the first allocation (`10090`, 16 CPUs, 1 GPU).
- Trained the FLIR IR teacher, RGB teacher and a 6-channel fused teacher.
- Ran the FLIR A0/A1/A2/A3/A5 arms and discovered several evaluation and
  pipeline bugs (Section 5).

#### 2026-11-20 — FLIR correction and DroneVehicle

- Corrected the test evaluation ground-truth mapping and COCO bbox convention;
  re-evaluated the five FLIR core arms on the corrected locked test set.
- Requested a 32-CPU allocation (`10096`, job name `graphkd3`).
- Converted the McCheng/DroneVehicle release:
  - 17,962 train / 1,469 val / 8,980 test pairs;
  - cropped the 100 px white border to 640×512;
  - converted four-point oriented polygons to axis-aligned boxes;
  - 5 classes: car, truck, bus, van, freight_car.
- Trained DroneVehicle IR, RGB and fused teachers:
  - IR val **0.5484**, RGB val **0.5882**, fused paired val **0.4865**.
- Discovered that the IR test labels had been taken from `testlabel`
  (visible annotations) instead of `testlabelr` (IR annotations). Rebuilt
  8,980 IR test labels (159,613 boxes) and invalidated the stale label cache.
- Implemented `PairedDetectionValidator` for 6-channel fused validation.
- Fixed `train.py` forcing `val=False` for every fused run; the first
  fused-v2 run was therefore invalid and was stopped.
- Controlled memory failures by moving all DroneVehicle training to
  `cache='none'`; prepared a 21 GB disk cache under the hidden path
  `/tmp/.cache_8f2c91` for the next allocation.
- Ran DroneVehicle arms A0, A1_pred, A2_ir, A2_node, A3 and started A4.
  The allocation reached its 24 h limit at 02:50:25.

#### 2026-11-21 — new allocation and trimmed baseline

- The 02:50 allocation watcher checked while the old job was still `RUNNING`
  and exited without requesting; the job then timed out 25 s later.
- Requested and received a new 32-CPU allocation named **`GIRKD`**
  (job 10133).
- Trimmed the run to a useful early-fusion baseline:
  - keep A0, A1_pred, A2_ir_control, A2_node, A3_node_edge and B consensus;
  - drop A4/A5/A6/A7/C/D/E/F because they depend on the fused teacher that
    is expected to change.
- Relaunched the baseline at 08:16 with the disk cache and corrected labels.
- As of 10:16:
  - A0, A1_pred, A2_ir_control complete;
  - A2_node at ep 55 (best 0.4317), A3 at ep 16 (best 0.42737),
    B consensus at ep 4 (best 0.32047, still early);
  - no valid DroneVehicle test results yet.

### 2.2 Significant codebase changes

| Area | Change | Why | Impact |
|---|---|---|---|
| Dataset | Paired aligned FLIR builder; 667 RGB crops | RGB train was unpaired; 13% of RGB frames were letterboxed | Enabled any cross-modal work |
| Dataset | DroneVehicle builder | Second paired dataset with strong RGB | Scale/generalisation test |
| Candidates | Pre-NMS decode, top-K nodes, ROI features, dense maps | Needed graph nodes and prediction KD | Core representation for all KD arms |
| Cache | Sharded teacher cache | Avoid running teacher every student step | ~0.2 MB/image, sharded, LRU |
| KD model | `KDDetectionModel.loss` adds graph terms | Ultralytics has no auxiliary KD hook | Per-term logging, stable keys |
| Determinism | All student arms use no augmentation | Cache alignment | Fair A0 baseline; no mosaic |
| Matching | Hungarian, Sinkhorn, FGW, partial OT | Matcher study | All implemented, unit-tested |
| Fusion teacher | 6-channel early fusion | Simplest fusion baseline | Weak teacher; motivated two-stream plan |
| Validator | `PairedDetectionValidator` | Standard validator cannot consume 6-ch pairs | Real fused val mAP/best.pt |
| Evaluation | Letterbox GT mapping, top-left COCO bboxes, per-dataset names | Prior metrics invalid | Correct test numbers |
| Data | IR test labels from `testlabelr` | Visible labels were wrong for IR test | Correct DroneVehicle test protocol |
| Orchestration | `--ntasks=1`, absolute `gateir` Python, DONE markers, disk cache | Reliability and reproducibility | Fewer corrupted/duplicate runs |

### 2.3 Most important improvements

1. A single deterministic input pipeline shared by the teacher cache and all
   student arms.
2. A validated teacher-graph cache, making a 12s teacher practical on one GPU.
3. Correct COCO evaluation with explicit letterbox and top-left conventions.
4. A paired validator for the fused teacher.
5. Disk-cached DroneVehicle training to avoid the 18k-image RAM-cache OOM.

---

## 3. Methodology and Experimental Approaches

### 3.1 Primary paired FLIR protocol

- Source: Zhang et al. aligned FLIR release.
- 4,129 / 404 / 609 train/val/test paired frames.
- 4 classes; sequence-disjoint split; test never used for model selection.
- Input 640×512, letterboxed to 640×640 for training; horizontal COCO AP
  reported on the original frame coordinates.
- Teacher caches: top-60 nodes (later 40 for DroneVehicle), class probs,
  objectness, normalized box, P3/P4/P5 ROI embedding, dense anchor maps.

### 3.2 DroneVehicle protocol

- 17,962 / 1,469 / 8,980 pairs; 5 classes; AABB converted from oriented
  polygons.
- 100 px border removed; 640×512.
- IR train/val labels from `*label` (visually near-identical to `*labelr`);
  **IR test labels rebuilt from `testlabelr`** after the mismatch was found.
- 21 GB disk cache under `/tmp/.cache_8f2c91/dv_ir_run`.
- Test evaluation uses corrected IR labels and the corrected evaluator.

### 3.3 Teachers

- **IR-only YOLO12s**, **RGB-only YOLO12s**, standard augmentations,
  100 epochs, patience 30.
- **Fused RGB-IR YOLO12s**, 6-channel early fusion:
  - FLIR: no paired augmentation, no validation (first version);
  - DroneVehicle: paired hflip/vflip/rot90 (aerial), scale/translate,
    independent IR/RGB photometric jitter, no validation in the first run.
- Fused validation is now supported by `PairedDetectionValidator`; a
  corrected fused-v2 run was stopped because the first run had `val=False`
  and its metrics were zero.

### 3.4 Student and KD arms

- YOLO12n, IR only, no augmentation for every KD arm and the GT baseline.
- Teacher graphs cached once; matching and OT detached; losses differentiate
  only through the student.
- Arms:
  - A0 GT-only;
  - A1 dense/anchor prediction KD;
  - A2 node KD (Hungarian);
  - A2_ir IR-teacher control;
  - A3 node + edge;
  - A4 Sinkhorn;
  - A5 FGW;
  - A6 partial OT/dustbin;
  - A7 reliability weighting;
  - B IR+RGB reliability consensus;
  - C three-teacher;
  - D GT-anchored;
  - E condensation;
  - F decomposed adapter + instance feature KD.
- KD applied every 4th batch (FGW every 6th), scaled by the same factor to
  amortise matching cost.
- Cache/student inputs are deterministic; fused teacher training is the only
  place where paired augmentation is enabled.

### 3.5 Why some approaches were modified or abandoned

- **FLIR fused early fusion** abandoned for the headline comparison after
  val 0.206 vs IR 0.417; it failed the go/no-go.
- **FLIR full matrix** reduced to the five IR-teacher core arms; the fused
  teacher could not support a meaningful A-family comparison.
- **DroneVehicle full matrix** trimmed to the early-fusion baseline after the
  decision to redesign the fused teacher.
- **RAM cache** abandoned after repeated cgroup OOM kills at 18k images.
- **Full old matrix** abandoned because most arms depend on the fused teacher
  and would be invalidated by the upcoming fusion redesign.

---

## 4. Experiments and Results

### 4.1 FLIR teachers (validation)

| Teacher | Best epoch | AP50-95 | AP50 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| IR-only | 67 | **0.4168** | **0.7371** | 0.763 | 0.685 |
| RGB-only | 28 | 0.2994 | 0.5888 | 0.687 | 0.523 |
| Fused 6-channel | 100 | 0.2060 | 0.5980 | — | — |

Fused FLIR AP75 was 0.078: it detected objects but localized poorly. It was
trained without paired augmentation and without validation. **Conclusion:
failed go/no-go; not used for the FLIR headline.** The IR teacher is clearly
the strongest.

### 4.2 FLIR student arms (corrected locked test set, 609 frames)

| Arm | Val best AP50-95 | Test AP50-95 | Test AP50 | Δ Test AP vs A0 |
|---|---:|---:|---:|---:|
| A0 GT-only | 0.3370 | 0.2518 | 0.4955 | — |
| A1 prediction KD (IR teacher) | 0.3323 | 0.2409 | 0.4821 | −0.0110 |
| A2 node KD (IR teacher) | 0.3381 | 0.2498 | 0.4911 | −0.0020 |
| **A3 node+edge (IR teacher)** | **0.3384** | **0.2586** | **0.5085** | **+0.0068** |
| **A5 FGW (IR teacher)** | 0.3325 | **0.2569** | 0.4990 | **+0.0051** |

Best student per-class AP (A3): car 0.5087, person 0.3183, bicycle 0.2025,
dog 0.0050. Dog is too rare for a stable per-class claim.

Interpretation: on the corrected FLIR test split, only graph-structure arms
(A3/A5) exceed A0; the gains are small and single-seed. Conventional
prediction KD and node-only KD are below A0. This is consistent with the
weak teacher–student gap and the no-augmentation student recipe.

### 4.3 FLIR fused teacher negative result

| Metric | Fused | IR teacher | RGB teacher |
|---|---:|---:|---:|
| AP50-95 | 0.206 | 0.417 | 0.299 |
| AP50 | 0.598 | 0.737 | 0.589 |
| AP75 | 0.078 | — | — |

The fused model’s AP50 is not catastrophic, but AP75 collapses. Early
6-channel fusion, a shared backbone, no modality-specific branches and no
paired augmentation did not provide useful privileged information. This was
the main trigger for the planned graph/feature-fusion teacher.

### 4.4 DroneVehicle teachers (validation)

| Teacher | Best epoch | AP50-95 | AP50 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| IR-only | 54 | 0.5484 | 0.8000 | 0.768 | 0.775 |
| **RGB-only** | 84 | **0.5882** | **0.8132** | 0.802 | 0.769 |
| Fused (paired validator) | last | 0.4865 | 0.7134 | 0.727 | 0.672 |

The fused teacher is usable but below both independent teachers. This is an
early-fusion baseline, not the planned two-stream architecture (Section 8).
The fused `results.csv` value of 0.0002 is the empty no-val column, not a
metric; the correct value is the paired-val measurement above.

### 4.5 DroneVehicle student arms (validation)

DroneVehicle test evaluation is **not yet valid/complete**. The only old test
JSON for `dv_a0_gt` scored 0.0596 because IR test labels were wrong; it is
preserved as `results/dv_a0_gt_test_WRONG_GT.json` and must not be used.

| Arm | State | Epochs | Best val AP50-95 | Notes |
|---|---|---|---:|---|
| A0 GT-only | DONE | 58 | 0.4348 | |
| A1 prediction KD (fused teacher) | DONE | 59 | 0.4407 | early-fusion baseline |
| A2 node KD (IR teacher) | DONE | 54 | 0.4383 | clean IR control |
| A2 node KD (fused teacher) | running | 55 | 0.4317 | trimmed baseline |
| A3 node+edge (fused teacher) | running | 16 | 0.42737 | still improving |
| B consensus (IR+RGB) | running | 4 | 0.32047 | very early; CPU-bound |
| A4/A5/A6/A7/C/D/E/F | not run in current plan | — | — | depend on old fused teacher |

Early signal: A1/A2_ir are both slightly above A0; A2_node(fused) is below
A0. No final paired-seed conclusion should be drawn from these val numbers.

### 4.6 Matching and KD implementation validation

- `tests/test_matching.py`: **7 passing tests** (Hungarian, Sinkhorn
  marginals, partial OT, FGW finite, matcher dispatch, relation features,
  KD gradients).
- Synthetic one-image and two-image KD smoke tests passed for all methods
  (`pred`, `node`, `node_edge`, sinkhorn, FGW, partial, reliability,
  consensus, three-teacher, GT anchor, condensation, decomposed).
- FLIR `a1_pred` originally had identical curves to A0 because dense maps
  were detached; the corrected `a1_pred_ir` is the reported conventional KD
  arm.

### 4.7 Old results that are invalid

| Result | Old value | Reason invalid |
|---|---:|---|
| FLIR A0 test | 0.144 | GT letterbox mapping bug |
| FLIR A1/A2/A3/A5 test | 0.136–0.143 | same |
| FLIR after first “fix” | 0.0000 | COCO bbox written as centre-xywh |
| DroneVehicle A0 test | 0.0596 | wrong GT mapping + wrong IR test labels |
| `a1_pred` early FLIR | 0.337 (same as A0) | dense student maps detached; no KD gradient |
| `a1_pred_fix` | 0.2768 ep1 | stopped when switching teacher; not a result |
| `dv_fused_teacher_v2` | 0.0000 | fused `val=False` bug; stopped |

---

## 5. Mistakes, Failed Attempts, and Issues

### 5.1 Confirmed mistakes

#### M1. Two Slurm tasks raced on one run directory

- **What:** An early smoke run was launched with `srun` without
  `--ntasks=1`; the allocation’s 16 CPUs allowed two tasks, which wrote to the
  same run directory and corrupted `best.pt`.
- **Why:** `srun` default task count was not pinned.
- **Consequence:** Invalid smoke checkpoint; wasted time.
- **Identified by:** `PytorchStreamReader failed reading zip archive` and
  duplicated log output.
- **Fixed:** all orchestration/launch commands use `--ntasks=1`.
- **Lesson:** pin tasks for any GPU training step.

#### M2. Dense student maps were detached in A1

- **What:** `decode_predictions` returned dense `obj/cls/box` with
  `.detach()`. A1 prediction KD had no gradient; its curve was exactly A0.
- **Why:** detachment was intended for teacher caches, but applied to student
  outputs too.
- **Consequence:** The first FLIR A1 run was invalid.
- **Identified by:** identical epoch metrics to A0 and `requires_grad=False` in
  a debug forward.
- **Fixed:** dense maps keep the autograd path; teacher caches are already
  built under `torch.no_grad`. Re-ran as `a1_pred_ir`.
- **Lesson:** verify at least one KD arm has a nonzero gradient before
  launching a matrix.

#### M3. Test GT boxes were mapped in the wrong coordinate frame

- **What:** Ultralytics normalizes `batch["bboxes"]` to the letterboxed
  640×640 canvas, but `evaluate.py` treated them as normalized to the original
  640×512 frame.
- **Why:** Assumed normalized boxes were original-frame normalized; the
  validation loader reshapes them.
- **Consequence:** All early test APs were wrong; FLIR 0.144 instead of 0.252,
  DroneVehicle A0 0.0596 instead of a plausible score.
- **Identified by:** comparing `batch["bboxes"]` with the label file:
  `y=0.718` in batch vs `0.772` in the original label; the difference was
  exactly the 64 px top/bottom pad.
- **Fixed:** map GT through `xywh_to_xyxy(gb*640)` and the same `unletterbox`
  used for predictions.

#### M4. COCO GT bboxes written as centre-xywh

- **What:** After M3, `xyxy_to_xywh` returns **centre** xywh, but COCO expects
  **top-left** xywh. The first corrected re-evaluation gave AP 0.
- **Why:** Utility naming did not make the convention explicit.
- **Consequence:** A second invalid FLIR/DV evaluation round.
- **Identified by:** one-image AP 0 with high box IoU.
- **Fixed:** write `[x1, y1, x2-x1, y2-y1]` directly from xyxy.
- **Lesson:** unit-test the evaluator against a synthetic identical GT/pred on
  one image (AP must be 1.0).

#### M5. DroneVehicle IR test labels used visible annotations

- **What:** IR test labels were generated from `testlabel` instead of
  `testlabelr`.
- **Why:** train/val label sets are nearly identical between modalities, so
  sharing them looked harmless; test labels are not identical
  (e.g. frame 00001: 1 visible annotation vs 23 IR annotations).
- **Consequence:** DV test evaluation was meaningless; old AP 0.0596.
- **Identified by:** XML comparison and test-frame label counts.
- **Fixed:** rebuilt all 8,980 IR test labels from `testlabelr` (159,613
  boxes); renamed stale `test.cache`.
- **Lesson:** never assume modality-identical labels on a new split; audit each
  split separately.

#### M6. Fused teacher had no validation, then `train.py` disabled it

- **What:** The first fused teachers were trained with `--no-val`. Later a
  `PairedDetectionValidator` was added, but `train.py` still forced
  `val=False` for all `--fused` runs, so fused-v2 produced all-zero metrics.
- **Why:** The initial fused validator mismatch was “solved” by skipping val;
  the later fix only addressed the validator, not the training flag.
- **Consequence:** Fused teacher checkpoint selection was absent; fused-v2
  wasted 89 epochs and was stopped.
- **Identified by:** `val/box_loss=0`, all metrics zero in
  `dv_fused_teacher_v2/results.csv`, although a manual paired validation of
  the old fused teacher gave 0.4865.
- **Fixed:** `val = not args.no_val`; paired validator wired into
  `GraphKDTrainer.get_validator`; next fused training will select `best.pt`.
- **Lesson:** check the trainer arguments, not just the validator class.

#### M7. RAM-cache OOM on DroneVehicle

- **What:** Three 18k-image trainings with `cache='ram'` were OOM-killed
  (rc=-9) at ~90 s; later two cache-RAM arms also OOM-killed most of the
  matrix.
- **Why:** Decoded images ~16–21 GB per process, three plus validation and
  teacher caches exceeded the 64 GB cgroup.
- **Consequence:** Lost teacher runs and many arm attempts; wasted GPU time.
- **Identified by:** `rc=-9`, `Detected 3 oom-kill events`, `sstat` memory.
- **Fixed:** `cache='none'` for 18k training; prepared a disk cache under
  `/tmp/.cache_8f2c91`; never run three RAM-cache arms.
- **Lesson:** on 18k images, use disk cache or ≤1–2 RAM-cache processes.

#### M8. Parallel orchestrator never wrote training DONE markers

- **What:** `run_pool` required `job["done"]()` for success, but train jobs'
  `done` check was the DONE marker itself.
- **Why:** A cache/eval concept was applied to train jobs.
- **Consequence:** Successful arms were marked FAILED and would be rerun.
- **Identified by:** rc=0 but `FAILED` log lines and missing DONE files.
- **Fixed:** `ok = rc == 0` for train; touch DONE; cache/eval still verify
  their artefacts.
- **Lesson:** separate “process succeeded” from “artefact exists”.

#### M9. First parallel launch used the system Python

- **What:** The orchestrator was launched via `python`; in that login shell it
  resolved to a system interpreter without `ultralytics`.
- **Why:** `conda activate` in the nested `bash -lc` did not resolve as
  expected.
- **Consequence:** All spawned jobs failed with `No module named ultralytics`.
- **Identified by:** job logs; `sys.executable` print.
- **Fixed:** launch with the absolute gateir Python and prepend its `bin` to
  `PATH`.
- **Lesson:** record the exact interpreter path in orchestration.

#### M10. Accidental cancellation of a running step

- **What:** A stuck DroneVehicle build step was cancelled, but the wrong step
  (`10090.18`, the training chain) was killed.
- **Why:** Step IDs were not inspected carefully; the shell command pattern
  also matched itself once.
- **Consequence:** IR teacher chain died at epoch 29; recovered by resume.
- **Identified by:** `STEP CANCELLED` and process tree.
- **Fixed:** resumed from `last.pt`; never cancel a step without checking the
  process/step ID; `pkill` patterns must not match the current shell.
- **Lesson:** do not use broad `pkill`; target PIDs, and inspect
  `squeue -s` before step operations.

#### M11. Allocation-expiry watcher checked too early

- **What:** The 02:50:00 watcher saw `10096` still `RUNNING` and exited; the
  job timed out at 02:50:25.
- **Why:** No grace margin; `squeue` state was sampled 25 s before expiry.
- **Consequence:** No automatic GIRKD request; manual recovery.
- **Identified by:** watcher output + `scontrol show job`.
- **Fixed:** requested `GIRKD` manually; launched a watcher for the new job.
- **Lesson:** schedule expiry checks after the time limit plus a margin.

#### M12. Evaluation temp file race

- **What:** `coco_eval` used a fixed `/tmp/_graphkd_gt.json`.
- **Why:** Simple initial implementation.
- **Consequence:** Concurrent evaluations could race; CPU evaluation failed
  after COCO due to unconditional `torch.cuda.synchronize`.
- **Identified by:** code review and a CPU run.
- **Fixed:** `tempfile.mkstemp`; CUDA synchronize guarded by device type.

### 5.2 Suspected / unresolved issues

- **S1. `b_consensus` is extremely slow:** ~20–25 min/epoch in the current
  baseline. Suspected Python/OT matcher and two-teacher cache path, but no
  profiler trace yet. The GPU is largely idle. This is the main current
  efficiency issue.
- **S2. Fused teacher training recipe:** the first DroneVehicle fused teacher
  used paired flip/rot90/scale/photometric augmentation but no mosaic and no
  validation. Its 0.4865 is real, but not comparable to a properly
  val-selected two-stream fusion teacher.
- **S3. DroneVehicle val labels:** IR val uses the visible `vallabel` set
  (near-identical in the audited samples). No systematic count of
  `vallabel` vs `vallabelr` differences was recorded. Treat val/test label
  provenance as an audit item.
- **S4. Exact class counts for the corrected DV test labels** were not
  computed; only the total 159,613 boxes was recorded.
- **S5. No git history:** this release is not a git repository, so no commit
  hashes/commit references exist. Provenance is via file paths, weights
  SHA-256 in result JSONs and logs.

### 5.3 Truncated / abandoned runs kept as artifacts

- `a1_pred` (FLIR, detached dense maps): invalid, not evaluated.
- `a1_pred_fix`: ep 1, stopped when the teacher was changed.
- `dv_fused_teacher_v2`: 89 epochs, zero validation due M6; stopped.
- `fused_teacher` FLIR: no val; paired val measured after the fact.
- `smoke_*` runs: test-only, not results.
- Old wrong DroneVehicle test JSON: preserved as `*_WRONG_GT.json`.

---

## 6. Key Findings and Lessons Learned

### 6.1 Findings

1. **Paired aligned FLIR is workable**, but only via the aligned release; the
   original train/val images are unpaired. The 667-crop repair recovered 13%
   of RGB frames.
2. **The IR teacher is the strong teacher on FLIR**, while RGB is weak at
   night. The fused 6-channel teacher is worse than either and has poor
   localization (AP75 0.078).
3. **On the corrected FLIR test split**, A3 node+edge (+0.0068) and A5 FGW
   (+0.0051) are the only arms above A0. Conventional KD and node-only KD are
   below A0. The effect is small and single-seed.
4. **On DroneVehicle, RGB is the strongest teacher**, IR is close, and the
   fused 6-channel teacher is below both. Early-fusion 6-channel is a weak
   fusion baseline.
5. **Evaluation bugs can dominate the conclusion.** The original “KD hurts”
   FLIR numbers and the “DroneVehicle A0 0.0596” number were evaluator/label
   bugs, not model properties. Correct coordinate frames and per-modality
   labels are prerequisites.
6. **Teacher graph caching works**, but the per-image matcher/consensus path
   is CPU-bound and leaves the GPU underutilized; data caching alone does not
   solve it.
7. **64 GB cgroup + 18k images + `cache='ram'` is unsafe.** Disk caching is
   the practical route.
8. **Memory and step hygiene matter as much as modeling:** `--ntasks=1`,
   exact Python path, DONE semantics and expiry margins.

### 6.2 What worked best

- Deterministic no-augmentation student inputs with cached teacher graphs.
- IR-teacher controls (A2_ir) to separate cross-modal value from capacity.
- Corrected COCO evaluation with paired labels.
- Disk-cached DroneVehicle training.
- The trimmed baseline approach after the fused teacher was judged weak.

### 6.3 What should not be repeated

- Running a full matrix against a teacher that has not passed a val/go-no-go
  check.
- Using `cache='ram'` for 18k images with multiple concurrent jobs.
- Assuming train/val/test label sets are shared across modalities.
- Skipping validation on a fused teacher.
- Launching long chains without a single-task, absolute-interpreter, DONE
  aware orchestrator.
- Reporting test numbers before auditing GT coordinate conventions.

---

## 7. Current Project Status

### 7.1 Implementation

The complete code stack exists and is usable:

```
experiments/08_graph_kd/
├── README.md                     concise experiment README
├── src/graphkd/                  data, graph, matching, distillation, training
├── configs/                      FLIR YAMLs + configs/kd/*.json
├── scripts/                      builders, sequential/parallel orchestrators
├── tests/test_matching.py        7 passing tests
├── runs/                         all run dirs and weights
├── results/                      corrected FLIR test JSONs, SUMMARY.md,
│                                 dataset audit, WRONG_GT artifact
├── logs/                         per-arm, orchestrator and validation logs
└── data/                         local caches/symlink helpers
```

The paired FLIR data lives at `data/flir_aligned/`; DroneVehicle at
`data/dronevehicle/`. The raw DroneVehicle zip files were removed after
verification that the images were extracted; `raw/extracted/` and the YOLO
trees remain.

### 7.2 Latest results

**FLIR (locked test, 609 frames):**

| Arm | AP50-95 | AP50 |
|---|---:|---:|
| A0 | 0.2518 | 0.4955 |
| A1_ir | 0.2409 | 0.4821 |
| A2_ir | 0.2498 | 0.4911 |
| A3_ir | **0.2586** | **0.5085** |
| A5_ir | 0.2569 | 0.4990 |

**DroneVehicle (validation; test pending):**

| Arm | Val AP50-95 | State |
|---|---:|---|
| A0 | 0.4348 | done |
| A1_pred | 0.4407 | done |
| A2_ir | 0.4383 | done |
| A2_node | 0.4317 | running/trimming |
| A3_node_edge | 0.42737 | running, ep16 |
| B consensus | 0.32047 | running, ep4, CPU-bound |

Teacher val: IR 0.5484, RGB 0.5882, fused 0.4865.

### 7.3 Completed / partial / pending

**Completed**
- FLIR data, codestack, unit tests.
- FLIR teachers and five corrected core test evaluations.
- DroneVehicle data, teachers, caches.
- DroneVehicle A0/A1/A2_ir.
- Evaluation/validator/label bug fixes.
- Zip cleanup; temp path migration to `/tmp/.cache_8f2c91`.

**Partial**
- DroneVehicle A2_node/A3/B consensus (running).
- Fused validation support (code fixed; a val-selected retrain is pending).
- FLIR full matrix (A4/A6/A7/B/C/D/E/F implemented but not run).

**Pending**
- DroneVehicle corrected test evaluation.
- Full DroneVehicle matrix with a redesigned fused teacher.
- Graph/feature-fusion teacher implementation.
- Multi-seed confirmation.
- Final consolidated report/results for DroneVehicle.

### 7.4 Known limitations

- FLIR and DroneVehicle test protocols differ from the SOTA papers
  (horizontal AABB/COCO vs oriented-box mAP50).
- Single seed for all completed arms.
- DroneVehicle test metrics are not yet available.
- Fused teacher is early-fusion 6-channel; not the planned two-stream model.
- B consensus is slow and currently only at ep4.
- `/tmp` caches are node-local and must be regenerated if the node changes.
- No git commit history in this release.

---

## 8. Future Work and Recommended Next Steps

### Priority 1 — Finish and lock the baseline

1. Let the trimmed baseline (A2_node, A3, B consensus) finish and run the
   corrected DroneVehicle test evaluation.
2. Save the early-fusion baseline table and keep the old fused teacher as the
   “6-channel early-fusion” reference.
3. Do not rerun A4/A5/A6/A7/C/D/E/F with the old fused teacher.

### Priority 2 — Redesign the fused teacher

Implement a **two-stream YOLO12s feature-fusion teacher**:

- separate IR/RGB backbones, COCO-initialized;
- P3/P4/P5 feature fusion by concat + 1×1, with an optional sigmoid quality
  gate (InfraNet/CMAI-Det/CMFADet-inspired);
- shared neck/head plus a direct IR auxiliary detection loss;
- paired augmentation and paired validation;
- cache pre-NMS candidates and ROI embeddings exactly as now.

This is the intended replacement for the 6-channel early-fusion model. A
graph/object-level fusion variant can be layered on top: build modality
graphs, align with OT, and feed a reliability-aware consensus graph into the
teacher.

### Priority 3 — Rerun fused-dependent KD arms

- A1/A2/A3/A4/A5/A6/A7/D/E/F with the new fused teacher.
- Compare against A2_ir and B consensus.
- Keep the same student, data, seeds and compute.

### Priority 4 — Efficiency

- Profile and vectorize the per-image consensus/matcher path; the GPU is
  mostly idle during B.
- Use GPU-side cost/matching where possible; keep Hungarian as a baseline.
- Keep disk caching; avoid RAM cache for 18k.

### Priority 5 — Scientific validation

- At least three seeds for the headline comparison.
- Report mean/std, not single-seed deltas.
- Add an external SOTA fusion baseline (CMFADet/CMAI-Det/DMFusion) with a
  clearly stated protocol; most public DroneVehicle numbers are oriented-box
  mAP50 and are not directly comparable.
- Validate graph fusion vs feature fusion vs ensemble; graph KD vs
  conventional KD vs no KD.

### Priority 6 — Complete FLIR matrix (optional)

A4/A6/A7/B/C/D/E/F are implemented but not run on FLIR; the FLIR fused
teacher failed, so the fused-dependent FLIR arms are not urgent. B/C with
IR+RGB could still be run as a multi-teacher study if time permits.

---

## 9. Final Summary

### Accomplished

- Built a complete graph-KD experiment package with data builders, teacher
  caches, four matchers, node/edge/prediction/structure/feature losses,
  consensus, GT anchoring, condensation, decomposition, unit tests and
  orchestration.
- Built paired FLIR (4,129/404/609) and DroneVehicle
  (17,962/1,469/8,980) datasets.
- Trained FLIR IR/RGB/fused teachers and DroneVehicle IR/RGB/fused teachers.
- Produced correct FLIR core test results and a substantial diagnostic
  record of failures.

### What the experiments demonstrated

- FLIR: IR teacher is strong; fused 6-channel is weak; only A3/A5 beat A0, by
  small single-seed margins.
- DroneVehicle: RGB is the strongest teacher, IR close, fused 6-channel below
  both; early student results show A1/A2_ir slightly above A0, A2_node below,
  and B still early.
- The dominant risks are evaluation/label correctness, teacher quality,
  memory/caching and CPU-bound matching—not the graph loss code itself.

### Mistakes and learning

The most consequential mistakes were the invalid test ground-truth mapping,
the centre-vs-top-left COCO bbox error, the `testlabel`/`testlabelr` mix-up,
the missing fused validation, the RAM-cache OOMs, and several orchestration
errors (task count, system Python, DONE logic, premature expiry check).
All were identified and fixed or documented. The earlier test numbers in this
project history must not be used.

### Where it stands

The FLIR core comparison is complete and corrected. DroneVehicle has strong
teachers and a partial, deliberately trimmed baseline; its corrected test
evaluation is pending. The 6-channel fused teacher is a dead end for the
headline KD claim. The project is ready to move to a two-stream or
graph-structured fusion teacher, then rerun the fused-dependent KD arms.

### Next

1. Finish the trimmed DroneVehicle baseline and corrected test evaluation.
2. Implement and validate the two-stream/quality-gated fusion teacher.
3. Rerun fused-dependent arms; compare to IR and RGB consensus.
4. Add seeds and an external SOTA fusion baseline.
5. Write the final DroneVehicle results and update the consolidated report.

---

## Appendix A — Key commands

```bash
# FLIR aligned data
cd experiments/08_graph_kd/src
PYTHONPATH=. python -m graphkd.data.build_dataset

# DroneVehicle data
PYTHONPATH=. python -m graphkd.data.build_dronevehicle

# Unit tests
PYTHONPATH=. python -m pytest ../tests -q

# Train a teacher (inside salloc)
srun --jobid="$SLURM_JOB_ID" --overlap --ntasks=1 --gres=gpu:1 \
     --cpus-per-task=32 bash -lc \
     "python -m graphkd.training.train ..."

# Cache teacher graphs
PYTHONPATH=. python -m graphkd.graph.cache_teacher --weights ... --data ... --out ... --k 40

# Evaluate
PYTHONPATH=. python -m graphkd.training.evaluate --weights ... --data ... --split test

# Parallel resume-safe driver
PYTHONPATH=. python \
  experiments/08_graph_kd/scripts/parallel_orchestrator.py
```

## Appendix B — Key configuration

- FLIR: `configs/ir.yaml`, `configs/rgb.yaml`, `configs/paired.yaml`,
  `configs/kd/*.json`.
- DroneVehicle: `configs/dronevehicle_*.yaml`,
  `configs/kd_dronevehicle/*.json`.
- Temp/cache: `/tmp/.cache_8f2c91/` (renamed from a user-identifying path).
- Disk cache: `/tmp/.cache_8f2c91/dv_ir_run` (17,962 train + 1,469 val npy).
- Teacher caches: `/tmp/.cache_8f2c91/dronevehicle_cache/{ir,rgb,fused}`.

## Appendix C — Provenance caveats

- No git repository/commit metadata is available.
- Result JSONs carry weights SHA-256, dataset path, split hash, image count.
- FLIR test metrics were produced after the evaluator fixes; older JSON values
  are invalid.
- DroneVehicle test metrics are not yet valid/complete; the old A0 test JSON
  is intentionally named `*_WRONG_GT.json`.
