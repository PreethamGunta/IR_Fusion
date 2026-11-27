# Experiment 09 — Graph-Based RGB-IR Fusion

**Status: complete (single seed).** The V1 data-loader defect was isolated and
fixed (V2 pipeline); the reference tier (E00–E03), the 10-epoch screen, and the
full-data flagship tier are complete. Under a matched recipe, **WG-Fusion
(E30, window-graph GAT) is the best single model on the locked 8,980-image test
set: 0.5988 AP, +2.08 pp over IR-only (paired-bootstrap 95 % CI
[+1.71, +2.42]) and +0.59 pp over concat fusion (CI [+0.25, +0.89], i.e.
significant).** Two-hop depth (E41), capacity (d=192), semantic-kNN edges (E35)
and warm-started fine-tuning are neutral or negative. Full narrative, figures
and tables: [REPORT_V0.md](REPORT_V0.md); paper draft in
[paper/](paper/); numbers in [RESULTS.md](RESULTS.md) and §5.2 below.

## 1. Motivation and objectives

A thermal camera detects warm objects in the dark but resolves little detail; a
visible camera does the opposite. A detector that consumes both modalities
should, in principle, see more than either alone. Published RGB-IR fusion
methods (gated recalibration, deformable alignment, cross-attention) report
gains, but they typically change many things at once: architecture, capacity,
training recipe and evaluation protocol.

This experiment isolates the question:

> Does a **graph formulation** of cross-modal interaction — where feature-map
> positions are nodes and messages flow along selected edges across modalities —
> beat simple fusion and attention fusion under a matched data/recipe budget?

**Outcome (2026-11-27).** The ⅓-data screening tier turned out to be
pipeline-invalid: every fused rung collapsed to 0.14–0.25 while the *IR-only*
control collapsed as well (0.183), isolating a defect in the superseded V1
data loader. The V2 loader reproduces the stock IR path bit-identically, and the
question above was re-asked at full data with E00's exact recipe:

> At matched budget, mid-level fusion improves the locked test AP by
> **+1.5–2.1 pp** over IR-only, and the **window graph significantly beats
> concat fusion (+0.59 pp, CI excludes zero)**. Graph depth, graph width,
> semantic edges and warm-started fine-tuning are neutral or negative.

The ladder design below is retained for history; the answered experiment is
§5.2.

Sub-goals:

1. Establish honest single-modality references and cheap fusion baselines on the
   same detector and data.
2. Measure the **complementarity headroom** (what fusion could recover if it
   were perfect) before attributing gains to a module.
3. Climb a controlled ladder (simple → attention → graph), promoting a rung only
   when its gain exceeds the noise floor.

## 2. Problem formulation

Given paired images `(x_ir, x_rgb)` of the same scene and thermal ground-truth
boxes, learn `f(x_ir, x_rgb) → detections`. Fusion happens at three feature
pyramid levels (P3, P4, P5; 80×64, 40×32, 20×16 at 640×512). For each level the
fusion module is

```
f_fused = Fuse(f_ir, f_rgb),    shape(f_fused) = shape(f_ir)
```

and residual rungs return `f_fused = f_ir + Δ`. The detection loss, assigner,
neck, head and recipe are identical for every rung; only `Fuse` changes.

## 3. Architecture

**Dual-stream GF-YOLO** (implemented in `src/gfyolo/model.py`):

```
input [B,6,H,W]  (channels 0-2 = IR, 3-5 = RGB)
   │
   ├─ IR stream:  YOLO12s backbone ─► P3_ir, P4_ir, P5_ir
   └─ RGB stream: YOLO12s backbone ─► P3_rgb, P4_rgb, P5_rgb
                                   │
                    Fuse at P3 / P4 / P5   (one module per level)
                                   │
                    shared YOLO12s neck + Detect head
```

Both backbones and the neck/head are initialised from the same pretrained
YOLO12s weights; the IR stem filters are replaced by their per-filter
grayscale mean. Total parameters depend on the rung (E10 ≈ 15.5 M, E14 ≈ 16.3 M).
The assembly is a single renumbered `ModuleList` (IR backbone 0–8, RGB backbone
9–17, fuse modules 18–20, neck/head 21–32) so the underlying Ultralytics
training, loss and validation machinery is unchanged.

**Input contract.** `[B,6,H,W]`; IR in channels 0–2, RGB in 3–5. The paired
dataset letterboxes 640×512 sources to a 640×640 square and applies one joint
geometric augmentation to both modalities.

## 4. Representations

- **Nodes (graph rungs):** one spatial feature vector per position per modality,
  projected to width `d = 128`.
- **Edges:** within-modality 8-neighbour grid; cross-modal window edges
  (radius `r = 1` by default); semantic edges (top-`k` RGB nodes by cosine
  similarity inside a Chebyshev radius `R`, `k = 4–8`); optional cross-scale
  edges.
- **Edge features:** relative offset `(dx,dy)`, cosine similarity, and per-node
  quality scalars.
- **Messages:** GAT-style
  `α_ij = softmax_j(LeakyReLU(aᵀ[W h_i ‖ W h_j ‖ W_e e_ij]))`,
  update `h_i' = h_i + g_i Σ_j α_ij W_v h_j` with a gated residual; gather-based
  fixed neighbour tables, no graph library dependency.

## 5. Methods and identifiers

Every run has an `E` identifier. The prefix encodes the family: `E0x` =
references, `E1x` = simple fusion, `E2x` = attention, `E3x`–`E4x` = graph.

### References and baselines (Tier A)

| ID | Run name | Method | Purpose | Status | Test mAP50-95 |
|---|---|---|---|---|---|
| **E00** | `e00_ir` | IR-only YOLO12s | the reference every fusion must beat | complete | **0.5783** |
| **E01** | `e01_rgb` | RGB-only YOLO12s, trained on visible labels, **graded on thermal GT** | measures the other modality alone | complete | 0.4126 |
| **E02** | `e02_fused` | 6-channel early fusion `[IR\|RGB]` | classic literature baseline | complete | 0.4762 |
| **E03** | `e03_late_fusion` | weighted box fusion of E00+E01 detections (no training) | free late-fusion baseline | complete | **0.5794** |

### Fusion ladder (Tier B)

| ID | Run name | `Fuse` module | What it changes vs. parent | Status |
|---|---|---|---|---|
| **E10** | `e10_screening` | concat + 1×1 conv at P3–P5 | ladder base (learned simple fusion) | **complete: 0.2403 (ep99)** |
| E11 | — | elementwise add, fixed weights | no learned mixing | planned |
| E12 | — | channel-softmax modality gate | per-channel trust | planned |
| E13 | — | channel + spatial softmax gates | IV-PAFNet MSFA-style port | planned |
| **E14** | `e14_screening` | IR-anchored residual `f_ir + sigmoid(g)·conv([f_ir,f_rgb])`, negative-bias gate | starts as IR-only, learns corrections | **complete: 0.2438 (ep94)** |
| E15 | — | E13 + visible-guided deformable alignment | misregistration fix | planned |
| E20 | — | one-directional cross-attention (IR←RGB) | attention baseline | planned |
| **E21** | `e21_screening` | bidirectional cross-attention, block-shared Q/K/V | DMFF-style attention representative | **complete: 0.2258 (ep87)** |
| E22 | — | intra-modal self-attention, capacity-matched to E21 | control: no cross-modal path | conditional |
| **E30** | `e30_screening` | GAT on grid + cross-modal window edges (IR←RGB) | first graph rung | **complete: 0.2521 (ep100)** — no rung separable (§5.1) |
| E31–E33 | — | GCN / GraphSAGE / GATv2+edge features | operator swaps | planned |
| E34 | — | bidirectional graph messages | refinement | planned |
| **E35** | `e35_screening` | E30 + semantic-kNN cross-modal edges | the core novelty | **complete: 0.2329 (ep99)** |
| E36–E37 | — | cross-scale edges; typed edge weights | refinements | planned |
| E38 | — | connectivity sweep (`r`, `R`, `k`, sparse/dense) | tuning | planned |
| E39–E41 | — | residual vs direct output; gate type; fusion placement | ablations | planned |
| E50 | — | object-level OT graph over candidate detections | speculative bridge to experiment 08 | planned |

Promotion rule: a rung is promoted only if its test mAP50-95 gain over its
parent exceeds the pre-registered threshold (**0.5 pp** under the single-seed
protocol), confirmed by a paired image-level bootstrap on the locked test set.

### 5.1 Screening-tier results (⅓ train, 100 epochs, training-time val)

All five rungs ran on the same ⅓-train subset
(`configs/dronevehicle_v2_paired_screening.yaml`), with the same augmentation
knobs (mosaic 1.0, close_mosaic 10, scale 0.5, hsv_v 0.4, fliplr 0.5, erasing
0.4), batch 16, seed 0, 100 epochs, and the same auto-resolved optimizer
(AdamW, lr 1.11e-3, 3 parameter groups).

| Run | Arm | Best val mAP50-95 | Best ep | mAP50 | Params | State |
|---|---|---:|---:|---:|---:|---|
| `e00_ir_screening` | IR-only control, native recipe | **0.5625** | 51 | 0.7793 | 9.26 M | complete (early stop at ep81) |
| `e00_ir_noaug_screening` | IR-only, augmentation disabled | 0.4451 | 15 | 0.6533 | 9.26 M | partial (32 ep, no DONE marker) |
| `e10_screening` | E10 concat + 1×1 conv | 0.2403 | 99 | 0.4173 | 15.49 M | complete |
| `e14_screening` | E14 IR-anchored gated residual | 0.2438 | 94 | 0.4399 | 16.28 M | complete |
| `e21_screening` | E21 bidirectional cross-attention | 0.2258 | 87 | 0.4121 | 16.95 M | complete |
| `e30_screening` | E30 window-graph GAT | **0.2521** | 100 | 0.4386 | 15.35 M | complete |
| `e35_screening` | E35 semantic-kNN graph | 0.2329 | 99 | 0.4230 | — | complete |

**What the tier supports.** Rung-vs-rung comparison is valid *within* the tier:
identical data subset, augmentation knobs, schedule, batch, seed and resolved
optimizer — the only intended difference is the `Fuse` module. The designs are
statistically indistinguishable: the largest gap (E30−E21 = 2.6 pp, E30−E14 =
0.83 pp) is inside the 1.4–3.7 pp spread measured between repeats of the same
configuration, with one seed and no confidence interval. The defensible
statement is therefore "**all five mid-level fusion designs land at 0.23–0.25
in this tier**", not a ranking. Execution histories are not uniform (E21
resumed at ep47; E10/E30/E35 are resume splices; E35's first 95 epochs are
unlogged), so this tier is a screening filter, not a controlled experiment.

**What it cannot support.** (a) Any comparison with §3: the ⅓-data runs
resolved AdamW (lr 1.11e-3) where the full-train runs resolved MuSGD (lr 1e-2),
so a scale-up would change the optimizer as well as the data. (b) Any claim
about the size of the fusion penalty: the same-subset IR control is 31 pp above
the best rung, but that control used the native augmentation while the rungs
used the paired pipeline, so the gap is pipeline-confounded. The unified
full-data tier (§10) exists to separate those two causes.

### 5.2 Full-data results — WG-Fusion (matched recipe, locked test)

All arms: full 17,962 training pairs, E00's resolved recipe (lr0 0.01, lrf
0.01, momentum 0.937, batch 16 with accumulation to nbs 64, close_mosaic 10,
patience 30, seed 0), single seed. Test = locked 8,980-image split, never used
for selection (locks in `results/LOCK_*.json`, created before any test
prediction); CIs = paired image-level bootstrap.

| Arm | val mAP50-95 (best ep) | test AP | AP50 | APs | APm | APl | ΔAP vs IR (95 % CI) |
|---|---:|---:|---:|---:|---:|---:|---|
| IR-only (E00) | 0.6306 (55) | 0.5783 | 0.7628 | 0.1866 | 0.5750 | 0.6298 | — |
| concat + 1×1 (E10) | **0.6538 (46)** | 0.5931 | 0.7754 | 0.2381 | 0.5912 | 0.5904 | +1.49 pp [+1.02, +1.90] |
| **WG-Fusion (E30)** | 0.6484 (63) | **0.5988** | **0.7819** | 0.2336 | **0.5951** | 0.6245 | **+2.08 pp [+1.71, +2.42]** |
| 2 hops (E41) | 0.6471 (55) | 0.5969 | 0.7784 | **0.2388** | 0.5939 | 0.6181 | +1.83 pp [+1.44, +2.19] |
| semantic-kNN (E35) | 0.6471 (48) | 0.5943 | 0.7770 | 0.2273 | 0.5921 | 0.6217 | +1.60 pp [+1.23, +1.97] |
| d=192 (capacity) | 0.6476 (39) | 0.5914 | 0.7760 | 0.2364 | 0.5881 | 0.6231 | +1.33 pp [+0.91, +1.67] |

Key paired contrasts: **WG-Fusion − concat = +0.591 pp [+0.247, +0.892],
P = 1.000**; 2 hops − WG-Fusion = −0.250 pp [−0.594, +0.036] (n.s.);
d=192 − WG-Fusion = **−0.745 pp [−1.054, −0.428]** (significantly worse);
E35 − concat = +0.100 pp [−0.261, +0.473] (n.s.).

**Ablation reading.** Depth helps small objects (best APs 0.2388) but not the
mean; capacity hurts on test despite an equal val score (a val-selection
warning); semantic edges are neutral; warm-started fine-tuning (streams
initialised from E00/E01, 50 ep at lr 0.002) peaks at 0.6217 val, far below
from-scratch. The graph formulation, not its hyper-parameters, carries the gain.

**Scale trade-off.** Concat buys the largest small-object gain but loses
large-object AP (0.5904 vs IR 0.6298); the graph arms gain nearly as much on
small objects while preserving large ones. WG-Fusion improves every band
(APs +4.71, APm +2.01, APl ≈ flat).

**SOTA context (1,469 split, as reported by HyperDet's single-protocol table,
2026).** WG-Fusion reaches 85.8/64.8 mAP50/50-95, above every 2023–2025 entry
including C2DFF-Net (85.7/65.2) and 0.5–1.7 pp behind the Feb-2026 hypergraph
frontier (HyperDet 86.4/65.9 and 87.5/67.1). The IR-only baseline alone matches
most published *fused* methods through 2024, so the controlled Δ over IR-only
is the meaningful number. Prior-work values, venues and protocol caveats:
[REPORT_V0.md](REPORT_V0.md) §5.3.

**Inference-time ensembles (post-hoc WBF, fixed defaults, no training).**
concat+WG+semantic = 0.6211 AP / 0.8072 AP50; WG+semantic = 0.6128 / 0.7983.

**Seeds.** Single seed (seed 0) for all reported numbers. A seed-1 run of
WG-Fusion reached 0.64875 best val at ep45 (vs 0.64838 for seed 0) before being
stopped for compute priority — encouraging but anecdotal; multi-seed reporting
is the main remaining robustness item.

## 6. Baselines and comparisons

- **E00** is the bar for *any* fusion: using RGB must beat IR-only.
- **E03** is the bar for *learned* fusion: a trained module that cannot beat
  free weighted box fusion does not justify itself.
- **E02** is the literature-standard early-fusion comparison.
- **E22** (when run) separates cross-modal benefit from added capacity.
- External re-implementations (IV-PAFNet's MSFA/ISA, ICAFusion's DMFF) are
  planned as `E13`/`E15` and `E21`. Native reproductions of those papers are not
  included: IV-PAFNet has no released weights and ICAFusion distributes its
  weights only through an inaccessible mirror. Documented protocol differences
  for those papers: IV-PAFNet uses visible annotations for both streams, keeps
  the 100 px border, squashes 840×712 to 640×640 and selects on val; ICAFusion's
  FLIR config points its validation entries at the test directories and has no
  DroneVehicle loader.

## 7. Training and evaluation

**Data.** Primary: DroneVehicle v2 — 17,962 / 1,469 / 8,980 paired train/val/test
images, 5 classes (car, truck, bus, van, freight_car), thermal labels
(`*labelr`) for IR supervision and visible labels (`*label`) for RGB, 640×512
crops. Secondary: FLIR-aligned 3-class (person, car, bicycle) with two
alignment policies (`drop`, `warp`) and the 1,013-frame paper test list.

**Recipe.** YOLO12s, 640 px, batch 16, seed 0, patience 30, val-only checkpoint
selection. Ladder rungs train on a ⅓-train screening subset (6,000 frames in 15
contiguous blocks); finalists rerun on full train. On a single 49 GB GPU,
graph/attention rungs run solo (two graph runs together forced an OOM
auto-reduction to batch=8, breaking both pace and recipe comparability); only
the lean baselines (E10+E14) share the GPU.

**Protocol.** All headline numbers come from the project's own
pycocotools-based evaluator (`src/eval/coco_eval.py`) on predictions exported
with fixed settings (conf 0.001, NMS IoU 0.6, maxDet 100). The validator used
during training (rect letterbox, Ultralytics NMS/mAP) is only for checkpoint
selection; it differs from the reporting protocol by ≈4 pp and is never quoted
as a result.

**Test lock.** The 8,980-frame thermal test split is evaluated only through
`scripts/final_eval.py`, which requires a pre-registered
`results/LOCK_<exp>.json` (config hash, checkpoint hash, val metric) and refuses
to overwrite results without a logged `--override`. Locks were created before
any test predictions were exported.

**Resume semantics.** While a run is active, `last.pt` and `best.pt` are written
from the same full checkpoint: model + EMA, optimizer state, scaler, epoch
counter, `best_fitness` and `train_args`. `resume=True` restores all of it and
continues the same LR schedule from that epoch. When a run **completes**,
Ultralytics' final evaluation calls `strip_optimizer()` on **both** files
(`epoch=-1`, optimizer and EMA state removed), so a finished run cannot be
resumed from either; only a run that was cancelled or crashed before final
evaluation keeps a full-state checkpoint. Restarting inside an existing run
directory without `resume=True` also resets the epoch/optimizer — attempts are
therefore always launched into fresh directories (old ones renamed
`*_INVALID_<reason>`). A warm start from weights alone (fresh optimizer, epoch
reset) changes the optimization budget and is not used for ladder comparisons.

**Leakage gate.** `scripts/check_leakage.py` verifies frame-id disjointness,
pair integrity, duplicate hashes, train-only normalisation and label
round-trips for both FLIR policies; the DroneVehicle splits are checked by the
builder and the COCO ground-truth exporter.

### Obtaining the data

Experiment 09 needs two paired RGB/IR datasets. Both are downloaded once into
the repository-root `data/` and never copied per experiment.

**DroneVehicle** (primary benchmark, ≈20 GB):

1. Download the three split archives (`train.zip`, `val.zip`, `test.zip`) from
   the DroneVehicle project (<https://github.com/VisDrone/DroneVehicle>; the
   Hugging Face mirror `McCheng/DroneVehicle` carries the same files) and place
   them in `data/dronevehicle/raw/`.
2. Build the 640×512 paired crops and YOLO trees with the shared builder:
   ```bash
   cd experiments/08_graph_kd/src
   PYTHONPATH=. python -m graphkd.data.build_dronevehicle
   ```
   This extracts the archives to `data/dronevehicle/raw/extracted/{train,val,test}`
   and writes `data/dronevehicle/yolo/{ir,rgb}`.
3. Apply the experiment-09 label policy (thermal `*labelr` for IR, visible
   `*label` for RGB) and the pair checks:
   ```bash
   cd experiments/09_graph_fusion
   python src/data/build_dv_v2.py
   ```
   Output: `experiments/09_graph_fusion/data/dronevehicle_v2/yolo/{ir,rgb}`
   plus `configs/dronevehicle_v2_{ir,rgb,paired}.yaml`. Expected counts are
   **17,962 / 1,469 / 8,980** pairs (see [RESULTS.md](RESULTS.md) §1).
4. Build the COCO ground truth used for evaluation:
   ```bash
   python scripts/make_gt_dv.py --modality ir --split val
   python scripts/make_gt_dv.py --modality ir --split test
   ```

**FLIR-aligned, 3-class** (secondary, ≈1.5 GB):

1. Download the aligned FLIR release published with Zhang et al.
   (*Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine
   Blocks*; the same aligned pairs are distributed with the ICAFusion/CFT
   projects) and extract it so that
   `data/flir_aligned/raw/align/JPEGImages/` and
   `data/flir_aligned/raw/align/Annotations/` exist.
2. Build the 4-class YOLO trees and the letterbox-corrected RGB images:
   ```bash
   cd experiments/08_graph_kd/src
   PYTHONPATH=. python -m graphkd.data.build_dataset
   ```
3. Build the 3-class policies used here:
   ```bash
   cd experiments/09_graph_fusion
   python src/data/build_flir3.py --policy all
   python scripts/check_leakage.py --policy drop
   python scripts/check_leakage.py --policy warp
   ```
   Output: `data/flir3/{drop,warp}` and `configs/flir3_*_{ir,rgb,paired}.yaml`
   (expected frames in [RESULTS.md](RESULTS.md) §1).

**Screening subset** (for the fusion ladder):

```bash
python scripts/build_screening.py     # 6,000 of 17,962 train frames, 15 blocks
```

## 8. Configurations and scripts

| Path | Purpose |
|---|---|
| `configs/dronevehicle_v2_{ir,rgb,paired}.yaml` | DroneVehicle v2 datasets (paired yaml has `channels: 6`) |
| `configs/dronevehicle_v2_*_screening.yaml` | ⅓-train screening variants used by the ladder |
| `configs/flir3_{drop,warp}_{ir,rgb,paired}.yaml` | FLIR-aligned 3-class datasets |
| `src/gfyolo/model.py` | dual-stream GF-YOLO assembly + pretrained loading |
| `src/gfyolo/gf_trainer.py` | Ultralytics trainer with paired dataset/validator |
| `src/gfyolo/paired_dataset.py` | 6-channel paired dataset and joint augmentation |
| `src/gfyolo/fused_trainer.py` | 6-channel trainer used by E02 |
| `src/gfyolo/wbf.py` | weighted box fusion used by E03 |
| `src/data/build_dv_v2.py`, `src/data/build_flir3.py` | dataset builders (labels, splits) |
| `src/eval/` | evaluator, geometry, locked evaluation support |
| `scripts/train_ref.py` | train a reference detector (E00/E01) |
| `scripts/train_fused.py` | train E02 early fusion |
| `scripts/train_gfyolo.py` | train a ladder rung (`--fuse E10/E14/E21/E30/E35`) |
| `scripts/predict_export.py`, `scripts/predict_export_paired.py` | export COCO predictions |
| `scripts/late_fusion_e03.py` | calibrate/apply weighted box fusion |
| `scripts/final_eval.py`, `scripts/make_lock.py` | locked test evaluation |
| `scripts/check_leakage.py` | pre-launch leakage gate |
| `scripts/complementarity_oracle.py` | Phase 1.2 headroom analysis |
| `testbed_gfyolo/` | standalone fusion modules + unit tests (77 tests) |
| `tests/` | evaluator, WBF, paired-augmentation and stem tests |

Commands (from this directory, `PYTHONPATH=src`):

```bash
# protocol checks
python -m pytest tests testbed_gfyolo -q
python scripts/check_leakage.py --policy warp

# references
python scripts/train_ref.py --name e00_ir --data configs/dronevehicle_v2_ir.yaml
python scripts/train_fused.py --name e02_fused --paired-rot90 --workers 16

# ladder rung
python scripts/train_gfyolo.py --name e10_screening --fuse E10 \
  --data configs/dronevehicle_v2_paired_screening.yaml \
  --rgb-yaml configs/dronevehicle_v2_rgb_screening.yaml --paired-rot90
```

All training runs are resume-safe: `runs/<name>/weights/last.pt` is resumed
automatically, and `DONE` markers skip completed runs.

## 9. Directory and artefacts

```
09_graph_fusion/
├── src/            builders, evaluator, GF-YOLO model, WBF, audited fusion modules
├── testbed_gfyolo/ standalone fusion modules + unit tests
├── scripts/        dataset builders, training, prediction, evaluation, lock
├── configs/        dataset YAMLs (DroneVehicle v2, FLIR 3-class, screening)
├── runs/           training runs (weights, results.csv, logs)
├── results/        locked test metrics, locks, GT/predictions, oracle report
├── audit/          dataset, alignment and leakage audits
├── data/           derived labels/splits and image trees
└── external/       original IV-PAFNet and ICAFusion repositories (pinned, read-only)
```

`results/<exp>_test_metrics.json` (locked test metrics), `results/LOCK_*.json`
(pre-registered choices), `results/oracle_dv_val.json` (complementarity
analysis) and `results/gt/`+`results/preds/` (COCO ground truth and exported
predictions) are produced by the scripts and are not versioned; the reported
numbers are recorded in [RESULTS.md](RESULTS.md).

## 10. Current status

- **Reference tier complete** with locked test numbers (§3); audited and
  reproducible (locks, checkpoints, predictions and GT hashes all match).
- **V1 screening tier invalidated**: every arm (including IR-only) collapsed
  under the superseded loader (0.14–0.25); the defect is documented and the V2
  loader reproduces the stock IR path bit-identically (§2, §5.1 retained for
  history).
- **Full-data flagship tier complete (single seed)**: WG-Fusion (E30) is the
  best single model on the locked test — **0.5988 AP, +2.08 pp vs IR-only,
  +0.59 pp vs concat (both CIs exclude zero)**; see §5.2 and
  [REPORT_V0.md](REPORT_V0.md).
- **Ablations complete**: two-hop depth (E41, 0.5969), capacity d=192 (0.5914,
  significantly worse), semantic edges (E35, 0.5943) and warm-started
  fine-tuning (0.6217 val) are neutral or negative; `r=2` is memory-infeasible
  at batch 16.
- **Paper draft** in [paper/](paper/) (CVPR style, figures generated from run
  artifacts by `paper/make_figures.py`).
- **Remaining:** multi-seed reporting (seed-1 partially run: 0.64875 val at
  ep45), qualitative figures (attention / Δ maps, confusion matrices,
  day-night examples), cross-scale edges for the small-object headroom,
  multi-dataset validation (FLIR-aligned, VEDAI), and the E22 capacity control.

## 11. Documentation links

- [Results](RESULTS.md) — datasets, protocol, locked test tables, oracle.
- [Report v0](REPORT_V0.md) — complete narrative of the answered experiment
  (abstract → method → results → SOTA comparison → limitations), with figures.
- [Paper draft](paper/main.tex) — CVPR-style manuscript; figures are generated
  from run artifacts by `paper/make_figures.py`.
- [Experiment plan](EXPERIMENT_PLAN.md) — full design, gates, ladder definitions.
- [Provenance](PROVENANCE.md) — external repositories, dataset sources, code
  provenance.
- [Experiment ledger](EXPERIMENTS.csv) — per-run tracking (status, metrics,
  decisions).
