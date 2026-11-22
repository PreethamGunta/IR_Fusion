# Experiment 09 — Graph-Based RGB-IR Fusion

**Status: active.** The reference tier (E00–E03) is complete with locked test
results; the fusion ladder (E10–E35) is implemented and the first rungs are
training. Numbers are in [RESULTS.md](RESULTS.md).

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
> beat simple fusion and attention fusion *under matched capacity, data, recipe
> and seed*?

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
| **E10** | `e10_screening` | concat + 1×1 conv at P3–P5 | ladder base (learned simple fusion) | training |
| E11 | — | elementwise add, fixed weights | no learned mixing | planned |
| E12 | — | channel-softmax modality gate | per-channel trust | planned |
| E13 | — | channel + spatial softmax gates | IV-PAFNet MSFA-style port | planned |
| **E14** | `e14_screening` | IR-anchored residual `f_ir + sigmoid(g)·conv([f_ir,f_rgb])`, negative-bias gate | starts as IR-only, learns corrections | training |
| E15 | — | E13 + visible-guided deformable alignment | misregistration fix | planned |
| E20 | — | one-directional cross-attention (IR←RGB) | attention baseline | planned |
| **E21** | — | bidirectional cross-attention, block-shared Q/K/V | DMFF-style attention representative | planned |
| E22 | — | intra-modal self-attention, capacity-matched to E21 | control: no cross-modal path | conditional |
| **E30** | — | GAT on grid + cross-modal window edges (IR←RGB) | first graph rung | planned |
| E31–E33 | — | GCN / GraphSAGE / GATv2+edge features | operator swaps | planned |
| E34 | — | bidirectional graph messages | refinement | planned |
| **E35** | — | E30 + semantic-kNN cross-modal edges | the core novelty | planned |
| E36–E37 | — | cross-scale edges; typed edge weights | refinements | planned |
| E38 | — | connectivity sweep (`r`, `R`, `k`, sparse/dense) | tuning | planned |
| E39–E41 | — | residual vs direct output; gate type; fusion placement | ablations | planned |
| E50 | — | object-level OT graph over candidate detections | speculative bridge to experiment 08 | planned |

Promotion rule: a rung is promoted only if its test mAP50-95 gain over its
parent exceeds the pre-registered threshold (**0.5 pp** under the single-seed
protocol), confirmed by a paired image-level bootstrap on the locked test set.

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
contiguous blocks); finalists rerun on full train.

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

- **Reference tier complete** with locked test numbers (see table above).
- **GF-YOLO testbed** implemented, reviewed and unit-tested (77 tests); dual-
  stream model integrated and smoke-tested (forward/backward, strides,
  parameter counts).
- **E10** completed its 100-epoch screening run: best validation mAP50-95
  0.2174 at epoch 87 on the ⅓-train subset (training protocol; not comparable
  to the full-train references), with a late-training collapse at epoch 100 to
  be diagnosed. **E14** is training on the same subset (best 0.1931 at the last
  recorded epoch, still improving). No ladder rung has a locked-test result.
- **E21, E30, E35, E22** pending; FLIR-aligned runs pending the DroneVehicle
  ladder.
- Paired image bootstrap on finalists pending.

## 11. Documentation links

- [Results](RESULTS.md) — datasets, protocol, locked test tables, oracle.
- [Experiment plan](EXPERIMENT_PLAN.md) — full design, gates, ladder definitions.
- [Provenance](PROVENANCE.md) — external repositories, dataset sources, code
  provenance.
- [Experiment ledger](EXPERIMENTS.csv) — per-run tracking (status, metrics,
  decisions).
