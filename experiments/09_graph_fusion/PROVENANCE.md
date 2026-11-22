# Experiment 09 — Provenance

## External repositories (pinned)

| Repository | Commit | License | Use |
|---|---|---|---|
| `external/IV-PAFNet` | `1de8cdd2ed02146232564153368d3e057ade67b2` | MIT | read-only reference for MSFA/ISA re-implementations |
| `external/IV-PAFNet_dataset` | `a3d84016d747208cf54f1ae8fbfcd3ef94aa29cf` | see repository README | released COCO annotations; protocol comparison |
| `external/ICAFusion` | `6c06b831c70f484b7a434d51cd71e2a2586d1998` | AGPL-3.0 | read-only reference; DMFF re-implemented, never copied |

The DroneVehicle COCO annotation archive used for protocol checks has SHA-256
`1e41468fd9fdf30b7ca00f86d9c21cdc40e68a3661908d99a5256bc75c6d888a`.

## Datasets

| Dataset | Source | Notes |
|---|---|---|
| DroneVehicle | Sun et al., IEEE TCSVT 2022 (UA-CMDet) | paired aerial RGB/IR; 5 classes; thermal labels (`*labelr`) for IR, visible labels (`*label`) for RGB |
| FLIR-aligned | Zhang et al., *Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine Blocks* | aligned RGB/IR pairs, 3 classes after dropping `dog` |
| FLIR ADAS v2 | FLIR Systems release (`data/flir_adas_v2`) | thermal/RGB frames and original checksums |

DroneVehicle v2 counts and the 28 missing source frames are documented in
`audit/dv_v2_audit.json`. FLIR alignment statistics and leakage checks are in
`audit/flir3_*.json`.

## Environment

All code runs in the project `gateir` environment: Python 3.11.16,
torch 2.14.0+cu126, ultralytics 8.4.138, pycocotools, opencv. The evaluator
depends only on pycocotools and torch; the fusion modules are pure PyTorch.

## Code provenance

- The evaluator (`src/eval/`), dataset builders (`src/data/`), audit scripts
  (`src/audit/`), trainers (`src/gfyolo/`) and experiment scripts are written
  for this experiment.
- `src/gfyolo/paired_dataset.py`, the stem-adaptation part of
  `src/gfyolo/fused_trainer.py` and its paired validator are adapted from
  experiment 08's GraphKD data/training modules (copied, not imported):

| File | Origin (08) | Source SHA-256 |
|---|---|---|
| `src/gfyolo/paired_dataset.py` | `src/graphkd/data/dataset.py` | `c20b2e169eb9c8c910943989da3a24ef2beb6f8c48a52131c5528808044b6caf` |
| `src/gfyolo/fused_trainer.py` (stem) | `src/graphkd/training/trainer.py` | `1c2e4599c49a1498525d599ac1c786bde53e263355763358d90a66f2f5629629` |
| `src/gfyolo/fused_trainer.py` (validator) | `src/graphkd/training/paired_validator.py` | `48a11790a19f41eaa9e83aa06b2428de7bae82027954ee5ffc318259c4b289d2` |

Adaptations: plain `DetectionModel` instead of the KD model; joint paired
augmentation split into geometric and photometric parts; the `[IR | RGB]` stem
channel order fixed and unit-tested.

- The fusion modules in `testbed_gfyolo/` implement the experiment's own design
  (concat, gated residual, bidirectional cross-attention, GAT with grid/window
  edges, semantic-kNN edges) in pure PyTorch with neighbour tables built from
  first principles; no graph library is used.
- `src/gfyolo/model.py` assembles the dual-stream detector from the stock
  YOLO12s architecture and the modules above.

## Ground truth and predictions

`results/gt/` contains COCO ground truth generated from the label trees by
`scripts/make_gt_dv.py`; the counts match the dataset audit exactly.
`results/preds/` contains exported predictions for the reported runs. Both are
regenerable and are excluded from version control.
