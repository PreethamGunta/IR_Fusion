# Provenance — Experiment 09

All paths are relative to the release root unless noted. No 08 source, config or
result file was modified; 08 images were read/hardlinked. The only 08 deletions
were the user-approved invalid/smoke runs and `data/cache` listed in
`AUDIT_REPORT.md`.

## External repositories (pinned)

| Repo | Commit SHA | License | Use |
|---|---|---|---|
| `external/IV-PAFNet` | `1de8cdd2ed02146232564153368d3e057ade67b2` | MIT | read converters/configs; Tier A reproduction |
| `external/IV-PAFNet_dataset` | `a3d84016d747208cf54f1ae8fbfcd3ef94aa29cf` | no separate dataset license; see repo README | released COCO annotations |
| `external/ICAFusion` | `6c06b831c70f484b7a434d51cd71e2a2586d1998` | AGPL-3.0 | read-only; DMFF will be re-implemented, not copied |

`DroneVehicle-COCO-annotations.tar` SHA-256:
`1e41468fd9fdf30b7ca00f86d9c21cdc40e68a3661908d99a5256bc75c6d888a`.

## Environment

Per user instruction, **all 09 work uses the existing `gateir` environment**
(`/home2/home/preetham_g/.conda/envs/gateir/bin/python`): Python 3.11.16,
torch 2.14.0+cu126, ultralytics 8.4.138. The temporary `gf09`/`ivpaf` clones
and the failed `icaf` prefix were removed to free quota; the frozen lists in
`environment/` are historical only. Tier A external reproduction that needs a
different Python/torch version is blocked until the user authorises recreating
such an env.

## Data

### DroneVehicle v2 (`data/dronevehicle_v2/`)

- Source images: `data/dronevehicle/raw/extracted/{split}/{split}/{split}img[r]`
  (840×712 with 100 px white border).
- Image files: hardlinks to the experiment-08 640×512 crops
  (`data/dronevehicle/yolo/{ir,rgb}/images/{split}`), which are read-only.
  These are **legacy** links: the shared-data rule now says large datasets live
  once under repo-root `data/` and experiments reference them by relative path.
  The local tree exists only because the corrected labels must sit next to
  images in Ultralytics' `images/` ↔ `labels/` layout; no dataset blocks are
  duplicated (same inodes).
- Labels: fresh YOLO labels written by `src/data/build_dv_v2.py`:
  - IR: `*labelr` for train, val **and test**;
  - RGB: `*label` for train, val and test.
- Counts: 17,962 / 1,469 / 8,980 pairs. The official 17,990 train labels imply
  28 images that are absent from the extracted archive (11 IR-only, 17
  RGB-only); they are listed in `audit/dv_v2_audit.json` and were not invented.
- Round-trip assertion: 0 mismatching frames per modality per split.

### FLIR 3-class (`data/flir3/{drop,warp}/`)

- Source: Zhang et al. aligned FLIR release, mirrored in experiment 08
  (`data/flir_aligned/`); images are hardlinks to the resolved raw targets.
- Classes: person, car, bicycle (dog dropped).
- Test: the complete paper validation list, 1,013 frames.
- Val: two contiguous frame-id blocks (3,900–4,400 and 7,000–7,120) with a
  ±10-frame buffer. A single tail holdout was rejected because frames
  5,500–8,859 contain 98.7% of the 667 letterboxed RGB frames.
- Alignment policies:
  - `drop`: remove pairs with robust |shift| > 2 px from train/val;
  - `warp`: translate flagged RGB frames by the estimated shift.
- No exact dHash duplicates across splits; frame-id overlaps are zero.

## Code provenance

The evaluator (`src/eval/coco_eval.py`), data builders and audit scripts are
new. No 08 module is imported. Copied 08 modules (adapted, source SHA-256
recorded; originals untouched):

| 09 file | Copied/adapted from 08 | Source SHA-256 |
|---|---|---|
| `src/gfyolo/paired_dataset.py` | `src/graphkd/data/dataset.py` | `c20b2e169eb9c8c910943989da3a24ef2beb6f8c48a52131c5528808044b6caf` |
| `src/gfyolo/fused_trainer.py` | `src/graphkd/training/trainer.py` (stem adaptation, fused trainer) | `1c2e4599c49a1498525d599ac1c786bde53e263355763358d90a66f2f5629629` |
| `src/gfyolo/fused_trainer.py` (validator) | `src/graphkd/training/paired_validator.py` | `48a11790a19f41eaa9e83aa06b2428de7bae82027954ee5ffc318259c4b289d2` |

09-side changes to the copies: plain `DetectionModel` (no KD machinery); the
paired augmentation split into `paired_geom`/`paired_phot` for the plan's
geometry test; the [IR | RGB] stem channel order fixed and unit-tested
(`tests/test_fused.py`, 4 tests) — 08's earlier fused teacher trained with
the halves swapped. Further 08 copies for Phase 2 (`graph/candidates.py`,
`matching/*`, `utils.py`) will be recorded here before use.

The GF-YOLO testbed modules (`testbed_gfyolo/fuse_modules.py`,
`neighbors.py`, 77 CPU tests) were created by the approved
`deepseek-v4.1-flash` subagent inside its dedicated `testbed_gfyolo/`
workspace (per the Subagent Rule in `rules.md`) and independently reviewed
by the main agent on 2026-11-21: tests re-run (77/77), six additional
probes passed (gate-init identity, gradient flow, border counts, semantic
kNN bounds/determinism, fp16), and a filesystem check confirmed no writes
outside the assigned directory. Known limitation: the PyG parity test is
blocked because PyG is not installed and no new packages are allowed.

The ultralytics integration (`src/gfyolo/model.py`, `gf_trainer.py`,
`scripts/train_gfyolo.py`, 2026-11-22) is main-agent code: it assembles the
stock yolo12s backbone twice plus the reviewed Fuse modules at P3/P4/P5
into one renumbered ModuleList (IR 0-8, RGB 9-17, Fuse 18-20, neck 21-32)
with remapped from-indices and a dual-entry `_predict_once`, importing —
not copying — the reviewed testbed modules.
