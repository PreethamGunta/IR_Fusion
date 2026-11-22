# Experiment 09 — Results (Phase 0)

No 09 model has been trained. This file contains only data/protocol tables.
All numbers are [VERIFIED] from the build/audit JSONs unless marked otherwise.

## Datasets

### DroneVehicle v2

| Split | Pairs | IR boxes | RGB boxes | IR classes (car,truck,bus,van,freight) |
|---|---:|---:|---:|---|
| train | 17,962 | 315,826 | 286,217 | 269,855 / 15,816 / 11,295 / 7,689 / 11,171 |
| val | 1,469 | 24,490 | 22,462 | 20,588 / 1,470 / 789 / 725 / 918 |
| test | 8,980 | 159,613 | 143,313 | 137,145 / 8,657 / 4,467 / 4,282 / 5,062 |

Official train labels = 17,990; 28 images missing from the extracted archive
(11 IR-only, 17 RGB-only), documented in `audit/dv_v2_audit.json`.

### FLIR 3-class

| Policy | Split | Frames | Person | Car | Bicycle |
|---|---|---:|---:|---:|---:|
| warp | train | 3,752 | 7,835 | 19,249 | 2,501 |
| warp | val | 365 | 1,081 | 1,348 | 65 |
| warp | test | 1,013 | 4,107 | 4,124 | 360 |
| drop | train | 1,387 | 2,675 | 6,864 | 927 |
| drop | val | 223 | 793 | 904 | 35 |
| drop | test | 1,013 | 4,107 | 4,124 | 360 |

`test_clean` (both policies) = 491 unflagged test frames.

## Alignment and leakage

| Split | Frames | cropped RGB | flagged >2 px |
|---|---:|---:|---:|
| warp train | 3,752 | 16.3% | 63.0% |
| warp val | 365 | 13.2% | 38.9% |
| test | 1,013 | 0% | 51.5% |

- Exact dHash duplicates across splits: 0.
- dHash NN median: test→train 16, val→train 16, train→train 13.
- Frame-id overlaps: 0.
- Pair integrity: true for every split.
- Normalization stats (train only) and SHA-256: `audit/flir3_leakage_*.json`.

## Gate A

**Formal Gate A (3-seed sigma_seed + external reproductions): waived** per the
minimal single-seed pass (plan §5.4). Substitute Gate A: protocol/leakage
checks 0.2–0.5 PASS [VERIFIED]; references E00–E03 trained and locked-test
evaluated with sane numbers (see the Tier A table below); the single-seed
caveat (deltas < 0.5 pp AP50-95 = "no detectable gain", paired image
bootstrap for the headline claim) is pre-registered. Tier A external
reproductions remain blocked (no accessible weights; `gateir`-only policy).

## Tier A references — minimal first pass, LOCKED TEST [VERIFIED 2026-11-22]

All runs: YOLO12s, 640 px, batch 16, seed 0, patience 30, val-only selection,
DroneVehicle v2 (thermal GT primary). Locked test = 8,980 frames, thermal GT
(`results/gt/dv_v2_ir_test.json`), scored by `src/eval/coco_eval.py` via
`scripts/final_eval.py` with locks (`results/LOCK_<exp>.json`) created before
any test access. Metrics: `results/<exp>_test_metrics.json`.

| Run | Val AP50-95 (ultralytics) | Val AP50-95 (locked protocol) | **Test AP50-95** | Test AP50 | Test AP75 |
|---|---:|---:|---:|---:|---:|
| **E00** `e00_ir` | 0.63055 (ep55) | 0.58757 | **0.57828** | 0.76281 | 0.69743 |
| **E01** `e01_rgb` | 0.58822 (ep84, visible GT) | 0.43578 (thermal) | **0.41264** | 0.65337 | 0.45111 |
| **E02** `e02_fused` | 0.50471 (ep100) | 0.46847 | **0.47619** | 0.69418 | 0.57086 |
| **E03** `e03_late_fusion` | 0.58833 (WBF grid) | — | **0.57939** | 0.78682 | 0.70746 |

Per-class test AP50-95 (car / truck / bus / van / freight_car):

| Run | car | truck | bus | van | freight_car |
|---|---:|---:|---:|---:|---:|
| E00 | 0.690 | 0.554 | 0.759 | 0.411 | 0.477 |
| E01 | 0.462 | 0.402 | 0.624 | 0.279 | 0.296 |
| E02 | 0.597 | 0.453 | 0.697 | 0.303 | 0.331 |
| E03 | 0.663 | 0.571 | 0.750 | 0.422 | 0.491 |

### Findings

1. **E00 (IR-only) is the reference to beat: test AP50-95 0.578.**
2. **E03 (free WBF late fusion) ≈ E00**: +0.11 pp AP50-95 (below the 0.5 pp
   promotion line) but consistent gains at looser IoU (+2.40 pp AP50,
   +1.00 pp AP75) and on truck (+1.6 pp) — RGB adds recall without hurting
   precision. Under the single-seed rule: **no detectable AP50-95 gain**;
   the paired image bootstrap will be run on the finalists before any
   stronger claim.
3. **E02 (6-channel early fusion) is 10.2 pp below IR-only** (0.476 vs
   0.578) — early fusion clearly hurts on DV v2, matching 08's finding and
   the literature's "usually mediocre". Caveat: E02 was still improving at
   its recipe-capped ep100, so part of the gap may be under-training.
4. **E01 graded on thermal GT is weak** (0.413) and drags WBF down via
   weight, despite decent visible-GT val (0.588): the thermal/visible label
   disagreement costs ~15 pp. The complementarity oracle (Phase 1.2) will
   quantify where RGB still helps (day scenes, IR-only misses).

### Protocol notes

- Ultralytics training-val (rect letterbox, own NMS/eval) vs the locked
  export protocol (square 640×640 letterbox, NMS IoU 0.6, pycocotools,
  conf 0.001, maxDet 100) differ by ~4 pp (E00 0.6306→0.5876, E02
  0.5047→0.4685). All cross-model comparisons use the locked protocol only.
- E03's WBF config (iou 0.6, weights 1.5:1 IR:RGB, members-scaling on) was
  chosen on a 12-config VAL grid; manifest + lock:
  `results/e03_config.json`, `LOCK_e03_late_fusion.json`.
- Exporter incidents (list-warmup OOM; tensor letterbox frame) are in
  `STATUS_LOG.md`; invalid outputs preserved as `*_INVALID_*`.

## Phase 1.2 complementarity oracle (DV val, thermal GT) [VERIFIED]

From the E00/E01 val predictions; thresholds fixed on val by max-F1
(IR 0.50 / RGB 0.45 at IoU 0.5); day/night split by the median TRAIN RGB
luminance. Full JSON: `results/oracle_dv_val.json`.

| Segment | Recall IR | Recall RGB | Best-of-either | **Oracle gain** |
|---|---:|---:|---:|---:|
| overall | 0.9179 | 0.7981 | 0.9497 | **+3.18 pp** |
| day | 0.9159 | 0.9125 | 0.9593 | +4.34 pp |
| night | 0.9195 | 0.7070 | 0.9420 | +2.25 pp |
| small objects | 0.8403 | 0.7200 | 0.9077 | **+6.73 pp** |

GT objects (val): 18,767 both, 3,712 IR-only, 778 RGB-only, 1,233 neither.

**Interpretation.** The oracle gain is ~3.2 pp overall and ~6.7 pp on small
objects — well above the plan's ~1.5 pp "expect fusion gains" line, so there
IS complementarity worth chasing. RGB nearly matches IR in day scenes
(0.913 vs 0.916) and collapses at night (0.707). Yet E03 (WBF) captured
almost none of it in AP50-95 (+0.11 pp; +2.40 pp at AP50, consistent with
the recall gains at IoU 0.5) and E02 early fusion lost 10 pp. The ladder's
job is to harvest this headroom — with small objects (P3-level fusion) the
most promising target.

## Current state (2026-11-22 ~02:25 UTC)

- **Reference tier E00–E03 of the minimal first pass: COMPLETE** [VERIFIED]
  (locked test numbers above). 08 remains paused/resumable.
- Allocation 10133 ends 2026-11-22 08:13; the E10+ ladder waits for the
  pending allocation (`girkd`, est. 2026-11-25 22:20).
- GF-YOLO testbed modules (subagent-built, reviewed 77/77 + probes) are
  ready in `testbed_gfyolo/`; integration into a dual-stream Ultralytics
  model is the next main-agent coding task.

## Next steps

1. Phase 1.2 complementarity oracle from the E00/E01 val predictions (CPU).
2. Paired image bootstrap on E00 vs E03 (the one headline comparison so far).
3. GF-YOLO integration (dual-stream model + Fuse at P3–P5), unit-tested.
4. E10→E14→E21→(E22 if E21>E14)→E30→E35 on the screening subset in the next
   allocation; finalists rerun on full DV; test stays locked.



