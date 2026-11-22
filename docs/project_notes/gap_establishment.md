# Gap Establishment Doctrine

**Status:** Active. All downstream work is gated on this document's exit criteria.

## Purpose

This document defines why, when, and how a significant weather gap must be established before any
GATE-IR pipeline code is written. It exists because the project has previously built experimental
infrastructure (M3FD Methods 1–3, Stage A/B/C scaffolding) before confirming that the underlying
weather degradation produces a gap large enough to justify the methodology.

## Core principle

**A gated pipeline cannot manufacture a gap that does not already exist in the data.**

The GATE-IR methodology (hard gating, per-condition enhancement, specialized detector) is only
justifiable if a naive mixed-trained baseline (Protocol 3B) already shows a significant gap between
clear and adverse conditions. If the maximum achievable gap is < 10%, the entire pipeline is
over-engineered for the problem size.

## Gap thresholds

| Threshold | Meaning | Action |
|---|---|---|
| ≥ 0.15 mAP@50-95 (Protocol 3B) | Significant gap. Pipeline construction is justified. | Proceed to Stage A/B/C. |
| 0.10–0.14 mAP@50-95 (Protocol 3B) | Marginal gap. Pipeline may be justified with strong efficiency numbers. | Proceed with caution; emphasize T1 (efficiency) over T2 (accuracy). |
| < 0.10 mAP@50-95 (Protocol 3B) | Insufficient gap. Pipeline cannot be justified on accuracy. | Pivot to T3 (negative/diagnostic result). |

**Absolute vs relative gaps:** On datasets where the clear baseline mAP@50-95 is low (< 0.55),
absolute gap thresholds may need to be relative (f.g., 25% of baseline). On datasets where the
clear baseline is high (> 0.60), absolute thresholds are appropriate.

## Why M3FD is a negative result

M3FD clear baseline: mAP@50-95 = 0.148.

| Method | Protocol | Condition | mAP@50-95 | Gap | Verdict |
|---|---|---|---|---|---|
| M1 classical | B | fog_heavy | 0.117 | 0.032 | FAIL |
| M2 depth | B | fog_heavy | 0.140 | −0.001 | FAIL |
| M3 cyclegan | B | fog | 0.098 | 0.049 | WEAK |
| M3 cyclegan | B | rain | 0.104 | 0.011 | FAIL |

No method sustains a Protocol 3B gap ≥ 0.05. The maximum observed gap is +0.049 (CycleGAN fog),
which is below even the marginal threshold of 0.10.

**Conclusion:** The M3FD synthetic-weather track is a negative result. It is cited as evidence that
classical, depth-based, and CycleGAN methods are insufficient, not as a baseline to beat. The project
does not re-enter this track unless a new generation method (CycleDiff) first demonstrates a 3B gap
≥ 0.15 on a different base dataset.

## Why UNIRI-TID alone is insufficient

UNIRI-TID clear-fog gap (YOLOv12s, matched object size): 0.094 APm.

This gap is real and persistent across architectures, but:
- mAP@50 is saturated at 0.96–0.97 — the benchmark cannot distinguish models on accuracy alone.
- Augmentation alone recovers ~28% of the gap (+0.029 fog mAP).
- A gated pipeline that recovers less than ~0.03 fog APm over an augmented baseline has no contribution.

**Conclusion:** UNIRI-TID is the primary dataset for baseline numbers and real-weather evidence, but
it cannot carry the paper alone. A larger gap on a multi-class dataset is required.

## FLIR ADAS v2 — gap establishment plan

**Dataset:** FLIR ADAS v2 (Kaggle `samdazel/teledyne-flir-adas-thermal-dataset-v2`)
**Modality:** Paired IR + RGB, 15 classes, ~26k images
**Status:** NOT YET DOWNLOADED

### Phase 1a — Natural gap measurement (if real weather exists)

If FLIR ADAS v2 contains real weather conditions (fog, rain, snow), measure the natural
clear-vs-adverse gap with YOLOv12s under Protocol 3B (mixed training). If the gap ≥ 0.15,
proceed directly to pipeline construction. No synthetic generation needed.

### Phase 1b — Synthetic generation (if no real weather exists)

If FLIR ADAS v2 contains only clear conditions, or the natural gap is < 0.10, apply synthetic
weather using Methods 1–4:

| Method | Description | Cost | Expected gap |
|---|---|---|---|
| M1 classical | Uniform veil (fog), streak overlay (rain), particles (snow) | Minutes | Small (M3FD: max 0.032) |
| M2 depth | Koschmieder fog with depth-based transmission | ~1–2 h | Small-Medium (M3FD: 0.076 A, −0.001 B) |
| M3 CycleGAN | Unpaired IR translation via DSERT-RoLL + Heavy_Snow | ~8–20 h | Medium (M3FD: 0.049 B fog) |
| M4 CycleDiff | Hybrid CycleGAN + diffusion refinement | ~30–60 h | Largest (unproven) |

**Generation target domains:**
- Fog: DSERT-RoLL Fog (3,282 thermal frames)
- Rain: DSERT-RoLL Heavy_Rain (1,938) + Light_Rain (2,214)
- Snow: Heavy_Snow (source TBD — must be acquired)

### Phase 1c — Gap decision

After Methods 1–4 are evaluated on FLIR ADAS v2:

| Best observed Protocol 3B gap | Action |
|---|---|
| ≥ 0.15 (any method, any condition) | Proceed to Phase 2 (pipeline construction). |
| 0.10–0.14 (any method, any condition) | Proceed to Phase 2 with caution; emphasize T1 (efficiency). |
| < 0.10 (all methods, all conditions) | Pivot to T3 (negative/diagnostic result). Pipeline construction becomes "build it and show it doesn't help." |

## Exit criteria for gap establishment

1. FLIR ADAS v2 is downloaded, inspected, and prepared with train/val/test splits.
2. DSERT-RoLL Heavy_Snow is acquired and integrated (if snow is a target condition).
3. Methods 1, 2, 3, and 4 are implemented and evaluated on FLIR ADAS v2 under Protocols 3A and 3B.
4. At least one method + condition produces a Protocol 3B gap ≥ 0.10 mAP@50-95.
5. The gap decision is recorded in `docs/handoff.md` and `MASTER_PLAN.md`.

## Anti-patterns to avoid

- **Building the pipeline before the gap is confirmed.** This assumes the gap exists, which the M3FD
  experiments proved false.
- **Using Protocol A as the primary metric.** A synthetic fog that passes Protocol A can still collapse
  in Protocol B (confirmed with Methods 1 and 2 on M3FD; Method 3 fog survived weakly at +0.049).
- **Reporting single-seed results.** A 0.15 mAP gap from a single seed is a promising signal, not a
  confirmed result. Run ≥ 3 seeds.
- **Treating PSNR as the win condition.** The generation method must be judged by detection mAP, not
  perceptual quality. A visually realistic fog that is easily adapted by YOLO under mixed training
  produces a small Protocol 3B gap and is therefore useless for GATE-IR.
- **Ignoring snow physics.** Snow is not white fog in thermal IR. A generation model that treats snow
  as "bright fog" will produce unrealistic outputs and a small or negative detection gap.
- **Circularity in weather labels.** Never derive weather labels from image statistics that the gate
  will later classify. Labels must come from the generation process (severity parameter) or from
  human/author-supplied annotations.

## Relationship to other documents

- `MASTER_PLAN.md` §3 (thesis) and §8 (methodology rules) incorporate this doctrine.
- `handoff.md` §"What to do next" orders tasks by this doctrine.
- `memory.md` §"Project goal" records the gap-first doctrine as a durable constraint.
- `dataset_generation_plan.md` implements the protocol described here.
