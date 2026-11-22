# GATE-IR — Handoff

Start here. This file tells a new session what to read, what exists, and what to do next.

## Read in this order

1. **`docs/memory.md`** — persistent context: who Preetham is, cluster constraints, environment, dataset facts, baseline numbers, known bugs.
2. **`docs/MASTER_PLAN.md`** — single source of truth for the project arc. Read §2 (strategic situation), §4 (assets), §7 (future directions) at minimum.
3. **`docs/handoff.md`** — this file: current state + what to do next.
4. **`docs/devlog.md`** — full session history, what broke, what was corrected. Read before proposing any new experiment.
5. **`docs/generating_weather_conditions.md`** — reference for future diffusion-based weather generation.

---

## The one-paragraph situation

GATE-IR is a three-stage thermal detection pipeline (weather gate → per-condition enhancement → YOLO). The UNIRI-TID baseline sweep (7 models, 6 valid) is complete. The M3FD synthetic weather experiments (Methods 1, 2, 3) are **fully complete with clean chunk-disjoint splits and verified provenance**. All results and model weights are on disk. No part of GATE-IR itself (Stage A, B, C) has been run. The next major direction is a larger IR dataset (DroneVehicle) and diffusion-based weather generation, expanding beyond rain/fog to snow, haze, etc.

---

## What exists and works

| | |
|---|---|
| env | `$HOME/.conda/envs/gateir` (torch 2.14+cu126, ultralytics 8.4.138) |
| UNIRI-TID data | `data/UNIRI-TID/` — 6,340 imgs, 6,276 annotated, single class `Human` |
| sequences | `sequence_index.csv` — 59 recordings, recovered from burned-in frame counter |
| conditions | `conditions.csv` — clear/fog/rain, validated against published split |
| splits | `splits.csv` + `data/UNIRI-TID/yolo/` — sequence-disjoint and condition-stratified |
| training (UNIRI) | `bench/train_detector.py` — trains + scores any Ultralytics model |
| scoring | `bench/eval/detect_eval.py` — per-condition + per-object-size AP via pycocotools |
| baselines (UNIRI) | `bench/results/*.json` — 7/7 done. RT-DETR-l invalid (29 epochs, 0.28 mAP). |
| M3FD split | `m3fd_experiments/resplit_m3fd.py` — chunk-disjoint temporal split (split_hash `effb9b3090ad2fcb`) |
| M3FD training | `m3fd_experiments/train_m3fd.py` — COCO-pretrained YOLOv12s, 100 epochs, early stop patience 25 |
| M3FD eval | `m3fd_experiments/eval_m3fd.py` — --out flag, provenance tracking |
| M3FD Method 1 | `m3fd_experiments/method1_classical/` — classical fog/rain, all 6 severity results |
| M3FD Method 2 | `m3fd_experiments/method2_depth/` — depth-based Koschmieder fog, label filtering |
| M3FD Method 3 | `m3fd_experiments/method3_cyclegan/` — CycleGAN generators, DSERT-RoLL target, all evals |
| M3FD results | `m3fd_experiments/results/` + per-method results/ — all clean, provenance-stamped |
| Final report | `m3fd_experiments/methods_report.tex` — corrected numbers, ready to compile |
| Exam prep | `m3fd_experiments/_old/exam_prep.md` — professor Q&A, defense cheat sheet |
| Unified pipeline | `m3fd_experiments/scripts/run_all_methods.sh` — all 3 methods from clean split |
| Stage 5 script | `m3fd_experiments/scripts/run_stage5_method2.sh` — resumable, skip-completed, isolated failures |
| Stage 6+7 script | `m3fd_experiments/scripts/run_remaining_stages.sh` — positional args, isolates fog/rain 3B |
| CycleGAN | `m3fd_experiments/method3_cyclegan/` — 4 generators (thermal_fog, thermal_rain, rgb_fog, rgb_rain) |
| DSERT-RoLL | `m3fd_experiments/method3_cyclegan/data/DSERT-RoLL/` — Fog 3,282 + Heavy_Rain 1,938 + Light_Rain 2,214 |
| debug | `DEBUG=1` → 2 frames/condition/split, 2 epochs. Quarantined in `bench/results/debug/` |
| submitting | `sbatch m3fd_experiments/scripts/submit_all_methods.sbatch` — 24h, `--requeue` |

---

## Baseline results — UNIRI-TID (test set, mAP@50-95)

| model | epochs | mAP@50-95 | clear | fog | rain |
|---|---|---|---|---|---|
| YOLOv8s | 71 | 0.6204 | 0.6536 | 0.5496 | 0.5917 |
| YOLO11s | 55 | 0.6194 | 0.6491 | 0.5341 | 0.6035 |
| **YOLOv12s** | 76 | **0.6296** | 0.6580 | **0.5807** | 0.6052 |
| YOLOv8s + aug | 69 | 0.6278 | 0.6558 | 0.5785 | 0.6019 |
| YOLOv10s | 69 | 0.6063 | 0.6361 | 0.5434 | 0.5800 |
| YOLOv8s-P2 | 100 | 0.6071 | 0.6402 | 0.5108 | 0.5931 |
| ⚠️ RT-DETR-l | 29 | 0.2778 | — | — | — |

APm (medium objects, 89–97% of all boxes):

| model | APm clear | APm fog | APm rain | clear−fog |
|---|---|---|---|---|
| YOLOv8s | 0.6623 | 0.5344 | 0.5951 | 0.128 |
| YOLO11s | 0.6585 | 0.5264 | 0.6038 | 0.132 |
| YOLOv12s | 0.6670 | 0.5733 | 0.6025 | 0.094 |
| YOLOv8s+aug | 0.6652 | 0.5691 | 0.6035 | 0.096 |
| YOLOv10s | 0.6459 | 0.5316 | 0.5873 | 0.114 |
| YOLOv8s-P2 | 0.6480 | 0.4954 | 0.5909 | 0.153 |

**Bar for GATE-IR on UNIRI-TID:** beat YOLOv8s+aug fog APm = 0.5691.

---

## M3FD synthetic weather results — COMPLETE (clean split)

Clear baseline: mAP@50-95 = 0.148 (split `effb9b3090ad2fcb`, 2520 train / 240 val / 600 test).

| Method | Protocol | Condition | mAP@50 | mAP@50-95 | Gap | Verdict |
|---|---|---|---|---|---|---|
| Clear baseline | — | clear | 0.268 | 0.148 | — | ref |
| M1 classical | A | fog_light | 0.268 | 0.148 | 0.001 | FAIL |
| M1 classical | A | fog_medium | 0.243 | 0.135 | 0.013 | FAIL |
| M1 classical | A | fog_heavy | 0.206 | 0.117 | 0.032 | FAIL |
| M1 classical | A | rain_light | 0.261 | 0.144 | 0.005 | FAIL |
| M1 classical | A | rain_medium | 0.189 | 0.094 | 0.054 | WEAK |
| M1 classical | A | rain_heavy | 0.079 | 0.041 | 0.108 | WEAK |
| M2 depth | A | fog_medium | 0.199 | 0.114 | 0.035 | FAIL |
| M2 depth | A | fog_heavy | 0.097 | 0.072 | 0.076 | WEAK |
| M2 depth | A | rain_medium | 0.094 | 0.054 | 0.054 | WEAK |
| M2 depth | A | rain_heavy | 0.011 | 0.011 | 0.137 | WEAK |
| M2 depth | B | clear test | 0.244 | 0.141 | — | ref |
| M2 depth | B | fog_heavy | 0.237 | 0.140 | −0.001 | FAIL |
| M3 cyclegan | A | fog | 0.075 | 0.048 | 0.101 | WEAK |
| M3 cyclegan | A | rain | 0.185 | 0.105 | 0.044 | FAIL |
| M3 cyclegan | B | clear test | 0.260 | 0.147 | — | ref |
| M3 cyclegan | B | fog | 0.193 | 0.098 | 0.049 | WEAK |
| M3 cyclegan | B | rain | 0.197 | 0.104 | 0.011 | FAIL |

All results carry `split_hash=effb9b3090ad2fcb` and `weights_sha256` provenance. Full report: `m3fd_experiments/methods_report.tex`.

---

## Open problems

### 1. Low clear baseline mAP on M3FD (0.148)

The YOLOv12s clear baseline scores 0.148 mAP@50-95 on M3FD. Likely causes: small dataset (2,520 train), class imbalance (Cars dominate, Motorcycles/Trucks/Lamps rare), small objects penalised by mAP@50-95. The absolute gap thresholds (0.15 A, 0.10 B) may need to be relative (fraction of baseline) rather than absolute.

### 2. CycleGAN 3A fog gap anomaly (+0.477 in contaminated run; +0.101 in clean run)

The clean run shows 0.101, which is more reasonable but still the largest gap. The `thermal_fog` generator may be transferring sensor characteristics (different thermal camera) alongside weather. Sample grids in `runs/thermal_fog/samples/` need visual inspection.

### 3. Stage A/B/C has not been run

No part of the actual GATE-IR pipeline exists as a trained model. Every number above is a baseline.

### 4. RT-DETR-l baseline is invalid

29 epochs / 0.28 mAP@50-95 — hit walltime without resume. Needs a rerun.

---

## What to do next

1. **E1 — Efficiency table** (params, GFLOPs, FP16 latency, FPS, peak memory for every model + all three GATE-IR stages). T1 lives or dies on these numbers.
2. **Rerun RT-DETR-l.** Resubmit as a standalone `sbatch --requeue` job.
3. **Decide: UNIRI-TID Stage A/B/C now, or pivot to DroneVehicle?** Key tradeoff: UNIRI-TID has a smaller but real and persistent gap; DroneVehicle offers larger scale and more classes.
4. **Build Stage A — WeatherGate.** Run E2 (gate accuracy on real weather: F1, confusion matrix, vs logistic regression / random forest) and E3 (gate confound test: hold out sequences *within* each condition).
5. **E4 — Oracle-gate and no-gate bounds.** Mandatory.
6. **Investigate diffusion-based weather generation** for the next dataset (see `docs/generating_weather_conditions.md`).
7. **Inspect CycleGAN sample grids** to diagnose the 3A fog gap.

---

## Things that will bite you

- **Never `scancel` the allocation.** Run inside it: `srun --jobid=<id> --overlap <cmd>`. Kill runs by Python PID.
- **QOS is 1 GPU, 1 running job, 24 h.** ~4–6 training runs per allocation.
- **Request `--time=6:00:00`** for backfill. `scontrol update` silently reverted once — re-check.
- **Nodes fail.** Always `sbatch --requeue`.
- **mAP@50 is saturated on UNIRI-TID (~97%).** Report mAP@50-95 only.
- **Splits must stay sequence/chunk-disjoint.**
- **Condition is confounded with camera distance.** Always report AP vs object size.
- **UNIRI-TID images are pseudo-coloured 8-bit RGB**, not radiometric.
- **Use `All_In_One_Anot_*` annotations only.**
- **YOLO silent label bug.** If images are at `data/IR/images/train/img.jpg`, labels MUST be at `data/IR/labels/train/img.txt`.
- **Protocol A gap is not the metric that matters.** Always run Protocol B. A synthetic fog that passes A can still collapse in B.
- **M3FD evaluation uses IR images only.** RGB fog/rain synthesis has no effect on detection metrics.
- **Method 2 rain IR = Method 1 rain IR** (byte-identical). Do not report as different.
- **Do not reinstall `thop`** — breaks RT-DETR under torch 2.14.
- **Cache matters.** Training is CPU-bound on JPEG decode. Use `--cache disk`.
- **Never derive weather labels from image statistics.** Fatal circularity for the gate.

---

## Working with Preetham

Discuss before producing a plan document — back-and-forth first. Surface findings that change direction immediately. He wants numbers, but not at the cost of numbers that turn out to be meaningless. Explicitly wants to know when something is a weak or indefensible claim.

---

## Open questions for Preetham to decide

- Which thesis: **T1** (efficiency/Pareto), **T2** (gate robustness), or **T3** (negative result)?
- Target venue and deadline?
- Is a Jetson Orin obtainable?
- Should Aryan Sood be looped in before Stage C is re-expressed through Ultralytics?
- Next dataset: **DroneVehicle** or continue with UNIRI-TID?
- Should diffusion-based weather generation be prioritised over Stage A/B/C?
