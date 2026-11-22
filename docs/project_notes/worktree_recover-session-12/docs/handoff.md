# GATE-IR — Handoff

Start here. This file tells a new session what to read, what exists, and what to do next.

## Read in this order

1. **`docs/gap_establishment.md`** — the gap-first doctrine, go/no-go rules, and why pipeline construction is deferred until a significant gap is confirmed.
2. **`docs/memory.md`** — persistent context: cluster constraints, environment, dataset facts, baseline numbers, known bugs, methodology rules.
3. **`docs/MASTER_PLAN.md`** — long-term project map. Read §2 (strategic situation), §4 (assets), §7 (future directions) at minimum.
4. **`docs/handoff.md`** — this file: current state + what to do next.
5. **`docs/devlog.md`** — full session history, what broke, what was corrected. Read before proposing any new experiment.
6. **`docs/generating_weather_conditions.md`** — reference for diffusion-based weather generation (used by CycleDiff).
7. **`docs/dataset_generation_plan.md`** — FLIR ADAS v2 experiment protocols, Method 3/4 specifications, and results tables.

---

## The one-paragraph situation

GATE-IR is a three-stage thermal detection pipeline (weather gate → per-condition enhancement → YOLO). The UNIRI-TID baseline sweep (7 models, 6 valid) is complete. The M3FD synthetic weather experiments (Methods 1, 2, 3) are **fully complete with clean chunk-disjoint splits and verified provenance**. All three methods failed Protocol 3B (fog gap +0.049 WEAK, rain gap +0.011 FAIL), making the synthetic-generation track a negative result. No part of GATE-IR itself (Stage A, B, C) has been run. The immediate next direction is the FLIR ADAS v2 dataset (paired IR+RGB, 15 classes, ~26k images) with a new Method 4 (CycleDiff) generating fog, snow, and rainy conditions. Pipeline construction is deferred until Protocol 3B on FLIR ADAS v2 confirms a gap ≥15% mAP@50-95.

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

**Bar for GATE-IR on UNIRI-TID:** beat YOLOv8s+aug fog APm = 0.5691. This bar is only meaningful if a larger gap can first be established on FLIR ADAS v2.

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

**Verdict:** No M3FD method sustains a Protocol 3B gap ≥ 0.05. The M3FD synthetic-weather track is a negative result. The project does not re-enter this track unless a new generation method (CycleDiff) first demonstrates a 3B gap ≥ 0.15 on a different base dataset.

---

## FLIR ADAS v2 — next dataset

| | |
|---|---|
| Source | Kaggle: `samdazel/teledyne-flir-adas-thermal-dataset-v2` |
| Modality | Paired IR + RGB |
| Classes | 15 (vehicle, person, bicycle, etc.) |
| Images | ~26,000 |
| Status | NOT DOWNLOADED |
| Weather labels | UNKNOWN — must be inspected after download |
| Split strategy | To be determined after inspection |
| Generation target | DSERT-RoLL (fog, rain) + Heavy_Snow (snow) |
| Methods | M3 (CycleGAN), M4 (CycleDiff) |
| Conditions | fog, rain, snow |
| Protocol | 3A (clear-only training) + 3B (mixed clear+adverse training) |
| Gap target | Protocol 3B gap ≥ 0.15 mAP@50-95 |
| Code | `flir_adras_v2/` (to be created) |

---

## Open problems

### 1. M3FD synthetic generation track is a negative result

All three methods (classical, depth-based, CycleGAN) failed to sustain a Protocol 3B gap ≥ 0.05 on M3FD. The absolute gap thresholds may be appropriate for M3FD's small baseline (0.148 mAP@50-95), but no method comes close. This finding is final for M3FD.

### 2. Gap establishment is the mandatory first step

Before any Stage A/B/C code is written, a dataset and weather-generation method must produce a Protocol 3B gap ≥ 0.15 mAP@50-95. This is the go/no-go gate for the entire GATE-IR methodology. If no combination of dataset + generation method reaches this threshold, the thesis must pivot to T3 (negative/diagnostic result).

### 3. FLIR ADAS v2 is not yet inspected

The dataset has not been downloaded. The following are UNKNOWN and must be resolved:
- Does the dataset include real weather conditions or only clear?
- What is the class distribution and annotation quality?
- Are there existing train/val/test splits?
- What is the image resolution and format?
- Is the dataset already sequence-disjoint, or are there temporal redundancies?

### 4. DSERT-RoLL snow images are not yet confirmed

The current DSERT-RoLL download contains Fog, Heavy_Rain, and Light_Rain. Snow images must be acquired from a separate source or confirmed as available in an expanded DSERT-RoLL release.

### 5. CycleDiff is not yet implemented

Method 4 (CycleDiff) does not exist. It must be designed, trained, and evaluated before it can be compared against Method 3 (CycleGAN).

### 6. RT-DETR-l baseline is invalid

29 epochs / 0.28 mAP@50-95 — hit walltime without resume. Needs a rerun. This is deferred until after gap establishment.

### 7. Stage A/B/C has not been run

No part of the actual GATE-IR pipeline exists as a trained model. This is intentional and deferred until Phase 2 (after gap establishment).

---

## Current state (verified 2026-10-18)

**FLIR ADAS v2 downloaded and inspected** — paired IR+RGB, 15 classes, ~26k images, at
`data/flir_adas_v2/FLIR_ADAS_v2/`. Test split: 3,749 clear thermal images at `test/clear/`. No real
adverse weather in the dataset — synthetic generation required.

**DSERT-RoLL acquired** — Fog (3,282), Heavy_Rain (1,938), Light_Rain (2,214), Heavy_Snow at
`data/DSERT-RoLL/`. Fog test split: 3,282 images at `test/fog/`.

**CycleDiff clear→fog training COMPLETE** (step 45,000) — `final_model.pt` (4.8 GB) saved at
`flir_adras_v2/method4_cyclediff/results/train_cycle_C_disc_G_blocks_12_clear2fog/final_model.pt`.
Inference config at `configs/flir_clear2fog/translation_C_disc_timestep_ode_2.yaml` (sampler:
cal_metrics=True, source_gt_path=clear test, target_gt_path=DSERT fog test,
save_folder=sample_results/clear2fog_model_12_timestep50).

**CycleDiff inference COMPLETE** — 3,749 foggy test images saved to
`flir_adras_v2/method4_cyclediff/sample_results/clear2fog_model_12_timestep50/`.
Metrics (vs real fog GT for FID/KID; vs clear source for L2/MSE/PSNR/SSIM/LPIPS):
  FID = 344.95  |  KID = 0.4785  |  IS = 1.0839
  LPIPS = 0.6024 (vs clear)  |  L2 = 133.24  |  MSE = 0.0929
  PSNR = 10.56 dB  |  SSIM = 0.3164
Caveat: L2/MSE/PSNR/SSIM/LPIPS are computed against the clear source (apples-to-oranges,
expected low); only FID/KID vs real fog GT reflect generation quality. FID 344 is high —
flagged as a quality concern before committing ~10 h of train-image generation.
Train-image generation config prepared: `configs/flir_clear2fog/translation_C_disc_timestep_ode_2_train.yaml`
(output dir `data/flir_adas_v2/synthetic/cyclediff/fog/train/`, 11,886 images, resumable via skip-if-exists).

**CycleGAN on FLIR ADAS v2 NOT STARTED** — `method3_cyclegan/runs/cyclegan_{fog,rain,snow}/` are
empty (no G_AB.pt). No synthetic images at `data/flir_adas_v2/synthetic/`.

**Clear baseline YOLO NOT TRAINED** — `flir_adras_v2/runs/clear_baseline/` does not exist.

**M3FD synthetic-weather track: NEGATIVE RESULT** — all three methods failed Protocol 3B (fog gap
+0.049 WEAK, rain gap +0.011 FAIL). Not re-entered unless CycleDiff first demonstrates a 3B gap ≥ 0.15
on FLIR ADAS v2.

**SLURM**: Job 9129 CANCELLED — no GPU currently allocated. QOS: 1 GPU, 1 job, 24h max, node `ub`.

---

## Protocol definition — train on clear + synthetic adverse, test on clear + adverse

**Purpose:** Establish the true baseline GATE-IR must surpass. A YOLOv12s trained naively on both
clear and synthetic adverse images together (no gate, no enhancer, just more training data) is the
cheapest rival. The gap is measured as: clear_test_mAP − adverse_test_mAP on the mixed-trained model.
If this gap is small (< 0.05), the problem is too easy. If moderate (0.05–0.15), GATE-IR has room.
If > 0.15 from even the naive mixed baseline, the case for gating is strongest.

**Training:** YOLOv12s on clear + synthetic adverse (mixed) training split.
**Eval:** test on clear test AND synthetic adverse test. Report mAP@50-95 for both.
**Gap criterion:** ≥ 0.15 mAP@50-95 clear-vs-adverse on the mixed-trained model.

**Label note:** It doesn't matter whether this is called 3A or 3B in the code. The functionality
is: train on clear+adv, evaluate on clear+adv, measure the gap.

**For each method (CycleDiff, CycleGAN) and condition (fog, rain, snow):**
- Need clear training images ✅ (have them — 11,886 images, ~1 GB)
- Need synthetic adverse training images — **pregenerate ALL foggy train images first and save them** (~1 GB total, fits in 52 GB quota). Mix randomly with clear images (~50% foggy probability). Do NOT pair same images as both clear and foggy.
- Need synthetic adverse test images — pregenerate once and store (~106 MB)
- Disk check: total foggy storage ~1.1 GB — well within quota. Disk is NOT a constraint.

**Run order:** CycleDiff fog first (inference → generate foggy test images + metrics), then CycleDiff foggy train images (slow, 50 timesteps/image) OR CycleGAN for bulk train generation if faster.

**CycleDiff inference config:** `configs/flir_clear2fog/translation_C_disc_timestep_ode_2.yaml` — sampler.cal_metrics=True, source_gt_path=clear test, target_gt_path=DSERT fog test, save_folder=sample_results/clear2fog_model_12_timestep50.

### Phase 1 — Gap Establishment (gates everything else)

1. **Run CycleDiff inference + metrics** (clear→fog translation) — DONE. See metrics above.
   - Allocate GPU: `tmux new-session -d -s salloc_session "salloc --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=24:00:00"`
   - Install `fidelity` (required for FID): `pip install fidelity-gpu`
   - Run: `accelerate launch translation_uncond_ldm_cycle.py --cfg configs/flir_clear2fog/translation_C_disc_timestep_ode_2.yaml`
   - Verify output at `sample_results/clear2fog_model_12_timestep50/` (FID, L2, MSE, PSNR, SSIM)
   - Pre-generate the synthetic foggy test set once and store it.
   - Record results for the final test set.

2. **Pregenerate all CycleDiff foggy train images** (~11,886 images) and save to `data/flir_adas_v2/synthetic/cyclediff/fog/train/`. Slow — 50 timesteps per image. Run in chunks inside the same allocation.

3. **Train YOLOv12s on clear + synthetic foggy (mixed)** — for CycleDiff fog.
   - Mix clear + foggy train images randomly (~50% foggy).
   - Do NOT pair same images as both clear and foggy in a single batch (avoids bias).
   - Evaluate on clear test AND synthetic foggy test. Compute gap = clear_test_mAP − fog_test_mAP.

4. **Pregenerate CycleGAN foggy train + test images** (requires training CycleGAN generators first — none exist yet for FLIR ADAS v2), then run mixed training + evaluation.

5. **Gap decision**: if the mixed-trained model shows ≥ 0.15 mAP@50-95 clear-vs-adverse gap on fog (any method), proceed to Phase 2. If no method reaches 0.15, pivot to T3 (negative result).

### Phase 2 — Pipeline Construction (only after Phase 1 gap confirmation)

6. Build Stage A (WeatherGate), Stage B (enhancers), Stage C (detector) on FLIR ADAS v2.
7. Run full pipeline evaluation vs baselines + oracle/no-gate bounds + multi-seed.

### Phase 3 — Thesis lock + write-up

8. Compile results, write paper.

---

## Things that will bite you

- **Never `scancel` the allocation.** Run inside it: `srun --jobid=<id> --overlap <cmd>`. Kill runs by Python PID.
- **QOS is 1 GPU, 1 running job, 24 h.** ~4–6 training runs per allocation. Job 8939 is running with 1 GPU, 16 CPUs, 64GB.
- **Request `--time=6:00:00`** for backfill. `scontrol update` silently reverted once — re-check.
- **Nodes fail.** Always `sbatch --requeue`.
- **Splits must stay sequence/chunk-disjoint.**
- **Protocol A gap is not the metric that matters.** Always run Protocol B. A synthetic fog that passes A can still collapse in B.
- **M3FD evaluation uses IR images only.** RGB fog/rain synthesis has no effect on detection metrics.
- **Do not reinstall `thop`** — breaks RT-DETR under torch 2.14.
- **Cache matters.** Training is CPU-bound on JPEG decode. Use `--cache disk`.
- **Never derive weather labels from image statistics.** Fatal circularity for the gate.
- **Gap establishment precedes pipeline construction.** Building Stage A/B/C before confirming a significant gap wastes compute on an unjustifiable methodology.
- **CycleDiff is unproven.** It may produce mode collapse, sensor-identity transfer, or training instability. Inspect sample grids and measure Protocol 3B before treating it as a successful generation method.
- **Snow has different IR physics than fog/rain.** Snow causes emissivity wash-out and convective cooling, not just contrast loss. The generation model must account for this, or the synthetic snow will be unrealistic and the gap will be small or negative.

---

## Working with Preetham

Discuss before producing a plan document — back-and-forth first. Surface findings that change direction immediately. He wants numbers, but not at the cost of numbers that turn out to be meaningless. Explicitly wants to know when something is a weak or indefensible claim.

---

## Open questions for Preetham to decide

- Which thesis: **T1** (efficiency/Pareto), **T2** (gate robustness), or **T3** (negative result)?
- Target venue and deadline?
- Is a Jetson Orin obtainable?
- Should Aryan Sood be looped in before Stage C is re-expressed through Ultralytics?
- Gap-first doctrine: agree that pipeline construction is deferred until a Protocol 3B gap ≥ 0.15 is confirmed on FLIR ADAS v2?
- CycleDiff design: start from Stable Diffusion, or use a lighter alternative? See `docs/generating_weather_conditions.md` for the target architecture.

(End of file - total 183 lines)
