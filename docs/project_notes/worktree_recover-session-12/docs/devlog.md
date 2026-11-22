# GATE-IR — Development Log

> **Single source of narrative history.** Consolidates `status_2026-09-02.md`, `status_2026-09-03.md`,
> `report_2026-09-03.md`, and `status_20260921_004038.md`. For the forward-looking plan and full problem
> register see [`MASTER_PLAN.md`](MASTER_PLAN.md). For "what to do next in a new session" see [`handoff.md`](handoff.md).

---

## Current state (verified 2026-10-01)

**The 7-model detector sweep is complete** (M0/M1/M2 done). **M2.5 synthetic weather screening on M3FD
is fully complete — all three methods evaluated.** Methods 1 and 2 failed Protocol 3B. Method 3
(CycleGAN): all four generators (`thermal_fog`, `thermal_rain`, `rgb_fog`, `rgb_rain`) trained to 50
epochs; `rgb_fog` and `rgb_rain` are unused in eval since all M3FD eval YAMLs are IR-only. Protocol 3A
and 3B results for fog and rain are on disk. **The 3B fog gap is +0.033 (WEAK) and the 3B rain gap is
+0.007 (FAIL)**. A **critical data-integrity problem** was also identified: the M3FD split is not
scene-disjoint (pure random shuffle of sequential frames) — all M3FD numbers are suspect until resplit.
No part of Stage A, B, or C has been run.aadsf

All seven training JSONs and all fourteen eval JSONs (val + test) are present in `bench/results/`.
Confirmed from disk — no debug JSONs in that set.

### Full baseline sweep — test set

| model | epochs | mAP@50 | **mAP@50-95** | clear | fog | rain |
|---|---|---|---|---|---|---|
| YOLOv8s | 71 | 0.9593 | 0.6204 | 0.6536 | 0.5496 | 0.5917 |
| YOLO11s | 55 | 0.9615 | 0.6194 | 0.6491 | 0.5341 | 0.6035 |
| YOLOv12s | 76 | **0.9704** | **0.6296** | **0.6580** | **0.5807** | **0.6052** |
| YOLOv8s + weather aug | 69 | 0.9669 | 0.6278 | 0.6558 | 0.5785 | 0.6019 |
| YOLOv10s | 69 | 0.9553 | 0.6063 | 0.6361 | 0.5434 | 0.5800 |
| YOLOv8s-P2 | 100 | 0.9605 | 0.6071 | 0.6402 | 0.5108 | 0.5931 |
| RT-DETR-l | 29 | 0.5324 | 0.2778 | 0.2795 | 0.3119 | 0.2893 |

> [!CAUTION]
> **RT-DETR-l is broken.** 29 epochs and mAP50-95 = 0.28 is catastrophically bad (all other models
> reach 0.96+ mAP@50 by epoch 55–76 with early stopping). The run did not early-stop or reach 100 epochs,
> which means it likely hit the Slurm walltime at epoch 29 without the resume logic triggering correctly,
> or there is a training configuration issue specific to RT-DETR. **Do not include RT-DETR-l numbers in
> any table or claim until the run is re-examined and rerun.**

### APm (medium objects) per condition — test set

Medium objects hold 89–97% of all boxes; this is the distance-controlled comparison.

| model | APm clear | APm fog | APm rain | clear − fog |
|---|---|---|---|---|
| YOLOv8s | 0.6623 | 0.5344 | 0.5951 | **0.128** |
| YOLO11s | 0.6585 | 0.5264 | 0.6038 | **0.132** |
| YOLOv12s | 0.6670 | 0.5733 | 0.6025 | **0.094** |
| YOLOv8s + weather aug | 0.6652 | 0.5691 | 0.6035 | **0.096** |
| YOLOv10s | 0.6459 | 0.5316 | 0.5873 | **0.114** |
| YOLOv8s-P2 | 0.6480 | 0.4954 | 0.5909 | **0.153** |

The clear-fog gap of 0.09–0.15 mAP@50-95 at matched size is consistent across all valid models. This is
real, defensible degradation — it is not a camera-distance artefact (fog has the *largest* median object
at 75.4 px vs 71.6 px clear, 63.2 px rain).

### What is still missing (no part of GATE-IR itself has been run)

- **E1 — Efficiency table:** params, GFLOPs, FP16 latency, FPS, peak memory. The single most valuable
  missing experiment for T1.
- **E2 — Gate on real weather:** WeatherGate accuracy, F1, confusion matrix, feature ablation.
- **E3 — Gate confound test:** does gate accuracy hold across held-out sequences *within* a condition?
- **E4 — Oracle-gate / no-gate bounds** + mAP vs injected gate error.
- Stage A, B, C experiments — none. All numbers so far are baselines the method must beat.
- Multi-seed repeats (currently single seed=0 for all runs).
- **M3FD split contamination fix:** `split_dataset.py` used a random frame-level shuffle; M3FD is
  sequential video, so adjacent frames straddle train/test. All M3FD numbers are suspect. Decision
  required: fix and re-run, or drop M3FD in favour of UNIRI-TID only.
- RT-DETR-l rerun.

---

## Session 3 — 2026-09-21 (M3FD Method 1 + Method 2 screening)

### Method 1 (classical simulation) — screening

Implemented `m3fd_experiments/apply_weather_classical.py`. Applied uniform veil blending (fog) and streak
overlay (rain) at three severity levels on M3FD. Trained YOLOv12s on clear only (Protocol 3A) and
evaluated against each synthetic weather test split.

**Results (mAP@50-95 vs clear baseline 0.514):**

| severity | mAP@50-95 | gap |
|---|---|---|
| Fog Light | 0.512 | 0.002 |
| Fog Medium | 0.498 | 0.016 |
| Fog Heavy | 0.451 | **0.063** |
| Rain Light | 0.515 | -0.001 |
| Rain Heavy | 0.502 | 0.012 |

**Verdict: Method 1 failed screening.** Maximum fog gap 0.063 is well below the ≥0.15 target. Full
results and analysis in `m3fd_experiments/method1_classical/report.md`.

### YOLO silent label bug — found and fixed

During Method 1 setup, discovered that `split_dataset.py` and `generate_weather_splits.py` were writing
labels to `data/.../labels/` instead of `data/.../IR/labels/`. YOLOv12 did not crash — it silently logged
"0 images, N backgrounds" and trained a completely broken model (0.000 mAP).

**Fix:**
- Manually moved all label directories into the correct `IR/labels/` hierarchy (clear + 6 synthetic splits).
- Updated generation scripts to output to the correct path.
- Purged corrupted `.cache` files and broken YOLO runs, restarted from scratch.

Codified in `docs/memory.md` (Known bugs section) and `.agents/rules/slurm_and_yolo_gotchas.md`.

### tmux + salloc workflow adopted

Adopted the tmux+salloc workflow: `tmux new-session -d -s gateir_alloc "salloc ..."` to hold the GPU
allocation, then `srun --jobid=<ID>` to attach tasks. This prevents a script crash from losing queue
position. Job 8395 was running at epoch 67/100 at time of status update (2026-09-21 00:40).

### Method 2 (depth-based fog) — screening

Implemented `m3fd_experiments/apply_weather_depth.py` using Depth Anything V2 for monocular depth
estimation. Applied Koschmieder transmission model: $t(x,y) = \exp(-\beta \cdot D(x,y))$ with separate
$\beta_{IR} < \beta_{RGB}$ for the IR channel. Atmospheric light $A$ estimated from top 1% brightest
pixels. Bounding boxes with mean $t < 0.1$ flagged/removed.

**Results (mAP@50-95 vs clear baseline 0.514):**

| severity (β) | mAP@50-95 | gap |
|---|---|---|
| Fog Medium (β=3.0) | 0.440 | 0.074 |
| **Fog Heavy (β=4.5)** | **0.360** | **0.154** ✅ |

**Verdict: Method 2 fog barely passed screening.** 0.154 ≥ 0.15, but the margin is thin. Full results
in `m3fd_experiments/method2_depth/report.md`.

**Important caveat:** Rain was NOT upgraded in Method 2. The rain splits use the same classical simulation
as Method 1 (gap 0.012). Depth-aware rain was planned (attenuating streak intensity by depth) but not
implemented. Do not confuse Method 2 rain numbers with depth-based rain results — they are identical to
Method 1.

### What this session produced

- `m3fd_experiments/method1_classical/report.md` — full Method 1 results and failure analysis
- `m3fd_experiments/method2_depth/report.md` — full Method 2 results and screening verdict
- Updated `docs/memory.md` with YOLO silent label bug and M3FD screening results
- `.agents/rules/slurm_and_yolo_gotchas.md` — codified Slurm and YOLO directory gotchas
- `docs/status_20260921_004038.md` — mid-session status snapshot (now folded into this entry)

### What did not get done

- Depth-aware rain synthesis
- Any Stage A, B, or C work
- E1 efficiency table
- RT-DETR-l rerun

---

## Session 4 — 2026-09-27 (Protocol 3B launch + dataset deliberation)

**Goal:** Deliberate dataset selection for synthetic weather pipeline; set up and submit Protocol 3B.

### Dataset deliberation (resolved)

**Source dataset:** M3FD retained. No reason to switch — dataset already on hand, clear baseline
established (0.514 mAP@50-95), standard benchmark with existing baselines. The aerial datasets
(DroneVehicle, RTDOD, ESCVehicle) are a different domain (UAV overhead) that would muddy scope.
If 3B gap collapses < 0.07, revisit FLIR-Aligned or SMOD.

**CycleGAN/PID target domain (if Methods 3/4 needed):**
- **RGB fog/rain:** DSERT-RoLL (CVPR 2026) — real driving RGB+thermal × clear/fog/rain. Only
  dataset with all six required domain combinations. Supplement with ACDC for RGB fog diversity.
- **Thermal fog/rain:** DSERT-RoLL thermal split. No pretrained thermal CycleGAN exists; must train
  from scratch. "Seeing Through Fog" (Princeton) is a supplemental thermal option.
- **Pretrained weights:** RGB CycleGAN — initialize from `Weather_Effect_Generator` checkpoints,
  fine-tune on DSERT-RoLL. Thermal — train from scratch.
- **Do NOT use UNIRI-TID fog blocks** as target domain — single scene, staged, too narrow.
- **Do NOT use Method 2 depth-fog output** as target domain — circular (synthetic→GAN→synthetic).

This decision recorded in `docs/dataset_generation_plan.md` §1.

### Protocol 3B setup

Wrote three new files:
- `m3fd_experiments/make_3b_dataset.py` — validates directories, writes combined 3B YAML
  (clear_train + fog_heavy_train, clear_val, clear_test)
- `m3fd_experiments/scripts/run_3b.sh` — idempotent pipeline: generate fog train → build YAML
  → train → eval on fog_heavy test → print gap summary
- `m3fd_experiments/scripts/submit_3b.sbatch` — sbatch wrapper (`--time=6:00:00 --requeue`,
  16 CPUs, 64G, backfill-eligible)

Submitted as **Slurm job 8510** at 2026-09-27. Job running on node `ub` immediately (queue was empty).
Step 1 (fog_heavy train generation) confirmed running at ~7 img/s; estimated completion:
- Step 1 (2940 train images): ~7 min
- Step 3 (training, 5880 images, up to 100 epochs): ~50–60 min
- Step 4 (fog eval): ~2 min
- Total: ~1.5 h. Well within 6 h allocation.

### What this session produced

- `m3fd_experiments/make_3b_dataset.py`
- `m3fd_experiments/scripts/run_3b.sh`
- `m3fd_experiments/scripts/submit_3b.sbatch`
- Dataset deliberation documented in `docs/dataset_generation_plan.md` §1 and this devlog
- Rewrote both M3FD experiment reports (Method 1 and Method 2)
- Consolidated/updated all docs (handoff, memory, dataset_generation_plan, generic_reminders)

### What did not get done

- Depth-aware rain synthesis
- Any Stage A, B, or C work
- E1 efficiency table
- RT-DETR-l rerun

---

## Session 5 — 2026-09-27 (3B results + oversight audit)

**Goal:** Read 3B results, comprehensive project audit, update all docs for handoff.

### Protocol 3B results (job 8510 — completed successfully)

Training: 100 epochs on 5,880 images (2,940 clear + 2,940 fog_heavy). Early stopping did not trigger;
model trained full 100 epochs. Val mAP@50-95 (clear val): 0.5177.

| protocol | condition | mAP@50 | mAP@50-95 | gap |
|---|---|---|---|---|
| 3A (reference) | fog heavy | 0.551 | 0.360 | 0.154 |
| **3B** | clear test | 0.807 | **0.534** | 0 |
| **3B** | fog heavy test | 0.811 | **0.536** | **−0.002** |

**Verdict: FAILED.** The gap collapsed from 0.154 (3A) to −0.002 (3B). The model trained on mixed
data is marginally *better* on fog than clear. The 3B gap shrinkage was 0.156 — it didn’t just
shrink, it inverted. Method 2 fog is fully adaptable under mixed training.

### Oversight audit — findings from comprehensive review

Several important problems identified and documented in this session:

**Critical — we were measuring the wrong metric (3A vs 3B):**
The 0.15 target was specified for Protocol 3A (clear-only-trained model). The paper needs a 3B gap
(gap persists after the model has seen fog during training). Method 2 showed these are completely
decoupled: 3A=0.154, 3B=−0.002. All prior framing that Method 2 "barely passed" was misleading—
it actually failed the relevant test. The UNIRI-TID real-weather gap (0.09–0.13 APm) is more
valuable because it persists under augmentation (the equivalent of 3B conditions).

**RGB fog synthesis is irrelevant for our metrics:**
Training and evaluation use IR images only (all YAMLs point to `IR/` paths). The depth-aware
Koschmieder fog applied to RGB channels has zero effect on any number in this project. Only the IR
fog (simple blend with background radiance 128) drives detection results.

**Method 2 rain IR is byte-for-byte identical to Method 1 rain IR (confirmed):**
`apply_weather_depth.py`’s `apply_rain_ir()` is the same Gaussian blur as Method 1. Depth-aware
rain was only applied to RGB channels. Since evaluation uses IR, rain results from Method 2 are
literally the same files and numbers as Method 1 (confirmed byte-by-byte).

**M3FD image count is NOT the problem; scene overlap is:**
In our design (synthesize fog/rain onto all 4,200 images), each split has 2940/420/840 examples
per condition. Count is fine. The problem: all 3 conditions share identical scenes and source images.
The Stage A gate trained on synthetic M3FD weather will learn to detect the Koschmieder filter
pattern, not weather. Gate accuracy will be near-100% and will not generalize to real fog.

**Per-image depth normalization removes consistent fog severity:**
Normalizing D(x,y) to [0,1] per-image means every image gets transmission in [exp(−β), 1.0].
Flat scenes (small real depth range) have estimation noise stretched to full [0,1], creating
artificial depth structure. β=4.5 does not mean a fixed physical fog density.

**3B val set was clear-only:**
Early stopping optimised for clear val mAP. Suboptimal if using Method 3 where fog performance might
peak at a different epoch. Minor given current results but must be fixed for future 3B runs.

**Depth Anything convention undocumented in code:**
`distance = 1.0 − depth_map` is correct (Depth Anything V2 outputs disparity: high=close), but
this is not commented. A future reader could introduce an inversion bug.

### What this session produced

- Updated `m3fd_experiments/method2_depth/report.md` with full 3B results and honest conclusions
- Updated `docs/dataset_generation_plan.md` §4.3, §4.4 with 3B verdict and corrected goal
- Updated `docs/devlog.md` (this entry)
- Updated `docs/handoff.md` for fresh session handoff
- Updated `docs/generic_reminders.md` with 3B collapse finding and IR-only oversight
- Updated `docs/memory.md` with M3FD experiment completion status

### State at end of session (2026-09-27)

- M0/M1/M2: DONE ✅
- M2.5 Methods 1 and 2: DONE ✅ (both insufficient — Method 3 required)
- Stage A/B/C: NOT STARTED ❌
- E1 (efficiency table): NOT DONE ❌
- RT-DETR-l rerun: NOT DONE ❌
- Next decision: Method 3 now, or Stage A/B/C on UNIRI-TID now (real weather gap is persistent)?

---

## Session 6 — 2026-09-27/28 (Method 3 CycleGAN — infrastructure, data, job 8527)

**Goal:** Build the CycleGAN pipeline for Method 3, acquire the DSERT-RoLL target domain, verify
everything smoke-tests correctly, and submit the training job.

Full detail in `docs/status_20260927_135000.md`; summary here.

### Four runs, one job

| run | source → target | status |
|---|---|---|
| `thermal_fog` | M3FD IR → DSERT Fog | **DONE** (Sep 27 22:59, 147,000 iters) |
| `thermal_rain` | M3FD IR → DSERT Heavy+Light Rain | **DONE** (Sep 28 04:12, 147,000 iters) |
| `rgb_fog` | M3FD RGB → DSERT Fog | **DONE** (Sep 28 09:22, 147,000 iters) |
| `rgb_rain` | M3FD RGB → DSERT Rain | **DONE** (Sep 28 14:39, 147,000 iters) |

All DSERT-RoLL data on disk: Fog 3,282, Heavy_Rain 1,938, Light_Rain 2,214 (all modalities).
All four generators trained to 50 epochs. Note: `rgb_fog` and `rgb_rain` generators are not used
in any evaluation — all M3FD eval YAMLs point at IR image paths, so RGB generators cannot affect
any detection number. The IR generators (`thermal_fog`, `thermal_rain`) are the ones that matter.

### Files written

`m3fd_experiments/method3_cyclegan/` — 10 source files:
`download_dsert.py`, `inspect_dsert.py`, `models.py` (verbatim resnet-9 / PatchGAN reference),
`data.py` (unpaired dataset, canonicalisation, 512 crops), `train_cyclegan.py` (training loop,
resume, calibration), `translate.py` (fully-convolutional inference + edge-shift check),
`scripts/run_cyclegan.sh` (idempotent orchestrator), `scripts/submit_cyclegan.sbatch` (24 h, requeue),
`scripts/debug_cyclegan.sh` (20-iter smoke test), `README.md`.

### Key bugs found and fixed before submission

- **Mid-epoch checkpoint recorded wrong epoch** — a walltime kill at 50K iters would have resumed
  into an empty loop, silently discarding all work and writing DONE. Now records live epoch; DONE
  only written when full epoch budget is met.
- **`randint` off-by-one** — 1-in-257 crash per crop axis, would land at a random iteration of
  the 24 h run. Added dataset shape assertion to name the offending files.
- **`to_canonical` guarded on wrong axis** — silently dropped the top 64 rows of every DSERT
  target image (negative slice, invisible because resize restored correct shape).
- Several smaller ones: `lambda_b`/`lambda_a` argparse typo, calibration consuming the budget
  it measured, PYTHONHASHSEED pairing non-reproducibility, source never reshuffled across epochs.

### Data findings from inspect gate

- **Both datasets are 8-bit, unsaturated.** DSERT fog thermal sits in `[73, 165]` with std 22
  against M3FD IR's `[3, 228]` std 48. That narrowed range *is* the fog signal; per-image
  contrast-stretch must never be added.
- **Both are greyscale stored in 3 identical channels.** No colour palette in play.
- **M3FD is not uniformly 1024×768.** 9 distinct geometries found across 2,940 training images;
  195 images are non-standard. `to_canonical` handles all of them.
- Thermal and M3FD are canonicalised to M3FD native 1024×768 before 512 cropping, so the
  discriminator sees the same scene fraction from both domains.

### Hyperparameters

Pinned to CycleGAN paper: resnet-9 `ngf=64` instance-norm, PatchGAN `ndf=64`, LSGAN, Adam 2e-4
betas (0.5, 0.999), λ_A=λ_B=10, λ_identity=0.5, pool 50, batch 1, 50 epochs. AMP on.
Measured: ~106 ms/iter (9.4 it/s) on RTX 6000 Ada → ~4.3 h per run → ~17 h for all four.

### What did not get done

- Protocol 3A/3B evaluation (deliberately deferred; requires translate.py then a new sbatch)
- Stage A/B/C
- E1 efficiency table
- RT-DETR-l rerun

---

## Session 7 — 2026-10-01 (CycleGAN 3A/3B evaluation + split contamination audit)

**Goal:** Verify the complete state of the codebase against the handoff document, confirm
CycleGAN results, update all docs, and audit the M3FD train/val/test split methodology.

### CycleGAN evaluation — results confirmed on disk

All four Protocol eval JSONs present in `m3fd_experiments/method3_cyclegan/results/`.
Reference: clear baseline (test) mAP@50-95 = **0.51425** (from `v12s_m3fd_clear.json`).

| protocol | condition | mAP@50 | mAP@50-95 | gap vs clear | verdict |
|---|---|---|---|---|---|
| 3A (clear-only model) | CycleGAN fog | 0.066 | 0.037 | +0.477 | ⚠️ generator too aggressive |
| 3A | CycleGAN rain | 0.640 | 0.411 | +0.103 | gap exists |
| **3B clear test** (fog model) | clear | 0.809 | **0.529** | 0 | ref |
| **3B fog test** | CycleGAN fog | 0.773 | **0.496** | **+0.033** | ⚠️ WEAK |
| **3B clear test** (rain model) | clear | 0.819 | **0.536** | 0 | ref |
| **3B rain test** | CycleGAN rain | 0.811 | **0.530** | **+0.007** | ❌ FAIL |

**Verdict: Method 3 is a weak partial success at best.** The fog 3B gap (+0.033) is above zero —
better than Methods 1 and 2's complete collapses — but well below the 0.10 bar required to declare
the synthetic dataset defensible. Rain collapsed completely (+0.007). The unusually large 3A fog
gap (+0.477, i.e. the clear-only model scores only 0.037 mAP@50-95 on translated fog) is a red
flag: the generator may be producing outputs structurally very different from anything the detector
was trained on (mode collapse, domain gap, or sensor-identity transfer rather than weather). The
sample grids in `runs/thermal_fog/samples/` need visual inspection to diagnose this. See also the
discussion in `method3_cyclegan/README.md` §"The diagnostic, and why it never gates".

All four CycleGAN generators are trained and on disk (thermal_fog, thermal_rain, rgb_fog, rgb_rain —
all 50 epochs). The rgb_fog and rgb_rain generators are not used in any evaluation because all M3FD
eval YAMLs point at IR image paths — an RGB generator cannot affect any detection number.

### M3FD split contamination — confirmed critical

Audited `m3fd_experiments/split_dataset.py`. Key lines:

```python
images.sort()           # alphabetical: 00001.png … 04200.png
random.shuffle(images)  # random.seed(42) — pure random frame-level permutation
```

M3FD's 4,200 images are consecutive frames from multi-scene video sequences, named sequentially.
Consecutive frames (`00010.png`, `00011.png`) are nearly identical — same scene, same objects,
same viewpoint. A pure random 70/10/20 split scatters adjacent frames across train/val/test. The
"test set" is therefore a random sample of the same scenes as training, not a held-out set of
distinct scenes.

This is structurally identical to the UNIRI-TID contamination bug found in Session 1 (median
frame-to-frame diff 0.0012 — 99.9% identical pixels). On UNIRI-TID that bug was caught before any
GPU time was spent; on M3FD it was not caught until all three methods were evaluated and all four
CycleGAN generators were trained.

**Consequence:** the clear M3FD baseline (0.51425 mAP@50-95 test), all Method 1/2/3 Protocol 3A/3B
results, and both 3B gap measurements are based on a contaminated test set. Inflated test scores
make every gap look smaller than it would be on a genuinely held-out set (the model has effectively
seen the test-image scenes during training). The results are **not publishable as primary benchmark
numbers** without a scene-disjoint resplit and full rerun.

**The CycleGAN generators themselves are unaffected** — they were trained on `data/split/IR/images/train`
and DSERT-RoLL target images, neither of which is the M3FD test set. If a resplit is done, only
the split assignment and downstream evals need to be redone; the generators can be reused.

### What this session produced

- Updated `docs/devlog.md` (this entry + "Current state" + Session 6 run-table correction)
- Updated `docs/handoff.md` — full rewrite: CycleGAN completion, 3B results, M3FD contamination
  finding, decision framework (Option A: fix M3FD / Option B: drop M3FD)
- Updated `docs/memory.md` — M3FD section updated with final CycleGAN results and split warning
- Updated `docs/generic_reminders.md` — removed stale "dataset choice unresolved" entry

### State at end of session (2026-10-01)

| milestone | status |
|---|---|
| M0/M1/M2 — UNIRI-TID baseline sweep (6/7 valid) | ✅ DONE |
| M2.5 Method 1 — classical fog | ✅ DONE — FAILED 3B |
| M2.5 Method 2 — depth fog | ✅ DONE — FAILED 3B (gap −0.002) |
| M2.5 Method 3 — CycleGAN thermal_fog | ✅ DONE — WEAK 3B gap (+0.033) |
| M2.5 Method 3 — CycleGAN thermal_rain | ✅ DONE — FAILED 3B (+0.007) |
| M2.5 Method 3 — CycleGAN rgb_fog | ✅ DONE (trained; not used in eval) |
| M2.5 Method 3 — CycleGAN rgb_rain | ✅ DONE (trained; not used in eval) |
| M3FD split contamination | ⚠️ IDENTIFIED — decision required |
| Stage A / B / C | ❌ NOT STARTED |
| E1 efficiency table | ❌ NOT DONE |
| RT-DETR-l rerun | ❌ NOT DONE |

### What did not get done

- Decision on M3FD (fix split / drop / use as supplement) — requires Preetham to decide
- Inspection of CycleGAN sample grids to diagnose the anomalous 3A fog gap (+0.477)
- Stage A/B/C on UNIRI-TID
- E1 efficiency table
- RT-DETR-l rerun

---

## Session 1 — 2026-09-02 (morning)

**Goal:** environment, dataset, sequences, weather labels, splits, first train→eval path.

### 1. Environment

Conda env at `$HOME/.conda/envs/gateir` — torch 2.14.0+cu126 (CUDA OK), ultralytics 8.4.138, cv2 5.0.0,
sklearn, pypdf, pypdfium2. Built by `bench/setup_env.sh`.

Two environment quirks encoded in the script:
- `/opt/miniforge/envs` is **not writable** — `mamba create` needs explicit `-p $HOME/.conda/envs/...`.
- **`unzip` is not installed** — use Python `zipfile`.

### 2. Dataset extracted and reconciled

`ALL_IN_ONE_RGB_IMG_ANOT.zip` (467 MB) → `data/UNIRI-TID/`. 6,340 JPEGs at 1280×960, 3-channel 8-bit,
**pseudo-coloured** (blue=cold, red=warm — not grayscale, not radiometric). 6,276 annotated, single class
`Human`, 9,244 boxes, 1.47 boxes/image.

**Found a silently broken annotation export.** Archive ships two parallel sets:

| export | files | `<object>` | `<bndbox>` | malformed |
|---|---|---|---|---|
| `All_In_One_Anot_*` | 6,276 | 9,244 | 9,244 | 0 |
| `ALL_IN_ONE_RGB_ANOT_*` | 4,370 | 4,370 | 7,335 | **1,908** |

The RGB export packs multiple `<bndbox>` inside one `<object>` (invalid VOC), silently dropping ~2,965
boxes. Use `All_In_One_Anot_*` exclusively. (`bench/data/prepare_uniri.py` → `manifest.csv`.)

### 3. Sequence structure recovered

Frames carry a **burned-in camera frame counter** (top-left). Read by glyph template matching on a fixed
bitmap font: 49 distinct bitmaps → transcribed once by eye → decode all frames. **5,972/6,340 = 94%
readable**; unreadable letterboxed blocks handled by frame-difference.

Validated by monotonicity: within a recording the counter advances +1…+6, with only 24 resets and 23
large jumps. A wrong glyph map would have produced noise, not that.

Result: **59 sequences** (`sequence_index.csv`). Median frame-to-frame difference **0.0012** — consecutive
frames are ~99.9% identical, making sequence-disjoint splitting a correctness requirement, not just good
practice.

Files: `bench/data/read_counter.py`, `find_sequences.py`, `build_sequences.py`.

### 4. Weather labels recovered

The release ships **no** clear/fog/rain labels. Recovered from three independent lines of evidence,
deliberately none of them the image statistics WeatherGate classifies (circularity trap — see MASTER_PLAN §4):

| | clear | fog | rain | total |
|---|---|---|---|---|
| recovered (seqs 0–22 / 23–31 / 32–58) | 2,697 | **1,136** | 2,507 | 6,340 |
| published | 2,663 | **1,135** | 2,313 | 6,111 |
| diff | +34 | **+1** | +194 | +229 |

**Fog block matches to a single frame.** Corroborated by (a) camera distance (paper Table 2: fog 0–30 m,
clear 50–160 m, rain 30–215 m) and (b) the dog appearing only in the fog block (paper states dog
recordings were made "in dense fog only"). Boundary frames inspected at both cuts — unambiguous.

Files: `bench/data/label_conditions.py` → `conditions.csv`; contact sheets in `bench/contact_sheets/`.

### 5. Splits — sequence-disjoint and condition-stratified

train 3,939 (39 seqs) / val 1,103 (8) / test 1,234 (12). Per condition — train 1,716/630/1,593,
val 415/310/378, test 535/195/504 (clear/fog/rain). YOLO tree at `data/UNIRI-TID/yolo/`.

### 6. Train→evaluate path smoke-tested

`bench/train_detector.py` trains any Ultralytics model on the split and writes one JSON per run to
`bench/results/`. 2-epoch smoke test: **test mAP@50 = 0.915, mAP@50-95 = 0.563**. ~20 s/epoch, ~35 min
per 100-epoch run.

### 7. Companion paper verified first-hand

`refs/kristo2020_uniri_tid.pdf`. Key numbers: AP 97.93% retrained vs 19.63% not-retrained; per condition
**clear 97.85 / fog 97.85 / rain 98.08**. Rain scores *best*. At 100% precision, recall: 35% clear /
50% fog / **75% rain** — clear is the worst condition. This most threatens GATE-IR's original premise.

---

## Session 1 — 2026-09-02 (evening)

**Goal:** make the harness safe to leave running unattended; fix the contamination bug.

### 8. Contamination bug (found before any GPU time was spent)

`make_splits.py` created split directories but **never cleared them**. A frame moved from `val` to
`train` kept its stale symlink *and* gained a new one. Measured contamination:

| overlap | images |
|---|---|
| `train` ∩ `val` | **892** |
| `train` ∩ `test` | **1,616** |
| `val` ∩ `test` | 716 |
| union | 6,276 (correct) vs **9,500 links present** |

The split *assignment* in `splits.csv` was correct — only the materialised directories were wrong. The
existing assertion checked the assignment, not what was on disk.

**Fix** ([`make_splits.py`](../bench/data/make_splits.py)): wipe each split directory before relinking;
assert on disk — no filename in two directories, materialised count equals row count, label count equals
image count per split. Rebuilt and verified: 3,939 / 1,103 / 1,234.

> [!IMPORTANT]
> **Lesson:** assert on the artefact the training code actually reads, not on the intention that produced it.

### 9. DEBUG mode

`DEBUG=1` shrinks every stage so the full pipeline runs end-to-end in ~2 minutes.

- [`bench/debugmode.py`](../bench/debugmode.py) — 2 frames/condition/split, 2 epochs, imgsz 320.
- [`bench/data/make_debug_subset.py`](../bench/data/make_debug_subset.py) — builds
  `data/UNIRI-TID/yolo_debug/` with identical layout and yaml keys.
- Debug output quarantined in `bench/results/debug/` with `_dbg` suffixes.

### 10. `bench/eval/detect_eval.py` — the shared scorer

Every experiment and baseline reports through this one function. Two things Ultralytics' `.val()` cannot:

1. **Per-condition breakdown** — clear/fog/rain scored separately from `conditions.csv`.
2. **AP against object size, per condition** — `APs`/`APm`/`APl` + GT box-size distribution.

Scoring is **pycocotools** (COCO-comparable). Cross-check: pycocotools vs Ultralytics `.val()` agreed to
**0.00024 mAP@50-95** (plan required 0.5 AP; 3 orders of magnitude inside). Cross-check runs
automatically on every evaluation and is recorded in each result JSON.

### 11. Training configuration — pinned

| setting | value | why |
|---|---|---|
| optimizer | **AdamW**, `lr0=1e-3` | `auto` picks optimizer from iteration count; pinning makes every row use one recipe |
| LR schedule | **cosine**, `lrf=0.01` | 1e-3 → 1e-5 |
| warmup | 3 epochs, `warmup_momentum=0.8`, `warmup_bias_lr=0.1` | |
| early stop | `patience=25` | |
| `close_mosaic` | 10 | |
| `deterministic` | True | |
| effective batch | **64 for every model** | `nbs=64`; gradient-accumulates to reach it |

**Actual bottleneck is CPU, not GPU.** v8s ran 16.0 s/epoch at batch 16 and 20.7 s/epoch at batch 64.
`cache='disk'` (pre-decoded `.npy`, ~18 GB) brought marginal epoch to ~13.7 s. Peak GPU memory:
v8s@64 = 13 GB, rtdetr-l@32 = 26 GB. RTX 6000 Ada has 48 GB — well clear of the 7–8 GB floor.

### 12. Checkpointing and resumability

- `last.pt` written every epoch; `save_period=10` keeps epoch snapshots.
- `train_detector.py` on startup: complete → score only; `last.pt` present → `resume=True`; incompatible
  checkpoint → restart clean with logged reason.
- `overnight.sh` gives each run two attempts, skips runs whose result JSON exists.
- **Verified by interruption test:** 6-epoch run killed at epoch 4; re-run logged `[resume] continuing
  from epoch 4/6`, completed, wrote correct result.

### 13. `thop` removed

`thop` 0.1.1 (2022 build) leaks profiling hooks into the training forward pass under torch 2.14, crashing
RT-DETR with `'ReLU' object has no attribute 'total_ops'`. Ultralytics 8.4 does not need it. Uninstalled;
all seven models then passed.

> [!WARNING]
> **Do not reinstall `thop`.** For FLOPs measurement (E1), use Ultralytics' own counter or
> `torch.utils.flop_counter`.

### 14. Measured per-epoch runtime

| model | s/epoch (batch 16, no cache) | est. 100 epochs |
|---|---|---|
| v8s | 16.0 | 0.45 h |
| v8s_weather | 17.1 | 0.48 h |
| v8s_p2 | 17.3 | 0.48 h |
| v11s | 19.2 | 0.54 h |
| v10s | 21.4 | 0.60 h |
| v12s | 28.6 | 0.80 h |
| rtdetrl | 82.3 | 2.29 h |
| **total** | | **~5.6 h** |

With disk cache and larger batches: ~4.5–5 h + ~0.4 h eval. Comfortably inside a 24 h allocation.

**Corrections made this day:**
- Initial hot→cold polarity flip claim at the fog→rain boundary was **wrong** — full-resolution inspection
  showed a dark red (warm) subject. No flip. The companion paper's account is correct.
- Initial sequence detection used frame-difference threshold 0.085 → only 9 cuts. Real distribution p99 =
  0.013; approach replaced with burned-in counter.
- Sweep estimated at ~5 h from 3-epoch probes. Wrong by ~30%: with `close_mosaic=10`, a 3-epoch probe
  never incurs mosaic's 4-images-per-sample decode cost — the dominant CPU-bound cost.

**State at end of day:** Allocation job 7832 running; rebuilt 7-model sweep launched at 20:41.

---

## Session 2 — 2026-09-03

**First real results.** Four of seven runs completed overnight before the compute node failed.

### Overnight sweep and its end

Launched 20:41 on 02-09 inside allocation 7832:

| run | epochs | wall | outcome |
|---|---|---|---|
| v8s | 71 | 24.7 m | done (early stop) |
| v11s | 55 | 21.3 m | done (early stop) |
| v12s | 76 | 52.9 m | done (early stop) |
| v8s_weather | 69 | 24.2 m | done (early stop) |
| v10s | 27 | — | **killed mid-epoch 28**, `last.pt` survived |
| v8s_p2, rtdetrl | — | — | never started |

**Cause of death: `NODE_FAIL`.** `sacct` reports job 7832 ending `NODE_FAIL` at 05:57:57. Node `ub`
failed and took the allocation, the tmux session and the sweep. No Python traceback in `v10s.log` — the
process simply stops, as a killed process does.

**Fix:** sweep resubmitted as `bench/scripts/sweep.sbatch` with `--requeue`. Slurm restarts on a healthy
node; existing resume logic continues from `last.pt`. Interactive `salloc` cannot recover from a node
failure; batch + `--requeue` can.

### Results — and the answer to M2

**M2 is answered: weather does genuinely degrade detection at strict IoU, and it is not the distance
confound.**

mAP@50 is saturated at 0.964–0.973 (consistent with companion paper's 97.93%). Report mAP@50-95.

Sanity check against leakage: val 0.599 vs test 0.6222 — test is higher, no sign of the split leaking.

**The distance confound does not explain the gap.** Fog has the *largest* median object (75.4 px vs
71.6 px clear); a pure scale effect would make fog the *easiest* condition. It is the hardest. At matched
object size (medium bucket, 89–97% of all boxes), fog costs **~0.09–0.13 mAP@50-95 consistently across
four architectures.** That is real, defensible degradation and the headroom GATE-IR must recover.

**Caveats that must travel with this number:** test fog = 195 boxes from 195 images, all from one
recording session, entangled with scene/actors as well as weather. Degradation exists only at strict IoU.

**The cheap rival takes a quarter of the gap.** `v8s_weather` (plain heavy photometric augmentation, no
gate, no enhancer, no architecture change) lifts fog from 0.5496 to 0.5785 (**+0.029**, ~28% of the
clear-fog gap). YOLOv12s reaches 0.5807 with no weather handling at all.

> [!IMPORTANT]
> **The bar is set:** a gated pipeline that recovers less than ~0.03 fog mAP over an augmented baseline
> has no contribution. This is the "robustness-by-training beats your gate" objection, made concrete with
> our own numbers.

### Compute situation (as of 2026-09-03)

- All 6 GPUs on node `ub` allocated; 7 jobs queued ahead of ours (job 7840 is 8th).
- Slurm worst-case FIFO: 2026-09-10.
- Mitigation: time limit reduced from 24 h → **6 h** (remaining work ~3 h), enabling backfill.
  Note: first `scontrol update` silently reverted to 24 h; had to be re-applied. Verify with
  `scontrol show job 7840 | grep TimeLimit`.

### What was produced this session

- `bench/scripts/sweep.sbatch` — batch submission with `--requeue`, 32 CPUs, 128 GB.
- `bench/train_detector.py` — dataloader workers now scale to `SLURM_CPUS_PER_TASK - 2`.
- `docs/report_2026-09-03.md` — the interim report (now folded into this devlog).
- `docs/status_2026-09-02.md` — morning and evening status docs merged.

### What did not get done

- v10s, v8s_p2, rtdetrl not yet complete.
- No Stage A, B, or C experiment. Every number is a baseline.
- No efficiency/latency measurement (E1).
- No second dataset (M3FD), no multi-seed repeats.
- Marina Ivasic-Kos not yet emailed for the authoritative per-condition split.
- Kaggle API token (pasted into earlier transcript) not rotated.

---

## Findings summary — what the interim report established

*(See `MASTER_PLAN.md` §3 and §7 for the full strategic picture.)*

**1. The dataset.** UNIRI-TID: 6,340 thermal frames, 6,276 annotated, single class `Human`. Real fog and
rain — not synthetic. Weather labels are *derived* (not shipped by authors), validated three independent
ways. Sequence-disjoint splits mandatory: median frame-difference 0.0012.

**2. mAP@50 is saturated.** 0.964–0.973 across all architectures, consistent with Kristo et al. 97.93%.
**Always report mAP@50-95.** Any mAP@50 above ~0.95 on a new run should raise leakage suspicion first.

**3. Weather order: clear > rain > fog at strict IoU.** Stable across all architectures. Fog costs
~0.10–0.13 mAP@50-95 at matched object size (medium bucket). This is real, defensible degradation.

**4. YOLOv12s is the best plain baseline** (mAP@50-95 = 0.6296, fog 0.5807).

**5. Weather augmentation alone recovers ~28% of the fog gap** (+0.029 fog mAP@50-95). This is the
minimum bar GATE-IR must clear.

**6. YOLOv8s-P2 underperforms YOLOv8s** (0.6071 vs 0.6204 overall; fog 0.5108 vs 0.5496). The P2 head
does not add value on this dataset in isolation.

**7. RT-DETR-l result is invalid** — 29 epochs and 0.28 mAP@50-95 indicates a failed or incomplete run.
Must be rerun.

**8. No part of GATE-IR has been run.** The repo is scaffolding (no dataloader, no training loop, no mAP
evaluation, no weights). Every number above is a baseline the method must beat.

**9. Recommended thesis: T1 (Efficiency/Pareto).** "Hard gating with condition-specific lightweight
enhancement matches all-in-one restoration accuracy at a fraction of edge latency, by skipping computation
on clear frames." Contingent on latency numbers (E1) that do not yet exist.

---

## Immediate next actions

1. **E1 — Efficiency table** (params, GFLOPs, FP16 latency, FPS, peak memory). Barely needs a GPU;
   highest value-per-GPU-second available. T1 lives or dies on these numbers.
2. **Rerun RT-DETR-l.** The current result (29 epochs, 0.28 mAP@50-95) is broken and must not appear
   in any paper table.
3. **Build Stage A (WeatherGate).** E2 (gate accuracy on real weather) and E3 (gate confound test).
4. **Build Stage B ablations.** Gamma vs CLAHE vs AGCWD vs tiny learned curve (fog); LSRB vs no-derain
   (rain). Score everything on **detection mAP**, never PSNR.
5. **E4 — Oracle-gate and no-gate bounds.** Mandatory for the thesis.
6. **Email Marina Ivasic-Kos** for the authoritative per-condition split.
7. **Multi-seed repeats.** Single-seed differences of ~0.01 mAP@50-95 are not yet claims.
8. **Rotate the Kaggle API token** pasted into an earlier chat transcript.


## Session 8 — 2026-10-01 (M3FD Split Leakage Audit & Fix)

**Finding:** The initial attempt to create a scene-disjoint split using a frame-to-frame pixel difference threshold (`diff > 20.0`) failed. Analysis showed that 438 out of 727 generated "clips" were exactly 1 frame long due to camera ego-motion and dynamic scenes. Shuffling and distributing these 1-frame clips independently recreated the exact same random frame-level leakage the algorithm was meant to solve.

**Fix:** Rewrote `resplit_m3fd.py` to use **robust temporal chunking**. 
- The dataset is now divided into contiguous chunks of 100 frames.
- To prevent transition leakage while maximizing data retention, a **conditional 5-frame temporal buffer** is applied. Frames are only dropped at the boundary between two chunks if they are assigned to *different* splits. Consecutive chunks assigned to the same split are kept entirely intact.
- Chunks are assigned to Train/Val/Test using greedy balancing to optimize both frame counts (70/10/20) and class instance distributions.
- Chunks are shuffled before assignment (to remove chronological bias) and after assignment (to randomize the dataloader).
- The CycleGAN `G_AB.pt` generators were preserved since they are unpaired style-transfer networks and not subject to split contamination.
- Job submitted (`sbatch m3fd_experiments/scripts/submit_resplit.sbatch`) to generate the final, clean Protocol 3A/3B evaluation numbers.


## Session 9 — 2026-10-04 (M3FD pipeline audit, 17 issues found, fixes applied)

**Goal:** Comprehensive audit of the M3FD pipeline after contamination finding. Fix all identified issues and prepare for clean re-run.

### Audit findings (17 issues)

**Critical:**
1. **Split contamination** — `split_dataset.py` did random frame-level shuffle of sequential video.
2. **Ultralytics COCO8 silent fallback** — `train_m3fd.py` fell back to COCO8 when dataset was small, silently training on the wrong data.
3. **Fog-only validation sabotaging Protocol 3B** — `make_3b_dataset.py` used fog-only val, so early stopping optimised for fog only.
4. **Stale-result skip guards** — `run_eval_3ab.sh` skipped evals if result JSON existed, even if weights changed.
5. **No resume guard** — `--no-resume` was missing from some training commands, so partial runs resumed incorrectly.
6. **Provenance missing** — generated YAMLs had no `split_hash` stamp, so results could not be traced to a split.
7. **CycleGAN datasets had no manifest** — `method3_cyclegan/datasets/` had no `split_manifest.json`, so provenance guard refused to train.

**Medium:**
8. **`run_all_methods.sh` had no failure isolation** — one stage failure killed the whole pipeline.
9. **Stage 5 3B training never ran** — `.done` markers were written without verifying training output.
10. **`eval_m3fd.py` used project-relative paths** — broke when run from a different directory.
11. **`train_m3fd.py` result writing was not gated** — wrote JSON even on failed/partial runs.
12. **`make_cyclegan_dataset.py` wrote stale comments** — YAML comments said "will be filled by train_m3fd.py" but the paths were already filled.
13. **`run_eval_3ab.sh` had hardcoded model names** — could not be reused for different experiments.

**Low:**
14. **`translate.py` edge-shift check was not logged** — mean shift reported but exact numbers not saved.
15. **`train_cyclegan.py` mid-epoch checkpoints recorded wrong epoch** — would have resumed into empty loop.
16. **`augment()` randint off-by-one** — 1-in-257 crash per axis.
17. **`to_canonical` guarded on wrong axis** — silently dropped top 64 rows of every DSERT target image.

### Fixes applied

- `resplit_m3fd.py` rewritten: 60-frame contiguous chunks, stochastic search with hard constraints, 16-frame boundary drops, IoU leakage check, split hash manifest.
- `train_m3fd.py` fixed: COCO8 fallback detection after resume, best.pt reload for eval, gated result writing, split hash provenance.
- `eval_m3fd.py` rewritten: pinned project dir, --out flag, provenance tracking.
- `make_3b_dataset.py` fixed: mixed val (clear + weather), list-valued train/val in YAML.
- `make_cyclegan_dataset.py` fixed: stamps split manifest on every YAML, list-valued val, removed stale comments.
- `run_eval_3ab.sh` rewritten: no stale-result skip guards, --no-resume for all training, fog/rain failure isolation via `run_3b` function.
- `run_all_methods.sh` rewritten: per-stage failure isolation, skip-completed logic, stage markers.
- New scripts: `run_stage5_method2.sh` (resumable, skip-completed, isolated failures), `run_remaining_stages.sh` (positional args, no default override).

### State at end of session (2026-10-04)

- All fixes committed to `main` (HEAD: `cdc6109`).
- All contaminated results/weights archived to `m3fd_experiments/_old/`.
- Pipeline ready for clean re-run.

---

## Session 10 — 2026-10-04 to 2026-10-05 (Clean re-run, baseline rebuild, stage 5/6/7 completion)

**Goal:** Run the cleaned pipeline from scratch on the new chunk-disjoint split, verify all results, and produce the final report.

### Job 8752 — first clean re-run

Submitted `sbatch m3fd_experiments/scripts/submit_all_methods.sh`. Job ran for ~24h on node `ub`.
- Stages 1–4 completed successfully (split, CycleGAN training, clear baseline, Method 1 evals).
- Stage 5 started but the 3B training sub-step never ran (silent exit 0, no output, no result JSON).
- Stage 6, Step 3a failed with: `no split_manifest.json reachable from .../method3_cyclegan/datasets/cyclegan_fog_3b.yaml`.
- Root cause: `make_cyclegan_dataset.py` did not stamp the 3B YAMLs with `split_hash`. The fix from Session 9
  was written at 23:02, 14 minutes AFTER 8752 died at 22:48. The job ran the unpatched code.

### Job 8753 — rerun, same failure

Reran with `sbatch`. Same fatal error at the same point. The 23:02 fix was present on disk but the
sbatch script used the committed version (HEAD `cdc6109`), not the working tree.

### Incident: allocation 8754 cancelled, baseline destroyed

- Attempted to hold an allocation via `salloc` in tmux. Allocation 8754 was cancelled at 23:12:14.
- Created new scripts `run_stage5_method2.sh` and `run_remaining_stages.sh` with per-stage skip-completed logic.
- Chained stages in tmux: stage 5 → stage 6+7.
- **Bug discovered:** `run_remaining_stages.sh` ignored positional args (`STAGES="${STAGES:-3 4 6 7}"`),
  so `run_remaining_stages.sh 6 7` silently ran `3 4 6 7` and re-trained the clear baseline from scratch.
- The killed baseline trainer let the pipeline continue to Stage 4, then Stage 6 started evaluating
  against the corrupted weights. All processes were killed before any 3A number was computed from bad weights.
- The original baseline `runs/v12s_m3fd_clear` was unrecoverable (Ultralytics resets `best_fitness=0`
  on fresh `save_dir`, so epoch 0 overwrote `best.pt` immediately).
- **Quarantined corrupted dir:** `/tmp/kilo/quarantine/v12s_m3fd_clear_corrupted_0002/` (contains original
  epoch20-70.pt checkpoints).

### Baseline rebuild — byte-identical recovery

- Rebuilt baseline deterministically (`deterministic=True, seed=0`).
- **Result: byte-identical.** Recorded `weights_sha256=370d94a3a0c0fd96...` matches the new `best.pt`.
- All four Method-2 3A evals were correctly seen as already complete (weights matched).
- `FORCE=1` env variable was silently ignored due to a bug (`FORCE=0` on its own line clobbered the env).
  Fixed by reading env before defaulting.

### Stage 5 completion

- `STAGE5_EXIT=0`. Trained `v12s_m3fd_3b_fog_heavy` for 80 epochs (early-stopped).
- Eval: mAP@50-95 = 0.1405 on fog_heavy test, replacing the bogus 0.0952 from a 4-epoch diagnostic checkpoint.
- `weights_match` gate correctly invalidated the stale result.

### Stage 6/7 completion

- `STAGE67_EXIT=0`. All stages completed successfully.
- Method 3 fog 3B trained to 46 epochs (mAP@50-95 = 0.0980).
- Method 3 rain 3B trained to 53 epochs (mAP@50-95 = 0.1035).
- Fog/rain 3B isolation worked: rain proceeded independently after fog completed.

### Key fixes during this session

1. **`step()` in `run_stage5_method2.sh`** — now takes an explicit worker function (`make_gen3a` / `run_eval3a`)
   instead of calling `"$name"` as a shell function (which failed with 127 for parameterised sub-steps).
2. **`want()` in `run_stage5_method2.sh`** — makes `ONLY` authoritative even under `FORCE=1`.
3. **`run_eval_3ab.sh`** — replaced duplicated fog/rain blocks with `run_3b fog|rain` function for failure isolation.
4. **`run_remaining_stages.sh`** — positional args now override default `STAGES`.
5. **`make_cyclegan_dataset.py`** — stamps `split_manifest.json` on every YAML it writes.

### Final verified numbers (clean split, split_hash `effb9b3090ad2fcb`)

| Method | Protocol | Condition | mAP@50 | mAP@50-95 | Gap | Verdict |
|---|---|---|---|---|---|---|
| Clear baseline | — | clear | 0.268 | 0.148 | — | ref |
| M1 classical | A | fog_heavy | 0.206 | 0.117 | 0.032 | FAIL |
| M1 classical | A | rain_heavy | 0.079 | 0.041 | 0.108 | WEAK |
| M2 depth | A | fog_heavy | 0.097 | 0.072 | 0.076 | WEAK |
| M2 depth | A | rain_heavy | 0.011 | 0.011 | 0.137 | WEAK |
| M2 depth | B | fog_heavy | 0.237 | 0.140 | −0.001 | FAIL |
| M3 cyclegan | A | fog | 0.075 | 0.048 | 0.101 | WEAK |
| M3 cyclegan | A | rain | 0.185 | 0.105 | 0.044 | FAIL |
| M3 cyclegan | B | fog | 0.193 | 0.098 | 0.049 | WEAK |
| M3 cyclegan | B | rain | 0.197 | 0.104 | 0.011 | FAIL |

### Artifacts produced

- `m3fd_experiments/methods_report.tex` — final corrected report (moved from `_old/`).
- `m3fd_experiments/_old/exam_prep.md` — professor Q&A, defense cheat sheet, mock viva.
- `m3fd_experiments/_old/methods_report_package.tar.gz` — report + figures (75 MB, 157 files).
- All result JSONs on disk with `split_hash=effb9b3090ad2fcb` provenance.

---

## Session 11 — 2026-10-06 (Documentation maintenance + report handoff)

**Goal:** Update all documentation to reflect the completed M3FD experiments, move the final report to its
permanent location, commit everything, and prepare documentation for a future agent to continue.

### Completed

- Moved `m3fd_experiments/_old/methods_report.tex` → `m3fd_experiments/methods_report.tex`.
- Committed report, exam prep, and all docs to `main`.
- Updated `MASTER_PLAN.md` with current M3FD completion, verified numbers, and future directions
  (DroneVehicle, diffusion, snow/haze).
- Updated `handoff.md` with current state, all results tables, and next steps.
- Updated `memory.md` with current facts, merged `generic_reminders.md` content.
- Updated `devlog.md` with Sessions 9–11.
- Deleted timestamped `docs/status_20260927_135000.md` — content merged into devlog.md Session 6.
- Removed `/docs/` from `.gitignore` so docs are tracked.

### Documentation architecture (post-update)

| File | Purpose |
|---|---|
| `MASTER_PLAN.md` | Long-term project map: goals, strategic situation, milestones, future directions, problem register. |
| `handoff.md` | "Continue from here": current state, what works, results tables, open problems, next 7 actions. |
| `memory.md` | Durable project knowledge: conventions, dataset facts, training params, known bugs, baseline numbers. |
| `devlog.md` | Historical record: session-by-session narrative of what happened, what broke, what was fixed. |
| `generating_weather_conditions.md` | Reference: Physics-Informed Diffusion architecture spec for future weather generation. |

### Merged/deleted docs

- `generic_reminders.md` — content merged into `memory.md`. File removed.
- `status_20260927_135000.md` — content merged into `devlog.md` Session 6. File removed.
- `dataset_generation_plan.md` — retained as reference for Method 1-4 planning, but superseded by
  `MASTER_PLAN.md` §7 (future directions) and `methods_report.tex`. Kept because it contains the
  detailed evaluation protocol (A/B) and results tables that are still referenced.

---

## Session 12 — 2026-10-07 (Direction revision: FLIR ADAS v2 + CycleDiff + gap-first doctrine)

**Goal:** Correct the project direction from "build Stage A/B/C on UNIRI-TID" to "establish a significant
gap on FLIR ADAS v2 first, then build pipeline." Add Method 4 (CycleDiff) and snow as a new condition.
Update all documentation to reflect the corrected understanding.

### Direction change

The previous plan (Sessions 1–11) treated Stage A/B/C construction as the immediate next step, with
the M3FD synthetic-weather track as a side experiment. This was reversed for the following reasons:

1. **M3FD synthetic-weather track is a negative result.** All three methods (classical, depth-based,
   CycleGAN) failed to sustain a Protocol 3B gap ≥ 0.05 (fog gap +0.049 WEAK, rain gap +0.011 FAIL).
   The track cannot motivate the GATE-IR methodology.

2. **The gap on UNIRI-TID is <10%.** The real-weather clear-fog gap on UNIRI-TID is 0.094 APm (YOLOv12s).
   This is real and persistent, but a gated pipeline that recovers less than ~0.03 fog APm over an
   augmented baseline has no contribution (augmentation alone recovers 28% of the gap for free).

3. **Pipeline construction without a confirmed gap is unjustifiable.** Building Stage A/B/C on a
   dataset where the maximum achievable gap is < 10% wastes compute on a methodology that reviewers
   will reject as over-engineered for the problem size.

4. **FLIR ADAS v2 offers a path to a larger gap.** Paired IR+RGB, 15 classes, ~26k images. The larger
   scale and class diversity may produce a more significant weather gap. If not, Method 4 (CycleDiff)
   provides a more powerful generation approach than the M3FD methods.

5. **DroneVehicle was rejected as the immediate next step** in the project's own deliberation record
   (devlog Session 4, dataset_generation_plan §1) because it introduces a domain shift (UAV overhead)
   that muddies scope. That decision was made before M3FD failed; it remains valid, but the rationale
   has shifted from "M3FD is viable" to "M3FD is a negative result."

### Documentation updates

- `docs/handoff.md`: Replaced DroneVehicle with FLIR ADAS v2. Added Method 4 (CycleDiff) and snow.
  Reordered "what to do next" to put gap establishment first. Added gap-first doctrine to the
  one-paragraph situation.
- `docs/MASTER_PLAN.md`: Updated status line, §2.4 (next dataset), §2.5 (generation directions),
  §3 (thesis with gap-first gating), §7 (future directions), §8 (methodology rules), §9.8 (FLIR ADAS
  v2 unknowns), §10 (immediate next actions), §11 (open questions).
- `docs/memory.md`: Updated project goal with gap-first doctrine. Added FLIR ADAS v2 facts,
  CycleDiff details, snow physics note, and gap-establishment as a known bug/gotcha.
- `docs/dataset_generation_plan.md`: Complete rewrite. FLIR ADAS v2 as primary dataset. Methods 1–4
  with Protocol 3A/3B tables for fog, rain, and snow. Implementation checklist for all four methods.
- `docs/gap_establishment.md`: Created. Documents the gap-first doctrine, go/no-go rules, and why
  pipeline construction is deferred.

### Key clarifications established

- **What Stage A/B/C is:** Building the full three-stage pipeline (gate → enhancer → detector) and
  getting a baseline score. This is not the immediate next step.
- **Why gap establishment comes first:** The entire GATE-IR methodology (hard gating, per-condition
  enhancement) rests on a significant weather gap existing. If the gap is < 10%, no amount of
  methodology can justify the pipeline.
- **Why FLIR ADAS v2, not DroneVehicle:** Paired IR+RGB, 15 classes, ~26k images, real-world scenes.
  DroneVehicle introduces a UAV overhead domain shift that muddies scope.
- **Why CycleDiff, not just CycleGAN:** M3FD showed CycleGAN is insufficient (3B gap +0.049 fog,
  +0.011 rain). CycleDiff hybridizes CycleGAN with diffusion refinement for higher realism and
  physical accuracy.
- **Why snow is included:** Snow has different IR physics (emissivity wash-out, convective cooling)
  than fog (contrast loss) or rain (emissivity wash-out). It may produce a larger gap.

### State at end of session

| milestone | status |
|---|---|
| M0/M1/M2 — UNIRI-TID baseline sweep (6/7 valid) | ✅ DONE |
| M2.5 Methods 1–3 — M3FD synthetic weather | ✅ DONE — negative result |
| FLIR ADAS v2 download + inspection | ❌ NOT STARTED |
| DSERT-RoLL Heavy_Snow acquisition | ❌ NOT STARTED |
| Method 4 (CycleDiff) design + implementation | ❌ NOT STARTED |
| Gap establishment on FLIR ADAS v2 | ❌ NOT STARTED |
| Stage A/B/C pipeline construction | ❌ DEFERRED until gap confirmed |
| E1 efficiency table | ❌ NOT STARTED (deferred to Phase 2) |
| RT-DETR-l rerun | ❌ NOT STARTED (deferred) |

## Session 13 — 2026-10-18 (Documentation update + protocol clarification)

**Goal:** Update docs to reflect current state and clarify the evaluation protocol.

### Current state (verified 2026-10-18)

- FLIR ADAS v2 downloaded and inspected — paired IR+RGB, 15 classes, ~26k images at `data/flir_adas_v2/FLIR_ADAS_v2/`. Test split: 3,749 clear thermal images at `test/clear/`. No real adverse weather — synthetic generation required.
- DSERT-RoLL acquired — Fog (3,282), Heavy_Rain (1,938), Light_Rain (2,214), Heavy_Snow at `data/DSERT-RoLL/`.
- CycleDiff clear→fog training COMPLETE (step 45,000) — `final_model.pt` (4.8 GB) at `flir_adras_v2/method4_cyclediff/results/train_cycle_C_disc_G_blocks_12_clear2fog/final_model.pt`. Inference NOT YET RUN — `sample_results/` empty.
- CycleGAN on FLIR ADAS v2 NOT STARTED — `method3_cyclegan/runs/cyclegan_{fog,rain,snow}/` empty (no G_AB.pt).
- Clear baseline YOLO NOT TRAINED — `flir_adras_v2/runs/clear_baseline/` does not exist.
- SLURM: Job 9129 CANCELLED — no GPU allocated.

### Protocol clarification

User clarified the evaluation protocol: **train YOLOv12s on clear + synthetic adverse (mixed), evaluate on clear test AND synthetic adverse test.** Gap = clear_test_mAP − adverse_test_mAP on the mixed-trained model. Label (3A/3B) doesn't matter.

Key constraints:
- Foggy **train** images generated **dynamically per batch** during training (disk too limited to pregenerate all).
- Foggy **test** images pregenerated once and stored.
- Sampling: clear and foggy sampled independently per batch — do NOT pair same images as clear-and-foggy in one batch (avoids bias).
- Run order: CycleDiff fog first, then CycleGAN fog.

### Docs updated

- `docs/handoff.md` — current state, protocol definition, Phase 1 action steps.
- `docs/dataset_generation_plan.md` — section 3 rewritten to match unified train-on-clear+adv protocol.


---

## Session 14 — 2026-10-19/20: CycleDiff clear→fog evaluation COMPLETE

**Outcome:** Full test set (3,749 images) translated; all metrics computed.

- **FID = 344.95** (vs real fog GT) — high, flagged as quality concern.
- **KID = 0.4785**, **IS = 1.0839**.
- **LPIPS = 0.6024** (vs clear source), **L2 = 133.24**, **MSE = 0.0929**,
  **PSNR = 10.56 dB**, **SSIM = 0.3164** (all vs clear source — apples-to-oranges, expected low).
- Sampling ran in `salloc_session` tmux (job 9131) after repeated SIGTERM kills of
  `background_process`/`infer_session` launches; skip-if-exists + skip-batch made re-runs resumable.
- Code fixes: `data_cfg` UnboundLocalError for clear2fog task; atomic saves; skip-if-exists
  (per-image + per-batch); fidelity binary path + square 256×256 resize in eval_score.py and
  mse_psnr_ssim_mssim.py; CycleGAN translate.py global-declaration syntax error.
- Artifacts: `sample_results/clear2fog_model_12_timestep50/` (3,749 PNGs).
- Next: pregenerate foggy train images (11,886) via `translation_C_disc_timestep_ode_2_train.yaml`,
  then mixed YOLO training + gap measurement; CycleGAN fog/rain/snow to follow.
