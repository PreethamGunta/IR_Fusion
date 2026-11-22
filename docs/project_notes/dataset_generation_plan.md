# Synthetic Adverse-Weather Dataset Generation Plan

**Goal:** Produce a paired IR-RGB object-detection dataset augmented with fog and rain, where a naively
trained YOLOv12s suffers ≥ 0.15 mAP@50-95 degradation between clear and adverse-weather conditions.
This establishes the need for GATE-IR and provides a controlled severity axis for ablations.

**Why this before Stage A/B/C:** The UNIRI-TID gap (~0.10 mAP@50-95 clear-vs-fog) is real but too
small to justify a full gated pipeline to a reviewer. Augmentation alone recovers 28% of it for free.
We need a larger, demonstrable gap first.

> [!NOTE]
> **Dataset choice deliberated 2026-09-27.** Source dataset: **M3FD retained** (already on hand,
> standard benchmark, established clear baseline 0.514 mAP@50-95). CycleGAN/PID target domain
> (if Methods 3/4 needed): **DSERT-RoLL** (CVPR 2026) for both RGB and thermal fog/rain — the
> only real driving dataset with RGB+thermal x clear/fog/rain. Supplement RGB with **ACDC** for
> fog diversity. Do NOT use UNIRI-TID fog blocks (too scene-specific) or Method 2 depth-fog output
> (circular — synthetic training data for a generator meant to produce realistic synthesis).
> See `docs/devlog.md` Session 4 for full rationale.

---

## 1. Dataset Selection

The source dataset must be:
- **Paired IR + RGB** (same scene, same moment, spatially aligned) — needed for the full GATE-IR story
  and for any RGB-guided KD later.
- **Object detection labels** (bounding boxes) — so mAP can be measured directly.
- **Not trivially easy** — avoid datasets where a single YOLOv12s already reaches mAP@50-95 > 0.80 on
  the clear split (leaves no headroom to demonstrate degradation).
- **Large enough** — at least ~3,000 images for a usable train/val/test split.

### Candidate datasets (ranked)

| # | Dataset | IR+RGB | Det. labels | Images | Classes | Notes |
|---|---|---|---|---|---|---|
| **1** | **M3FD** | ✅ | ✅ | 4,200 | 6 (person, car, bus, motorcycle, lamp, truck) | Preferred. Already on hand (HF `nonameplease/M3FD_Detecion`). Multi-class, mixed urban+road scenes, moderate detection difficulty. |
| **2** | **KAIST Multispectral Pedestrian** | ✅ | ✅ | ~95,000 | 1 (person) | Very large. Diverse day/night. Annotation quality issues (many false negatives) — must use sanitised annotations (Liu et al. 2016 or MSDS). |
| **3** | **LLVIP** | ✅ | ✅ | 15,488 | 1 (person) | High quality, night-only, well-aligned. Single class limits multi-class ablations. |
| **4** | **FLIR ADAS** | ⚠️ near-paired | ✅ | ~10,000 | 3 (person, car, bicycle) | RGB and IR from different cameras; spatial alignment requires homography. More work to align. |
| 5 | VEDAI | ✅ | ✅ | 1,210 | 9 (vehicle classes) | Aerial, small objects. Interesting but unusual domain. |

**Start with M3FD.** It is already on hand, has 6 classes (gives richer AP tables than single-class),
and is a standard benchmark. If M3FD's baseline mAP is too high (making the gap too small), move to KAIST.

> [!NOTE]
> Check M3FD's clear-condition baseline before generating weather. If YOLOv12s already reaches
> mAP@50-95 > 0.75 on the clear split, degradation to ~0.60 gives a 0.15 gap which is sufficient.
> If the clear baseline is lower (< 0.55), the absolute gap may be small even with aggressive fog.

---

## 2. Method Ordering

Try methods cheapest-first. Stop as soon as the proof-of-gap condition is met (≥ 0.15 mAP@50-95 gap
on the medium-severity test split). Each method produces a separate test split for evaluation.

### Method 0 — Baseline (no weather, no augmentation)
Train YOLOv12s on the clear split only. Evaluate on the clear test split.
This is the anchor; all subsequent rows in the results table are relative to this.

---

### Method 1 — Albumentations / OpenCV classical simulation *(cheapest)*
**What it does:** Image-processing approximations applied at inference time to the test split (no
retraining of the generator). Extremely fast, no GPU needed.

**Fog:**
- Blend each image with a uniform white/grey veil: `I_fog = α·I + (1−α)·A` where `A ≈ 200` (grey) and
  `α = exp(−β·d)`. Use a fixed transmission `α ∈ {0.5, 0.3, 0.15}` for light/medium/heavy fog (no depth
  map; uniform degradation across the image).
- Albumentations: `RandomFog(fog_coef_lower, fog_coef_upper)` or manual blend.

**Rain:**
- Overlay synthetic rain streaks: random bright diagonal lines at 60–80° with varying density.
- Albumentations: `RandomRain(slant_lower, slant_upper, drop_length, drop_width, drop_color, blur_value, brightness_coefficient, rain_type)`.
- Optionally add a slight darkening to simulate wet-surface reflections.

**IR treatment:** Apply the same spatial mask (where rain/fog is applied) but reduce contrast instead
of adding a bright veil. For fog: reduce local contrast by `σ·(I − mean(I))` with `σ < 1`. For rain:
no rain streaks in IR (thermal doesn't produce bright rain streaks); instead apply a slight spatial
smoothing to simulate emissivity wash-out.

**Severity levels to test:** light (barely visible), medium (clearly degraded), heavy (objects partially
obscured). Each produces a separate test split.

**Cost:** minutes on CPU. Implement as `bench/data/apply_weather_classical.py`.

---

### Method 2 — Depth-based transmission map fog *(moderate cost, still no training)*
**What it does:** Adds spatial realism by making fog strength a function of scene depth. Objects further
away are more obscured. Requires a monocular depth estimate.

**Steps:**
1. Run a pretrained depth estimator (MiDaS or Depth Anything v2) on the **RGB** images to get a relative
   depth map `D(x,y)`.
2. Compute transmission: `t(x,y) = exp(−β · D(x,y))` where `β` controls fog density.
3. Apply Koschmieder to RGB: `I_fog = I·t + A·(1−t)` where `A` is atmospheric light (estimated as
   the top-1% brightest pixels, or a fixed value ≈ 210).
4. For IR: apply `t_IR(x,y) = exp(−β_IR · D(x,y))` with `β_IR ≈ 0.3·β` (LWIR penetrates fog better)
   and blend with a low-temperature background radiance instead of a bright veil.

**Rain:** Same as Method 1 for the streak pattern, but use `D` to attenuate streak intensity by depth
(streaks near background are lighter than those in the foreground).

**Bounding-box check:** Boxes for occluded objects (t < 0.1) should be flagged and optionally removed
from training/eval to avoid penalising the detector for genuinely invisible objects.

**Cost:** ~1–2 h on CPU for 4,200 images (depth estimation is the bottleneck; can batch on GPU in minutes).
Implement as `bench/data/apply_weather_depth.py`.

---

### Method 3 — CycleGAN-based unpaired translation *(moderate training cost)*
**What it does:** Train a CycleGAN to translate clear → foggy/rainy using unpaired foggy/rainy images
as the target domain. Can produce texturally realistic weather without requiring paired clear/degraded data.

**Source of target-domain images:** RESIDE (RGB foggy), Rain100H (RGB rainy), or frames from UNIRI-TID's
fog/rain blocks (IR foggy/rainy).

**Limitation:** CycleGAN can hallucinate content and may corrupt object boundaries, making bounding boxes
unreliable. Must verify with a box-overlap assertion after translation.

**IR channel:** Train a separate CycleGAN for IR, or use paired RGB→IR translation as an intermediate step.

**Cost:** ~8–20 h training (single GPU). Use the existing `bench/scripts/sweep.sbatch` pattern with
`--time=20:00:00 --requeue`. Do not use the full 200-epoch spec — 50 epochs is usually sufficient for
domain texture transfer.

**When to try:** Only if Methods 1–2 fail to produce a ≥ 0.15 mAP gap.

---

### Method 4 — Physics-Informed Diffusion (PID) *(highest cost, highest realism)*
**What it does:** A dual-stream latent diffusion model (parallel RGB + IR U-Nets with cross-modal
attention) conditioned on physical priors (transmission maps, TeV decomposition). Produces the most
physically accurate and perceptually realistic results. Full architecture in `docs/generating_weather_conditions`.

**When to try:** Only if CycleGAN (Method 3) produces results that are visually unconvincing or that
cause bounding-box alignment problems.

**Key requirements (from `docs/generating_weather_conditions`):**
- Parallel dual-stream U-Nets sharing a latent space via cross-modal attention.
- RGB branch: Koschmieder model with depth-based transmission map.
- IR branch: Temperature-Emissivity-Texture (TeV) decomposition; emissivity wash-out for rain; path
  radiance for fog. Not a copy of the RGB degradation.
- PatchNCE loss to guarantee bounding-box coordinates stay valid across modalities.
- Composite training objective: `L = L_diffusion + λ1·L_PatchNCE + λ2·L_TeV + λ3·L_perceptual`.

**Base model:** Fine-tune from a Stable Diffusion backbone (HuggingFace `diffusers`).
**Training data:** Bootstrap from RESIDE + Rain100H for the RGB branch; fine-tune the IR branch on
available thermal data (UNIRI-TID clear + synthesised weather from Methods 1–2 as pseudo-pairs).
**Cost:** 40–80 h training. Requires careful checkpoint/resume logic. Plan for ≥2 allocations.

---

## 3. Evaluation Protocol

There are **two distinct evaluations** with different purposes:

### 3A — Preliminary screening (train on clear only)
**Purpose:** fast check that the synthetic weather produces a visually meaningful degradation. If a model
that has *never seen fog or rain* cannot detect objects in synthetic fog, the synthesis is realistic
enough to be useful. This is cheap (no new training runs needed once the clear baseline exists) and
sufficient to decide whether to proceed with a given synthesis method.

**Training:** YOLOv12s trained on the **clear-only training split**.
**Eval:** test on clear, light, medium, heavy synthetic weather.
**Gap criterion (screening):** ≥ 0.15 mAP@50-95 clear-vs-heavy for the model to proceed.

### 3B — Final baseline (train on clear + adverse — the real bar to beat)
**Purpose:** establish the true baseline GATE-IR must surpass. A YOLOv12s trained naively on *both*
clear and adverse-weather images together (no gate, no enhancer, just more training data) is the cheapest
rival. If GATE-IR cannot beat this, the gated pipeline has no contribution.

This is the same setup as the UNIRI-TID sweep: the model sees all conditions at training time, so its
performance ceiling is already high. **The existing UNIRI-TID v12s result (APm gap of 0.094) is exactly
this number on UNIRI-TID** — trained on 1,716 clear + 630 fog + 1,593 rain images. The synthetic
dataset must reproduce (or exceed) this gap under the same mixed-training regime.

**Training:** YOLOv12s trained on the **full training split (clear + synthetic fog + synthetic rain)**.
**Eval:** test on clear, light, medium, heavy synthetic weather (same splits as 3A).
**Gap criterion (final bar):** GATE-IR must beat the 3B mAP on fog/rain. If 3B already achieves a
small gap (< 0.05), the problem is too easy for a paper. If 3B achieves a moderate gap (0.05–0.15),
there is room for GATE-IR. A large gap (> 0.15) from even the mixed-training naive baseline makes the
strongest case for a gated approach.

> [!NOTE]
> On UNIRI-TID, YOLOv8s + weather augmentation (the closest equivalent to 3B) achieved APm fog = 0.5691,
> a gap of 0.096 over YOLOv8s clear (0.6652). GATE-IR must beat 0.5691 on fog to have a contribution.
> The synthetic dataset experiment needs to produce the equivalent 3B number for M3FD.

### Protocol steps (for each method and severity level)

1. **Split:** sequence-disjoint train/val/test on the source (clear) dataset.
   Apply weather **only to val and test** for screening (3A).
   For the final baseline (3B), weather is applied to **all three splits**.
2. **3A training:** YOLOv12s on clear-only train split. One run per method.
3. **3B training:** YOLOv12s on clear+fog+rain train split. One run per method that passes 3A screening.
4. **Evaluation metrics:** mAP@50-95 overall, per-condition, and APm (medium objects).
5. **Multi-seed:** ≥ 3 seeds, report mean ± std before making any gap claim.

---

## 4. Results Table (to fill in)

> **3A** = trained on clear only · **3B** = trained on clear + fog + rain (the true bar to beat)
>
> APm = AP on medium-sized objects (≥ 32px, ≤ 96px side) — the most informative metric given box sizes.

### 4.0 UNIRI-TID reference (real weather — YOLOv12s trained on mixed, the 3B equivalent)

| condition | mAP@50 | mAP@50-95 | APm | clear − adverse (APm) |
|---|---|---|---|---|
| clear | 0.9704 | 0.6296 | 0.6670 | 0 |
| fog | 0.9728* | 0.5807 | 0.5733 | **0.094** |
| rain | — | 0.6052 | 0.6025 | 0.045 |

*mAP@50 is saturated on this dataset; ignore it. All numbers from YOLOv12s 76-epoch run.*

---

### 4.1 Source dataset baseline (M3FD, clear split only — no weather)

| training | epochs | mAP@50 | mAP@50-95 | APm | notes |
|---|---|---|---|---|---|
| 3A: clear only | — | — | — | — | anchor for screening |
| 3B: clear+aug | — | — | — | — | with standard augmentation, no synthetic weather |

---

### 4.2 Method 1 — Classical simulation

**Status: COMPLETE (3A screening done; 3B not needed — failed screening).**

| training | severity | mAP@50 | mAP@50-95 | APm | clear − adverse (APm) |
|---|---|---|---|---|---|
| **3A** (clear-only) | clear | 0.787 | 0.514 | — | 0 |
| 3A | light fog | 0.782 | 0.512 | — | 0.002 |
| 3A | medium fog | 0.764 | 0.498 | — | 0.016 |
| 3A | heavy fog | 0.697 | 0.451 | — | **0.063** (Target ≥ 0.15) |
| 3A | light rain | 0.788 | 0.515 | — | -0.001 |
| 3A | medium rain | 0.782 | 0.511 | — | 0.003 |
| 3A | heavy rain | 0.768 | 0.502 | — | 0.012 |
| **3B** (mixed) | clear | — | — | — | not run (failed screening) |
| 3B | medium fog | — | — | — | not run |
| 3B | heavy fog | — | — | — | not run |
| 3B | medium rain | — | — | — | not run |
| 3B | heavy rain | — | — | — | not run |

**3A gap criterion met? No ✔ (failed) — proceeded to Method 2.**
See `m3fd_experiments/method1_classical/report.md` for full analysis.

---

### 4.3 Method 2 — Depth-based transmission map

**Status: DONE — both 3A and 3B run. Fog passed 3A but FAILED 3B. Method 3 required.**

> [!CAUTION]
> **3B gap collapsed to zero.** 3A passed (0.154) but after mixed training the gap is −0.002.
> Method 2 fog is a deterministic filter — YOLO fully adapts to it with exposure. This is the
> critical metric for the paper and it is not met. See `m3fd_experiments/method2_depth/report.md`.
>
> Rain rows use **Method 1 classical IR** verbatim (byte-identical). Depth-aware rain was not
> implemented for IR; the RGB-only depth rain has no effect since evaluation uses IR images only.

| protocol | condition | mAP@50 | mAP@50-95 | gap vs clear |
|---|---|---|---|---|
| **3A** (clear-only model) | clear | 0.787 | 0.514 | 0 |
| 3A | fog medium (β=3.0) | 0.677 | 0.440 | 0.074 |
| 3A | fog heavy (β=4.5) | 0.551 | 0.360 | **0.154** |
| 3A | rain heavy (classical IR) | 0.768 | 0.502 | 0.012 |
| **3B** (mixed clear+fog model) | clear test | 0.807 | 0.534 | 0 |
| 3B | fog heavy test | 0.811 | 0.536 | **−0.002** ❌ |

**3A: passed fog (0.154 ≥ 0.15). 3B: FAILED fog (−0.002). Rain: never adequate.**
**Conclusion: Method 2 fog is insufficient. Proceed to Method 3 (CycleGAN).**
Full analysis in `m3fd_experiments/method2_depth/report.md`.

---

### 4.4 Method 3 — CycleGAN translation

**Status: DONE — 3A and 3B both run. Fog WEAK (+0.033 3B gap). Rain FAILED (+0.007). Below bar.**

> [!IMPORTANT]
> The goal for Method 3 is a **3B gap ≥ 0.10**, not just a 3A gap. Method 2 showed that a large 3A
> gap can coexist with a 3B gap of zero. CycleGAN output has stochastic texture variation that makes
> it harder for a model to fully adapt under mixed training. Target domain: **DSERT-RoLL** (real
> driving RGB+thermal fog/rain). All four generators trained to 50 epochs (147,000 iters each).
> `rgb_fog`/`rgb_rain` completed but unused in eval (all eval YAMLs are IR-only).

> [!WARNING]
> **The M3FD split is not scene-disjoint.** All numbers below are on a contaminated test set
> (random frame-level split of sequential video). These results are not publishable without a
> scene-disjoint resplit and rerun. The generators themselves are unaffected and can be reused.

| protocol | condition | mAP@50 | mAP@50-95 | gap vs clear | verdict |
| training | severity | mAP@50 | mAP@50-95 | clear − adverse | notes |
|---|---|---|---|---|---|
| **3A** (clear-only model) | clear (ref) | — | 0.564 | 0 | clean baseline |
| 3A | CycleGAN fog | — | 0.037 | +0.527 | ⚠️ catastrophic collapse |
| 3A | CycleGAN rain | — | 0.411 | +0.153 | gap exists |
| **3B** (fog mixed model) | clear test | — | 0.285 | 0 | ref |
| 3B | CycleGAN fog test | — | 0.496 | **-0.211** | ❌ FAIL (negative gap) |
| **3B** (rain mixed model) | clear test | — | 0.274 | 0 | ref |
| 3B | CycleGAN rain test | — | 0.530 | **-0.256** | ❌ FAIL (negative gap) |

**3B gap criterion met? No.** Both fog and rain produced massive negative gaps (-0.21 and -0.25), meaning the mixed model performs >20 mAP worse on *clear* images than on synthetic weather. The synthetic generation destroys clear-weather performance rather than simulating realistic degradation.

Method 3 is the final synthetic method tried. No method has met the threshold on M3FD.
**Decision (2026-10-01):** Drop M3FD. Proceed to Stage A/B/C using UNIRI-TID.

---

### 4.4 Status Update (2026-10-01): The Splitting Pothole & Re-evaluation

The initial Method 3 (CycleGAN) evaluation produced suspect results because the M3FD dataset split was found to be highly contaminated (a purely random frame-level shuffle, which leaked consecutive video frames across train/test).

A first attempt at fixing this using pixel-difference thresholding (`diff > 20.0`) failed, creating hundreds of 1-frame clips and recreating the exact same leakage. 
The old, flawed evaluation results for Method 3 have been archived in `m3fd_experiments/_old/`.

**Current Corrected State:**
M3FD has been resplit using a robust **Stochastic Multi-Objective Class-Stratified Chunking** algorithm:
1. The dataset was chunked into contiguous 100-frame blocks.
2. A conditional 5-frame drop (temporal buffer) was enforced at boundaries where the assigned split changes, preventing transition leakage.
3. A stochastic search balanced both the target frame quotas (70/10/20) and the class distributions across Train/Val/Test, ensuring no class is missing from the validation set (which previously skewed the `best.pt` selection).

The `v12s_m3fd_clear` baseline and the Method 3 CycleGAN Protocol 3A/3B pipeline are currently re-running on this clean, mathematically sound split. The new results will replace the archived numbers upon completion.

---

### 4.5 Method 4 — Physics-Informed Diffusion

| training | severity | mAP@50 | mAP@50-95 | APm | clear − adverse (APm) |
|---|---|---|---|---|---|
| **3A** | clear | — | — | — | 0 |
| 3A | medium fog | — | — | — | — |
| 3A | heavy fog | — | — | — | **← screening ≥ 0.15** |
| 3A | medium rain | — | — | — | — |
| 3A | heavy rain | — | — | — | — |
| **3B** | clear | — | — | — | 0 |
| 3B | heavy fog | — | — | — | **← bar to beat** |
| 3B | heavy rain | — | — | — | — |

**3A gap criterion met? ☐ Yes / ☐ No**

---

## 5. Implementation Checklist

- [x] **Download and verify M3FD** — confirmed image count, label format (YOLO), class names.
- [x] **Establish M3FD clear baseline** — YOLOv12s trained; mAP@50-95=0.51425, mAP50=0.787 (test). ⚠️ split contaminated.
- [x] **Implement Method 1** (`m3fd_experiments/method1_classical/`)
  - [x] Fog: uniform veil blend at 3 severity levels
  - [x] Rain: streak overlay at 3 severity levels
  - [x] IR: contrast reduction (fog), spatial smoothing (rain)
- [x] **Run Method 1 evaluation** — failed 3A (max fog gap 0.063 < 0.15).
- [x] **Implement Method 2** (`m3fd_experiments/method2_depth/`)
  - [x] Depth estimation (Depth Anything v2, batched on GPU)
  - [x] Transmission map for RGB and IR separately
  - [x] Atmospheric light estimation
- [x] **Run Method 2 evaluation** — passed 3A (0.154) but **FAILED 3B (−0.002)**.
- [x] **Method 3 (CycleGAN)** — run because Methods 1–2 insufficient.
  - [x] `thermal_fog` generator (M3FD IR → DSERT Fog) — DONE, 50 epochs
  - [x] `thermal_rain` generator (M3FD IR → DSERT Rain) — DONE, 50 epochs
  - [x] `rgb_fog` generator (M3FD RGB → DSERT Fog) — DONE, 50 epochs (unused in eval)
  - [x] `rgb_rain` generator (M3FD RGB → DSERT Rain) — DONE, 50 epochs (unused in eval)
  - [x] Protocol 3A eval — fog 0.037 (+0.477 gap), rain 0.411 (+0.103 gap)
  - [x] Protocol 3B eval — fog **+0.033 (WEAK)**, rain **+0.007 (FAIL)**
- [ ] **Method 4 (PID)** — only if Method 3 insufficient. ← DECISION REQUIRED first.
- [x] **Fix M3FD split** — scene-disjoint resplit done using stochastic chunking.
- [ ] **Select the best method** — the one meeting the gap criterion with the most realistic output.
- [ ] **Build the full dataset** — apply best method to train+val+test splits.
- [ ] **Hand off to GATE-IR Stage A/B/C experiments.**

---

## 6. Key Constraints and Gotchas

**Circularity trap (same as UNIRI-TID):** If weather labels are derived from image statistics that
WeatherGate classifies, gate accuracy is ~100% by construction. Labels must come from the generation
process (severity parameter), not from image statistics.

**Bounding-box validity:** Heavy fog/rain can occlude objects entirely. Before training or evaluating
on a generated split, assert that every bounding box still corresponds to a visible object (t > 0.05
for that box region). Remove or flag invisible boxes.

**IR physics must be treated separately from RGB.** A common mistake is applying the same JPEG filter
to both channels. LWIR fog causes contrast loss + path radiance, not a bright scattering veil. LWIR
rain causes emissivity wash-out (wet surfaces), not bright rain streaks. The IR treatment in every method
above is deliberately different from the RGB treatment.

**Do not train the weather generator on UNIRI-TID features.** If you fine-tune a generator on UNIRI-TID's
fog/rain blocks to produce the M3FD synthetic weather, you risk learning UNIRI-TID's scene-specific
appearance (one location, one set of actors) rather than physical weather properties.

**Alignment check after generation.** Run a quick `cv2.matchTemplate` or SSIM sanity check between
the generated IR and RGB images to confirm spatial alignment was not broken by the generation process.

**Single-seed is not enough for a gap claim.** Run at least 3 seeds, report mean±std. A 0.15 mAP gap
from a single seed is a promising signal, not a confirmed result.

