# Synthetic Adverse-Weather Dataset Generation Plan — FLIR ADAS v2

**Goal:** Produce a paired IR-RGB object-detection dataset augmented with fog, rain, and snow, where a
naively trained YOLOv12s suffers ≥ 0.15 mAP@50-95 degradation between clear and adverse-weather
conditions under Protocol 3B (mixed training). This establishes the need for GATE-IR and provides a
controlled severity axis for ablations.

**Why this before Stage A/B/C:** The UNIRI-TID real-weather gap (~0.10 mAP@50-95 clear-vs-fog at
matched object size) is real but too small to justify a full gated pipeline to a reviewer. Augmentation
alone recovers ~28% of it for free. The M3FD synthetic-weather track produced negative results (all
methods failed Protocol 3B with gaps < 0.05). A larger, demonstrable gap on a multi-class dataset is
required before any GATE-IR pipeline code is written.

> **Dataset choice deliberated 2026-10-07.** Source dataset: **FLIR ADAS v2** (paired IR+RGB,
> 15 classes, ~26k images, Kaggle `samdazel/teledyne-flir-adas-thermal-dataset-v2`). Target domains:
> **DSERT-RoLL** (CVPR 2026) for fog and rain real thermal imagery; **Heavy_Snow** (source TBD) for
> snow. Do NOT use M3FD as the primary dataset (synthetic-weather track is a negative result). Do NOT
> use UNIRI-TID fog blocks as target domain (too scene-specific, single camera). Do NOT use Method 2
> depth-fog output as target domain (circular — synthetic training data for a generator meant to
> produce realistic synthesis).
>
> **Gap-first doctrine:** Pipeline construction (Stage A/B/C) is deferred until Protocol 3B on FLIR
> ADAS v2 confirms a gap ≥ 0.15 mAP@50-95. If no combination of dataset + generation method reaches
> this threshold, the thesis must pivot to T3 (negative/diagnostic result). See `docs/gap_establishment.md`.

---

## 1. Dataset Selection

The source dataset must be:
- **Paired IR + RGB** (same scene, same moment, spatially aligned) — needed for Method 4 (CycleDiff)
  cross-modal generation and for future RGB-guided KD.
- **Object detection labels** (bounding boxes) — so mAP can be measured directly.
- **Not trivially easy** — avoid datasets where a single YOLOv12s already reaches mAP@50-95 > 0.80 on
  the clear split (leaves no headroom to demonstrate degradation).
- **Large enough** — at least ~10,000 images for a usable train/val/test split.
- **Multi-class** — enables richer per-class AP tables than single-class datasets.

### Selected dataset

| # | Dataset | IR+RGB | Det. labels | Images | Classes | Notes |
|---|---|---|---|---|---|---|
| **1** | **FLIR ADAS v2** | ✅ | ✅ | ~26,000 | 15 (vehicles, person, bicycle, etc.) | Selected. Paired IR+RGB, automotive/urban scenes, larger scale than M3FD. Multi-class. Source: Kaggle `samdazel/teledyne-flir-adas-thermal-dataset-v2`. |

**FLIR ADAS v2 is the primary dataset.** It has not been downloaded or inspected yet. The following
are UNKNOWN and must be resolved after download:
- Does the dataset include real weather conditions or only clear?
- What is the class distribution and annotation quality?
- Are there existing train/val/test splits?
- What is the image resolution and format?
- Is the dataset already sequence-disjoint, or are there temporal redundancies?

If the dataset contains real weather conditions, the natural clear-vs-adverse gap is measured first
(Protocol 0). If not, the dataset becomes the base for Method 3 (CycleGAN) and Method 4 (CycleDiff)
generation.

---

## 2. Method Ordering

Try methods cheapest-first. Stop as soon as the proof-of-gap condition is met (≥ 0.15 mAP@50-95 gap
under Protocol 3B). Each method produces a separate test split for evaluation.

**UPDATE 2026-10-07:** Methods 1 and 2 are IGNORED for FLIR ADAS v2. They are skipped entirely —
Method 1 produced only +0.032 fog gap on M3FD (too small), and Method 2 showed structural failure
(deterministic invertibility). Focus directly on Methods 3 (CycleGAN) and 4 (CycleDiff).

### Method 0 — Natural gap (if real weather exists in FLIR ADAS v2)

**What it does:** If FLIR ADAS v2 contains real weather conditions (fog, rain, snow), train YOLOv12s
on the mixed split and measure the natural clear-vs-adverse gap. This is the strongest possible
evidence because it uses real degradation, not synthetic.

**Protocol 3B equivalent:** YOLOv12s trained on clear + adverse, evaluated on clear test vs adverse
test.

**Gap criterion:** ≥ 0.15 mAP@50-95 clear-vs-adverse. If this exists natively, no synthetic
generation is needed and the project can proceed directly to pipeline construction.

### Method 1 — Albumentations / OpenCV classical simulation *(cheapest)*

**What it does:** Image-processing approximations applied at inference time to the test split (no
retraining of the generator). Extremely fast, no GPU needed.

**Fog:**
- Blend each image with a uniform white/grey veil: `I_fog = α·I + (1−α)·A` where `A ≈ 200` (grey) and
  `α = exp(−β·d)`. Use a fixed transmission `α ∈ {0.5, 0.3, 0.15}` for light/medium/heavy fog (no depth
  map; uniform degradation across the image).

**Rain:**
- Overlay synthetic rain streaks: random bright diagonal lines at 60–80° with varying density.

**Snow:**
- Overlay synthetic snow particles: white circular blobs with Gaussian falloff, combined with a
  slight contrast reduction to simulate emissivity wash-out.

**IR treatment:** Apply the same spatial mask but reduce contrast instead of adding a bright veil.
For fog: reduce local contrast by `σ·(I − mean(I))` with `σ < 1`. For rain: spatial smoothing to
simulate emissivity wash-out. For snow: stronger contrast reduction + high-frequency noise.

**Severity levels to test:** light (barely visible), medium (clearly degraded), heavy (objects partially
obscured). Each produces a separate test split.

**Cost:** minutes on CPU.

---

### Method 2 — Depth-based transmission map fog *(moderate cost, still no training)*

**What it does:** Adds spatial realism by making fog strength a function of scene depth. Objects further
away are more obscured. Requires a monocular depth estimate.

**Steps:**
1. Run a pretrained depth estimator (MiDaS or Depth Anything v2) on the **RGB** images to get a relative
   depth map `D(x,y)`.
2. Compute transmission: `t(x,y) = exp(−β · D(x,y))` where `β` controls fog density.
3. Apply Koschmieder to RGB: `I_fog = I·t + A·(1−t)` where `A` is atmospheric light.
4. For IR: apply `t_IR(x,y) = exp(−β_IR · D(x,y))` with `β_IR ≈ 0.3·β` (LWIR penetrates fog better)
  and blend with a low-temperature background radiance instead of a bright veil.

**Rain:** Same as Method 1 for the streak pattern, but use `D` to attenuate streak intensity by depth.

**Snow:** Use depth to modulate snow accumulation — more snow on horizontal surfaces (roads, hoods)
than vertical surfaces (walls, pedestrians).

**Bounding-box check:** Boxes for occluded objects (t < 0.1) should be flagged and optionally removed
from training/eval.

**Cost:** ~1–2 h on CPU for depth estimation.

---

### Method 3 — CycleGAN-based unpaired translation *(moderate training cost)*

**What it does:** Train a CycleGAN to translate clear → foggy/rainy/snowy using unpaired adverse
images as the target domain. Can produce texturally realistic weather without requiring paired
clear/degraded data.

**Source:** FLIR ADAS v2 clear IR images.
**Target:** DSERT-RoLL thermal fog/rain + Heavy_Snow.

**Limitation:** CycleGAN can hallucinate content and may corrupt object boundaries. Must verify with
a box-overlap assertion after translation.

**Cost:** ~8–20 h training (single GPU).

**When to try:** Only if Methods 1–2 fail to produce a ≥ 0.15 mAP gap.

---

### Method 4 — CycleDiff *(hybrid CycleGAN + diffusion, highest cost, highest realism)*

**What it does:** A hybrid architecture that combines CycleGAN's unpaired style-transfer capability
with diffusion-based refinement. The CycleGAN provides the initial weather translation (fast,
unpaired), and a lightweight diffusion model refines the output for perceptual realism and physical
accuracy.

**Architecture (design in progress):**
- **Stage 1 — CycleGAN backbone:** ResNet-9 generator (same as Method 3) trained on FLIR ADAS v2
  (clear) → DSERT-RoLL + Heavy_Snow (adverse). Produces initial weather translation.
- **Stage 2 — Diffusion refiner:** A lightweight latent diffusion model (e.g., SD 2.1-base or
  ControlNet) fine-tuned on the CycleGAN outputs vs real adverse images. Applies 5–10 denoising
  steps to refine texture, reduce artifacts, and improve physical realism.

**Conditioning:** Severity control via classifier-free guidance (CFG) scale or a learned severity
embedding. Fog, rain, and snow each have their own conditioning vector.

**IR-specific physics:**
- Fog: contrast loss + path radiance (not a bright veil).
- Rain: emissivity wash-out (wet surfaces), not bright streaks.
- Snow: emissivity wash-out + convective cooling + high-frequency texture.

**Cost:** CycleGAN: ~8–20 h. Diffusion fine-tuning: ~20–40 h. Total: ~30–60 h.

**When to try:** If Method 3 (CycleGAN) produces results that are visually unconvincing or that
cause bounding-box alignment problems. Method 4 is the highest-cost option and should only be
attempted after Methods 1–3 are exhausted.

---

## 3. Evaluation Protocol

There is **one evaluation** — train on clear + synthetic adverse, test on clear + adverse:

**Purpose:** Establish the true baseline GATE-IR must surpass. A YOLOv12s trained naively on both
clear and synthetic adverse images together (no gate, no enhancer, just more training data) is the
cheapest rival. The gap is measured as: clear_test_mAP − adverse_test_mAP on the mixed-trained model.
If this gap is small (< 0.05), the problem is too easy. If moderate (0.05–0.15), GATE-IR has room.
If > 0.15 from even the naive mixed baseline, the case for gating is strongest.

**Label note:** It doesn't matter whether this is called 3A or 3B in the code. The functionality
is: train on clear+adv, evaluate on clear+adv, measure the gap.

**Training:** YOLOv12s on clear + synthetic adverse (mixed) training split.
**Eval:** test on clear test AND synthetic adverse test. Report mAP@50-95 for both.
**Gap criterion:** ≥ 0.15 mAP@50-95 clear-vs-adverse on the mixed-trained model.

**For each method (CycleDiff, CycleGAN) and condition (fog, rain, snow):**
- Need clear training images ✅ (have them — 11,886 images, ~1 GB)
- Need synthetic adverse training images — **pregenerate ALL foggy train images first and save them** (~1 GB total, fits in 52 GB quota). Mix randomly with clear images (~50% foggy probability). Do NOT pair same images as both clear and foggy.
- Need synthetic adverse test images — pregenerate once and store (~106 MB)
- Disk check: total foggy storage ~1.1 GB — well within quota. Disk is NOT a constraint.

### Protocol steps (for each method and severity level)

1. **Split:** sequence-disjoint / chunk-disjoint train/val/test on the source (clear) dataset.
   Apply weather to train and test splits (mixed training).
2. **Generate synthetic adverse train images dynamically** during training (per-batch sampling,
   independent clear/foggy pairs — no pregeneration of all train images due to disk constraints).
3. **Generate synthetic adverse test images once** and store them for evaluation.
4. **Train YOLOv12s** on clear + synthetic adverse (mixed) training split. One run per method.
5. **Evaluate** on clear test AND synthetic adverse test. Report mAP@50-95 for both.
6. **Gap** = clear_test_mAP − adverse_test_mAP on the mixed-trained model.
7. **Multi-seed:** ≥ 3 seeds, report mean ± std before making any gap claim.

---

## 4. Results Table (to fill in)

> **3A** = trained on clear only · **3B** = trained on clear + fog + rain + snow (the true bar to beat)
>
> APm = AP on medium-sized objects (≥ 32px, ≤ 96px side) — the most informative metric given box sizes.

### 4.0 FLIR ADAS v2 baseline (clear split only — no weather)

| training | epochs | mAP@50 | mAP@50-95 | APm | notes |
|---|---|---|---|---|---|
| 3A: clear only | — | — | — | — | anchor for screening |
| 3B: clear+aug | — | — | — | — | with standard augmentation, no synthetic weather |

### 4.1 Method 1 — Classical simulation

| training | severity | mAP@50 | mAP@50-95 | APm | clear − adverse (APm) |
|---|---|---|---|---|---|
| **3A** (clear-only model) | clear | — | — | — | 0 |
| 3A | light fog | — | — | — | — |
| 3A | medium fog | — | — | — | — |
| 3A | heavy fog | — | — | — | **← screening ≥ 0.15** |
| 3A | light rain | — | — | — | — |
| 3A | medium rain | — | — | — | — |
| 3A | heavy rain | — | — | — | — |
| 3A | light snow | — | — | — | — |
| 3A | medium snow | — | — | — | — |
| 3A | heavy snow | — | — | — | — |
| **3B** (mixed) | clear | — | — | — | 0 |
| 3B | medium fog | — | — | — | — |
| 3B | heavy fog | — | — | — | — |
| 3B | medium rain | — | — | — | — |
| 3B | heavy rain | — | — | — | — |
| 3B | medium snow | — | — | — | — |
| 3B | heavy snow | — | — | — | — |

### 4.2 Method 2 — Depth-based transmission map

| training | severity | mAP@50 | mAP@50-95 | APm | clear − adverse (APm) |
|---|---|---|---|---|---|
| **3A** (clear-only model) | clear | — | — | — | 0 |
| 3A | medium fog | — | — | — | — |
| 3A | heavy fog | — | — | — | **← screening ≥ 0.15** |
| 3A | medium rain | — | — | — | — |
| 3A | heavy rain | — | — | — | — |
| 3A | medium snow | — | — | — | — |
| 3A | heavy snow | — | — | — | — |
| **3B** (mixed) | clear | — | — | — | 0 |
| 3B | heavy fog | — | — | — | **← bar to beat** |
| 3B | heavy rain | — | — | — | — |
| 3B | heavy snow | — | — | — | — |

### 4.3 Method 3 — CycleGAN translation

| training | severity | mAP@50 | mAP@50-95 | APm | clear − adverse (APm) |
|---|---|---|---|---|---|
| **3A** (clear-only model) | clear | — | — | — | 0 |
| 3A | CycleGAN fog | — | — | — | **← screening ≥ 0.15** |
| 3A | CycleGAN rain | — | — | — | — |
| 3A | CycleGAN snow | — | — | — | — |
| **3B** (mixed) | clear | — | — | — | 0 |
| 3B | CycleGAN fog | — | — | — | **← bar to beat** |
| 3B | CycleGAN rain | — | — | — | — |
| 3B | CycleGAN snow | — | — | — | — |

### 4.4 Method 4 — CycleDiff

| training | severity | mAP@50 | mAP@50-95 | APm | clear − adverse (APm) |
|---|---|---|---|---|---|
| **3A** (clear-only model) | clear | — | — | — | 0 |
| 3A | CycleDiff fog | — | — | — | **← screening ≥ 0.15** |
| 3A | CycleDiff rain | — | — | — | — |
| 3A | CycleDiff snow | — | — | — | — |
| **3B** (mixed) | clear | — | — | — | 0 |
| 3B | CycleDiff fog | — | — | — | **← bar to beat** |
| 3B | CycleDiff rain | — | — | — | — |
| 3B | CycleDiff snow | — | — | — | — |

---

## 5. Implementation Checklist

- [x] **Download and verify FLIR ADAS v2** — Kaggle dataset, paired IR+RGB, 15 classes, ~26k images. Complete (12GB).
- [x] **Inspect FLIR ADAS v2** — weather metadata, class distribution, splits, resolution, format. Complete (see inspect_report.json).
- [x] **Acquire DSERT-RoLL (Fog, Rain, Heavy_Snow)** — Fog (3282), Heavy_Rain (1938), Light_Rain (2214), Heavy_Snow (980 thermal).
  - NOTE: Light_Snow download is in progress in background (tmux: dl_lightsnow). Training proceeds with Heavy_Snow for snow condition; Light_Snow will be added to training pool when download completes. Light_Snow is OPTIONAL - we proceed with Heavy_Snow only.
- [ ] **Establish FLIR ADAS v2 clear baseline** — YOLOv12s trained; mAP@50-95 on clear split.
- [~] **Method 1 (classical)** — SKIPPED. Produced only +0.032 fog gap on M3FD.
- [~] **Method 2 (depth-based)** — SKIPPED. Structural failure: deterministic invertibility.
- [~] **Implement Method 3** (`flir_adras_v2/method3_cyclegan/`) — implemented
  - [x] CycleGAN generators for fog, rain, snow (train_cyclegan.py, translate.py, make_cyclegan_dataset.py)
  - [x] DSERT-RoLL Fog + Rain + Heavy_Snow target domains (verified via --check-data)
- [--] **Run Method 3 evaluation** — Protocol 3A + 3B. (Pending clear baseline + translator training)
- [ ] **Implement Method 4** (`flir_adras_v2/method4_cyclediff/`)
  - [ ] CycleDiff hybrid architecture (CycleGAN + diffusion refiner)
  - [ ] Severity conditioning
  - [x] Configs updated with correct paths
  - [x] VAE pretrained weight (kl-f4) downloaded
  - [x] train_cyclediff.sh orchestrator created
- [--] **Run Method 4 evaluation** — Protocol 3A + 3B. (Pending VAE/LDM stages + cycle training)
- [ ] **Select the best method** — the one meeting the gap criterion with the most realistic output.
- [ ] **Build the full dataset** — apply best method to train+val+test splits.
- [ ] **Hand off to GATE-IR Stage A/B/C experiments** (only after gap ≥ 0.15 confirmed).

---

## 6. Key Constraints and Gotchas

**Gap-first doctrine:** The entire GATE-IR methodology rests on a significant weather gap existing.
If no method reaches a Protocol 3B gap ≥ 0.15 on FLIR ADAS v2, the thesis pivots to T3 (negative
result) and Stage A/B/C becomes a "build it and show it doesn't help" study.

**Circularity trap:** If weather labels are derived from image statistics that WeatherGate classifies,
gate accuracy is ~100% by construction. Labels must come from the generation process (severity
parameter), not from image statistics.

**Bounding-box validity:** Heavy fog/rain/snow can occlude objects entirely. Before training or
evaluating on a generated split, assert that every bounding box still corresponds to a visible object
(t > 0.05 for that box region). Remove or flag invisible boxes.

**IR physics must be treated separately from RGB.** LWIR fog causes contrast loss + path radiance,
not a bright scattering veil. LWIR rain causes emissivity wash-out, not bright rain streaks.
LWIR snow causes emissivity wash-out + convective cooling + high-frequency texture, not a white veil.

**Do not train the weather generator on UNIRI-TID features.** If you fine-tune a generator on
UNIRI-TID's fog/rain blocks to produce FLIR ADAS v2 synthetic weather, you risk learning UNIRI-TID's
scene-specific appearance (one location, one set of actors) rather than physical weather properties.

**Alignment check after generation.** Run a quick `cv2.matchTemplate` or SSIM sanity check between
the generated IR and RGB images to confirm spatial alignment was not broken by the generation process.

**Single-seed is not enough for a gap claim.** Run at least 3 seeds, report mean±std. A 0.15 mAP gap
from a single seed is a promising signal, not a confirmed result.

**Snow is not white fog.** In thermal IR, snow does not appear as a bright veil. It appears as
emissivity wash-out (wet surfaces have higher emissivity, reducing contrast) and convective cooling
(surfaces lose heat to the cold snow). A generation model that treats snow as "bright fog" will
produce unrealistic outputs and a small or negative detection gap.
