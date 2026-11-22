# GATE-IR — Master Plan

**Status:** M3FD synthetic weather experiments complete — negative result (all methods fail Protocol 3B, gap < 0.05). Next: FLIR ADAS v2 dataset with Method 3 (CycleGAN) and Method 4 (CycleDiff) generating fog, rain, and snow. Gap establishment precedes pipeline construction.
**Last updated:** 2026-10-07
**Scope:** the arc from here to a submittable paper. History lives in `docs/devlog.md`. Current next actions live in `docs/handoff.md`.

---

## 1. What GATE-IR claims

A three-stage pipeline for thermal (LWIR) person/object detection under adverse weather:

| Stage | Component | Size | Purpose |
|---|---|---|---|
| **A — Gate** | MLP over 5 handcrafted image statistics | ~4.7k params | classify Clear / Fog / Rain |
| **B — Preprocess** | adaptive gamma (fog) / LSRB+LCN (rain) / bypass (clear) | **0 params** / ~1.9k params | condition-specific enhancement |
| **C — Detect** | YOLOv8s + P2 head + ViT neck | ~11M params | detection |
| §3.4 — KD | CycleGAN→pseudo-RGB + YOLOv8-L teacher + feature mimic | ~28M | **deferred, see §6** |

Selling points: works on thermal, works under fog/rain, runs real-time on edge hardware.

---

## 2. Strategic situation

### 2.1 UNIRI-TID baseline — done, but the premise is shaky

The 7-model sweep on UNIRI-TID is complete (M0/M1/M2 done). YOLOv12s achieves mAP@50-95 = 0.6296 overall, with a clear-fog gap of 0.094 APm at matched object size. The gap is real and persistent across architectures, but:

- The published companion paper already reports ~98% AP@50 in every weather condition — the benchmark is saturated at IoU 0.5.
- Rain scores *higher* than clear in the published baseline (98.08 vs 97.85), because rain increases the thermal contrast between warm subjects and cold backgrounds.
- Condition is confounded with camera distance (fog 0–30 m, clear 50–160 m, rain 30–215 m).

These facts force the paper away from an accuracy claim and toward either T1 (efficiency/Pareto), T2 (gate-error robustness), or T3 (negative/diagnostic result).

### 2.2 M3FD synthetic weather — complete and clean

All three methods (classical, depth-based, CycleGAN) have been evaluated on M3FD with a chunk-disjoint temporal split (split_hash `effb9b3090ad2fcb`). The split was rebuilt after an audit found the original was a random frame-level shuffle of sequential video. All contaminated results were archived to `m3fd_experiments/_old/` and replaced with clean re-runs.

**Current verified numbers (Protocol A / B, mAP@50-95):**

| Method | A Fog Gap | B Fog Gap | A Rain Gap | B Rain Gap |
|---|---|---|---|---|
| 1 — classical | 0.032 | — | 0.108 | — |
| 2 — depth fog | 0.076 | −0.001 | 0.137 | — |
| 3 — CycleGAN | 0.101 | 0.049 | 0.044 | 0.011 |

Key findings:
- No method passes the B gap ≥ 0.10 threshold.
- Method 2's deterministic Koschmieder fog is fully learnable (gap inverts from 0.076 to −0.001 under mixed training).
- Method 3's CycleGAN creates the largest A gap (0.101 fog) but the B gap collapses to 0.049 — the learned domain is harder but still adaptively learnable.
- The clear baseline mAP@50-95 is 0.148, which is low. All relative comparisons are valid, but absolute gap thresholds may need to be relative (fraction of baseline) rather than absolute.

### 2.3 Good news — we have real fog and real rain with boxes

UNIRI-TID remains the primary dataset. The M3FD experiments serve as a controlled synthetic-weather benchmark and a proof-of-concept for the generation pipeline.

### 2.4 The next dataset direction

M3FD is a completed negative result for synthetic weather generation. The next dataset is **FLIR ADAS v2** (Teledyne FLIR, paired IR+RGB, 15 classes, ~26k images). This provides:
- Paired RGB+IR for cross-modal generation (CycleDiff, future KD)
- 15 classes for richer per-class AP tables
- Larger scale (~26k vs M3FD's 4.2k) for stronger baseline detectors
- Real-world automotive/urban scenes

The dataset must be inspected after download to determine whether it contains real weather conditions or requires synthetic generation. If real weather conditions exist, the natural clear-vs-adverse gap is measured first. If not, the dataset becomes the base for Method 3 (CycleGAN) and Method 4 (CycleDiff) generation.

**Generation target domains:** DSERT-RoLL provides fog and rain real thermal imagery. Heavy_Snow must be acquired from a separate source or an expanded DSERT-RoLL release. See `docs/dataset_generation_plan.md` for the full Method 3/4 specifications.

**Gap-first doctrine:** Pipeline construction (Stage A/B/C) is deferred until Protocol 3B on FLIR ADAS v2 confirms a gap ≥ 0.15 mAP@50-95. If no combination of dataset + generation method reaches this threshold, the thesis must pivot to T3 (negative/diagnostic result). See `docs/gap_establishment.md`.

### 2.5 Future weather-generation directions

- **Method 4 — CycleDiff:** hybrid CycleGAN + diffusion model for unpaired IR weather translation. Trained on FLIR ADAS v2 (source) → DSERT-RoLL (fog, rain) + Heavy_Snow (target). Architecture design in progress.
- **Expand conditions beyond rain/fog:** snow, haze, low-light, sensor noise. Snow is included in the FLIR ADAS v2 generation plan.
- **Continue evaluating classical, depth-based, and CycleGAN approaches** where applicable.

---

## 3. The thesis, restated

The original framing ("weather degrades thermal detection, so enhance per-condition to recover accuracy") is weakened by the evidence. Candidate replacements, best first:

**T1 — Efficiency / Pareto (recommended, but gated on gap establishment).**
"Hard weather-gating matches all-in-one restoration accuracy at a fraction of the edge latency, by skipping enhancement entirely on the majority of frames."
Wins on the accuracy-latency Pareto frontier rather than on mAP. Survives saturation, because parity plus a genuine latency win is a real result. Requires the efficiency table and gate-error robustness to be strong. **This thesis is only viable if a Protocol 3B gap ≥ 0.15 is first confirmed on FLIR ADAS v2.** If the maximum achievable gap is < 0.10, the methodology cannot be justified.

**T2 — Gate-error robustness.**
"Condition-specific routing degrades gracefully under gate misclassification, unlike soft-routing MoE restorers." This is the vulnerability reviewers probe hardest; owning it is a contribution. Gated on T1 being viable.

**T3 — Negative/diagnostic result (fallback, still publishable).**
"Weather-adaptive enhancement does not help thermal detection, and here is the controlled evidence."
Consistent with the IA-YOLO / BAD-Net / D-YOLO literature, which repeatedly finds that cascading restoration before detection hurts. A careful negative result on real weather data has value. **This becomes the default thesis if gap establishment fails (no method reaches 0.15 gap on FLIR ADAS v2).**

**Go/no-go:** if GATE-IR cannot dominate the Pareto frontier against a soft-router and an all-in-one restorer, T1 fails and we fall back to T3 rather than forcing an accuracy claim. If no dataset + generation method produces a gap ≥ 0.15, T3 is the only defensible thesis and pipeline construction becomes a "build it and show it doesn't help" study.

---

## 4. Assets and established facts

**Data.** `data/UNIRI-TID/` — 6,340 JPEGs at 1280×960; 6,276 annotated; single class `Human` (no `Dog`, despite the abstract); ~2.2 boxes/image; COCO+VOC+YOLO annotations all present.

> **Important and easy to miss:** the images are **pseudo-colored 8-bit RGB**, not grayscale or 14-bit raw radiometric data (blue = cold, red = warm). GATE-IR's "thermal variance / entropy" features therefore operate on a *palette-mapped* image, not on temperature. The palette is non-monotonic, so grayscale conversion distorts temperature ordering. `bit_depth=14` and `ch: 1` in the spec are both wrong for this data, and README ablation G1 (8- vs 14-bit) is not runnable. The upside: 3-channel input loads COCO-pretrained weights with no stem surgery.

**Sequence recovery (done).** Frames carry a **burned-in on-screen counter**, read by glyph template matching (`counter.csv`, `counter_templates.json`; 5,972/6,340 = 94% readable) to give true temporal order and **59 sequences** (`sequence_index.csv`). Validated by monotonicity: within a recording the counter advances by +1…+6, with only 24 resets and 23 large jumps. Median frame-to-frame difference is **0.0012**, i.e. consecutive frames are ~99.9% identical, which is why sequence-disjoint splitting is mandatory rather than merely tidy.

**Weather labels (RESOLVED — was the blocker).** Recovered to `conditions.csv` from three independent lines of evidence:
1. Archive ordering: flat numbering grouped by scenario → seqs 0–22 clear, 23–31 fog, 32–58 rain.
2. Camera distance (companion paper Table 2): fog 0–30 m, clear 50–160 m, rain 30–215 m.
3. The dog appears inside the fog block, matching the paper's statement.

**Splits (rebuilt, condition-stratified).** Sequence-disjoint *and* stratified per condition: train 3,939 (39 seqs) / val 1,103 (8) / test 1,234 (12), YOLO tree at `data/UNIRI-TID/yolo/` (`nc: 1`, `names: ['Human']`). Per-condition — train 1,716/630/1,593, val 415/310/378, test 535/195/504 (clear/fog/rain).

> **The circularity trap.** Recovering labels by clustering on image statistics is fatal: WeatherGate's five features *are* image statistics. Labels derived from those features, then used to train a classifier on those features, give ~100% accuracy **by construction**. Labels must be human-marked or author-supplied.

**Environment.** Conda env `gateir` under `$HOME/.conda/envs`: torch 2.14.0+cu126, ultralytics 8.4.138, cv2 5.0.0, sklearn 1.9.0. `/opt/miniforge/envs` is not writable. `unzip` is not installed — use Python `zipfile`.

**Compute.** Node `ub`, 6× RTX 6000 Ada. QOS `normal`: **1 GPU, 1 running job, 24 h max**. Pattern: one long `salloc` inside `tmux` running experiments sequentially.

**Secondary dataset.** FLIR ADAS v2 (Kaggle: `samdazel/teledyne-flir-adas-thermal-dataset-v2`, paired IR+RGB, 15 classes, ~26k images) — **not yet downloaded or inspected.** Target for Method 3 (CycleGAN) and Method 4 (CycleDiff) generation, plus the eventual GATE-IR testbed. If the dataset contains real weather conditions, the natural gap is measured first. If not, fog, rain, and snow are synthesized via CycleGAN and CycleDiff using DSERT-RoLL (fog, rain) and Heavy_Snow (snow) as target domains.

**Synthetic weather generation (M3FD — negative result).** Three methods evaluated and verified on M3FD with a clean chunk-disjoint split:
1. Classical uniform veil (fog) + Gaussian blur (rain) — Albumentations/OpenCV.
2. Depth-based Koschmieder fog via Depth-Anything-V2 — IR-specific β_IR = 0.3β, BG=128, label filtering at t_IR < 0.1.
3. CycleGAN translation trained on DSERT-RoLL real thermal fog/rain — ResNet-9 generator, PatchGAN discriminator, 50 epochs/126k iters each.

All three methods failed to sustain a Protocol 3B gap ≥ 0.05 on M3FD. The M3FD synthetic-weather track is a negative result. It is not re-entered unless a new generation method (CycleDiff) first demonstrates a 3B gap ≥ 0.15 on a different base dataset.

**FLIR ADAS v2 — planned.** Not yet downloaded. Planned methods:
- Method 3 (CycleGAN): unpaired IR translation using DSERT-RoLL target domains.
- Method 4 (CycleDiff): hybrid CycleGAN + diffusion model for IR weather translation.

---

## 5. Milestones

### M0 — Foundations ✅
Env built; UNIRI-TID extracted and reconciled; counter OCR'd; 59 sequences recovered; sequence-disjoint splits verified; YOLO tree emitted; companion paper obtained and read.

### M1 — Weather labels + harness + first numbers ✅
Condition labelling → `conditions.csv`. Sequence-disjoint, condition-stratified splits. Working train→eval path (`bench/train_detector.py` + `bench/eval/detect_eval.py`). Overnight fine-tunes: 7-model sweep (v8s, v11s, v12s, v8s_weather, v10s, v8s_p2, rtdetrl). RT-DETR-l invalid (29 epochs, 0.28 mAP) — needs rerun.

### M2 — Establish whether weather actually hurts ✅
Yes, fog genuinely degrades detection at strict IoU, and it is not the camera-distance confound. Fog has the largest median object (75.4 px) yet the worst per-condition mAP@50-95. At matched object size (medium bucket, 89–97% of boxes) fog costs 0.09–0.13 mAP@50-95 consistently across all architectures.

### M2.5 — Synthetic weather dataset on M3FD ✅ COMPLETE (negative result)
All three methods evaluated on a clean chunk-disjoint split (split_hash `effb9b3090ad2fcb`). None sustain a Protocol 3B gap ≥ 0.05 (fog gap +0.049 WEAK, rain gap +0.011 FAIL). Results verified, archived contaminated results moved to `_old/`. Detailed results in `m3fd_experiments/methods_report.tex` and `m3fd_experiments/_old/exam_prep.md`. The M3FD synthetic-weather track is a negative result. The project does not re-enter this track unless a new generation method (CycleDiff) first demonstrates a 3B gap ≥ 0.15 on a different base dataset.

### M3 — Gap establishment on FLIR ADAS v2 (precedes pipeline construction)
Before any Stage A/B/C code is written, confirm a Protocol 3B gap ≥ 0.15 mAP@50-95 on FLIR ADAS v2:
- Download and inspect FLIR ADAS v2 (paired IR+RGB, 15 classes, ~26k images). ✅ DONE
  - Weather: 9617 unknown, 623 clear, 432 partly_cloudy, 69 overcast, 1 snowy (train)
  - Val: 1018 unknown, 113 clear, 8 partly_cloudy, 5 overcast
  - Test video: 3749 unknown
  - No real adverse weather → synthetic generation required.
- Acquire DSERT-RoLL (Fog, Heavy_Rain, Light_Rain) + Heavy_Snow images for target domains. ✅ Fog/Rain DONE
- **Methods 1 and 2 are SKIPPED** for FLIR ADAS v2 — they produced negative/negligible results on M3FD
  (max gap +0.049 fog on Method 3, Methods 1/2 structurally insufficient). Focus directly on:
  - Method 3 (CycleGAN): unpaired IR translation using DSERT-RoLL + Heavy_Snow target domains.
  - Method 4 (CycleDiff): hybrid CycleGAN + diffusion model for fog, rain, snow generation.
- Run Protocol 3A and 3B for Method 3 (CycleGAN) and Method 4 (CycleDiff).
- Gap decision: if any method + condition produces Protocol 3B gap ≥ 0.15, proceed to M4 (pipeline construction). If no method reaches 0.15, pivot to T3 (negative result).

### M4 — GATE-IR Stage A/B/C on FLIR ADAS v2 (only after M3 gap confirmation)
Hold the pipeline fixed and sweep one stage at a time:
- **Gate:** handcrafted MLP over 5 image statistics → Clear / Fog / Rain / Snow.
- **Fog:** adaptive gamma vs CLAHE vs AGCWD vs one deep dehazer (DEA-Net).
- **Rain:** LSRB vs a lightweight rival (DRT) vs a heavy SOTA (DRSformer/Restormer).
- **Snow:** new enhancer branch (emissivity wash-out, convective cooling).
- **Detector:** YOLOv8s vs +P2 vs +ViT neck vs BiFPN vs YOLOv10/11/12 vs RT-DETRv2 or D-FINE.

### M5 — Gating-specific evidence (the core of T1/T2)
Oracle vs no-gate vs GATE-IR; always-apply-both-enhancers; gate-error sensitivity; gate latency as a fraction of total pipeline latency; head-to-head against an all-in-one restorer on both accuracy and compute.

### M6 — Efficiency and the edge story
Per-stage latency breakdown; expected latency under gating versus always-on restoration; TensorRT FP16/INT8; accuracy-latency Pareto plot. Jetson Orin numbers if hardware can be borrowed.

### M7 — Generalization, then write-up
Cross-dataset transfer; real-vs-synthetic weather gap; out-of-taxonomy inputs (sensor noise, dead pixels); severity sweeps; failure-mode analysis with qualitative examples. Then the paper.

---

## 6. Deferred: CycleGAN + cross-modal distillation (§3.4)

Deferred. Decision is over-determined:
1. Most expensive component — 20–40 h at 200 epochs, exceeding 24 h walltime.
2. Most attackable — FreqKD (2026) shows naive RGB→IR feature mimicry failing to beat baseline.
3. UNIRI-TID has no RGB at all.
4. A + B + C is a complete story without it.

If it returns, M3FD serves it independently, and an InfraNet/FreqKD-style train-only RGB-guided scheme is the better design than CycleGAN pseudo-RGB.

---

## 7. Future directions (beyond current plan)

- **Cross-domain transfer:** After the thesis is locked on FLIR ADAS v2, test whether the gated pipeline
  transfers to UNIRI-TID (ground-level, single-class, real weather) or DroneVehicle (UAV overhead,
  vehicle classes). Cross-domain transfer is a Phase-5 activity, not the immediate next step.
- **CycleDiff refinement:** Once the basic CycleDiff hybrid (CycleGAN + diffusion) is validated on
  FLIR ADAS v2, explore architecture improvements (dual-stream U-Nets with cross-modal attention,
  TeV decomposition for IR, PatchNCE for alignment). Full spec in `docs/generating_weather_conditions.md`.
- **Expand weather conditions:** haze, low-light, sensor noise — the gate has no class for these, so
  what does it do?
- **Per-class analysis:** FLIR ADAS v2's 15 classes enable richer per-class AP breakdowns than
  UNIRI-TID's single class.

---

## 8. Standing methodology rules

These exist because violating any one of them invalidates results downstream.

1. **Gap establishment precedes pipeline construction.** Do not write Stage A/B/C code until a Protocol 3B gap ≥ 0.15 mAP@50-95 is confirmed on the target dataset. Building the pipeline first assumes the gap exists.
2. **Sequence-disjoint splits, always.** If mAP@50 exceeds ~0.95 on a new run, suspect leakage before celebrating.
3. **Report mAP@50-95 and APm.** mAP@50 is saturated on this benchmark and hides everything.
4. **One eval function.** Everything must route through one `detect_eval.py`, cross-checked once against `pycocotools`.
5. **No circular labels.** Assert the condition-label source is human/author-derived, never clustered from WeatherGate's features.
6. **Restoration is judged by mAP, not PSNR.** The task-driven literature exists precisely because higher PSNR does not imply higher mAP.
7. **Timing hygiene.** CUDA warmup, `torch.cuda.synchronize()` around timed regions, ≥3 repeats as mean±std.
8. **Multi-seed.** ≥3 seeds with mean±std for headline numbers.
9. **Report negatives.** If LSRB hurts, or the ViT neck gains nothing, that goes in the paper.
10. **Protocol B is the metric that matters.** Always run Protocol B. A synthetic fog that passes Protocol A can still collapse in Protocol B (confirmed with Methods 1 and 2 on M3FD; Method 3 fog survived weakly at +0.049, rain collapsed to +0.011).

---

## 9. Known problems and drawbacks

### 9.1 Threats to the core thesis

- **`[S]` The benchmark is saturated.** Kristo et al. already report ~98% AP@50 on UNIRI-TID in every weather condition.
- **`[S]` The premise appears false on our own data.** GATE-IR opens with "adverse weather severely degrades thermal detection." The published baseline shows rain 98.08 > clear 97.85 — rain is better than clear. Only fog shows genuine contrast loss.
- **`[S]` The rain branch may be solving a non-problem.** If rain improves contrast, LSRB deraining can only remove signal.
- **`[M]` "Why gate at all?"** All-in-one restoration is the dominant competing paradigm. We must show hard gating buys something — latency is the plausible answer, accuracy is not.
- **`[M]` Restoration-before-detection is documented to hurt.** IA-YOLO, D-YOLO, TogetherNet and BAD-Net all find that cascading an off-the-shelf restorer before a detector often fails to help.
- **`[M]` Robustness-by-training may beat the whole pipeline.** A detector simply trained with weather augmentation and no preprocessing is the cheapest rival.

### 9.2 Novelty and prior art

- **`[S]` Near-identical prior art exists and must be cited.** DTRDNet, Aloufi et al., WM-MoE.
- **`[M]` The "naive combination" objection is real.** GATE-IR reads as engineering integration rather than contribution. Needs one defensible thesis, proven, not five components asserted.
- **`[M]` Comparing only against YOLOv8 is outdated in 2026.** Reviewers will expect YOLOv10/11/12 and at least one DETR-family detector.
- **`[M]` Adaptive gamma is thin as a "fog enhancer".** Zero learnable parameters.
- **`[M]` The ViT neck is incremental unless isolated.** Transformer necks are well-trodden.

### 9.3 Dataset problems

- **`[S]` Condition is confounded with camera distance.** Fog 0–30 m, clear 50–160 m, rain 30–215 m.
- **`[S]` Weather is confounded with recording session.** Every fog frame comes from a single ~13-minute shoot.
- **`[S]` Fog has only 9 sequences.** With whole-sequence granularity that is the practical floor.
- **`[M]` One of the two shipped annotation exports is silently broken.** Use `All_In_One_Anot_*` exclusively.
- **`[M]` The images are pseudo-colored, not radiometric.** 8-bit 3-channel JPEGs with a colour palette.
- **`[M]` Video-frame leakage.** Adjacent frames are near-duplicates. Sequence-disjoint splitting is mandatory.
- **`[M]` Narrow domain.** Single site, single camera, night-only, staged actors, single class `Human`.
- **`[L]` No out-of-taxonomy condition.** The gate has three classes and no "snow", "low-light", "sensor noise".

### 9.4 Specification and code defects

- **`[S]` Nothing in the repo is runnable on real data.** No dataloader, no detector training loop, no NMS/box decode, no mAP evaluation. `main.py --mode demo` runs on `torch.rand`.
- **`[M]` CLAHE.forward train/eval operator mismatch.** `preprocessing/rain_remover.py` uses differentiable LCN when `training=True` but OpenCV CLAHE when `False`.
- **`[M]` `theory.md` §2.3's weather-signature table is asserted, never measured.**
- **`[M]` `theory.md` §2.1 cites He et al. CVPR 2009 — the Dark Channel Prior paper — for the thermal fog model.** That is an RGB haze paper.
- **`[M]` Spec/data mismatch on input format.** The code assumes 14-bit single-channel raw; the data is 8-bit 3-channel.
- **`[L]` Stage B modules are very small.** FogEnhancer has zero learnable parameters; LSRB has ~1.9k.

### 9.5 Evaluation and methodology hazards

- **`[S]` One scoring function, or the comparison is meaningless.** Everything must route through one `detect_eval.py`.
- **`[M]` No confidence intervals or multi-seed runs.**
- **`[M]` Missing bounds.** Oracle-gate and no-gate lower bound are mandatory.
- **`[M]` Gate-error sensitivity is unmeasured.**
- **`[M]` Real-time claims need a named edge device.** Desktop FPS is discounted.
- **`[L]` Class imbalance and object size.** Single class on UNIRI-TID; AP-small must be reported.

### 9.6 Compute and logistics

- **`[S]` One GPU, one job, 24 h, serial.** Realistic throughput is ~4–6 training runs per day.
- **`[M]` No escape hatch on compute.** The 3-GPU research QOS is not available to students.
- **`[L]` Environment quirks, already solved:** `unzip` absent; `/opt/miniforge/envs` not writable.

### 9.7 M3FD-specific findings

- **Split contamination identified and fixed.** Original `split_dataset.py` did a random frame-level shuffle of sequential video. Fixed by `resplit_m3fd.py` with 60-frame temporal chunks, stochastic search, 16-frame boundary drops, IoU leakage check.
- **All contaminated results archived to `m3fd_experiments/_old/`.**
- **CycleGAN 3A fog gap (+0.477) is a red flag.** The clear-only model scores only 0.037 mAP@50-95 on translated fog — possible mode collapse, domain gap, or sensor-identity transfer. Needs visual inspection of sample grids.
- **Method 2 rain IR = Method 1 rain IR** (byte-identical). Depth-aware rain was RGB-only.
- **RGB generators trained but unused.** `rgb_fog` and `rgb_rain` complete 50 epochs but all eval YAMLs point to IR paths.
- **Label filtering removes hardest cases.** Boxes with mean t_IR < 0.1 are removed from both training and test fog, making evaluation easier.
- **All three methods failed Protocol 3B.** No method sustains a gap ≥ 0.05. The M3FD synthetic-weather track is a negative result.

### 9.8 FLIR ADAS v2 — unknown properties

- **Dataset not yet downloaded.** All properties (weather labels, splits, class balance, resolution) are UNKNOWN until inspection.
- **Snow physics differ from fog/rain.** Snow causes emissivity wash-out and convective cooling in IR, not just contrast loss. A generation model that treats snow as "white fog" will produce unrealistic outputs and a small or negative gap.
- **CycleDiff is unproven.** It may produce mode collapse, sensor-identity transfer, or training instability. Inspect sample grids and measure Protocol 3B before treating it as a successful generation method.
- **Gap-first doctrine.** Pipeline construction (Stage A/B/C) is deferred until a Protocol 3B gap ≥ 0.15 is confirmed. If no method reaches this threshold, the thesis pivots to T3.

---

## 10. Immediate next actions

### Phase 1 — Gap Establishment (gates everything else)

1. **Download FLIR ADAS v2** from Kaggle. Inspect dataset structure, classes, splits, and any weather metadata. If real weather conditions exist, train a baseline and measure the natural clear-vs-adverse gap.
2. **Acquire DSERT-RoLL Heavy_Snow** images for the snow target domain. Inspect and integrate with the existing DSERT-RoLL fog/rain data.
3. **Design and implement Method 4 (CycleDiff).** Architecture design in progress. See `docs/generating_weather_conditions.md` for the target specification. Train on FLIR ADAS v2 (source) → DSERT-RoLL (fog, rain) + Heavy_Snow (target).
4. **Run Protocol 3A and 3B** on FLIR ADAS v2 for Methods 3 (CycleGAN) and 4 (CycleDiff). Measure the gap for each condition.
5. **Gap decision:** if any method + condition produces a Protocol 3B gap ≥ 0.15, proceed to Phase 2 (pipeline construction). If no method reaches 0.15, pivot to T3 (negative result).

### Phase 2 — Pipeline Construction (only after Phase 1 gap confirmation)

6. Build Stage A (WeatherGate), Stage B (enhancers), Stage C (detector) on FLIR ADAS v2.
7. Run full pipeline evaluation vs baselines + oracle/no-gate bounds + multi-seed.

### Phase 3 — Thesis lock + write-up

8. Compile results, write paper.

> **Slurm caution.** Run work *inside* the existing allocation with `srun --jobid=<id> --overlap`, and stop runs by killing the Python PID. **Never `scancel` the job itself**.

---

## 11. Open questions for Preetham

- Gap-first doctrine: agree that pipeline construction (Stage A/B/C) is deferred until a Protocol 3B gap ≥ 0.15 is confirmed on FLIR ADAS v2?
- Which thesis: **T1** (efficiency/Pareto), **T2** (gate robustness), or accept **T3** (negative result)?
- Target venue and deadline?
- Is a Jetson Orin obtainable?
- Should Aryan Sood be looped in before Stage C is re-expressed through Ultralytics?
- CycleDiff design: start from Stable Diffusion, or use a lighter alternative? See `docs/generating_weather_conditions.md` for the target architecture.
- FLIR ADAS v2: confirm the Kaggle access token is available and the dataset can be downloaded within cluster constraints.
- DSERT-RoLL Heavy_Snow: confirm availability and licensing before committing to the snow condition.
