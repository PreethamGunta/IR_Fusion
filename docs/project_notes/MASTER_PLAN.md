# GATE-IR — Master Plan

**Status:** M3FD synthetic weather experiments complete. M2.5 clean results on disk. Next: larger IR dataset (DroneVehicle) + diffusion-based weather generation.
**Last updated:** 2026-10-06
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

M3FD is essentially complete as a synthetic-weather testbed. The next major dataset direction is a larger IR dataset such as **DroneVehicle** (UAV overhead, vehicle classes, multi-spectral). This provides:
- Larger scale for stronger baseline detectors
- Different viewpoint (overhead vs ground-level) for cross-domain generalisation
- More classes for richer AP tables

### 2.5 Future weather-generation directions

- **Diffusion-based approaches:** Physics-Informed Diffusion (PID) with parallel RGB+IR U-Nets and cross-modal attention. Full architecture spec in `docs/generating_weather_conditions.md`.
- **Expand conditions beyond rain/fog:** snow, haze, low-light, sensor noise.
- **Continue evaluating classical, depth-based, CycleGAN approaches** where applicable.

---

## 3. The thesis, restated

The original framing ("weather degrades thermal detection, so enhance per-condition to recover accuracy") is weakened by the evidence. Candidate replacements, best first:

**T1 — Efficiency / Pareto (recommended).**
"Hard weather-gating matches all-in-one restoration accuracy at a fraction of the edge latency, by skipping enhancement entirely on the majority of frames."
Wins on the accuracy-latency Pareto frontier rather than on mAP. Survives saturation, because parity plus a genuine latency win is a real result. Requires the efficiency table and gate-error robustness to be strong.

**T2 — Gate-error robustness.**
"Condition-specific routing degrades gracefully under gate misclassification, unlike soft-routing MoE restorers." This is the vulnerability reviewers probe hardest; owning it is a contribution.

**T3 — Negative/diagnostic result (fallback, still publishable).**
"Weather-adaptive enhancement does not help thermal detection, and here is the controlled evidence."
Consistent with the IA-YOLO / BAD-Net / D-YOLO literature, which repeatedly finds that cascading restoration before detection hurts. A careful negative result on real weather data has value.

**Go/no-go:** if GATE-IR cannot dominate the Pareto frontier against a soft-router and an all-in-one restorer, T1 fails and we fall back to T3 rather than forcing an accuracy claim.

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

**Secondary dataset.** M3FD (`nonameplease/M3FD_Detecion`, HF, 2.2 GB IR + YOLO labels, 4,200 imgs, 6 classes, paired visible) — for cross-dataset generalization, class diversity, and paired RGB for future diffusion/KD work. **All three synthetic weather methods have been evaluated and verified on M3FD with a clean chunk-disjoint split.**

**Synthetic weather generation (completed on M3FD).** Three methods evaluated:
1. Classical uniform veil (fog) + Gaussian blur (rain) — Albumentations/OpenCV.
2. Depth-based Koschmieder fog via Depth-Anything-V2 — IR-specific β_IR = 0.3β, BG=128, label filtering at t_IR < 0.1.
3. CycleGAN translation trained on DSERT-RoLL real thermal fog/rain — ResNet-9 generator, PatchGAN discriminator, 50 epochs/126k iters each.

All results on disk in `m3fd_experiments/method{1,2,3}_*/results/` and `m3fd_experiments/results/`. Provenance: every result JSON carries `split_hash=effb9b3090ad2fcb` and `weights_sha256`.

---

## 5. Milestones

### M0 — Foundations ✅
Env built; UNIRI-TID extracted and reconciled; counter OCR'd; 59 sequences recovered; sequence-disjoint splits verified; YOLO tree emitted; companion paper obtained and read.

### M1 — Weather labels + harness + first numbers ✅
Condition labelling → `conditions.csv`. Sequence-disjoint, condition-stratified splits. Working train→eval path (`bench/train_detector.py` + `bench/eval/detect_eval.py`). Overnight fine-tunes: 7-model sweep (v8s, v11s, v12s, v8s_weather, v10s, v8s_p2, rtdetrl). RT-DETR-l invalid (29 epochs, 0.28 mAP) — needs rerun.

### M2 — Establish whether weather actually hurts ✅
Yes, fog genuinely degrades detection at strict IoU, and it is not the camera-distance confound. Fog has the largest median object (75.4 px) yet the worst per-condition mAP@50-95. At matched object size (medium bucket, 89–97% of boxes) fog costs 0.09–0.13 mAP@50-95 consistently across all architectures.

### M2.5 — Synthetic weather dataset on M3FD ✅ COMPLETE
All three methods evaluated on a clean chunk-disjoint split (split_hash `effb9b3090ad2fcb`). Results verified, archived contaminated results moved to `_old/`. Detailed results in `m3fd_experiments/methods_report.tex` and `m3fd_experiments/_old/exam_prep.md`.

### M3 — Stage-wise baseline swaps (UNIRI-TID)
Hold the pipeline fixed and sweep one stage at a time:
- **Gate:** handcrafted MLP vs small CNN vs CLIP zero-shot vs no-gate vs oracle-gate vs soft-router.
- **Fog:** adaptive gamma vs CLAHE vs AGCWD vs one deep dehazer (DEA-Net).
- **Rain:** LSRB vs a lightweight rival (DRT) vs a heavy SOTA (DRSformer/Restormer).
- **Detector:** YOLOv8s vs +P2 vs +ViT neck vs BiFPN vs YOLOv10/11/12 vs RT-DETRv2 or D-FINE.

### M4 — Gating-specific evidence (the core of T1/T2)
Oracle vs no-gate vs GATE-IR; always-apply-both-enhancers; gate-error sensitivity; gate latency as a fraction of total pipeline latency; head-to-head against an all-in-one restorer on both accuracy and compute.

### M5 — Efficiency and the edge story
Per-stage latency breakdown; expected latency under gating versus always-on restoration; TensorRT FP16/INT8; accuracy-latency Pareto plot. Jetson Orin numbers if hardware can be borrowed.

### M6 — Generalization, then write-up
Cross-dataset transfer (UNIRI-TID → M3FD / DroneVehicle / LLVIP); real-vs-synthetic weather gap; out-of-taxonomy inputs (snow, sensor noise, dead pixels); severity sweeps; failure-mode analysis with qualitative examples. Then the paper.

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

- **Larger IR dataset:** DroneVehicle (UAV overhead, vehicle classes, multi-spectral) for stronger baselines and cross-domain generalisation.
- **Diffusion-based weather generation:** Physics-Informed Diffusion (PID) with parallel RGB+IR U-Nets and cross-modal attention. Full architecture in `docs/generating_weather_conditions.md`.
- **Expand weather conditions:** snow, haze, low-light, sensor noise — the gate has no class for these, so what does it do?
- **Per-class analysis:** M3FD's 6 classes (People, Car, Bus, Motorcycle, Lamp, Truck) enable richer AP breakdowns than UNIRI-TID's single class.

---

## 8. Standing methodology rules

These exist because violating any one of them invalidates results downstream.

1. **Sequence-disjoint splits, always.** If mAP@50 exceeds ~0.95 on a new run, suspect leakage before celebrating.
2. **Report mAP@50-95 and APm.** mAP@50 is saturated on this benchmark and hides everything.
3. **One eval function.** Everything must route through one `detect_eval.py`, cross-checked once against `pycocotools`.
4. **No circular labels.** Assert the condition-label source is human/author-derived, never clustered from WeatherGate's features.
5. **Restoration is judged by mAP, not PSNR.** The task-driven literature exists precisely because higher PSNR does not imply higher mAP.
6. **Timing hygiene.** CUDA warmup, `torch.cuda.synchronize()` around timed regions, ≥3 repeats as mean±std.
7. **Multi-seed.** ≥3 seeds with mean±std for headline numbers.
8. **Report negatives.** If LSRB hurts, or the ViT neck gains nothing, that goes in the paper.

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

---

## 10. Immediate next actions

1. **E1 — Efficiency table** (params, GFLOPs, FP16 latency, FPS, peak memory). T1 lives or dies on these numbers.
2. **Rerun RT-DETR-l.** Current result is broken (29 epochs, 0.28 mAP).
3. **Decide: proceed with UNIRI-TID Stage A/B/C, or pivot to DroneVehicle?** If UNIRI-TID's gap is enough, defer M2.5 to M6. If a larger dataset is needed, start DroneVehicle.
4. **Build Stage A — WeatherGate.** Run E2 (gate accuracy) and E3 (gate confound test).
5. **E4 — Oracle-gate / no-gate bounds.** Mandatory.
6. **Inspect CycleGAN sample grids** to diagnose the anomalous 3A fog gap (+0.477).
7. **Investigate diffusion-based weather generation** for the next dataset (see `docs/generating_weather_conditions.md`).

> **Slurm caution.** Run work *inside* the existing allocation with `srun --jobid=<id> --overlap`, and stop runs by killing the Python PID. **Never `scancel` the job itself**.

---

## 11. Open questions for Preetham

- Given the M3FD results, which thesis do we commit to — **T1 (efficiency/Pareto)**, **T2 (gate robustness)**, or accept **T3 (negative result)**?
- Target venue and deadline?
- Is any edge device (Jetson Orin) obtainable?
- Should the collaborator (Aryan Sood) be looped in before Stage C is re-expressed through Ultralytics?
- Should the next dataset be **DroneVehicle** (larger, multi-class, UAV) or continue with UNIRI-TID?
