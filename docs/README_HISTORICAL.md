# GATE-IR — Current State & Working README

**Last verified:** 2026-11-04 (direct filesystem, git, SLURM, and result-JSON inspection)
**Repo:** `/home2/home/preetham_g/Gate_IR` (branch `main`, 15 commits ahead of `origin/main`)
**Purpose of this file:** stop the next session from re-discovering the layout, the numbers, and the traps. Read this before exploring. Deeper history lives in `docs/`.

> This file replaces the old root README as the "start here" document. The original UNIRI-TID harness README is preserved at `docs/README_UNIRI_HARNESS.md`.

---

## 0. TL;DR

- The **core GATE-IR pipeline (Stage A/B/C) has never been built or run.** Everything on disk is baselines and synthetic-weather studies.
- **UNIRI-TID baselines are complete** (7-model sweep, 6 valid). **M3FD synthetic-weather experiments are complete** (Methods 1/2/3). **FLIR ADAS v2 fog experiments are complete for the detector runs**, but the clear-vs-fog comparison is currently **invalid** because clear and fog were evaluated on different test sets.
- **The worktree `recover-session-12` is mostly stale, but not entirely.** It still holds ~20 GB of DSERT snow data and the only git-tracked history of the FLIR code. Do not delete it without preserving those.
- **Nothing can be re-run from the FLIR YAMLs as-is**: every YAML and result JSON still points at the deleted worktree path.
- **SLURM job `9639`** is the live allocation (1 GPU, 16 CPUs, 48 GB, 1-day limit). Use `srun --jobid=9639 --overlap ...`. Never cancel the allocation.
- **`KAGGLE_TOKEN` is committed to git history** in commit `6a5f15b` ("hi"). Rotate it and scrub it before pushing.

### 0.1 Run status (paused 2026-11-09 14:45)

- **Phase 1 — complete.** CycleGAN filter screen (YOLOv8s 60) + main comparison (YOLOv12s 100, both arms).
- **Phase 2 — partial.** CycleDiff-75: B1 trained to early stop (65 epochs) and evaluated. G1 with the CycleDiff-trained U-Net stopped at **epoch 27/100**, `last.pt` saved and resumable; its filtered fog images remain in `data/filtered/unet_cd75` (1.2 GB).
- **Phase 3 — stopped** at B3 training (`runs/b3_unet_yolov8s_40` is partial); no B3/B0/B2 result JSONs.
- SLURM job **9743** released. Resume recipe: `flir_gate_experiments/results/RUN_STATUS.md`.
- Results: `flir_gate_experiments/results/REPORT.md` and `all_results.md`.

**Headline — YOLOv12s, 100-epoch budget (shared early stop, patience 25), matched 3,749 VTT frames, CycleGAN fog:**

| Arm | Clear mAP50 | Fog mAP50 | Clear mAP50-95 | Fog mAP50-95 | Gap (50-95) |
|---|---:|---:|---:|---:|---:|
| B1 — clear + unfiltered fog | 0.2849 | 0.2430 | 0.1525 | 0.1266 | 0.0259 |
| **G1 — clear + U-Net-dehazed fog** | **0.3063** | **0.2856** | **0.1683** | **0.1535** | **0.0148** |
| Δ | +0.0214 | **+0.0426 (+17.5 %)** | +0.0158 | **+0.0269 (+21.3 %)** | **−42.9 %** |

Both arms early-stopped: B1 at epoch 55 (best 30), G1 at epoch 57 (best 32). The U-Net filter also raises **clear** mAP, so the gain is not a clear/fog tradeoff.

**CycleDiff-75, YOLOv12s 100-epoch budget — B1 complete, G1 pending:**

| Arm | Clear mAP50 | Fog mAP50 | Clear mAP50-95 | Fog mAP50-95 | Gap (50-95) |
|---|---:|---:|---:|---:|---:|
| B1 — clear + unfiltered fog | 0.2746 | 0.2747 | 0.1491 | 0.1495 | −0.0004 |

CycleDiff-75 B1 is fog-robust (fog marginally *above* clear). The earlier CycleGAN-trained-U-Net transfer test (YOLOv8s, 40 epochs) **hurt** on CycleDiff-75 (−0.0107 fog, −0.0072 clear), which is why the pending G1 arm retrains the U-Net on target-generator pairs.

**Historical note:** the old "clear 0.3274 vs fog 0.1445" gap was mostly domain shift (in-domain 1,200-frame clear set vs out-of-domain 3,749-frame VTT fog). On matched VTT frames the real synthetic-fog gap is ~0.02.

---

## 1. Project goal

### 1.1 Long-term thesis

GATE-IR: a three-stage thermal detection pipeline for adverse weather.

| Stage | Component | Purpose |
|---|---|---|
| **A — Gate** | WeatherGate: small MLP over 5 image statistics | Classify Clear / Fog / Rain per frame |
| **B — Enhance** | Fog → adaptive gamma/AGCWD; Rain → LSRB + CLAHE; Clear → passthrough | Condition-specific cleanup, skipped on clear frames |
| **C — Detect** | YOLOv8s + P2 head + ViT neck | Object detection |

**Thesis direction: T1 — Efficiency / Pareto.**
"Hard gating matches all-in-one restoration accuracy at a fraction of the edge latency, by skipping enhancement entirely on the majority of frames."

Fallbacks:
- **T2 — Gate-error robustness:** graceful degradation under misclassification.
- **T3 — Negative/diagnostic result:** weather-adaptive enhancement does not help thermal detection, with controlled evidence.

The efficiency claim is the only one that survives the saturated UNIRI-TID benchmark. See `docs/MASTER_PLAN.md` §2–3.

### 1.2 Immediate workstream

The recent code is a controlled **synthetic-weather benchmark on FLIR ADAS v2 thermal**:
- Compare **CycleGAN** vs **CycleDiff** fog synthesis.
- Train YOLOv8s detectors under Protocol 3A (clear + synthetic adverse).
- Evaluate on clear vs synthetic-fog test sets.
- Produce a short report (`flir_adas_experiments/reports/report.pdf`).

The next proposed step (not started) is the **GATE-IR method on the FLIR synthetic fog data** — see §9.

---

## 2. Repository map

### 2.1 Top level

| Path | What it is |
|---|---|
| `README.md` | This file — verified current state and working notes |
| `docs/` | Project documentation (plan, handoff, memory, devlog, status) |
| `bench/` | UNIRI-TID baseline harness (data prep, training, scoring) |
| `data/UNIRI-TID/` | UNIRI-TID dataset (images, annotations, recovered conditions, splits) |
| `m3fd_experiments/` | M3FD synthetic-weather experiments (Methods 1/2/3), complete |
| `flir_adas_experiments/` | FLIR ADAS v2 code, data, runs, results, report |
| `flir_gate_experiments/` | New GATE-IR-on-FLIR experiment: binary gate + fog filter + matched detector arms |
| `GATE-IR/` | Original GATE-IR architecture scaffolding (Stage A/B/C modules; no training loop) |
| `weights/` | Pretrained detector weights (`yolov8s.pt`, `yolo11s.pt`, `yolo12s.pt`, `rtdetr-l.pt`, etc.) |
| `.kilo/worktrees/` | Old Kilo worktrees (`recover-session-12`, `abstracted-side`) |
| `_trash/` | Ignored scratch/trash |
| `refs/` | Reference PDFs (e.g. Kristo et al. 2020) |
| `datasets/coco8/` | Tiny Ultralytics sanity dataset |
| `.gitignore` | Ignore rules (currently modified, uncommitted) |
| `opencode.json` | OpenCode config (`snapshots: false`), untracked |

### 2.2 `flir_adas_experiments/` — FLIR ADAS v2

```
flir_adas_experiments/
  data/
    flir_adas_v2/                # 19 GB — canonical data + synthetic + YOLO tree
    flir_adras_v2/               # 4.7 GB — older synthetic copy (cyclegan + cyclediff)
  flir_adras_v2/                 # 53 GB — code, method3/4, runs, results, logs
  reports/                       # report.tex, report.pdf, 12 sample assets
```

#### `flir_adas_experiments/flir_adras_v2/` — code and runs

| Path | Description |
|---|---|
| `train_detector.py` | YOLO training for Protocol 3A/3B. `REPO = flir_adas_experiments`; `DATA = flir_adas_experiments/data/flir_adas_v2`. Defaults to `yolov12s.pt`; actual FLIR runs used `yolov8s.pt`. |
| `eval_detector.py` | THE FLIR scorer. Writes JSON with mAP50-95, mAP50, P, R, weights sha256, data path, split hash, n_images, git provenance. |
| `prepare.py` | FLIR → YOLO conversion + sequence-disjoint split (by video ID). Produces `yolo/thermal/` and `split_manifest.json`. |
| `inspect_dataset.py` | Post-download dataset inspection → `inspect_report.json`. |
| `download.py`, `download_bg.py` | Kaggle download helpers. |
| `convert_video_thermal_test_yolo.py` | Converts VTT test frames/annotations to YOLO labels (`labels_vtt_test`). |
| `blend_cyclediff_fog.py` | Creates the 75 % fog / 25 % clear blend from CycleDiff fog. |
| `run_fog_pipeline.sh` | End-to-end fog pipeline: translate CycleGAN train, blend CycleDiff 75 %, build YAML/symlinks/labels, train + evaluate. **Hardcodes old worktree path.** |
| `run_yolo_full.sh` | Full YOLO training/eval driver. **Hardcodes old worktree path.** |
| `run_yolo_gap.sh` | Gap-establishment driver. **Hardcodes old worktree path.** |
| `run_all_protocols.sh` | Protocol 3A/3B driver. **Hardcodes old worktree path.** |
| `run_evaluation.sh` | 3A/3B evaluation driver. **Hardcodes old worktree path.** |
| `resume_pipeline.sh`, `wait_and_run.sh`, `regen_cyclegan_train.sh` | Resume/queue helpers. **Hardcode old worktree path.** |
| `submit_yolo_gap.sbatch` | SLURM submission wrapper. |
| `run_yolo_full.log` | Log from a full YOLO run. |
| `method1_classical/` | (FLIR) classical fog/rain/snow generation — empty/stub here; the complete version is under `m3fd_experiments/method1_classical/`. |
| `method2_depth/` | (FLIR) depth-based fog — empty/stub here; complete version under `m3fd_experiments/method2_depth/`. |
| `method3_cyclegan/` | CycleGAN translators for FLIR (fog, rain, snow). |
| `method4_cyclediff/` | CycleDiff hybrid (CycleGAN + latent diffusion). |
| `runs/` | YOLO training/eval run directories (ignored by git). |
| `results/` | Result JSONs (5 files, see §4.1). |
| `logs/` | Training/eval logs (ignored by git). |
| `scripts/` | SLURM/srun wrappers and DSERT download scripts. |

##### `method3_cyclegan/` (FLIR)

| Path | Description |
|---|---|
| `train_cyclegan.py` | ResNet-9 generator + PatchGAN discriminator. `REPO` logic assumes the old worktree; `SNOW_ROOT` points at `flir_adas_experiments/data/DSERT-RoLL` (does not exist yet). |
| `translate.py` | Applies a trained generator. `train` source = `images_thermal_train`; `test` source = `video_thermal_test` (VTT). |
| `models.py` | Generator/discriminator/loss definitions (reused from M3FD). |
| `runs/fog/` | Fog translator: `G_AB.pt`, `G_BA.pt`, `DONE`, `samples/` (1,074), `SOURCE.json`, `train.log`. |
| `runs/rain/` | Rain translator: same layout, 348 samples. |
| `runs/snow/` | Snow translator: same layout, 134 samples. `SOURCE.json` points at the worktree's `data/DSERT-RoLL`. |
| `datasets/` | Empty (old FLIR 3A/3B YAMLs were removed). |
| `results/` | Empty (old FLIR 3A/3B results were removed). |

##### `method4_cyclediff/` (FLIR)

| Path | Description |
|---|---|
| `train_uncond_ldm.py`, `train_uncond_ldm_cycle.py`, `train_vae.py` | Latent diffusion / VAE training. |
| `translation_uncond_ldm_cycle.py` | CycleDiff translation/sampling. |
| `compute_lpips.py` | LPIPS metric. |
| `run_train_and_translate.sh`, `submit_train_and_translate.sbatch` | Pipeline wrappers. |
| `configs/`, `ddm/`, `taming/`, `torch_utils/`, `unet/`, `util/` | Model/training library code. |
| `pretrain_weight/` | Pretrained VAE/LDM weights (ignored). |
| `results/`, `sample_results/`, `logs/`, `metrics/` | Outputs (ignored). |
| `retrain.log`, `infer*.log`, `lpips.log` | Run logs. |

#### `flir_adas_experiments/reports/`

| Path | Description |
|---|---|
| `report.tex` | 2-page report: data paths, sample images, detection table. |
| `report.pdf` | Compiled PDF (1.47 MB, valid). |
| `report_assets/` | 12 PNG/JPG sample images: clear boxed, CycleGAN, CycleDiff full, 75 % blend. |

### 2.3 `m3fd_experiments/` — M3FD synthetic weather (complete)

| Path | Description |
|---|---|
| `resplit_m3fd.py` | Chunk-disjoint temporal split (`split_hash=effb9b3090ad2fcb`; 2,520 train / 240 val / 600 test). |
| `train_m3fd.py` | YOLOv12s training, 100 epochs, early stop 25. |
| `eval_m3fd.py` | Evaluation with `--out` and provenance. |
| `provenance.py` | Split hashing and run records. |
| `severity.py` | Severity calibration. |
| `make_3b_dataset.py`, `make_debug_subset.py` | Protocol-3B and debug helpers. |
| `method1_classical/` | Classical fog/rain generation + 6 result JSONs. |
| `method2_depth/` | Depth-Anything-V2 Koschmieder fog + 5 result JSONs. |
| `method3_cyclegan/` | CycleGAN translators (fog, rain; RGB variants), DSERT-RoLL target, results. |
| `results/` | Top-level result JSONs (5). |
| `runs/` | 30 YOLO run/eval directories. |
| `logs/` | SLURM/pipeline logs. |
| `figures/` | Sample grids. |
| `scripts/` | `run_all_methods.sh`, `run_stage5_method2.sh`, `run_remaining_stages.sh`, `submit_all_methods.sbatch`. |
| `methods_report.tex` | Final corrected report (no PDF yet). |
| `data/` | 21 GB — M3FD images, split, synthetic, synthetic_depth. |
| `method3_cyclegan/data/DSERT-RoLL/` | 3.8 GB — Fog + Heavy_Rain + Light_Rain (no snow). |

### 2.4 `bench/` — UNIRI-TID baseline harness

| Path | Description |
|---|---|
| `data/prepare_uniri.py` | Annotations → canonical YOLO labels. |
| `data/read_counter.py` | Glyph-match the burned-in frame counter. |
| `data/build_sequences.py` | Counter → 59 recordings. |
| `data/label_conditions.py` | Recordings → clear/fog/rain. |
| `data/make_splits.py` | Sequence-disjoint, condition-stratified split. |
| `data/make_debug_subset.py` | Tiny mirror for `DEBUG=1`. |
| `eval/detect_eval.py` | THE UNIRI scorer — per-condition + per-size AP via pycocotools. |
| `train_detector.py` | Train one model + score it. |
| `scripts/overnight.sh` | 7-model baseline sweep. |
| `scripts/sweep.sbatch` | Batch submission, resumable. |
| `scripts/estimate_runtime.py` | Per-epoch cost estimate. |
| `results/` | 21 JSONs: 7 models × train/eval. |
| `runs/` | YOLO run dirs. |
| `weights/` | Local copies of pretrained weights. |

### 2.5 `GATE-IR/` — architecture scaffolding

**Important:** this is scaffolding only. `main.py --mode demo` runs on `torch.rand`; there is no dataloader, training loop, mAP evaluation, or trained weights.

| Path | Description |
|---|---|
| `gate/weather_gate.py` | `WeatherGate`: 5 features (variance, min, max, entropy, Laplacian variance) → 3-layer MLP → Clear/Fog/Rain. Assumes 14-bit single-channel RAW; FLIR is 8-bit 3-channel pseudo-colour. |
| `preprocessing/fog_enhancer.py` | `FogEnhancer` (adaptive gamma) + `AdaptiveContrastEnhancer` (gamma + percentile stretch). |
| `preprocessing/rain_remover.py` | `RainRemover` (LSRB + LCN/CLAHE). Known bug: differentiable LCN in train mode, OpenCV CLAHE in eval mode. |
| `preprocessing/weather_router.py` | `WeatherRouter` / `WeatherPipeline`: routes clear→identity, fog→FogEnhancer, rain→RainRemover. |
| `models/yolov8_thermal.py`, `.yaml` | YOLOv8s + P2 + transformer neck scaffolding. |
| `training/cyclegan.py`, `train_cyclegan.py` | IR→pseudo-RGB CycleGAN scaffolding (deferred). |
| `training/distillation.py` | Feature-mimic KD scaffolding (deferred). |
| `main.py` | Demo/test/inference entry point (synthetic data). |
| `docs/theory.md` | Theoretical background. |

### 2.6 `docs/`

| Path | Description |
|---|---|
| `MASTER_PLAN.md` | Long-term project map: claims, strategic situation, milestones M0–M6, deferred KD, known problems. |
| `handoff.md` | "Continue from here": what works, results tables, open problems, next actions. |
| `memory.md` | Durable facts: cluster constraints, environment, dataset facts, conventions, known bugs. |
| `devlog.md` | Session-by-session history (51 KB). Read before proposing new experiments. |
| `generating_weather_conditions.md` | Physics-Informed Diffusion architecture spec for future weather generation. |
| `dataset_generation_plan.md` | Method 1–4 planning + protocol A/B reference. |
| `status_20261030_0624.md` | Snapshot of the FLIR fog session (untracked). |
| `README_UNIRI_HARNESS.md` | Preserved original root README (UNIRI-TID harness). |
| `gap_establishment.md` | **Missing** — referenced by the FLIR README but not present. |

### 2.7 Worktrees

| Path | State |
|---|---|
| `.kilo/worktrees/recover-session-12` | Branch `efd97a1`. `flir_adras_v2/` and `data/flir_adas_v2/` moved out; working tree shows them as deleted. Still holds **20 GB `data/DSERT-RoLL`** (Fog 26 MB, Heavy_Snow 4.1 GB, Light_Snow 16 GB, test/train splits) and a 279 MB old `m3fd_experiments/` copy. **Only git-tracked copy of the FLIR code (941 files).** |
| `.kilo/worktrees/abstracted-side` | 94 MB, old branch, uncommitted M3FD changes. Not current. |

### 2.8 `flir_gate_experiments/` — new GATE-IR-on-FLIR experiment

Self-contained, relative-path experiment. See `flir_gate_experiments/README.md`.

| Path | Description |
|---|---|
| `src/paths.py` | All paths derived from `Path(__file__).resolve().parents[2]`; no hardcoded absolute paths. |
| `src/gate.py` | Binary Clear/Fog gate: 5 GATE-IR features (variance, min, max, entropy, Laplacian variance) + MLP. |
| `src/filters.py` | `gamma` (FogEnhancer), `clahe`, `gamma_clahe`, `dcp` (dark channel prior, box-filter guided refinement). |
| `src/unet_dehazer.py` | Small paired U-Net (fog→clear) + SSIM loss. |
| `src/train_unet.py` | Train the U-Net on (clear, CycleGAN fog) pairs. |
| `src/apply_unet.py` | Apply the trained U-Net to a directory. |
| `src/build_datasets.py` | Builds matched b0/b1/b2/b3/g1 datasets and YAMLs. |
| `src/build_screen.py` | Builds `b1` + one `g1_<filter>` dataset per filter. |
| `src/build_cyclediff.py` | Builds the CycleDiff-75 B1/G1 datasets. |
| `src/build_ablations.py` | Builds B0/B2/B3 datasets for both fog domains. |
| `src/train_detector.py` | YOLO training, one recipe for every arm (`--weights yolo12s.pt` for the main runs). |
| `src/eval_detector.py` | Matched clear/fog evaluation + provenance JSON. |
| `src/route.py` | Materialises `gated_learned` / `gated_oracle` test sets. |
| `src/summarize.py`, `src/summarize_screen.py`, `src/summarize_results.py` | Markdown result tables. |
| `scripts/run_phase1_cyclegan.sh` | CycleGAN prep → 60-epoch filter screen → 100-epoch B1/winner. |
| `scripts/run_phase2_cyclediff.sh` | CycleDiff-75 with the winning filter. |
| `scripts/run_phase3_ablations.sh` | B0/B2/B3 ablations. |
| `scripts/run_all_remaining.sh` | Chains Phase 1 → 2 → 3, continuing on stage failure. |
| `scripts/run_prep.sh`, `run_training.sh`, `run_eval.sh` | Earlier gamma_clahe-only drivers (superseded by the phase scripts). |
| `configs/` | Generated YAMLs. |
| `data/` | Generated symlink trees + filtered images (ignored). |
| `runs/` | YOLO runs (ignored). |
| `results/` | Result JSONs + gate/routing metrics. |
| `logs/` | Logs (ignored). |

---

## 3. Data map

### 3.1 FLIR ADAS v2 — original

Root: `flir_adas_experiments/data/flir_adas_v2/FLIR_ADAS_v2/`

| Directory | Count | Description |
|---|---:|---|
| `images_thermal_train/data/` | 10,742 | Thermal training frames (video frames, 15 classes). |
| `images_thermal_val/data/` | 1,144 | Thermal validation frames. |
| `video_thermal_test/data/` | 3,749 | Thermal VTT test frames (8 videos). |
| `test/clear/` | 3,749 | Clear version of the VTT test frames. |
| `images_rgb_train/`, `images_rgb_val/`, `video_rgb_test/` | — | Paired RGB (not used for detection metrics). |

### 3.2 Synthetic fog

Root: `flir_adas_experiments/data/flir_adas_v2/synthetic/`

| Path | Test | Train | Val | Notes |
|---|---:|---:|---:|---|
| `cyclegan/fog/images/` | 3,749 | 10,742 | 500 | CycleGAN 100 % fog; labels 3,749 / 8,189 / 500. |
| `cyclediff/fog/images/` | 3,749 | 11,886 | 500 | CycleDiff full fog; **282 broken train symlinks** (11,604 valid). |
| `cyclediff_75fog/fog/images/` | 3,749 | 9,051 | 500 | 75 % fog / 25 % clear blend. |
| `cyclegan/rain/images/` | 3,749 | 10,742 | 500 | Images exist; **no detector trained/evaluated**. |
| `cyclegan/snow/images/` | 3,749 | 10,742 | 500 | Images exist; **no detector trained/evaluated**. |

### 3.3 YOLO splits

Root: `flir_adas_experiments/data/flir_adas_v2/yolo/`

| Path | Count | Description |
|---|---:|---|
| `thermal/images/{train,val,test}` | 9,051 / 1,635 / 1,200 | Sequence-disjoint split (`split_hash=e4ee55238820b842`, seed 42). |
| `thermal/labels/{train,val,test}` | 9,051 / 1,635 / 1,200 | 15 classes. |
| `thermal/labels_vtt_test/` | 3,749 | VTT test labels (classes person/car etc.). |
| `thermal/split_manifest.json` | — | Split provenance. |
| `train_3a/images/train_clear` | 9,051 | Symlinks to clear train. |
| `train_3a/images/train_adv` | 8,189 (currently CycleGAN) | Symlinks to the synthetic adverse train set; **overwritten between runs**. |
| `train_3a/images/val` | 1,635 | Val. |
| `train_3a/images/test` | 1,200 | Clear test (from train-pool videos). |
| `3a_cyclegan_fog.yaml`, `3a_cyclediff_fog.yaml`, `3a_cyclegan_rain.yaml`, `3a_cyclegan_snow.yaml` | — | 3A YAMLs. **All `path:` entries point to the deleted worktree.** |

**Critical split fact:** the 1,200 clear test frames are sequence-disjoint from train/val, but they come from `images_thermal_train`/`val` videos. The synthetic fog test frames come from `video_thermal_test` (VTT). The two sets have **zero filename overlap**. They are different scenes/domains.

### 3.4 M3FD

Root: `m3fd_experiments/data/` (21 GB)

- `M3FD/` — 4,200 paired IR+RGB images, 6 classes.
- `split/` — chunk-disjoint split (`effb9b3090ad2fcb`): 2,520 / 240 / 600.
- `synthetic/` — Method 1/2/3 outputs.
- `synthetic_depth/` — depth-based fog outputs.
- `3b_fog_heavy/` — Protocol-3B set.

### 3.5 UNIRI-TID

Root: `data/UNIRI-TID/`

- 6,340 JPEGs, 1280×960, 3-channel 8-bit pseudo-colour (blue=cold, red=warm).
- 6,276 annotated, single class `Human`.
- `conditions.csv` — clear/fog/rain recovered from burned-in counter (59 sequences).
- `sequence_index.csv`, `counter.csv`, `counter_templates.json`, `manifest.csv`.
- `yolo/` — sequence-disjoint, condition-stratified YOLO tree (3,939 / 1,103 / 1,234).
- **Use `All_In_One_Anot_*` annotations only**; `ALL_IN_ONE_RGB_ANOT_*` is malformed.

### 3.6 DSERT-RoLL

- Main copy: `m3fd_experiments/method3_cyclegan/data/DSERT-RoLL/` — 3.8 GB, **Fog + Heavy_Rain + Light_Rain** (no snow).
- Worktree copy: `.kilo/worktrees/recover-session-12/data/DSERT-RoLL/` — 20 GB, **Fog + Heavy_Snow (4.1 GB) + Light_Snow (16 GB)** + train/test splits.
- **These are complementary, not duplicates.** The FLIR snow translator's `SOURCE.json` points at the worktree copy.

---

## 4. Verified results

### 4.1 FLIR ADAS v2 fog (YOLOv8s, Protocol 3A)

Source: `flir_adas_experiments/flir_adras_v2/results/*.json`

| Result | Weights | n | mAP50-95 | mAP50 | P | R |
|---|---|---:|---:|---:|---:|---:|
| CycleGAN fog — clear test | `3a_cyclegan_fog-3/best.pt` | 1,200 | 0.32739 | 0.54193 | 0.65078 | 0.51818 |
| CycleGAN fog — 100 % fog test | `3a_cyclegan_fog-3/best.pt` | 3,749 | 0.14448 | 0.27182 | 0.40460 | 0.28190 |
| CycleDiff 75 % — clear test | `3a_cyclediff_fog-8/best.pt` | 1,200 | 0.32702 | 0.54752 | 0.68111 | 0.52710 |
| CycleDiff 75 % — 75 % fog test | `3a_cyclediff_fog-8/best.pt` | 3,749 | 0.16221 | 0.30371 | 0.39515 | 0.30286 |
| CycleDiff full — 100 % fog test | `3a_cyclediff_fog-3/best.pt` | 3,749 | 0.00000 | 0.00000 | 0.00031 | 0.00136 |

**Matched VTT re-score (2026-11-04, `flir_gate_experiments/`).** The existing B1 model, evaluated on the **same 3,749 VTT frames** for both conditions:

| Condition | mAP50-95 | mAP50 | P | R |
|---|---:|---:|---:|---:|
| Clear VTT | 0.16686 | 0.30548 | 0.42152 | 0.31926 |
| Fog VTT | 0.14448 | 0.27182 | 0.40460 | 0.28190 |
| **Gap** | **0.02238** | 0.03366 | | |

This is the corrected baseline. The old clear number (0.3274) was an **in-domain 1,200-frame** evaluation; on the VTT domain the model scores 0.1669. **The old clear−fog gap of 0.183 was mostly domain shift, not weather.** The real synthetic-fog gap on matched frames is ≈0.022 mAP@50-95. Any GATE-IR claim must be judged against this smaller gap.

**Caveats:**
1. **Clear and fog are different test sets** (1,200 vs 3,749, zero overlap). The clear−fog gaps are therefore not a clean weather effect. Fix before using these numbers in a paper.
2. The report's row "CycleDiff fog (100 %, clear test) = 0.3270 / 0.5475 / 0.6811 / 0.5271" is **mislabeled** — it is the 75 % model's clear result (same weight hash `d49334addc02143c`). No clear eval exists for the full CycleDiff model (`3117bdc24e93ab6a`).
3. Training: CycleGAN fog-3 = 100 epochs (resumed from fog-2); CycleDiff fog-8 = 100 epochs. Last-epoch val mAP50-95 ≈ 0.2827 and 0.2976 respectively.
4. `3a_cyclediff_fog-8` was trained on the 75 % blend, but its provenance labels `synthetic_source` as the full CycleDiff set. Metadata caveat.

**Incomplete FLIR conditions:**
- **Rain:** CycleGAN rain translators trained, but detector runs (`3a_cyclegan_rain-2`, `-3`) stopped after **1 epoch** (results.csv has one data row). No rain eval JSON.
- **Snow:** translation done (3,749 test images), but no detector run (`runs/3a_cyclegan_snow/` contains only `provenance` and `split_hash`).

### 4.2 M3FD synthetic weather (clean split `effb9b3090ad2fcb`)

Clear baseline: mAP@50-95 = **0.148** (mAP50 0.268).

| Method | Protocol | Condition | mAP50 | mAP50-95 | Gap | Verdict |
|---|---|---|---|---:|---:|---|
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
| M3 CycleGAN | A | fog | 0.075 | 0.048 | 0.101 | WEAK |
| M3 CycleGAN | A | rain | 0.185 | 0.105 | 0.044 | FAIL |
| M3 CycleGAN | B | clear test | 0.260 | 0.147 | — | ref |
| M3 CycleGAN | B | fog | 0.193 | 0.098 | 0.049 | WEAK |
| M3 CycleGAN | B | rain | 0.197 | 0.104 | 0.011 | FAIL |

**Key findings:**
- No method passes the 0.10 Protocol-B gap.
- Deterministic fog (Method 2) is fully learnable: A-gap 0.076 → B-gap −0.001.
- CycleGAN fog has the largest A-gap (0.101) but B-gap collapses to 0.049.
- Always run Protocol B; A-gap is not the metric that matters.
- Method 2 rain IR = Method 1 rain IR (byte-identical); do not report as different methods.
- RGB fog/rain generators exist but have no effect on IR detection metrics.

### 4.3 UNIRI-TID baselines (test set, mAP@50-95)

| Model | Epochs | mAP50-95 | APm clear | APm fog | APm rain | clear−fog (APm) |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8s | 71 | 0.6204 | 0.6623 | 0.5344 | 0.5951 | 0.128 |
| YOLO11s | 55 | 0.6194 | 0.6585 | 0.5264 | 0.6038 | 0.132 |
| **YOLOv12s** | 76 | **0.6296** | **0.6670** | **0.5733** | **0.6025** | **0.094** |
| YOLOv8s + aug | 69 | 0.6278 | 0.6652 | 0.5691 | 0.6035 | 0.096 |
| YOLOv10s | 69 | 0.6063 | 0.6459 | 0.5316 | 0.5873 | 0.114 |
| YOLOv8s-P2 | 100 | 0.6071 | 0.6480 | 0.4954 | 0.5909 | 0.153 |
| ⚠️ RT-DETR-l | 29 | 0.2778 | — | — | — | INVALID |

- Bar for GATE-IR on UNIRI-TID: beat **YOLOv8s+aug fog APm = 0.5691**.
- mAP@50 is saturated (0.96–0.97); always report mAP@50-95 and APm.
- RT-DETR-l is invalid (29 epochs, walltime without requeue) — needs a rerun.

---

## 5. What is done

- [x] UNIRI-TID data reconciled, 59 sequences recovered, conditions labelled, sequence-disjoint splits.
- [x] 7-model UNIRI-TID baseline sweep (6 valid).
- [x] M3FD clean chunk-disjoint split + Methods 1/2/3 complete + `methods_report.tex`.
- [x] FLIR ADAS v2 downloaded, inspected, YOLO split built.
- [x] FLIR CycleGAN fog/rain/snow translators trained.
- [x] FLIR CycleDiff fog translator trained.
- [x] FLIR CycleGAN fog detector trained (100 epochs) + evaluated.
- [x] FLIR CycleDiff 75 % detector trained (100 epochs) + evaluated.
- [x] FLIR CycleDiff full detector trained + evaluated (collapsed).
- [x] FLIR report (`report.tex` / `report.pdf`) + 12 sample assets.
- [x] FLIR 75 % blend generated.
- [x] **GATE-IR-on-FLIR experiment (2026-11-05):** binary Clear/Fog gate, 5-filter screen (gamma/CLAHE/DCP/U-Net/gamma+CLAHE), paired U-Net dehazer, YOLOv12s B1-vs-G1 main comparison, CycleDiff-75 transfer test. Results: `flir_gate_experiments/results/REPORT.md`.
- [x] **Matched VTT evaluation:** all new FLIR runs use the same 3,749 clear/fog frames, fixing the old 1,200-vs-3,749 mismatch.
- [x] Symlinks mostly repaired (178,466 total; 282 broken remain).
- [x] Docs: `MASTER_PLAN`, `handoff`, `memory`, `devlog`, `generating_weather_conditions`, status snapshot.

---

## 6. What is not done / open issues

### 6.1 Blocking correctness (fix before new experiments)

- [x] **Matched VTT evaluation (new experiment).** The GATE-IR-on-FLIR runs use direct YAMLs and the same 3,749 VTT frames for clear and fog; the old 1,200-vs-3,749 mismatch no longer applies to those results.
- [ ] **Retire or rewrite the old 8 `flir_adras_v2` YAMLs** — they still point at the deleted worktree. The new experiment uses `flir_gate_experiments/configs/*_direct.yaml`.
- [ ] **Update stale result JSON paths** in the old `flir_adras_v2/results/` (they record the old worktree location).
- [ ] **Repair or remove the 282 broken CycleDiff-train symlinks** (11,604 of 11,886 valid).
- [ ] **Correct the old `flir_adas_experiments/reports/report.tex` table**: remove/relabel the "CycleDiff 100 % clear test" row. The new `flir_gate_experiments/results/REPORT.md` supersedes it for the new experiment.

### 6.2 Incomplete experiments

- [ ] **Rain:** detector runs stopped after 1 epoch; no eval JSONs.
- [ ] **Snow:** translation done; no detector trained/evaluated.
- [ ] **Protocol 3B on FLIR:** not run.
- [ ] **B3 / B0 / B2 ablations:** Phase 3 was stopped at B3 training; no B3/B0/B2 result JSONs (`runs/b3_unet_yolov8s_40` is partial).
- [ ] **CycleDiff-trained U-Net:** the current U-Net (trained on CycleGAN fog) **hurts** on CycleDiff-75. Retrain on CycleDiff pairs to test whether the filter can transfer when trained on the target generator.
- [ ] **Gate-routed mixed-condition evaluation:** the headline uses per-condition (oracle) routing; the real mixed-stream pipeline (8.6 % clear false-positive rate) is not measured.
- [ ] **100-epoch YOLOv12s:** the main comparison is 40 epochs (compute-limited).
- [ ] **Multi-seed / mean±std:** none of the FLIR or M3FD headline numbers are multi-seed.
- [ ] **Real-fog validation:** all FLIR fog is synthetic (CycleGAN / CycleDiff).
- [x] **FLIR binary Clear/Fog gate:** trained and evaluated (`results/gate_cyclegan.pkl`); 100 % on CycleGAN fog, 91.4 % clear (8.6 % FPR), 66 % cross-generator on CycleDiff-75.
- [ ] **Full GATE-IR Stage A/B/C on UNIRI-TID:** still not built.

### 6.3 Core GATE-IR

- [ ] Stage A WeatherGate trained/evaluated.
- [ ] Stage B enhancers wired and evaluated.
- [ ] Stage C detector trained/evaluated.
- [ ] E1 efficiency table (params, GFLOPs, FP16 latency, FPS, peak memory).
- [ ] E4 oracle-gate / no-gate bounds.
- [ ] RT-DETR-l rerun.
- [ ] DroneVehicle download.
- [ ] Diffusion weather generation (snow/haze, PID).

### 6.4 Repo hygiene / risks

- [ ] **FLIR code is untracked in `main`.** It exists only in the worktree branch `efd97a1` (941 files). Commit it to `main` before retiring the worktree.
- [ ] **Preserve the worktree's DSERT snow data** (20 GB) before deleting the worktree. `runs/snow/SOURCE.json` points at it; main's DSERT copy has no snow.
- [ ] **Rotate `KAGGLE_TOKEN` and scrub it from commit `6a5f15b`** before pushing.
- [ ] Commit the `.gitignore` change (all `data/` dirs ignored).
- [ ] Commit `docs/status_20261030_0624.md` (or move its content into `devlog.md`).
- [ ] Add `docs/gap_establishment.md` or fix the FLIR README reference.
- [ ] Decide what to do with `abstracted-side` and the 4.7 GB `data/flir_adras_v2` copy.

---

## 7. Known pitfalls / do not repeat

1. **A question is not an instruction.** Answer first; do not change files until told to act. (See `rules.md` at the repo root.)
2. **Never delete training artifacts** (checkpoints, logs, samples, result dirs, data) without explicit confirmation.
3. **Never cancel the SLURM allocation.** Use `srun --jobid=9639 --overlap`; kill Python PIDs, not the allocation.
4. **Report mAP@50-95 and APm, never mAP@50 alone.** mAP@50 is saturated on UNIRI-TID.
5. **Sequence/chunk-disjoint splits only.** Consecutive video frames are ~99.9 % identical.
6. **Never derive weather labels from image statistics.** WeatherGate's features are image statistics; that would be circular.
7. **Protocol A gap ≠ Protocol B gap.** Always run B.
8. **Restoration is judged by mAP, not PSNR.**
9. **Do not reinstall `thop`** — it breaks RT-DETR under torch 2.14.
10. **Training is CPU-bound on JPEG decode.** Use `cache='disk'`; raising batch size does not help.
11. **FLIR images are 8-bit 3-channel pseudo-colour**, not 14-bit single-channel RAW. The `GATE-IR` modules assume 14-bit; adapt before using.
12. **The 1,200 clear test and 3,749 fog test are different domains.** Do not compare them directly.
13. **`train_3a/images/train_adv` is overwritten** between CycleGAN/CycleDiff runs. Rebuild it before reproducing an old run.
14. **The report's "CycleDiff 100 % clear test" row is the 75 % model.** Do not cite it as the full model.

---

## 8. Environment, SLURM, how to run

### Environment

```bash
source /opt/miniforge/etc/profile.d/conda.sh
conda activate gateir
# torch 2.14.0+cu126, ultralytics 8.4.138, cv2 5.0.0, sklearn 1.9.0
```

Python: `$HOME/.conda/envs/gateir/bin/python`

### SLURM

Current allocation: **job `9639`** (`interact`, 1 GPU, 16 CPUs, 48 GB, 1-day limit).

```bash
# Run inside the existing allocation
srun --jobid=9639 --overlap <command>

# Check it
squeue -u $USER
scontrol show job 9639 | grep -E 'TimeLimit|NodeList|TRES'
```

- Do **not** `scancel 9639` (the allocation).
- `scancel 9639.<step>` for a step is safe.
- tmux sessions present: `ablation_j`, `salloc_session`, `srun_s`.

### Reproduce an evaluation (after fixing YAML paths)

```bash
cd /home2/home/preetham_g/Gate_IR/flir_adas_experiments
$HOME/.conda/envs/gateir/bin/python flir_adras_v2/eval_detector.py \
  --weights flir_adras_v2/runs/3a_cyclegan_fog-3/weights/best.pt \
  --data data/flir_adas_v2/synthetic/cyclegan/fog/synthetic_fog.yaml \
  --name 3a_cyclegan_fog_adv_recheck --split test --imgsz 640 --force
```

### Train a detector (current code)

```bash
cd /home2/home/preetham_g/Gate_IR/flir_adas_experiments
$HOME/.conda/envs/gateir/bin/python flir_adras_v2/train_detector.py \
  --mode 3a --condition fog --method cyclegan --epochs 100 --batch-size 16 --weights yolov8s.pt
```

Note: `train_detector.py` defaults to `yolov12s.pt`; the existing FLIR runs used `yolov8s.pt`.

---

## 9. Active next experiment: GATE-IR on FLIR synthetic fog

**Status: started 2026-11-04.** Code and data live in `flir_gate_experiments/`. Prep is complete; the detector arms are training inside SLURM job `9639`. The design below is what is being run.

### 9.0 First verified results (2026-11-04)

**Matched B1 baseline** (existing CycleGAN-fog model, same 3,749 VTT frames):

| Condition | mAP50-95 | mAP50 | P | R |
|---|---:|---:|---:|---:|
| Clear VTT | 0.16686 | 0.30548 | 0.42152 | 0.31926 |
| Fog VTT | 0.14448 | 0.27182 | 0.40460 | 0.28190 |
| Gap | **0.02238** | 0.03366 | | |

**B1 YOLOv8s 60-epoch screen (same matched VTT frames):** clear **0.16464**, fog **0.14134**, gap **0.02330**. This is consistent with the 100-epoch B1 (0.16686 / 0.14448 / 0.02238), so the 60-epoch screen is a fair reference for the filter arms.

**CycleGAN filter screen (YOLOv8s, 60 epochs, matched VTT):**

| Filter | Clear mAP50-95 | Fog mAP50-95 | Gap |
|---|---:|---:|---:|
| B1 (none) | 0.16464 | 0.14134 | 0.02330 |
| gamma (FogEnhancer) | 0.15883 | 0.13694 | 0.02189 |
| CLAHE | 0.16357 | 0.13352 | 0.03005 |
| DCP | 0.16505 | 0.13940 | 0.02565 |
| **U-Net dehazer** | **0.16757** | **0.15241** | **0.01516** |
| gamma_clahe | 0.16104 | 0.13838 | 0.02266 |

**U-Net is the clear winner:** +0.0111 fog mAP50-95 and +0.0029 clear over B1, reducing the clear−fog gap from 0.0233 to 0.0152 (~35 %). The classical filters (gamma/CLAHE/DCP) do **not** help. This makes the learned paired dehazer the best Stage-B candidate and the likely winner for the YOLOv12s main comparison and CycleDiff-75.

**Main comparison — YOLOv12s, 40 epochs, matched VTT:**

| Arm | Clear mAP50-95 | Fog mAP50-95 | Gap |
|---|---:|---:|---:|
| B1 (no filter) | 0.14639 | 0.12340 | 0.02299 |
| **G1 U-Net** | **0.15178** | **0.14333** | **0.00845** |
| Δ | +0.0054 | **+0.0199** | **−63 %** |

This is the headline result: the learned U-Net dehazer improves fog mAP@50-95 by **~0.020 absolute (16 % relative)** and shrinks the clear−fog gap by **63 %** on YOLOv12s. The classical filters do not.

**CycleDiff-75 transfer (YOLOv8s, 40 epochs, matched VTT):**

| Arm | Clear mAP50-95 | Fog mAP50-95 | Gap |
|---|---:|---:|---:|
| B1 (no filter) | 0.16645 | 0.16186 | 0.00459 |
| G1 U-Net | 0.15929 | 0.15113 | 0.00816 |
| Δ | −0.0072 | **−0.0107** | +0.0036 |

**The U-Net does not transfer across generators.** Trained on CycleGAN fog, it hurts on CycleDiff-75. This is the strongest caveat in the study: the filter learned CycleGAN-specific artefacts rather than general fog removal. A CycleDiff-trained U-Net (and ideally a mixed-generator / real-fog training set) is required before claiming general dehazing.

**Gate (5 features + MLP, trained on CycleGAN fog):**

| Test set | Accuracy | F1 | FNR (fog→clear) | FPR (clear→fog) |
|---|---:|---:|---:|---:|
| Video-disjoint val | 0.9894 | 0.9895 | 0.0000 | 0.0212 |
| VTT clear | 0.9141 | — | — | 0.0859 |
| VTT fog (CycleGAN) | 1.0000 | 1.0000 | 0.0000 | — |
| VTT fog (CycleDiff 75 %) | 0.6612 | 0.7961 | **0.3388** | — |

**Interpretation (important):**
- The gate is **near-perfect on the same generator** but drops to 66 % on a different generator. It is partly detecting CycleGAN artefacts, not "fog". Cross-generator numbers must be reported.
- The **matched** clear−fog gap is only ~0.022 mAP50-95. The filter has little headroom to "recover", so a small or null effect is the most likely outcome. That is still a useful result (T3).
- The old 0.327 clear number was in-domain; do not use it in any comparison.

### 9.1 Hypothesis

On FLIR synthetic fog, a binary clear/fog gate that routes fog frames through a defogging filter and leaves clear frames untouched will:
- recover detection accuracy on fog compared with the mixed-training baseline (clear + fog, no gate/filter), and
- cost less than applying the filter to every frame (always-on restoration).

This is the T1 (efficiency/Pareto) claim at small scale.

### 9.2 Proposed design

1. **Stage A — binary gate:** clear vs fog. Reuse the GATE-IR feature set (variance, min, max, entropy, Laplacian variance) + tiny MLP, adapted to 8-bit 3-channel FLIR input. Optionally compare with a tiny CNN.
2. **Stage B — fog filter:** start with the existing `FogEnhancer` (adaptive gamma + contrast stretch) and classical alternatives (CLAHE, AGCWD, DCP). If they are too weak, train a small paired U-Net dehazer on (clear, synthetic fog) pairs.
3. **Stage C — YOLOv8s detector:** no knowledge distillation. Train on the same distribution seen at inference (clear + filtered fog).
4. **Comparison baseline:** YOLOv8s trained on clear + unfiltered fog (the current 3A setup), no gate/filter.
5. **Evaluation:** matched clear/fog test frames (VTT), mAP@50-95 + APm, plus latency.

### 9.3 Run order (decided 2026-11-04)

The main model is **YOLOv12s**. The old B1 numbers above are YOLOv8s; the headline B1/G1 comparison will be YOLOv12s, trained with the same recipe.

1. **CycleGAN, main numbers + filter sweep.**
   - Filters: `gamma` (FogEnhancer), `clahe`, `dcp`, `unet` (paired dehazer), `gamma_clahe`.
   - 60-epoch screen: YOLOv12s B1 (clear + unfiltered fog) and G1 (clear + filtered fog) for every filter.
   - Evaluate all on the matched 3,749 VTT clear/fog frames, plus learned-gate and oracle-gate routing.
   - 100-epoch confirmation: B1 and the winning filter.
2. **CycleDiff-75.** Use only the winning filter; train a CycleDiff-75 gate; train B1 and G1 at the same recipe; report matched VTT clear vs CycleDiff-75 fog.
3. **Ablations last.** B0 (clear-only), B2 (fog-only), B3 (always-on filter) for the methods already run.

### 9.3b Experiment matrix (reference)

| ID | Detector training data | Inference pipeline | Purpose |
|---|---|---|---|
| **B0** | clear only | none | domain-gap lower bound |
| **B1** | clear + synthetic fog | none | current 3A baseline (exists) |
| **B2** | synthetic fog only | none | does clear data help? |
| **B3** | clear + filtered fog | filter on all frames (no gate) | always-on restoration baseline |
| **G1** | clear + filtered fog | learned gate: clear→identity, fog→filter | proposed GATE-IR |
| **G2** | clear + filtered fog | oracle gate (ground-truth labels) | upper bound |
| **G3** | same as G1 | G1 with x % gate flips | gate-error robustness (T2) |
| **G4** | same as G1 | filter ablation (gamma/CLAHE/DCP/U-Net) | pick filter by mAP |
| **G5** | train on CycleGAN fog, test on CycleDiff fog (and vice versa) | gate + filter | cross-generator generalization |

### 9.4 Metrics

- **Detection:** mAP@50-95, mAP@50, P, R; per-condition (clear/fog); per-size (APs/APm/APl) where possible.
- **Gate:** accuracy, F1, confusion matrix, **false-negative rate** (fog→clear, the costly error), false-positive rate (clear→fog, wasted compute).
- **Efficiency:** gate latency, filter latency, detector latency, end-to-end latency, FPS, params, FLOPs, peak memory. Report the accuracy–latency Pareto frontier for B1, B3, G1, G2.
- **Seeds:** ≥3 seeds, mean±std, for headline numbers (compute permitting).

### 9.5 Open questions for Preetham

1. **Filter choice:** start with the cheap `FogEnhancer`/CLAHE/DCP, or train a paired U-Net dehazer immediately? The cheap filters may not improve mAP; the U-Net is stronger but adds training cost and a generator-specificity risk.
2. **Test-set fix:** evaluate all conditions on the 3,749 VTT frames (recommended), or regenerate fog from the 1,200 held-out frames?
3. **Gate training data:** train on CycleGAN fog only, CycleDiff fog only, or both? Cross-generator test is recommended either way.
4. **Scope:** is a synthetic-only proof of concept acceptable for now, or is a real-fog sanity check required (DSERT-RoLL real fog frames, no boxes, gate-only evaluation)?
5. **Compute:** how many seeds/epochs fit the current 24 h allocation? A 100-epoch YOLO run is ~1.7–3.7 h on this data; the full matrix will need multiple allocations.

### 9.6 Recommended compute plan

1. Fix YAML paths + matched clear eval (minutes).
2. Train/validate the gate (minutes).
3. Run B0, B2, B3, G1, G2 at 100 epochs (≈2–4 h each; ~10–20 h total).
4. Run G3/G5 (cheap, mostly inference).
5. If the cheap filter fails, train the U-Net dehazer and rerun B3/G1.

---

## 10. Docs index

| File | Read it for |
|---|---|
| `README.md` (this file) | Current verified state, layout, issues, active next experiment |
| `flir_gate_experiments/README.md` | The GATE-IR-on-FLIR experiment: design, arms, how to run |
| `docs/MASTER_PLAN.md` | Long-term plan, thesis options, milestones, known problems |
| `docs/handoff.md` | What works, results tables, next actions |
| `docs/memory.md` | Durable facts, cluster constraints, environment, conventions |
| `docs/devlog.md` | Full session history, what broke, what was fixed |
| `docs/generating_weather_conditions.md` | Diffusion/PID architecture spec |
| `docs/dataset_generation_plan.md` | Method 1–4 planning + Protocol A/B |
| `docs/status_20261030_0624.md` | FLIR fog session snapshot |
| `docs/README_UNIRI_HARNESS.md` | Original UNIRI-TID harness README |
| `rules.md` (repo root) | Hard rules: artifact preservation, irreversible changes, SLURM, question-vs-instruction, relative paths |

---

## 11. Change log

| Date | Change |
|---|---|
| 2026-11-04 | Created this README from a direct audit. Preserved the old root README as `docs/README_UNIRI_HARNESS.md`. Recorded verified layout, results, issues, and the proposed GATE-IR-on-FLIR experiment. |
