# GATE-IR — Persistent Context

> This file consolidates the key facts, constraints, and conventions that every coding session
> must know. It is the single place for "things the AI assistant should always remember." Update
> it when new constraints are discovered, not retrospectively.
>
> For the current plan, see `MASTER_PLAN.md`. For what to do next, see `handoff.md`.

---

## About Preetham

- Preetham is a student researcher driving GATE-IR toward publication. Not the original code author
  (Aryan Sood wrote the repo); Preetham is running the experimental/publication side.
- **Discuss first, plan second.** He wants back-and-forth before a plan document is written. Surface
  tradeoffs in prose; write the plan only once he confirms. When he asks a question about earlier work,
  answer that question — do not treat it as a signal to redo the plan.
- Explicitly wants to know when something is a weak or indefensible claim, not to be told it looks fine.
- Works under hard compute constraints (see Cluster section below).

---

## Project goal

GATE-IR: a three-stage thermal detection pipeline for adverse weather.
- **Stage A:** WeatherGate (MLP classifier: Clear / Fog / Rain / Snow)
- **Stage B:** Per-condition enhancer (gamma/AGCWD for fog; LSRB for rain; snow-specific enhancer; identity for clear)
- **Stage C:** YOLOv8s + P2 head + ViT neck detector

The code repo (`GATE-IR/`) is **architecture scaffolding only** — no dataloader, no training loop,
no mAP evaluation, no trained weights. `main.py --mode demo` runs on `torch.rand`. The entire
experimental harness was built from scratch in `bench/`.

**Current thesis direction: T1 — Efficiency / Pareto.**
*"Hard gating matches all-in-one restoration accuracy at a fraction of edge latency, by skipping
enhancement on clear frames."* This thesis is only viable if a Protocol 3B gap ≥ 0.15 mAP@50-95
is first confirmed on the target dataset. If the maximum achievable gap is < 0.10, the methodology
cannot be justified and the thesis must pivot to T3 (negative/diagnostic result).

**Next phase:** FLIR ADAS v2 dataset acquisition + Method 4 (CycleDiff) generation for fog, rain, and snow.
Gap establishment precedes pipeline construction. See `docs/gap_establishment.md`.

**Critical doctrine — gap establishment precedes pipeline construction.**
Stage A/B/C code is not written until a Protocol 3B gap ≥ 0.15 mAP@50-95 is confirmed on the
target dataset. Building the pipeline first assumes the gap exists, which the M3FD experiments
proved false (all methods failed Protocol 3B with gaps < 0.05).

---

## Cluster constraints

- Node `ub`, 6× RTX 6000 Ada (48 GB each).
- QOS `normal`: **1 GPU, 1 running job at a time, 24 h max**. Everything is serial; ~4–6 training
  runs per 24 h allocation. The 3-GPU `research` QOS is **not available to students**.
- GPU queue routinely runs 7+ deep.
- **Never `scancel` the allocation** — the queue is too long to give up a held GPU.
- **Preferred Allocation Workflow (salloc + tmux):**
  - `tmux new-session -d -s gateir_alloc "salloc --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=24:00:00"`
  - Execute tasks by attaching to the job ID from any shell: `srun --jobid=<JOBID>.<STEP> <cmd>`.
  - Append decimals (e.g., `.1`, `.2`) to the Job ID to distinguish parallel or sequential `srun` steps cleanly.
  - `scancel <JOBID>.<STEP>` (cancel a step) is safe. `scancel <JOBID>` (cancel the job) drops the GPU entirely.
- **Nodes fail.** If using unattended sweeps that span multiple days, use `sbatch --requeue`.
- Request `--time=6:00:00` (real work ~3 h), not 24 h — shorter jobs get backfill sooner.
  Watch out: `scontrol update` silently reverted once; re-check with `scontrol show job <id> | grep TimeLimit`.
- Training is **CPU-bound on JPEG decode**, not GPU-bound. `cache='disk'` is the throughput fix;
  raising batch size does not help. Peak GPU memory: v8s@64 = 13 GB, rtdetr-l@32 = 26 GB of 48 GB.

---

## Environment

```
conda env:  $HOME/.conda/envs/gateir
torch:      2.14.0+cu126   (CUDA OK)
ultralytics: 8.4.138
cv2:        5.0.0
sklearn:    1.9.0
```

- `/opt/miniforge/envs` is **not writable** — always use `-p $HOME/.conda/envs/<name>` with conda.
- `unzip` is **not installed** — use Python `zipfile`.
- **Do not reinstall `thop`** — `thop 0.1.1` leaks profiling hooks into the forward pass under
  torch 2.14, crashing RT-DETR with `'ReLU' object has no attribute 'total_ops'`.
- Weight files live in `weights/` (moved there 2026-09-20). Scripts reference `weights/*.pt`.

---

## UNIRI-TID dataset — key facts

**Still the primary dataset for all baseline numbers. Do not discard.**

- 6,340 JPEGs, 1280×960, **3-channel 8-bit, pseudo-coloured** (blue=cold, red=warm). NOT radiometric.
- 6,276 annotated, single class `Human`. ~1.47 boxes/image.
- **Use `All_In_One_Anot_*` annotations only.** `ALL_IN_ONE_RGB_ANOT_*` is malformed: packs multiple
  `<bndbox>` inside one `<object>`, silently dropping ~2,965 boxes.
- Weather labels were **not shipped** — recovered from the burned-in frame counter (OCR of 49 glyphs,
  94% readable). 59 sequences: seqs 0–22 clear, 23–31 fog, 32–58 rain. Fog count matches the published
  1,135 to within 1 frame. See `bench/data/label_conditions.py` and `conditions.csv`.
- **Circularity trap:** WeatherGate's 5 features are image statistics. Deriving weather labels from
  image statistics and training the gate on them → ~100% accuracy by construction, fatal in review.
  Labels must come from the frame counter reconstruction, never from clustering on features.
- Sequence-disjoint splits are **mandatory**: median frame-to-frame diff = 0.0012 (~99.9% identical).
- Camera distance confound: fog=0–30 m, clear=50–160 m, rain=30–215 m. Always report AP vs object size
  alongside per-condition AP. Never attribute a gap to weather alone without the size control.
- Split sizes: train 3,939 / val 1,103 / test 1,234. Per condition (test): 535 clear / 195 fog / 504 rain.

### UNIRI-TID baseline results (test set, mAP@50-95)

| model | epochs | mAP@50-95 | APm clear | APm fog | APm rain | clear−fog (APm) |
|---|---|---|---|---|---|---|
| YOLOv8s | 71 | 0.6204 | 0.6623 | 0.5344 | 0.5951 | 0.128 |
| YOLO11s | 55 | 0.6194 | 0.6585 | 0.5264 | 0.6038 | 0.132 |
| **YOLOv12s** | 76 | **0.6296** | **0.6670** | **0.5733** | **0.6025** | **0.094** |
| YOLOv8s+aug | 69 | 0.6278 | 0.6652 | 0.5691 | 0.6035 | 0.096 |
| YOLOv10s | 69 | 0.6063 | 0.6459 | 0.5316 | 0.5873 | 0.114 |
| YOLOv8s-P2 | 100 | 0.6071 | 0.6480 | 0.4954 | 0.5909 | 0.153 |
| ⚠️ RT-DETR-l | 29 | 0.2778 | — | — | — | INVALID |

All models trained on the full mixed split (clear+fog+rain). mAP@50 is saturated at 0.96–0.97 on this
dataset — **always report mAP@50-95 and APm**. The 3B baseline to beat for GATE-IR on UNIRI-TID is
**YOLOv8s+aug fog APm = 0.5691** (cheapest rival, no gate or enhancer).

RT-DETR-l is invalid (29 epochs, 0.28 mAP) — likely hit walltime without `--requeue`. Ignore it for now.

---

## FLIR ADAS v2 dataset — planned (not yet acquired)

- **Source:** Kaggle `samdazel/teledyne-flir-adas-thermal-dataset-v2`
- **Modality:** Paired IR + RGB
- **Classes:** 15 (vehicles, person, bicycle, etc.)
- **Images:** ~26,000
- **Status:** NOT DOWNLOADED — all properties UNKNOWN until inspection
- **Weather labels:** UNKNOWN — must be inspected after download
- **Split strategy:** UNKNOWN — must be determined after inspection
- **Generation target:** DSERT-RoLL (fog, rain) + Heavy_Snow (snow) — see below
- **Methods to evaluate:** M3 (CycleGAN), M4 (CycleDiff)
- **Conditions:** fog, rain, snow
- **Protocol:** 3A (clear-only training) + 3B (mixed clear+adverse training)
- **Gap target:** Protocol 3B gap ≥ 0.15 mAP@50-95
- **Code directory:** `flir_adras_v2/` (to be created)

## DSERT-RoLL — target domains for weather generation

- Fog: 3,282 thermal frames + 3,282 RGB frames
- Heavy_Rain: 1,938 thermal + 1,938 RGB
- Light_Rain: 2,214 thermal + 2,214 RGB
- **Heavy_Snow:** NOT YET CONFIRMED — must be acquired from DSERT-RoLL expansion or a separate source
- Used as target domain for CycleGAN (M3) and CycleDiff (M4) training on M3FD and FLIR ADAS v2

## M3FD dataset — secondary / synthetic weather source (negative result)

- `nonameplease/M3FD_Detecion` on HuggingFace. 2.2 GB. 4,200 paired IR+RGB images.
- 6 classes: person, car, bus, motorcycle, lamp, truck. YOLO labels included.
- **All three synthetic weather methods have been evaluated** (Methods 1, 2, 3 — all done). All failed
  Protocol 3B with gaps < 0.05. The M3FD synthetic-weather track is a negative result.
- Split: chunk-disjoint temporal split via `resplit_m3fd.py` (split_hash `effb9b3090ad2fcb`).
  Train 2520 / val 240 / test 600. 60-frame chunks, 16-frame boundary drops, IoU leakage check.
- **Do not re-enter the M3FD synthetic-weather track** unless a new generation method (CycleDiff)
  first demonstrates a 3B gap ≥ 0.15 on a different base dataset.

### M3FD synthetic weather results — all methods complete (clean split, negative result)

| protocol | condition | method | mAP@50 | mAP@50-95 | gap vs clear | verdict |
|---|---|---|---|---|---|---|
| A | clear (baseline) | — | 0.268 | 0.148 | 0 | ref |
| A | fog_light | M1 classical | 0.268 | 0.148 | 0.001 | FAIL |
| A | fog_medium | M1 classical | 0.243 | 0.135 | 0.013 | FAIL |
| A | fog_heavy | M1 classical | 0.206 | 0.117 | 0.032 | FAIL |
| A | rain_light | M1 classical | 0.261 | 0.144 | 0.005 | FAIL |
| A | rain_medium | M1 classical | 0.189 | 0.094 | 0.054 | WEAK |
| A | rain_heavy | M1 classical | 0.079 | 0.041 | 0.108 | WEAK |
| A | fog_medium | M2 depth | 0.199 | 0.114 | 0.035 | FAIL |
| A | fog_heavy | M2 depth | 0.097 | 0.072 | 0.076 | WEAK |
| A | rain_medium | M2 depth | 0.094 | 0.054 | 0.054 | WEAK |
| A | rain_heavy | M2 depth | 0.011 | 0.011 | 0.137 | WEAK |
| B | clear test | M2 mixed | 0.244 | 0.141 | 0 | ref |
| B | fog_heavy | M2 mixed | 0.237 | 0.140 | −0.001 | FAIL |
| A | fog | M3 cyclegan | 0.075 | 0.048 | 0.101 | WEAK |
| A | rain | M3 cyclegan | 0.185 | 0.105 | 0.044 | FAIL |
| B | clear test | M3 fog model | 0.260 | 0.147 | 0 | ref |
| B | fog | M3 fog model | 0.193 | 0.098 | 0.049 | WEAK |
| B | rain | M3 rain model | 0.197 | 0.104 | 0.011 | FAIL |

**Verdict:** No method sustains a Protocol 3B gap ≥ 0.05. The M3FD synthetic-weather track is a
negative result. It is cited as evidence that classical, depth-based, and CycleGAN methods are
insufficient, not as a baseline to beat.

---

## Eval harness — conventions

All experiments report through `bench/eval/detect_eval.py`. **One scorer for everything.**

- Per-condition breakdown (clear / fog / rain / snow) from the dataset's condition labels.
- Per-object-size AP (APm = medium bucket, 89–97% of boxes on the target dataset).
- pycocotools backend, cross-checked against Ultralytics `.val()` (Δ ≤ 0.002 mAP on every run).
- Result JSON: `bench/results/<name>.json` (training) + `bench/results/eval_<name>_test.json` (eval).
- Debug JSONs: `bench/results/debug/` with `_dbg` suffix; carry `"debug": true`. Never mistake for real.

---

## Training conventions — pinned for all baselines

| setting | value | reason |
|---|---|---|
| optimizer | AdamW, lr0=1e-3 | `auto` picks by iteration count; pinning makes every row use the same recipe |
| LR schedule | cosine, lrf=0.01 | 1e-3 → 1e-5 |
| warmup | 3 epochs, momentum=0.8, bias_lr=0.1 | |
| early stop | patience=25 | |
| close_mosaic | 10 | mosaic off for last 10 epochs |
| effective batch | 64 (all models) | nbs=64; gradient-accumulates to reach it |
| deterministic | True | reproducible |
| hsv_h, hsv_s | 0.0 | thermal pseudo-colour palette must not be hue/sat jittered |

**Resume logic:** `train_detector.py` checks `last.pt` on startup and resumes automatically.
`overnight.sh` skips runs whose result JSON already exists.

---

## Known bugs and gotchas

- **YOLO Silent Label Bug** (found Session 3, 2026-09-21): YOLOv12 is highly sensitive to directory
  structure. If image paths are `data/IR/images/train/img.jpg`, labels **must** be at
  `data/IR/labels/train/img.txt`. If placed at `data/labels/train/img.txt` (missing the `IR/` component),
  YOLO will *not crash* — it silently logs "0 images, N backgrounds", treats every image as background,
  and trains a model yielding 0.000 mAP. **Always grep training logs for "backgrounds" count** to
  confirm labels were found. Fix: manually move label directories into `IR/labels/` hierarchy and purge
  `.cache` files after fixing label paths.
- `preprocessing/rain_remover.py` — `CLAHE.forward` uses differentiable LCN when `training=True`
  but OpenCV CLAHE when `False`. Two different operators; the module trains against one and is evaluated
  with another. Fix before any Stage B rain experiment. (This file lives in the gitignored original
  `GATE-IR/` scaffolding; it does not exist in the current worktree but the bug is documented.)
- `models/yolov8_thermal.yaml` is already valid Ultralytics syntax. Porting Stage C is "register
  `TransformerBlock` as a parseable module", not a rewrite.
- **Protocol A is not the metric that matters.** Always run Protocol B. A synthetic fog that
  passes A can still collapse to 0 in B (confirmed with Method 2; Method 3 fog survived weakly
  at +0.049, rain collapsed to +0.011).
- **Method 2 rain IR = Method 1 rain IR** (byte-identical). Do not report or describe them as
  different. Depth-aware rain was only applied to RGB.
- **RGB fog/rain generators were trained** — `rgb_fog` and `rgb_rain` completed 50 epochs. However
  they produce no eval-side effect — all M3FD eval YAMLs point at IR image paths, so no RGB generator
  can affect any detection number.
- **M3FD evaluation uses IR images only.** RGB fog/rain synthesis has no effect on detection
  metrics. Only IR synthesis matters.
- **Cache matters.** Training is CPU-bound on JPEG decode. Use `--cache disk`. Raising batch size
  does not help throughput.
- **Never derive weather labels from image statistics.** Fatal circularity for the gate — see
  MASTER_PLAN §4.
- **M3FD split contamination was fixed** by `resplit_m3fd.py`. The original `split_dataset.py` did a
  random frame-level shuffle of sequential video. All contaminated results archived to `_old/`.
- **Gap establishment precedes pipeline construction.** Building Stage A/B/C before confirming a
  significant gap wastes compute on an unjustifiable methodology. See `docs/gap_establishment.md`.
- **Snow has different IR physics than fog/rain.** Snow causes emissivity wash-out and convective
  cooling, not just contrast loss. A generation model that treats snow as "white fog" will produce
  unrealistic outputs and a small or negative gap.
- **CycleDiff is unproven.** It may produce mode collapse, sensor-identity transfer, or training
  instability. Inspect sample grids and measure Protocol 3B before treating it as a successful
  generation method.

---

## Key references

- **Kristo et al. 2020** — `refs/kristo2020_uniri_tid.pdf` — companion paper for UNIRI-TID.
  Table 2 (camera distance per condition) and Table 3 (AP numbers at IoU=0.5) are the most important.
- **DSERT-RoLL** (CVPR 2026) — real driving RGB+thermal dataset under clear/fog/rain. Used as
  CycleGAN target domain for M3FD Method 3. Snow images must be confirmed or acquired separately.
- **Depth-Anything-V2-Small-hf** — monocular depth estimator used in M3FD Method 2.
- **FLIR ADAS v2** (Kaggle: `samdazel/teledyne-flir-adas-thermal-dataset-v2`) — paired IR+RGB,
  15 classes, ~26k images. Next dataset for GATE-IR experiments. Not yet downloaded or inspected.
- **docs/generating_weather_conditions.md** — Physics-Informed Diffusion architecture spec for
  future weather generation work (CycleDiff design reference).
- **docs/gap_establishment.md** — gap-first doctrine, go/no-go rules, and why pipeline construction
  is deferred until a significant gap is confirmed.

---

## SLURM Allocation (Facts)

- `salloc` reserves resources; `srun` launches work inside the allocation
- `salloc` shell stays in persistent `tmux` session `salloc_session`
- Use `srun --jobid=<JOB_ID>` from `srun_training` tmux to target existing allocation
- SLURM jobs expire after 24h; new `salloc` needed for extension
- QOS limits: 1 GPU per user, 24h max, 32 CPUs max, 144GB max memory
- New allocation command: `salloc -p gpu --gres=gpu:1 --time=24:00:00 --cpus-per-gpu=32 --mem=128G`

## CycleDiff Training Configuration (Optimized)

- batch_size: 18 (benchmarked - optimal balance of throughput vs memory)
- num_workers: 24 (with 32 CPU allocation)
- gradient_accumulate_every: 1 (halves iteration time vs ga=2)
- train_num_steps: 45000
- save_every: 5000 (auto-cleanup keeps last 2 checkpoints)
- bf16 mixed precision enabled
- prefetch_factor: 4 on DataLoader

## Batch Size Benchmark Results

| batch_size | Avg time/step | Peak GPU Memory | Throughput | Status |
|---|---|---|---|---|
| 16 | 2.407s | 37.58 GiB | 6.6 samples/s | Stable |
| 18 | 2.428s | 41.33 GiB | 7.4 samples/s | Stable (CURRENT) |
| 20 | 2.562s | 44.63 GiB | 7.8 samples/s | Unstable (OOM) |

### Notes:
- GPU utilization fluctuates 4-95% - bottleneck is data loading between GPU compute segments
- batch_size=18 chosen as optimal: stable, 12% throughput gain over 16, fits in 48GB GPU
- With ga=1, iteration time ~2.2s/it, ETA ~27h (fits in 24h allocation with renewal)
- Checkpoint saves atomic (tmp file + rename) to prevent corruption

## Policy and Rules

All project rules and policies are in `docs/rules.md` you must read this first. Must follow.
Key points:
- Never delete training artifacts without explicit user confirmation
- Never make irreversible changes casually
- Keep SLURM allocations alive between runs
- Never delete `.git/` directory



