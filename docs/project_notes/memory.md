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
- **Stage A:** WeatherGate (MLP classifier: Clear / Fog / Rain)
- **Stage B:** Per-condition enhancer (gamma/AGCWD for fog; LSRB for rain; identity for clear)
- **Stage C:** YOLOv8s + P2 head + ViT neck detector

The code repo (`GATE-IR/`) is **architecture scaffolding only** — no dataloader, no training loop,
no mAP evaluation, no trained weights. `main.py --mode demo` runs on `torch.rand`. The entire
experimental harness was built from scratch in `bench/`.

**Current thesis direction: T1 — Efficiency / Pareto.**
*"Hard gating matches all-in-one restoration accuracy at a fraction of edge latency, by skipping
enhancement on clear frames."* This is contingent on: (a) a dataset with a large enough gap to
motivate the pipeline, and (b) latency numbers that do not yet exist.

**Next major direction:** larger IR dataset (DroneVehicle) + diffusion-based weather generation (snow, haze, etc.).

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

## M3FD dataset — secondary / synthetic weather source

- `nonameplease/M3FD_Detecion` on HuggingFace. 2.2 GB. 4,200 paired IR+RGB images.
- 6 classes: person, car, bus, motorcycle, lamp, truck. YOLO labels included.
- **All three synthetic weather methods have been evaluated** (Methods 1, 2, 3 — all done).
- Split: chunk-disjoint temporal split via `resplit_m3fd.py` (split_hash `effb9b3090ad2fcb`).
  Train 2520 / val 240 / test 600. 60-frame chunks, 16-frame boundary drops, IoU leakage check.
- **UNIRI-TID alone cannot carry the paper:** one site, one camera, night-only, staged actors, single class.

### M3FD synthetic weather results — all methods complete (clean split)

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

**Key insights:**

- **A gap ≠ B gap.** Method 2 fog had A=0.154 (contaminated) / 0.076 (clean) that collapsed to −0.001 in B.
  Deterministic filters (Methods 1–2) are invertible; YOLO learns to undo them under mixed training.
  Always run Protocol B.
- **CycleGAN fog 3B gap (+0.049) is weak** — above zero, but below the 0.10 bar.
- **The anomalous 3A fog gap (+0.477 in contaminated run; +0.101 in clean run)** — the clear-only model
  scores poorly on translated fog. Possible causes: mode collapse, domain gap, or sensor-identity
  transfer rather than weather. Sample grids in `runs/thermal_fog/samples/` need inspection.
- **Method 2 rain IR = Method 1 rain IR** (byte-identical). Depth-aware rain was RGB-only; evaluation
  is IR-only. Never report these as different methods for rain.
- **RGB fog/rain generators were trained** — `rgb_fog` and `rgb_rain` completed 50 epochs. However they
  produce no eval-side effect: all M3FD eval YAMLs point at IR image paths.
- **Label filtering removes hardest cases.** Boxes with mean t_IR < 0.1 are removed from both training
  and test fog, making evaluation easier.

---

## Eval harness — conventions

All experiments report through `bench/eval/detect_eval.py`. **One scorer for everything.**

- Per-condition breakdown (clear / fog / rain) from `conditions.csv`.
- Per-object-size AP (APm = medium bucket, 89–97% of UNIRI-TID boxes).
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
  with another. Fix before any Stage B rain experiment.
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

---

## Key references

- **Kristo et al. 2020** — `refs/kristo2020_uniri_tid.pdf` — companion paper for UNIRI-TID.
  Table 2 (camera distance per condition) and Table 3 (AP numbers at IoU=0.5) are the most important.
- **DSERT-RoLL** (CVPR 2026) — real driving RGB+thermal dataset under clear/fog/rain. Used as
  CycleGAN target domain for M3FD Method 3.
- **Depth-Anything-V2-Small-hf** — monocular depth estimator used in M3FD Method 2.
- **docs/generating_weather_conditions.md** — Physics-Informed Diffusion architecture spec for
  future weather generation work.
