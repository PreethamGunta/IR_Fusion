# GATE-IR — benchmark harness and baselines

Benchmarking code for **GATE-IR** (Gated Adaptive Thermal Enhancement for Infrared Detection): a thermal
person-detection pipeline that classifies the weather in each frame, applies a cleanup matched to that
condition, and skips the cleanup entirely when the weather is clear.

This repository holds the **harness and the baselines** — dataset preparation, training, and evaluation.
The GATE-IR model itself is not here; every number produced by this code is a baseline the method has to beat.

---

## The pipeline being evaluated

| stage | what it does | size |
|---|---|---|
| **A — WeatherGate** | 5 image statistics (variance, min, max, entropy, Laplacian variance) → small MLP → clear / fog / rain | ~4.7k params |
| **B — Enhancer** | fog → adaptive gamma; rain → LSRB + CLAHE; **clear → passthrough** | 0 / ~1.9k params |
| **C — Detector** | YOLOv8s with a P2 (stride-4) head for small objects and a transformer neck | — |

Stage D (CycleGAN pseudo-RGB + cross-modal distillation) is deferred.

The efficiency argument is Stage B's cost — nothing for fog, ~1.9k parameters for rain — against all-in-one
restoration models that run a full network on *every* frame regardless of conditions.

## Dataset — UNIRI-TID

Kristo, Ivasic-Kos & Pobar, IEEE DataPort, DOI [10.21227/yec9-yy29](https://doi.org/10.21227/yec9-yy29).
6,340 thermal frames (6,276 annotated), single class `Human`, with **real** fog and rain rather than synthetic
corruption. Companion paper: *Thermal Object Detection in Difficult Weather Conditions Using YOLO*, IEEE Access
2020 — mandatory citation, and the source of the published per-condition numbers.

Two things had to be solved before any result was trustworthy:

**The weather labels are not distributed.** The published 2,663 clear / 1,135 fog / 2,313 rain split is described
in the paper but not shipped. It is reconstructed here from the **burned-in camera frame counter**, read by glyph
template matching → 59 recordings → condition blocks. Validated three independent ways: published counts match
fog to **±1 frame**; per-condition camera distances match the paper's Table 2; the dog appears only in the fog
block, as the paper states.

> **This deliberately uses no image statistics.** WeatherGate's five features *are* image statistics, so
> clustering labels out of that same feature space would make the gate ~100% accurate by construction. Never
> derive the labels that way.

**Splits must be sequence-disjoint.** These are video frames — consecutive frames are ~99.9% identical (median
difference 0.0012), so a random split puts near-duplicates on both sides and inflates mAP for free. Splits here
are sequence-disjoint **and** condition-stratified: train 3,939 / val 1,103 / test 1,234.

## Results so far

Test set, four baselines, sequence-disjoint split:

| model | epochs | mAP@50 | **mAP@50-95** | clear | fog | rain |
|---|---|---|---|---|---|---|
| YOLOv8s | 71 | 0.9643 | 0.6222 | 0.6536 | 0.5496 | 0.5917 |
| YOLO11s | 55 | 0.9647 | 0.6198 | 0.6491 | 0.5341 | 0.6035 |
| YOLOv12s | 76 | 0.9728 | **0.6324** | 0.6580 | 0.5807 | 0.6052 |
| YOLOv8s + weather aug | 69 | 0.9717 | 0.6271 | 0.6558 | 0.5785 | 0.6019 |

**Report mAP@50-95, never mAP@50** — at IoU 0.5 this benchmark is saturated (0.964–0.973, matching the published
97.93% AP), so gains there are meaningless.

**Fog is genuinely harder, and it is not a distance artefact.** Condition is confounded with camera range in this
dataset (fog 0–30 m, clear 50–160 m, rain 30–215 m), so a per-condition gap could just be object scale. It isn't:
fog has the *largest* median box (75.4 px vs clear 71.6, rain 63.2) yet the worst AP. Restricting to same-sized
objects (the medium bucket, 89–97% of all boxes), **fog costs ~0.09–0.13 mAP@50-95** across all four
architectures. Caveat: test fog is 195 boxes from a single recording session.

**The bar for GATE-IR.** Plain heavy augmentation — no gate, no enhancer, no architecture change — already
recovers ~28% of the clear-fog gap (+0.029 fog mAP). A gated pipeline that beats that by less has no
contribution.

Always report `size_profile` beside `per_condition`; that pairing is what makes a weather claim defensible.

## Layout

```
bench/
  data/          dataset preparation
    prepare_uniri.py      annotations -> canonical YOLO labels
    read_counter.py       glyph-match the burned-in frame counter
    build_sequences.py    counter -> 59 recordings
    label_conditions.py   recordings -> clear/fog/rain
    make_splits.py        sequence-disjoint, condition-stratified split
    make_debug_subset.py  tiny mirror of the split for DEBUG runs
  eval/
    detect_eval.py        THE scorer — per-condition and per-object-size AP
  scripts/
    sweep.sbatch          batch submission (resumable)
    overnight.sh          the 7-model baseline sweep
    estimate_runtime.py   measure per-epoch cost before committing to a run
  train_detector.py       train one model + score it
  debugmode.py            the DEBUG switch
  results/                one JSON per run
```

## Running it

```bash
bash bench/setup_env.sh                 # conda env `gateir` (torch cu126, ultralytics 8.4.138)

python bench/data/prepare_uniri.py      # then, in order:
python bench/data/read_counter.py
python bench/data/build_sequences.py
python bench/data/label_conditions.py
python bench/data/make_splits.py

sbatch bench/scripts/sweep.sbatch       # the baseline sweep
python bench/eval/detect_eval.py --run v8s --split test
```

**Test your changes first:** `DEBUG=1` runs the entire pipeline — all seven models, training and per-condition
scoring — on 2 frames per condition per split in about two minutes.

```bash
python bench/data/make_debug_subset.py
DEBUG=1 bash bench/scripts/overnight.sh
```

Debug output is quarantined in `bench/results/debug/` with `_dbg` names and `"debug": true`, so it can never be
mistaken for a real result.

## How results are scored

Every model — baseline or GATE-IR variant — goes through the same `train_detector.py` and the same
`detect_eval.py`, so no comparison depends on differing evaluation code.

`detect_eval.py` scores with **pycocotools** (COCO-comparable numbers) and reports what Ultralytics' `.val()`
cannot: a **per-condition** breakdown and **AP against object size**. Each evaluation automatically cross-checks
its overall figure against `.val()`; observed agreement is ≤ 0.003 mAP. Inference runs once per split and
detections are re-scored per subset, so the breakdown is nearly free.

Training config is pinned rather than left to Ultralytics' `auto`: AdamW at lr0=1e-3, cosine decay, 3-epoch
warmup, `patience=25` early stop, `close_mosaic=10`, effective batch 64 for every model. `auto` picks the
optimiser from the iteration count, which would let a heavier model silently get a different one — the table
would then be comparing optimisers as much as architectures.

## Notes for anyone running this

- **Training is CPU-bound on JPEG decode**, not GPU-bound. Raising batch size does not help; `cache='disk'`
  does. Peak GPU memory is 13–26 GB of 48 GB. Dataloader workers scale to `SLURM_CPUS_PER_TASK - 2`.
- **Runs are resumable.** `last.pt` is written every epoch; re-running the same command continues from it rather
  than restarting the schedule. `sweep.sbatch` uses `--requeue`, so a node failure restarts automatically. This
  is not theoretical — a `NODE_FAIL` killed a sweep mid-run on 2026-09-03.
- **Do not install `thop`.** Version 0.1.1 leaks profiling hooks into the training forward pass under torch 2.14
  and crashes RT-DETR. Ultralytics 8.4 has its own FLOPs counter.
- **Use the `All_In_One_Anot_*` annotations only.** The `ALL_IN_ONE_RGB_ANOT_*` export is malformed and silently
  drops ~2,965 boxes.
- The images are **pseudo-coloured 8-bit RGB**, not radiometric 14-bit single-channel — so `bit_depth=14` and
  `ch: 1` from the original spec do not apply to this data.
- Rebuilding the split **wipes and relinks** `data/UNIRI-TID/yolo/`, by design. An earlier version did not, and
  left 892 images in both `train` and `val`. Assertions check what is on disk, not just the assignment.

## Data and weights

Neither is in this repository. UNIRI-TID comes from IEEE DataPort at the DOI above; detector weights download
automatically on first use.
