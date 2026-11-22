"""Train one detector on UNIRI-TID and evaluate it through the shared harness.

Every baseline and GATE-IR variant goes through this same script so that the
numbers are produced by identical training and evaluation code -- otherwise a
reviewer can ask whether our mAP and a baseline's mAP mean the same thing.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parents[1]
sys.path.insert(0, str(BENCH))
import debugmode
from eval.detect_eval import evaluate, report

UNIRI = ROOT / "data" / "uniri_tid"
RESULTS = BENCH / "results"
RUNS = BENCH / "runs"

# keep ultralytics' config/weights inside the project
os.environ.setdefault("YOLO_CONFIG_DIR", str(BENCH / ".ultralytics"))


def epochs_done(run_dir):
    """How many epochs a previous attempt at this run got through."""
    path = run_dir / "results.csv"
    if not path.exists():
        return 0
    try:
        return sum(1 for _ in csv.DictReader(open(path)))
    except OSError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/yolov8s.pt", help="pretrained .pt (relative to repo root) or a model .yaml for scratch")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    # 64 is Ultralytics' nominal batch size, so accumulate==1 and no gradient
    # accumulation is involved -- and it keeps the 48 GB card busy, where 16 left
    # it mostly idle waiting on JPEG decode.
    ap.add_argument("--batch", type=int, default=64)
    # Scale to whatever Slurm actually gave us, leaving 2 cores for the main
    # process and the GPU feed. Worker count affects prefetch only -- not sample
    # order, augmentation or optimisation -- so runs stay comparable across
    # allocations of different sizes.
    _cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    ap.add_argument("--workers", type=int, default=max(2, min(_cpus - 2, 28)),
                    help="dataloader workers (default: allocated CPUs - 2)")
    ap.add_argument("--cache", default="disk", choices=("disk", "ram", "False"),
                    help="decoded-image cache. Training here is CPU-bound on JPEG decode of "
                         "1280x960 frames, not GPU-bound, so this is the setting that actually "
                         "buys throughput; 'disk' persists across runs so only the first pays.")
    ap.add_argument("--patience", type=int, default=25,
                    help="early stop after this many epochs without val mAP improvement")
    ap.add_argument("--save-period", type=int, default=10,
                    help="also keep an epoch snapshot every N epochs; last.pt is written every epoch regardless")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing checkpoint and start over")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weather-aug", action="store_true",
                    help="heavier photometric augmentation, the 'robustness by training' baseline")
    ap.add_argument("--data", default=None,
                    help="override dataset YAML (for preprocessing variants)")
    args = ap.parse_args()

    from ultralytics import YOLO

    debugmode.banner("train_detector")
    if args.data:
        DATA = Path(args.data)
    elif debugmode.enabled():
        # tiny subset, tiny schedule; results are quarantined below so a debug
        # JSON can never be picked up as a real result
        DATA = UNIRI / "yolo_debug" / "uniri_debug.yaml"
        args.epochs = debugmode.EPOCHS
        args.imgsz = debugmode.IMGSZ
        args.batch = debugmode.BATCH
    else:
        DATA = UNIRI / "yolo" / "uniri.yaml"
    if not DATA.exists():
        raise SystemExit(f"no dataset yaml at {DATA}"
                         + (" -- run bench/data/make_debug_subset.py" if debugmode.enabled() else ""))

    if debugmode.enabled():
        args.workers = 2
        args.save_period = -1
        args.cache = "False"
    args.cache = False if args.cache == "False" else args.cache

    RESULTS.mkdir(exist_ok=True)
    run_dir = RUNS / args.name
    last, best = run_dir / "weights" / "last.pt", run_dir / "weights" / "best.pt"
    done = epochs_done(run_dir)

    kw = dict(
        data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, save_period=args.save_period, cache=args.cache,
        seed=args.seed, project=str(RUNS), name=args.name, exist_ok=True,
        pretrained=args.model.endswith(".pt"), val=True, plots=False, verbose=True,

        # --- optimisation, pinned rather than left to ultralytics' "auto" ------
        # "auto" picks the optimiser from the iteration count, so a heavier model
        # could silently get a different one and the table would be comparing
        # optimisers as much as architectures. AdamW at lr0=1e-3 is what auto
        # would choose at this dataset size (<10k iterations); fixing it makes
        # every row use the same recipe.
        optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005,
        cos_lr=True,                     # cosine decay 1e-3 -> 1e-5
        warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
        patience=args.patience,          # early stop once val mAP stops improving
        close_mosaic=10,                 # last 10 epochs on undistorted images
        deterministic=True,

        # thermal is pseudo-coloured with a fixed palette: hue/sat jitter would
        # destroy the palette's meaning, so only value jitter is enabled
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.4,
        degrees=0.0, fliplr=0.5, flipud=0.0,
    )
    if args.weather_aug:
        # crude stand-in for degraded weather: stronger brightness/contrast swings,
        # blur-like scale jitter and erasing. No preprocessing module involved.
        kw.update(hsv_v=0.8, scale=0.7, translate=0.2, erasing=0.4, mosaic=1.0)

    # Resume policy. Ultralytics writes last.pt every epoch, so an interrupted
    # run -- a dead allocation, a killed process -- loses at most one epoch.
    # `resume=True` restores optimizer, EMA, LR schedule and epoch counter from
    # the checkpoint, so it continues rather than restarting the schedule.
    t0 = time.time()
    if done >= args.epochs and best.exists() and not args.no_resume:
        print(f"[resume] {args.name}: {done}/{args.epochs} epochs already done -- scoring only", flush=True)
        train_s, resumed = 0.0, "eval-only"
        model = YOLO(str(best))
    elif last.exists() and not args.no_resume:
        print(f"[resume] {args.name}: continuing from epoch {done}/{args.epochs}", flush=True)
        resumed = f"from-epoch-{done}"
        model = YOLO(str(last))
        try:
            model.train(resume=True)
        except Exception as exc:
            # a checkpoint from a different config cannot be resumed into this one
            print(f"[resume] failed ({type(exc).__name__}: {exc}) -- restarting clean", flush=True)
            resumed = "restarted"
            model = YOLO(args.model)
            model.train(**kw)
        train_s = time.time() - t0
    else:
        resumed = None
        model = YOLO(args.model)
        model.train(**kw)
        train_s = time.time() - t0

    out = {}
    for split in ("val", "test"):
        # project/name pinned so Ultralytics does not scatter a runs/ dir at the repo root
        m = model.val(data=str(DATA), split=split, imgsz=args.imgsz, verbose=False,
                      project=str(RUNS), name=f"{args.name}_eval_{split}", exist_ok=True)
        out[split] = {
            "mAP50": round(float(m.box.map50), 5),
            "mAP50_95": round(float(m.box.map), 5),
            "precision": round(float(m.box.mp), 5),
            "recall": round(float(m.box.mr), 5),
        }

    # the per-condition and per-object-size breakdown, through the shared scorer
    best = RUNS / args.name / "weights" / "best.pt"
    detail = {}
    for split in ("val", "test"):
        try:
            e = evaluate(best, split=split, name=args.name, imgsz=args.imgsz,
                         data_root=DATA.parent if args.data else None,
                         yaml_path=DATA if args.data else None)
            report(e)
            detail[split] = dict(per_condition=e["per_condition"],
                                 size_profile=e["size_profile"],
                                 crosscheck=e.get("crosscheck"))
        except Exception as exc:
            # a broken breakdown must not throw away a finished training run
            detail[split] = dict(error=f"{type(exc).__name__}: {exc}")

    rec = dict(name=args.name, model=args.model, epochs=args.epochs, imgsz=args.imgsz,
               batch=args.batch, workers=args.workers, seed=args.seed,
               weather_aug=args.weather_aug, debug=debugmode.enabled(), resumed=resumed,
               optimizer="AdamW", lr0=0.001, cos_lr=True, patience=args.patience,
               epochs_trained=epochs_done(run_dir),
               train_seconds=round(train_s, 1), metrics=out, detail=detail)
    dest = RESULTS / ("debug" if debugmode.enabled() else "")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{args.name}.json").write_text(json.dumps(rec, indent=2))
    print("\n" + json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
