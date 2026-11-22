"""Train one detector on M3FD and score it, with provenance that can be audited.

Every baseline and variant goes through this script so the numbers come out of
identical training and evaluation code -- otherwise a reviewer can ask whether
our mAP and a baseline's mAP mean the same thing.

Three properties this file is built around, all of them fixes for ways the
previous version could produce a plausible number from the wrong thing:

1. RESUME MUST NOT CHANGE THE EXPERIMENT.  Ultralytics `check_resume()` restores
   *every* argument from the checkpoint and only honours an override for a fixed
   whitelist (imgsz, batch, device, close_mosaic, augmentments, save_period,
   workers, cache, patience, time, freeze, val, plots, channels_last,
   distill_model, save_dir).  `data`, `epochs`, `seed`, `optimizer` and `lr0`
   are NOT overridable.  It also has two silent substitutions:

     * `if not Path(ckpt_args["data"]).exists(): ckpt_args["data"] = self.args.data`
       -- so a checkpoint whose dataset YAML has been deleted silently adopts
       whatever data this call passed, or, if this call passed no data at all,
       the `coco8.yaml` default.
     * `resume=True` (a bool) makes it call `get_latest_run()`, which is
       `glob("./**/last*.pt", recursive=True)` -- it picks the most recently
       modified `last*.pt` anywhere under the working directory.  From the repo
       root that can be a different experiment entirely.

   So: resume is always given an explicit checkpoint path, and the checkpoint's
   own recorded `data`/`epochs`/`nc` must match the CLI before we will touch it.
   Mismatch is a hard error, never a silent restart -- an archived result proved
   why, carrying `"resumed": "eval-only"` over a pre-resplit checkpoint and
   scoring it against a different split.

   The old code raised its "silently fell back to COCO8" RuntimeError *inside*
   the `try` whose `except` meant "incompatible checkpoint -> restart clean", so
   it swallowed the one error it claimed to detect and then launched an
   unplanned full retrain.  Verification is now outside the try and raises.

2. EARLY STOPPING IS NOT AN INCOMPLETE RUN.  The previous gate was
   `epochs_done < args.epochs -> do not write the result`.  With the default
   patience=25 that fires on every healthy run that stops early, so the clear
   baseline -- the single number every Protocol 3A gap is measured against --
   wrote no JSON, the pipeline's `[ -f weights ]` guard passed anyway, and the
   job exited 0 having silently produced no reference.  A run that returns
   normally from `model.train()` finished, by full schedule or by early stop;
   only a kill or a crash prevents reaching this point at all.  The gate is now
   "did we get here", and `early_stopped` / `best_epoch` are recorded so the
   distinction is visible in the result rather than inferred.

3. EVERY RESULT CARRIES ITS SPLIT.  `split_hash` resolved through a two-element
   candidate list whose entries were byte-identical (a no-op dressed as a
   fallback), so it was `null` for every Method 1/2/3 artifact.  It now resolves
   by walking up from the dataset YAML's `path:` -- see provenance.py.
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
sys.path.insert(0, str(ROOT / "experiments" / "05_detector_bench"))
import debugmode

import provenance

RESULTS = BENCH / "results"
RUNS = BENCH / "runs"

# keep ultralytics' config/weights inside the project
os.environ.setdefault("YOLO_CONFIG_DIR", str(BENCH / ".ultralytics"))

# Arguments that change what gets trained. If any of these differs between a
# checkpoint and this command line, resuming would silently train a different
# experiment -- so they are compared, not overridden.
RESUME_CRITICAL = ("data", "epochs", "nc", "imgsz")


def epochs_done(run_dir):
    """How many epochs a previous attempt at this run got through."""
    path = Path(run_dir) / "results.csv"
    if not path.exists():
        return 0
    try:
        with open(path) as f:
            return sum(1 for _ in csv.DictReader(f))
    except OSError:
        return 0


def best_epoch(run_dir):
    """1-based epoch of the best val fitness in results.csv, or None."""
    path = Path(run_dir) / "results.csv"
    if not path.exists():
        return None
    best, arg = None, None
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                col = next((k for k in row if k.endswith("metrics/mAP50-95(B)")), None)
                if not col:
                    continue
                try:
                    val = float(row[col])
                except (TypeError, ValueError):
                    continue
                if best is None or val > best:
                    best, arg = val, int(float(row.get("epoch", 0))) + 1
    except OSError:
        return None
    return arg


def recorded_train_args(run_dir, last):
    """The `data`/`epochs`/`nc` the previous attempt recorded, or {}.

    args.yaml is ultralytics' own dump of self.args and is cheap to read;
    the checkpoint's `train_args` is the same dict and is the authority when
    args.yaml is missing.
    """
    recorded = {}
    args_yaml = Path(run_dir) / "args.yaml"
    if args_yaml.exists():
        try:
            import yaml
            recorded = yaml.safe_load(args_yaml.read_text()) or {}
        except Exception:
            recorded = {}
    if not recorded:
        try:
            import torch
            ckpt = torch.load(str(last), map_location="cpu", weights_only=False)
            recorded = ckpt.get("train_args") or {}
        except Exception:
            recorded = {}
    return recorded if isinstance(recorded, dict) else {}


def same_dataset(a, b):
    """Compare two dataset paths that may be relative to different cwds."""
    if not a or not b:
        return False
    pa, pb = Path(str(a)), Path(str(b))
    if pa.is_absolute() and pb.is_absolute():
        return pa.resolve() == pb.resolve()
    return os.path.abspath(str(pa)) == os.path.abspath(str(pb))


def check_resume_compatible(name, recorded, data, epochs, nc):
    """Hard-fail if resuming would change the experiment. Returns a note list."""
    problems = []
    if recorded.get("data") is not None and not same_dataset(recorded["data"], data):
        problems.append(f"data: checkpoint={recorded['data']!r} vs requested={data!r}")
    for key, want in (("epochs", epochs), ("nc", nc)):
        got = recorded.get(key)
        if got is not None and int(got) != int(want):
            problems.append(f"{key}: checkpoint={got} vs requested={want}")
    if problems:
        raise SystemExit(
            f"REFUSING TO RESUME {name}: the checkpoint at this run directory was "
            f"trained on a different experiment.\n  "
            + "\n  ".join(problems)
            + "\nUltralytics cannot override these on resume, so continuing would "
              "train/record one thing and claim another. Re-run with --no-resume "
              "(and archive the old run) or move the checkpoint aside."
        )


def verify_trained_on(model, requested_data, name):
    """Post-training assertion: the trainer really used our dataset.

    Runs *after* training and raises on mismatch. It cannot prevent a wasted
    run, but it guarantees a mismatched run never reaches a result JSON.
    """
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        print(f"[verify] {name}: no trainer attached -- skipping dataset assertion")
        return
    actual = str(getattr(trainer.args, "data", "") or "")
    save_dir = str(getattr(trainer, "save_dir", "") or "")
    for label, value in (("data", actual), ("save_dir", save_dir)):
        if "coco8" in value.lower():
            raise SystemExit(
                f"FATAL: {name} trained on {label}={value!r}. Ultralytics "
                f"substitutes coco8 when it cannot reconcile the requested "
                f"dataset with the checkpoint. No result will be written.")
    if actual and not same_dataset(actual, requested_data):
        raise SystemExit(
            f"FATAL: {name} trained on {actual!r} but {requested_data!r} was "
            f"requested. No result will be written.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/yolov8s.pt",
                    help="pretrained .pt (relative to repo root) or a model .yaml for scratch")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    _cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    ap.add_argument("--workers", type=int, default=max(2, min(_cpus - 2, 28)),
                    help="dataloader workers (default: allocated CPUs - 2)")
    ap.add_argument("--cache", default="disk", choices=("disk", "ram", "False"),
                    help="decoded-image cache. Training is CPU-bound on decode of "
                         "1024x768 PNG frames, so this is the setting that buys "
                         "throughput; 'disk' persists across runs.")
    ap.add_argument("--patience", type=int, default=25,
                    help="early stop after this many epochs without val mAP improvement")
    ap.add_argument("--save-period", type=int, default=10)
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any existing checkpoint and start over")
    ap.add_argument("--allow-missing-split-hash", action="store_true",
                    help="proceed even when no split_manifest.json is reachable "
                         "from the dataset YAML (the result's split_hash will "
                         "be null and cannot be traced to a split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weather-aug", action="store_true",
                    help="heavier photometric augmentation, the 'robustness by "
                         "training' baseline")
    args = ap.parse_args()

    from ultralytics import YOLO

    debugmode.banner("train_detector")
    if debugmode.enabled():
        # tiny subset, tiny schedule; results are quarantined below so a debug
        # JSON can never be picked up as a real result
        orig_data = Path(args.data)
        debug_yaml = orig_data.parent / (orig_data.stem + "_debug.yaml")
        if not debug_yaml.exists():
            import subprocess
            subprocess.run([sys.executable, str(BENCH / "make_debug_subset.py"),
                            str(orig_data)])
        DATA = debug_yaml
        args.epochs = debugmode.EPOCHS
        args.imgsz = debugmode.IMGSZ
        args.batch = debugmode.BATCH
    else:
        DATA = Path(args.data)

    if not DATA.exists():
        raise SystemExit(f"no dataset yaml at {DATA}")

    # --- provenance: which split, which commit, which job --------------------
    current_split_hash = provenance.split_hash_for_dataset(DATA)
    if current_split_hash is None and not args.allow_missing_split_hash:
        raise SystemExit(
            f"no split_manifest.json reachable from {DATA} (looked upward from "
            f"the YAML and from its path:). Refusing to train a run that cannot "
            f"be traced to a split. Pass --allow-missing-split-hash to override.")
    if current_split_hash is None:
        print(f"WARNING: no split manifest found for {DATA}; split_hash will be null")

    RESULTS.mkdir(exist_ok=True)
    run_dir = RUNS / args.name
    last, best = run_dir / "weights" / "last.pt", run_dir / "weights" / "best.pt"
    done = epochs_done(run_dir)

    n_train = provenance.count_images(DATA, "train")
    n_val = provenance.count_images(DATA, "val")
    n_test = provenance.count_images(DATA, "test")
    if not n_train:
        raise SystemExit(f"{DATA}: no train images found under its train: entries")

    kw = dict(
        data=str(DATA), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, save_period=args.save_period, cache=args.cache,
        seed=args.seed, project=str(RUNS), name=args.name, exist_ok=True,
        pretrained=args.model.endswith(".pt"), val=True, plots=False, verbose=True,

        # --- optimisation, pinned rather than left to ultralytics' "auto" ----
        # "auto" picks the optimiser from the iteration count, so a heavier
        # model could silently get a different one. AdamW at lr0=1e-3 is what
        # auto would choose at this dataset size; fixing it makes every row use
        # the same recipe.
        optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005,
        cos_lr=True,
        warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
        patience=args.patience,
        close_mosaic=10,
        deterministic=True,

        # thermal is pseudo-coloured with a fixed palette: hue/sat jitter would
        # destroy the palette's meaning, so only value jitter is enabled
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.4,
        degrees=0.0, fliplr=0.5, flipud=0.0,
    )
    if args.weather_aug:
        kw.update(hsv_v=0.8, scale=0.7, translate=0.2, erasing=0.4, mosaic=1.0)

    if debugmode.enabled():
        args.workers = 2
        args.save_period = -1
        args.cache = "False"
    args.cache = False if args.cache == "False" else args.cache

    nc = len(provenance.read_yaml_dataset(DATA).get("names") or []) or None

    # --- resume policy ------------------------------------------------------
    # Ultralytics writes last.pt every epoch, so an interrupted run -- a dead
    # allocation, a killed process -- loses at most one epoch. `resume` restores
    # optimizer, EMA, LR schedule and epoch counter, so it continues rather than
    # restarting the schedule.
    t0 = time.time()
    resumed = None
    trainer_stopped_early = False
    did_train = False

    if args.no_resume:
        print(f"[resume] disabled by --no-resume: training {args.name} from scratch",
              flush=True)
        model = YOLO(args.model)
        model.train(**kw)
        did_train = True
    else:
        recorded = recorded_train_args(run_dir, last) if last.exists() else {}
        if done >= args.epochs and best.exists():
            print(f"[resume] {args.name}: {done}/{args.epochs} epochs already done "
                  f"-- scoring best.pt only", flush=True)
            check_resume_compatible(args.name, recorded, DATA, args.epochs, nc)
            resumed = "eval-only"
            model = YOLO(str(best))
        elif last.exists():
            check_resume_compatible(args.name, recorded, DATA, args.epochs, nc)
            print(f"[resume] {args.name}: continuing from epoch {done}/{args.epochs} "
                  f"via {last}", flush=True)
            resumed = f"from-epoch-{done}"
            model = YOLO(str(last))
            # Explicit path, never the bare True: see module docstring.
            model.train(resume=str(last))
            did_train = True
        else:
            print(f"[resume] {args.name}: no checkpoint -- training from scratch",
                  flush=True)
            model = YOLO(args.model)
            model.train(**kw)
            did_train = True

    train_s = (time.time() - t0) if did_train else 0.0

    # Verification is deliberately outside any try/except: a mismatch must fail
    # the run, not be absorbed and turned into a retrain.
    if resumed != "eval-only":
        verify_trained_on(model, DATA, args.name)
        trainer_stopped_early = bool(getattr(getattr(model, "trainer", None), "stop", False))

    # Always reload best.pt for evaluation rather than scoring whatever EMA state
    # is left in memory from the last training step.
    if best.exists():
        model = YOLO(str(best))
    elif last.exists():
        model = YOLO(str(last))
    else:
        raise SystemExit(f"{args.name}: training produced no weights")

    out = {}
    for split in ("val", "test"):
        m = model.val(data=str(DATA), split=split, imgsz=args.imgsz, verbose=False,
                      project=str(RUNS), name=f"{args.name}_eval_{split}", exist_ok=True)
        out[split] = {
            "mAP50": round(float(m.box.map50), 5),
            "mAP50_95": round(float(m.box.map), 5),
            "precision": round(float(m.box.mp), 5),
            "recall": round(float(m.box.mr), 5),
        }

    # Reaching this point means model.train() returned normally, i.e. the run
    # finished -- either on the full schedule or by early stopping. A kill or a
    # crash never gets here, so gating on epoch count would only ever discard
    # healthy runs.
    final_epochs = epochs_done(run_dir)
    weights_path = best if best.exists() else last
    rec = dict(
        name=args.name, model=args.model, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, workers=args.workers, seed=args.seed,
        weather_aug=args.weather_aug, debug=debugmode.enabled(), resumed=resumed,
        optimizer="AdamW", lr0=0.001, cos_lr=True, patience=args.patience,
        epochs_trained=final_epochs, early_stopped=trainer_stopped_early,
        best_epoch=best_epoch(run_dir),
        train_seconds=round(train_s, 1),
        data=str(DATA), split_hash=current_split_hash,
        n_images={"train": n_train, "val": n_val, "test": n_test},
        weights_path=str(weights_path),
        weights_sha256=provenance.sha256_file(weights_path),
        metrics=out,
        **provenance.run_record(),
    )
    dest = RESULTS / ("debug" if debugmode.enabled() else "")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{args.name}.json").write_text(json.dumps(rec, indent=2))
    print("\n" + json.dumps(rec, indent=2))
    if trainer_stopped_early:
        print(f"\n[note] early stopping fired after {final_epochs}/{args.epochs} "
              f"epochs (best at {rec['best_epoch']}); best.pt is the selected model.")


if __name__ == "__main__":
    main()