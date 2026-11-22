"""Train one detector on a dataset YAML produced by ``build_dataset.py``.

Deliberately small: the protocol compares arms trained with one recipe, so the
only variable is the data pipeline. Resume is always an explicit ``last.pt``
path (never ``resume=True`` without a checkpoint), which is the same guard used
in workstream 03.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from . import paths

os.environ.setdefault("YOLO_CONFIG_DIR", str(paths.EXP_ROOT / ".ultralytics"))


def _sha256(path: Path) -> str | None:
    import hashlib
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _epochs_done(run_dir: Path) -> int:
    path = run_dir / "results.csv"
    if not path.exists():
        return 0
    try:
        with open(path) as f:
            return sum(1 for _ in csv.DictReader(f))
    except OSError:
        return 0


def _best_epoch(run_dir: Path) -> int | None:
    path = run_dir / "results.csv"
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


def _train_image_count(data_yaml: Path) -> int:
    """Number of images in the YAML's train entries (0 if unreadable)."""
    try:
        import yaml
        cfg = yaml.safe_load(data_yaml.read_text())
        root = Path(cfg.get("path", data_yaml.parent))
        entries = cfg.get("train", [])
        entries = [entries] if isinstance(entries, str) else list(entries or [])
        n = 0
        for entry in entries:
            p = Path(entry)
            p = p if p.is_absolute() else root / p
            if p.is_dir():
                n += sum(1 for f in p.iterdir() if f.suffix.lower() in paths.IMG_EXTS)
            elif p.is_file():
                n += 1
        return n
    except Exception:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a detector for one 07 arm")
    ap.add_argument("--data", required=True, help="dataset YAML path (or config name)")
    ap.add_argument("--name", required=True, help="run name under runs/")
    ap.add_argument("--weights", default=str(paths.REPO_ROOT / "models" / "pretrained" / "yolo12s.pt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="ram", choices=["disk", "ram", "false"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    paths.ensure_dirs()
    data = Path(args.data)
    if not data.is_absolute():
        candidate = paths.CONFIGS / (args.data if args.data.endswith(".yaml") else f"{args.data}.yaml")
        data = candidate
    if not data.exists():
        raise SystemExit(f"dataset YAML not found: {data}")
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")

    run_dir = paths.RUNS / args.name
    result_path = paths.RESULTS / f"train_{args.name}.json"
    started = time.time()

    # A 48 GB allocation cannot hold an 18k-image RAM cache plus workers and
    # augmentation buffers: one sensor arm was SIGKILLed after two minutes.
    # Large datasets fall back to on-the-fly decoding and fewer workers.
    n_train = _train_image_count(data)
    cache = args.cache
    workers = args.workers
    if n_train > 15000:
        if cache == "ram":
            print(f"[train] {n_train} train images: overriding cache=ram -> false")
            cache = "false"
        workers = min(workers, 12)

    from ultralytics import YOLO

    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"--resume requested but {last} is missing")
        print(f"[train] resume {last}")
        model = YOLO(str(last))
        # `cache` and `workers` are on Ultralytics' resume-override whitelist,
        # so a checkpoint saved with cache=ram cannot re-trigger the OOM that
        # killed the first sensor attempt.
        model.train(resume=True, cache=False if cache == "false" else cache,
                    workers=workers)
    else:
        model = YOLO(str(weights))
        model.train(
            data=str(data),
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=workers,
            device=args.device,
            seed=args.seed,
            optimizer="auto",
            close_mosaic=10,
            cache=False if cache == "false" else cache,
            project=str(paths.RUNS),
            name=args.name,
            exist_ok=True,
            plots=True,
        )

    result = {
        "name": args.name,
        "data": str(data.resolve()),
        "weights": str(weights.resolve()),
        "weights_sha256": _sha256(weights),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "cache": cache,
        "workers": workers,
        "seed": args.seed,
        "epochs_done": _epochs_done(run_dir),
        "best_epoch": _best_epoch(run_dir),
        "run_dir": str(run_dir),
        "elapsed_seconds": round(time.time() - started, 1),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": _git_commit(),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    (run_dir / "TRAIN_DONE").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    print(json.dumps(result, indent=2))
    print(f"[train] -> {result_path}")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(paths.REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()
