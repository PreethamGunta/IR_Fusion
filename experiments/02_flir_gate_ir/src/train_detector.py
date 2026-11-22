"""Train a YOLOv8s detector for one of the GATE-IR-on-FLIR datasets.

Uses the same recipe for every arm (yolov8s, 100 epochs, batch 16, imgsz 640,
AdamW auto, close_mosaic 10, patience 25) so the only difference between arms
is the data pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path

from .paths import CONFIGS, PRETRAINED, REPO_ROOT, RUNS, ensure_dirs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Train YOLOv8s for a GATE-IR-on-FLIR arm")
    ap.add_argument("--dataset", required=True,
                    help="dataset name; reads configs/<dataset>_train.yaml")
    ap.add_argument("--name", default=None, help="run name (default: <dataset>_yolov8s)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--weights", default=str(PRETRAINED / "yolov8s.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="ram", choices=["disk", "ram", "false"])
    ap.add_argument("--resume", action="store_true",
                    help="resume runs/<name>/weights/last.pt (keeps partial progress)")
    args = ap.parse_args()

    ensure_dirs()
    name = args.name or f"{args.dataset}_yolov8s"
    cfg = CONFIGS / f"{args.dataset}_train.yaml"
    if not cfg.exists():
        raise FileNotFoundError(f"missing {cfg}; run build_datasets.py first")
    if not Path(args.weights).exists():
        raise FileNotFoundError(args.weights)

    from ultralytics import YOLO

    if args.resume:
        last = RUNS / name / "weights" / "last.pt"
        if not last.exists():
            raise FileNotFoundError(f"--resume requested but {last} is missing")
        print(f"[resume] {last}")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(args.weights)
        cache = False if args.cache == "false" else args.cache
        model.train(
            data=str(cfg),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=str(RUNS),
            name=name,
            exist_ok=True,
            cache=cache,
            workers=args.workers,
            device=args.device,
            patience=25,
            close_mosaic=10,
            deterministic=True,
            seed=args.seed,
            optimizer="auto",
            cos_lr=False,
            val=True,
            plots=True,
            verbose=True,
        )

    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    prov = {
        "dataset": args.dataset,
        "name": name,
        "data_yaml": str(cfg),
        "weights_init": str(Path(args.weights).resolve()),
        "weights_init_sha256": sha256_file(Path(args.weights)),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "cache": args.cache,
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "platform": platform.platform(),
    }
    (run_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
