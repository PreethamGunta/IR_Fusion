"""Evaluate one detector on one matched clear/fog test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .paths import REPO_ROOT, RESULTS, RUNS, ensure_dirs

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


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


def count_images(data_yaml: Path, split: str = "test") -> int:
    d = yaml.safe_load(data_yaml.read_text())
    root = Path(d["path"])
    entries = d.get(split, d.get("val"))
    entries = [entries] if isinstance(entries, str) else list(entries)
    n = 0
    for entry in entries:
        p = Path(entry)
        p = p if p.is_absolute() else root / p
        if p.is_dir():
            n += sum(1 for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
        elif p.is_file():
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a YOLO detector on a matched split")
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    out_file = args.out or (RESULTS / f"{args.name}.json")
    if out_file.exists() and not args.force:
        print(f"SKIP: {out_file} exists (use --force)")
        return
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    if not args.data.exists():
        raise FileNotFoundError(args.data)

    from ultralytics import YOLO

    n_images = count_images(args.data, args.split)
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=str(RUNS),
        name=f"eval_{args.name}",
        exist_ok=True,
        plots=True,
    )

    res = {
        "name": args.name,
        "weights": str(args.weights.resolve()),
        "weights_sha256": sha256_file(args.weights),
        "data": str(args.data.resolve()),
        "split": args.split,
        "n_images": n_images,
        "imgsz": args.imgsz,
        "mAP50_95": round(float(metrics.box.map), 5),
        "mAP50": round(float(metrics.box.map50), 5),
        "precision": round(float(metrics.box.mp), 5),
        "recall": round(float(metrics.box.mr), 5),
        "git_commit": git_commit(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"-> {out_file}")


if __name__ == "__main__":
    main()
