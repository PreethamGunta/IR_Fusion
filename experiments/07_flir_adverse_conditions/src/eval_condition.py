"""Evaluate one detector on a 07 dataset YAML and write a provenance JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import paths


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _resolve_data(value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = paths.CONFIGS / (value if value.endswith(".yaml") else f"{value}.yaml")
    return p


def _count_images(data_yaml: Path, split: str) -> int:
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg.get("path", data_yaml.parent))
    entries = cfg.get(split, cfg.get("val"))
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a detector on a 07 split")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths.ensure_dirs()
    data = _resolve_data(args.data)
    weights = Path(args.weights)
    if not data.exists():
        raise SystemExit(f"dataset YAML not found: {data}")
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")
    out_file = Path(args.out) if args.out else paths.RESULTS / f"eval_{args.name}.json"
    if out_file.exists() and not args.force:
        print(f"[eval] SKIP existing {out_file}")
        return

    n_images = _count_images(data, args.split)
    if n_images == 0:
        raise SystemExit(
            f"{data}: 0 images visible for split {args.split!r}; refusing to "
            f"report an mAP computed over nothing"
        )

    started = time.time()
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=str(paths.RUNS),
        name=f"eval_{args.name}",
        exist_ok=True,
        plots=False,
    )

    result = {
        "name": args.name,
        "weights": str(weights.resolve()),
        "weights_sha256": _sha256(weights),
        "data": str(data.resolve()),
        "split": args.split,
        "n_images": n_images,
        "imgsz": args.imgsz,
        "mAP50_95": round(float(metrics.box.map), 5),
        "mAP50": round(float(metrics.box.map50), 5),
        "precision": round(float(metrics.box.mp), 5),
        "recall": round(float(metrics.box.mr), 5),
        "elapsed_seconds": round(time.time() - started, 1),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[eval] -> {out_file}")


if __name__ == "__main__":
    main()
