#!/usr/bin/env python3
"""Train E02: 6-channel early fusion [IR | RGB] on paired DV v2 data.

Mirrors scripts/train_ref.py (resume-safe, DONE marker, ref_meta.json) but
uses the fused trainer from src/gfyolo. The paired dataset supplies joint
geometry augs (rot90 on for DroneVehicle) and per-modality photometric
jitter; standard Ultralytics augmentation is off inside the paired loader
(the 08 fused-teacher recipe).

Usage (inside the allocation):
  PYTHONPATH=src python scripts/train_fused.py \
      --name e02_fused --data configs/dronevehicle_v2_paired.yaml \
      --rgb-yaml configs/dronevehicle_v2_rgb.yaml --paired-rot90
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
PRET = REPO / "models" / "pretrained"
SRC = EXP / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="e02_fused")
    ap.add_argument("--data", default="configs/dronevehicle_v2_paired.yaml")
    ap.add_argument("--rgb-yaml", default="configs/dronevehicle_v2_rgb.yaml")
    ap.add_argument("--model", default="yolo12s.yaml")
    ap.add_argument("--pretrained", default=str(PRET / "yolo12s.pt"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--paired-rot90", action="store_true")
    args = ap.parse_args()

    from gfyolo.fused_trainer import FusedDetectionTrainer
    from ultralytics import YOLO

    data = Path(args.data)
    if not data.is_absolute():
        data = EXP / data
    rgb = Path(args.rgb_yaml)
    if not rgb.is_absolute():
        rgb = EXP / rgb

    run_dir = EXP / "runs" / args.name
    last = run_dir / "weights" / "last.pt"
    resume = last.exists()

    FusedDetectionTrainer.paired_yaml = str(data)
    FusedDetectionTrainer.rgb_yaml = str(rgb)
    FusedDetectionTrainer.paired_rot90 = bool(args.paired_rot90)

    model_path = str(last) if resume else args.model
    m = YOLO(model_path)

    kw = dict(
        data=str(data), epochs=args.epochs, patience=args.patience,
        batch=args.batch, imgsz=args.imgsz, seed=args.seed,
        workers=args.workers, deterministic=True,
        project=str(EXP / "runs"), name=args.name, exist_ok=True,
        plots=False, val=True, save=True, cache=False,
    )
    if resume:
        kw["resume"] = True
    else:
        kw["pretrained"] = args.pretrained
        kw["augmentations"] = []  # disable Blur/CLAHE: breaks modality pairing

    t0 = time.time()
    m.train(trainer=FusedDetectionTrainer, **kw)
    dt = time.time() - t0

    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": args.name,
        "model": args.model,
        "data": str(data),
        "rgb_yaml": str(rgb),
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "paired_rot90": bool(args.paired_rot90),
        "resumed": resume,
        "wall_h": round(dt / 3600, 3),
        "params_m": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 3),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "ref_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (run_dir / "DONE").write_text("ok\n")
    print(f"[train_fused] {args.name} DONE in {dt / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
