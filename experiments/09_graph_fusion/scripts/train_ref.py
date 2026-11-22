#!/usr/bin/env python3
"""Train a Tier-A reference detector (E00 IR-only / E01 RGB-only) on 09 v2 data.

Resume-safe: rerun with the same --name to continue from last.pt.
All paths are derived from this file's location; nothing is hardcoded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
PRET = REPO / "models" / "pretrained"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name, e.g. e00_ir / e01_rgb")
    ap.add_argument("--data", required=True, help="dataset yaml, relative to EXP or absolute")
    ap.add_argument("--model", default="yolo12s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from ultralytics import YOLO

    data = Path(args.data)
    if not data.is_absolute():
        data = EXP / data
    run_dir = EXP / "runs" / args.name
    last = run_dir / "weights" / "last.pt"
    resume = last.exists()

    if resume:
        print(f"[train_ref] resuming {args.name} from {last}", flush=True)
        model = YOLO(str(last))
    else:
        model = YOLO(str(PRET / args.model))

    t0 = time.time()
    model.train(
        data=str(data),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=args.seed,
        workers=args.workers,
        project=str(EXP / "runs"),
        name=args.name,
        exist_ok=True,
        resume=resume,
        deterministic=True,
        val=True,
        cache=False,
    )
    dt = time.time() - t0

    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": args.name,
        "model": args.model,
        "data": str(data),
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "resumed": resume,
        "wall_h": round(dt / 3600, 3),
        "params_m": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 3),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "ref_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (run_dir / "DONE").write_text("ok\n")
    print(f"[train_ref] {args.name} DONE in {dt / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
