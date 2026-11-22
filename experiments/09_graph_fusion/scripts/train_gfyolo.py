#!/usr/bin/env python3
"""Train a GF-YOLO ladder rung (E10/E14/E21/E30/E35) on paired DV data.

Resume-safe (last.pt + DONE marker + meta), same pattern as train_ref.py /
train_fused.py. Example (screening subset):

  PYTHONPATH=src python scripts/train_gfyolo.py --name e10_screening \
      --fuse E10 --data configs/dronevehicle_v2_paired_screening.yaml \
      --rgb-yaml configs/dronevehicle_v2_rgb_screening.yaml --paired-rot90
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
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fuse", required=True,
                    choices=["E10", "E14", "E21", "E30", "E35"])
    ap.add_argument("--data", required=True, help="paired yaml (train may be the screening list)")
    ap.add_argument("--rgb-yaml", required=True)
    ap.add_argument("--pretrained", default=str(PRET / "yolo12s.pt"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--paired-rot90", action="store_true")
    ap.add_argument("--d", type=int, default=128)
    args = ap.parse_args()

    from gfyolo.gf_trainer import GFTrainer
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

    GFTrainer.paired_yaml = str(data)
    GFTrainer.rgb_yaml = str(rgb)
    GFTrainer.paired_rot90 = bool(args.paired_rot90)
    GFTrainer.fuse = args.fuse
    # only the attention/graph rungs take the projection width d
    GFTrainer.fuse_kwargs = ({"d": args.d}
                             if args.fuse in ("E21", "E30", "E35") else {})

    m = YOLO(str(last) if resume else GFTrainer.model_cfg)
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
        kw["augmentations"] = []

    t0 = time.time()
    m.train(trainer=GFTrainer, **kw)
    dt = time.time() - t0

    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": args.name, "fuse": args.fuse, "d": args.d,
        "data": str(data), "rgb_yaml": str(rgb),
        "epochs_requested": args.epochs, "patience": args.patience,
        "batch": args.batch, "imgsz": args.imgsz, "seed": args.seed,
        "paired_rot90": bool(args.paired_rot90), "resumed": resume,
        "wall_h": round(dt / 3600, 3),
        "params_m": round(sum(p.numel() for p in m.model.parameters()) / 1e6, 3),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "ref_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (run_dir / "DONE").write_text("ok\n")
    print(f"[train_gfyolo] {args.name} DONE in {dt / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
