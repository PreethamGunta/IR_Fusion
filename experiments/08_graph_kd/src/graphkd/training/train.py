"""Train teachers, the GT-only student, and graph-KD students.

Examples:
  python -m graphkd.training.train --name ir_student --model yolo12n.pt \
      --data configs/ir.yaml --epochs 100 --patience 30
  python -m graphkd.training.train --name a3_node_edge --model yolo12n.pt \
      --data configs/ir.yaml --kd-config configs/kd/a3_node_edge.json --epochs 100
  python -m graphkd.training.train --name fused_teacher --model yolo12s.yaml \
      --pretrained ../models/pretrained/yolo12s.pt --fused --data configs/paired.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from graphkd import paths
from graphkd.data.dataset import NO_AUG
from graphkd.training.trainer import GraphKDTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="yolo12n.pt")
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--data", default=str(paths.CONFIGS / "ir.yaml"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="ram")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--paired-yaml", default=None)
    ap.add_argument("--rgb-yaml", default=None)
    ap.add_argument("--paired-rot90", action="store_true")
    ap.add_argument("--kd-config", default=None)
    ap.add_argument("--fused", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--exist-ok", action="store_true")
    args = ap.parse_args()

    paths.ensure_dirs()
    kd_cfg = None
    if args.kd_config:
        kd_cfg = json.loads(Path(args.kd_config).read_text())
        kd_cfg.setdefault("nc", 4)
        GraphKDTrainer.kd_cfg = kd_cfg
    else:
        GraphKDTrainer.kd_cfg = None
    GraphKDTrainer.fused = bool(args.fused)
    GraphKDTrainer.paired = bool(args.fused)
    GraphKDTrainer.paired_yaml = args.paired_yaml
    GraphKDTrainer.rgb_yaml = args.rgb_yaml
    GraphKDTrainer.paired_augment = bool(args.fused)
    GraphKDTrainer.paired_rot90 = bool(args.paired_rot90)

    model_path = args.model
    if args.resume:
        model_path = str(paths.RUNS / args.name / "weights" / "last.pt")
    m = YOLO(model_path)

    kw = dict(
        data=args.data, epochs=args.epochs, patience=args.patience,
        batch=args.batch, imgsz=args.imgsz, device=args.device,
        workers=args.workers, seed=args.seed, deterministic=True,
        project=str(paths.RUNS), name=args.name, exist_ok=args.exist_ok or args.resume,
        cache=args.cache, plots=False, val=(not args.no_val),
        save=True, save_period=-1, verbose=True, fraction=args.fraction,
        pretrained=args.pretrained if args.pretrained else True,
    )
    if args.resume:
        kw.pop("pretrained", None)
        kw["resume"] = True
    if kd_cfg is not None or args.no_aug:
        kw.update({k: v for k, v in NO_AUG.items() if k != "augmentations"})
    # UV/Ultralytics does not accept augmentations=None; an empty list disables
    # the default Blur/CLAHE albumentations that would break cache alignment.
    if kd_cfg is not None or args.no_aug:
        kw["augmentations"] = []

    m.train(trainer=GraphKDTrainer, **kw)


if __name__ == "__main__":
    main()
