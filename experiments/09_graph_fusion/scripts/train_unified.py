#!/usr/bin/env python3
"""Train any arm through the unified pipeline (``src/gfdata/unified.py``).

  --mode ir    IR-only YOLO12s (thermal labels)
  --mode rgb   RGB-only YOLO12s (thermal labels)
  --mode pair  GF-YOLO dual stream, --fuse E10/E14/E21/E30/E35

Full dataset, full augmentation (mosaic + close_mosaic schedule, flips,
scale/translate, HSV/value jitter, erasing). Speed is controlled with
``--epochs`` -- never by shrinking the data.

Examples
--------
  python scripts/train_unified.py --name e00_ir_unified --mode ir \
      --data configs/dronevehicle_v2_ir.yaml
  python scripts/train_unified.py --name e30_unified --mode pair --fuse E30 \
      --data configs/dronevehicle_v2_paired.yaml \
      --rgb-yaml configs/dronevehicle_v2_rgb.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
PRET = REPO / "models" / "pretrained"
for p in (str(EXP / "src"), str(EXP)):
    if p not in sys.path:
        sys.path.insert(0, p)


def write_code_state(run_dir: Path) -> None:
    """Record git HEAD + file hashes before training (provenance, audit §1.3)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in (EXP / "src").rglob("*.py")]
                   + [EXP / "scripts" / "train_unified.py"])
    hashes = {str(p.relative_to(EXP)): hashlib.sha256(p.read_bytes()).hexdigest()[:16]
              for p in files}
    try:
        head = subprocess.run(["git", "-C", str(EXP.parents[1]), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        head = "unknown"
    (run_dir / "code_state.json").write_text(json.dumps({
        "git_head": head,
        "sha256_16": hashes,
        "snapshot_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", required=True, choices=["ir", "rgb", "pair"])
    ap.add_argument("--fuse", default="E30")
    ap.add_argument("--data", required=True)
    ap.add_argument("--rgb-yaml", default=None)
    ap.add_argument("--pretrained", default=str(PRET / "yolo12s.pt"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--fuse-warmup", type=int, default=0,
                    help="epochs to hold the fuse contribution at exactly zero "
                         "(B1); 0 = off, the previous behaviour")
    ap.add_argument("--fuse-ramp", type=int, default=1,
                    help="epochs to ramp the fuse contribution 0 -> 1 after "
                         "the warm-up (only used when --fuse-warmup > 0)")
    ap.add_argument("--close-mosaic", type=int, default=None,
                    help="override close_mosaic (default: Ultralytics' 10); "
                         "set 0 for short mechanism probes so the whole run "
                         "keeps mosaic on, matching the early phase of a "
                         "longer schedule")
    # Pin the optimizer: `optimizer=auto` silently switches between AdamW
    # (6k-image runs, lr 1.11e-3) and MuSGD (full-train, lr 1e-2) by iteration
    # count (audit 2026-11-23). The defaults below reproduce the auto-resolved
    # full-train settings exactly, including warmup_bias_lr=0.0.
    ap.add_argument("--optimizer", default="MuSGD")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    args = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    from gfdata.unified import make_dataset

    data = Path(args.data)
    data = data if data.is_absolute() else EXP / data
    rgb = None
    if args.mode == "pair":
        if not args.rgb_yaml:
            ap.error("--mode pair requires --rgb-yaml")
        rgb = Path(args.rgb_yaml)
        rgb = rgb if rgb.is_absolute() else EXP / rgb

    run_dir = EXP / "runs" / args.name
    last = run_dir / "weights" / "last.pt"
    resume = last.exists()

    class UnifiedRefTrainer(DetectionTrainer):
        """Single-stream arm: same unified dataset, stock validator."""

        data_yaml = str(data)
        modality = args.mode

        def build_dataset(self, img_path, mode="train", batch=None):
            split = mode if mode in ("train", "val", "test") else "train"
            return make_dataset(self.data_yaml, split, modality=self.modality,
                                mode=mode, imgsz=self.args.imgsz,
                                batch=batch or self.args.batch,
                                cache=self.args.cache or None)

    kw = dict(data=str(data), epochs=args.epochs, patience=args.patience,
              batch=args.batch, imgsz=args.imgsz, seed=args.seed,
              workers=args.workers, deterministic=True,
              project=str(EXP / "runs"), name=args.name, exist_ok=True,
              plots=False, val=True, save=True, cache=False,
              optimizer=args.optimizer, lr0=args.lr0,
              momentum=args.momentum, warmup_bias_lr=0.0)
    if args.close_mosaic is not None:
        kw["close_mosaic"] = int(args.close_mosaic)

    if args.mode == "pair":
        from gfyolo.gf_trainer import GFTrainer
        GFTrainer.paired_yaml = str(data)
        GFTrainer.rgb_yaml = str(rgb)
        GFTrainer.fuse = args.fuse
        GFTrainer.fuse_kwargs = ({"d": args.d}
                                 if args.fuse in ("E21", "E30", "E35") else {})
        GFTrainer.fuse_warmup_epochs = args.fuse_warmup
        GFTrainer.fuse_ramp_epochs = args.fuse_ramp
        model = YOLO(str(last) if resume else GFTrainer.model_cfg)
        trainer_cls = GFTrainer
        if resume:
            kw["resume"] = True
        else:
            kw["pretrained"] = args.pretrained
            kw["augmentations"] = []
    else:
        model = YOLO(str(last) if resume else args.pretrained)
        trainer_cls = UnifiedRefTrainer
        if resume:
            kw["resume"] = True

    t0 = time.time()
    write_code_state(run_dir)
    model.train(trainer=trainer_cls, **kw)
    dt = time.time() - t0

    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": args.name, "mode": args.mode, "fuse": args.fuse,
        "fuse_warmup_epochs": args.fuse_warmup,
        "fuse_ramp_epochs": args.fuse_ramp,
        "close_mosaic": kw.get("close_mosaic", "ultralytics-default"),
        "data": str(data), "rgb_yaml": str(rgb) if rgb else None,
        "epochs_requested": args.epochs, "patience": args.patience,
        "batch": args.batch, "imgsz": args.imgsz, "seed": args.seed,
        "workers": args.workers, "resumed": resume,
        "optimizer": args.optimizer, "lr0": args.lr0,
        "momentum": args.momentum, "warmup_bias_lr": 0.0,
        "wall_h": round(dt / 3600, 3), "pipeline": "gfdata.unified",
        "hyp": "full augmentation (mosaic+close_mosaic, flips, affine, HSV/value, erasing)",
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "ref_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (run_dir / "DONE").write_text("ok\n")
    print(f"[train_unified] {args.name} DONE in {dt / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
