#!/usr/bin/env python3
"""Train any arm through the unified pipeline **V2** (``src/gfdata/unified_v2.py``).

  --mode ir    IR-only YOLO12s (thermal labels)      -> E00 by identity
  --mode rgb   RGB-only YOLO12s (thermal labels)
  --mode pair  dual-stream GF-YOLO, --fuse E10/...   -> IR|RGB, 6 channels

Differences from ``train_unified.py`` (V1)
------------------------------------------
1. The dataset comes from ``gfdata.unified_v2``, which owns no augmentation
   code: the IR branch *is* Ultralytics' stock ``YOLODataset`` and the RGB
   branch is the same chain replayed under a restored RNG state.
2. The recipe defaults reproduce **E00's resolved configuration**, not V1's
   drifted one: ``momentum=0.937`` and ``warmup_bias_lr=0.1`` (V1 used 0.9 and
   0.0). Those were two of the three remaining suspects in postmortem §2E-3,
   so they must be held at E00's values when the data path is the variable
   under test.
3. Short probes are made *prefixes* of E00's schedule:
     * ``--match-e00-schedule`` (default) rescales ``lrf`` so that the learning
       rate at epoch k equals E00's learning rate at epoch k. Without this a
       10-epoch run has already decayed to ``lrf`` by epoch 10 while E00 is
       still near its peak LR, and the comparison is meaningless.
     * ``--close-mosaic`` defaults to 0 for probes: E00 keeps mosaic on until
       epoch 90, so a probe's epochs 0..N-1 must also have mosaic on.
       (Ultralytics closes mosaic at ``epochs - close_mosaic``; leaving the
       default 10 on a 10-epoch run would close it at epoch 0.)

Examples
--------
  python scripts/train_unified_v2.py --name e00_ir_v2_10ep --mode ir \
      --data configs/dronevehicle_v2_ir.yaml --epochs 10
  python scripts/train_unified_v2.py --name e30_v2 --mode pair --fuse E30 \
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

#: E00 reference schedule (runs/e00_ir/args.yaml).
E00_EPOCHS = 100
E00_LRF = 0.01
E00_WARMUP = 3.0


def write_code_state(run_dir: Path) -> None:
    """Record git HEAD + file hashes before training (provenance)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in (EXP / "src").rglob("*.py")]
                   + [EXP / "scripts" / "train_unified_v2.py"])
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


def prefix_lrf(epochs: int, warmup: float = E00_WARMUP,
               ref_lrf: float = E00_LRF) -> float:
    """Final LR fraction that makes an N-epoch run a prefix of E00's schedule.

    Ultralytics decays linearly after warm-up, over ``epochs - warmup``
    epochs, from ``lr0`` to ``lr0 * lrf``. Matching the per-epoch slope of the
    100-epoch E00 run gives the same LR at every epoch ``k <= N``.
    """
    if epochs >= E00_EPOCHS:
        return ref_lrf
    span_probe = max(1.0, epochs - warmup)
    return 1.0 - (1.0 - ref_lrf) * span_probe / (E00_EPOCHS - warmup)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", required=True, choices=["ir", "rgb", "pair"])
    ap.add_argument("--fuse", default="E30")
    ap.add_argument("--data", required=True)
    ap.add_argument("--rgb-yaml", default=None)
    ap.add_argument("--pretrained", default=str(PRET / "yolo12s.pt"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8,
                    help="E00 used 8; V1 used 16")
    # -- E00 recipe (resolved values from runs/e00_ir/args.yaml) -----------
    ap.add_argument("--optimizer", default="MuSGD")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--lrf", type=float, default=None,
                    help="final LR fraction; default = E00 prefix (see "
                         "--no-match-e00-schedule)")
    ap.add_argument("--momentum", type=float, default=0.937)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup-epochs", type=float, default=E00_WARMUP)
    ap.add_argument("--warmup-momentum", type=float, default=0.8)
    ap.add_argument("--warmup-bias-lr", type=float, default=0.1)
    ap.add_argument("--no-match-e00-schedule", action="store_true",
                    help="use lrf=E00_LRF even for a short probe (NOT "
                         "comparable to E00's early epochs)")
    ap.add_argument("--close-mosaic", type=int, default=0,
                    help="0 (default) keeps mosaic on for the whole probe, "
                         "matching E00's epochs 0-89")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--project", default="runs",
                    help="run directory root (relative to the experiment or absolute)")
    ap.add_argument("--fuse-r", type=int, default=1,
                    help="graph window radius for E30/E35/E41 (r=1 -> 3x3 window)")
    ap.add_argument("--init-ir", default=None,
                    help="warm-start the IR stream (+ neck/head) from this "
                         "DV-trained checkpoint; requires --init-rgb")
    ap.add_argument("--init-rgb", default=None,
                    help="warm-start the RGB stream backbone from this "
                         "DV-trained checkpoint")
    ap.add_argument("--fuse-warmup", type=int, default=0)
    ap.add_argument("--fuse-ramp", type=int, default=1)
    ap.add_argument("--no-strict-sync", action="store_true",
                    help="skip the per-sample IR/RGB replay assertion (pair "
                         "mode only; not recommended)")
    ap.add_argument("--pipeline", choices=["v2", "v1"], default="v2",
                    help="DIAGNOSTIC ONLY: 'v1' swaps in the superseded "
                         "gfdata.unified loader while keeping this launcher's "
                         "recipe, to isolate loader vs recipe as the cause of "
                         "the e00_ir_unified collapse")
    args = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    from gfdata.unified_v2 import make_dataset_v2

    data = Path(args.data)
    data = data if data.is_absolute() else EXP / data
    rgb_yaml = None
    if args.mode == "pair":
        if not args.rgb_yaml:
            ap.error("--mode pair requires --rgb-yaml")
        rgb_yaml = Path(args.rgb_yaml)
        rgb_yaml = rgb_yaml if rgb_yaml.is_absolute() else EXP / rgb_yaml

    lrf = args.lrf if args.lrf is not None else (
        E00_LRF if args.no_match_e00_schedule else prefix_lrf(args.epochs))

    proj = Path(args.project)
    proj = proj if proj.is_absolute() else EXP / proj
    run_dir = proj / args.name
    last = run_dir / "weights" / "last.pt"
    resume = last.exists()

    rgb_data = check_det_dataset(str(rgb_yaml)) if rgb_yaml else None

    def _model_stride(model) -> int:
        """Model stride as a plain int (``model.stride`` is a tensor here)."""
        s = getattr(model, "stride", 32)
        try:
            if hasattr(s, "flatten"):
                return int(s.flatten().max().item())
            if isinstance(s, (list, tuple)):
                return int(max(s))
            return int(s)
        except Exception:
            return 32

    class V2RefTrainer(DetectionTrainer):
        """Single-stream arm (IR/RGB) through the V2 dataset."""

        modality = args.mode
        rgb_info = rgb_data
        strict = not args.no_strict_sync

        def build_dataset(self, img_path, mode="train", batch=None):
            if args.pipeline == "v1":
                # DIAGNOSTIC ABLATION: identical launcher/recipe, superseded
                # loader. V1 ignores the trainer hyp and applies its own
                # DEFAULT_HYP (mosaic 1.0, flips, HSV, erasing 0.4), which is
                # part of what is under test.
                from gfdata.unified import make_dataset as make_unified
                split = mode if mode in ("train", "val", "test") else "train"
                return make_unified(
                    str(data), split, modality=self.modality, mode=mode,
                    imgsz=self.args.imgsz, batch=batch or self.args.batch,
                    cache=self.args.cache or None,
                )
            return make_dataset_v2(
                cfg=self.args, img_path=img_path,
                batch=batch or self.args.batch, data=self.data, mode=mode,
                rect=False, stride=_model_stride(self.model),
                modality=self.modality,
                rgb_data=self.rgb_info,
                rgb_img_path=(self.rgb_info[mode] if self.rgb_info else None),
                strict_sync=self.strict,
            )

    kw = dict(data=str(data), epochs=args.epochs, patience=args.patience,
              batch=args.batch, imgsz=args.imgsz, seed=args.seed,
              workers=args.workers, deterministic=True,
              project=str(proj), name=args.name, exist_ok=True,
              val=True, save=True, cache=False,
              optimizer=args.optimizer, lr0=args.lr0, lrf=lrf,
              momentum=args.momentum, weight_decay=args.weight_decay,
              warmup_epochs=args.warmup_epochs,
              warmup_momentum=args.warmup_momentum,
              warmup_bias_lr=args.warmup_bias_lr,
              close_mosaic=args.close_mosaic)

    if args.mode == "pair":
        from gfyolo.gf_trainer import GFTrainer

        class V2GFTrainer(GFTrainer):
            """Dual-stream arm through the V2 dataset (GFTrainer's own
            ``build_dataset`` still points at V1, so it is overridden here)."""

            strict = not args.no_strict_sync

            def build_dataset(self, img_path, mode="train", batch=None):
                return make_dataset_v2(
                    cfg=self.args, img_path=img_path,
                    batch=batch or self.args.batch, data=self.data, mode=mode,
                    rect=False, stride=_model_stride(self.model),
                    modality="pair", rgb_data=rgb_data,
                    rgb_img_path=(rgb_data[mode] if rgb_data else None),
                    strict_sync=self.strict,
                )

        GFTrainer.paired_yaml = str(data)
        GFTrainer.rgb_yaml = str(rgb_yaml)
        GFTrainer.fuse = args.fuse
        GFTrainer.fuse_kwargs = ({"d": args.d}
                                 if args.fuse in ("E21", "E30", "E35", "E41")
                                 else {})
        if args.fuse in ("E30", "E35", "E41") and args.fuse_r != 1:
            GFTrainer.fuse_kwargs["r"] = args.fuse_r
        GFTrainer.fuse_warmup_epochs = args.fuse_warmup
        GFTrainer.fuse_ramp_epochs = args.fuse_ramp
        GFTrainer.init_ir = args.init_ir
        GFTrainer.init_rgb = args.init_rgb
        if bool(args.init_ir) != bool(args.init_rgb):
            ap.error("--init-ir and --init-rgb must be given together")
        model = YOLO(str(last) if resume else GFTrainer.model_cfg)
        trainer_cls = V2GFTrainer
        if resume:
            kw["resume"] = True
        else:
            kw["pretrained"] = args.pretrained
            kw["augmentations"] = []
    else:
        model = YOLO(str(last) if resume else args.pretrained)
        trainer_cls = V2RefTrainer
        if resume:
            kw["resume"] = True

    t0 = time.time()
    write_code_state(run_dir)
    model.train(trainer=trainer_cls, **kw)
    dt = time.time() - t0

    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": args.name, "mode": args.mode, "fuse": args.fuse,
        "project": str(proj),
        "pipeline": "gfdata.unified_v2",
        "data": str(data), "rgb_yaml": str(rgb_yaml) if rgb_yaml else None,
        "epochs_requested": args.epochs, "patience": args.patience,
        "batch": args.batch, "imgsz": args.imgsz, "seed": args.seed,
        "workers": args.workers, "resumed": resume,
        "optimizer": args.optimizer, "lr0": args.lr0, "lrf": lrf,
        "momentum": args.momentum, "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": args.warmup_momentum,
        "warmup_bias_lr": args.warmup_bias_lr,
        "close_mosaic": args.close_mosaic,
        "fuse_r": args.fuse_r,
        "init_streams": ({"ir": args.init_ir, "rgb": args.init_rgb}
                         if args.init_ir else None),
        "schedule": ("e00-prefix" if not args.no_match_e00_schedule
                     and args.epochs < E00_EPOCHS else "native"),
        "strict_sync": not args.no_strict_sync,
        "wall_h": round(dt / 3600, 3),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "ref_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (run_dir / "DONE").write_text("ok\n")
    print(f"[train_unified_v2] {args.name} DONE in {dt / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
