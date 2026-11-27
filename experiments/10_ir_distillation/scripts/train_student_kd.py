#!/usr/bin/env python3
"""Train an IR-only YOLO12s/n student by distilling the frozen E30 teacher.

    python scripts/train_student_kd.py --name student_12s_decomp \
        --student-cfg models/pretrained/yolo12s.pt \
        --teacher-ckpt ../09_graph_fusion/runs/e30_v2_100ep/weights/best.pt \
        --kd-mode decomposed

Resume is automatic when ``<project>/<name>/weights/last.pt`` exists (the
projector sidecar ``kd_proj.pt`` is restored too). Pass ``--fresh`` to start a
new run in an existing directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

EXP10 = Path(__file__).resolve().parents[1]
REPO = EXP10.parents[1]
EXP09 = REPO / "experiments" / "09_graph_fusion"
for _p in (str(EXP10 / "src"), str(EXP09 / "src"), str(EXP09)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

E00_EPOCHS = 100
E00_LRF = 0.01
E00_WARMUP = 3.0


def prefix_lrf(epochs: int) -> float:
    """E00's LR at epoch k for a shorter prefix schedule."""
    if epochs >= E00_EPOCHS:
        return E00_LRF
    value = 1.0 - (1.0 - E00_LRF) * (epochs - E00_WARMUP) / (E00_EPOCHS - E00_WARMUP)
    return float(min(1.0, max(E00_LRF, value)))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_state() -> dict:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        head = None
    files = [
        EXP10 / "src/irkd/kd_trainer.py",
        EXP10 / "src/irkd/losses.py",
        EXP10 / "src/irkd/teacher_hooks.py",
        EXP10 / "scripts/train_student_kd.py",
    ]
    return {
        "git_head": head,
        "files": {str(p.relative_to(REPO)): sha256(p) for p in files if p.exists()},
        "python": sys.executable,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--student-cfg", default="models/pretrained/yolo12s.pt")
    ap.add_argument("--teacher-ckpt",
                    default="../09_graph_fusion/runs/e30_v2_100ep/weights/best.pt")
    ap.add_argument("--kd-mode", choices=["none", "direct", "decomposed"],
                    default="decomposed")
    ap.add_argument("--data", default="configs/dv_paired.yaml")
    ap.add_argument("--rgb-yaml", default="configs/dv_rgb.yaml")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--lrf", type=float, default=None)
    ap.add_argument("--lambda-pres", type=float, default=0.25)
    ap.add_argument("--lambda-comp", type=float, default=0.50)
    ap.add_argument("--lambda-direct", type=float, default=1.0)
    ap.add_argument("--kd-warmup-epochs", type=int, default=3)
    ap.add_argument("--kd-alpha", type=float, default=1.0)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="dataset fraction (smoke tests)")
    ap.add_argument("--no-val", action="store_true",
                    help="skip validation (CPU integration smoke only)")
    ap.add_argument("--no-plots", action="store_true",
                    help="skip Ultralytics plots (results.png, curves, confusion matrix, batches)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing last.pt and start over")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    from irkd.kd_trainer import KDDetectionTrainer

    def resolve(value: str) -> Path:
        q = Path(value)
        if q.is_absolute():
            return q
        for base in (Path.cwd(), REPO, EXP10):
            if (base / q).exists():
                return (base / q).resolve()
        return (EXP10 / q).resolve()

    student_cfg = resolve(args.student_cfg)
    teacher_ckpt = resolve(args.teacher_ckpt)
    data_yaml = resolve(args.data)
    rgb_yaml = resolve(args.rgb_yaml)

    for label, p in (("student", student_cfg), ("teacher", teacher_ckpt),
                     ("data", data_yaml), ("rgb-yaml", rgb_yaml)):
        if not p.exists():
            raise SystemExit(f"missing {label}: {p}")

    proj = Path(args.project)
    proj = proj if proj.is_absolute() else (EXP10 / proj)
    run_dir = proj / args.name
    last = run_dir / "weights" / "last.pt"
    resume = last.exists() and not args.fresh
    lrf = args.lrf if args.lrf is not None else prefix_lrf(args.epochs)

    _teacher = str(teacher_ckpt)
    _rgb_yaml = str(rgb_yaml)

    if last.exists() and not args.fresh:
        _ck = torch.load(last, map_location="cpu", weights_only=False)
        if int(_ck.get("epoch", -1)) < 0:
            raise SystemExit(
                f"{last} is a completed/stripped checkpoint (epoch=-1); it cannot "
                "be resumed. Use --fresh to start a new run, or delete/rename the "
                "run directory deliberately."
            )

    class RunTrainer(KDDetectionTrainer):
        kd_mode = args.kd_mode
        teacher_ckpt = _teacher
        rgb_yaml = _rgb_yaml
        lambda_pres = args.lambda_pres
        lambda_comp = args.lambda_comp
        lambda_direct = args.lambda_direct
        kd_warmup_epochs = args.kd_warmup_epochs
        kd_alpha = args.kd_alpha
        strict_sync = True

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "code_state.json").write_text(json.dumps(code_state(), indent=1))

    model = YOLO(str(last) if resume else str(student_cfg))
    kw = dict(
        trainer=RunTrainer, data=str(data_yaml), epochs=args.epochs,
        patience=args.patience, batch=args.batch, imgsz=args.imgsz,
        seed=args.seed, workers=args.workers, deterministic=True,
        project=str(proj), name=args.name, exist_ok=True,
        val=not args.no_val, save=True, plots=not args.no_plots, cache=False,
        fraction=args.fraction,
        optimizer="MuSGD", lr0=0.01, lrf=lrf, momentum=0.937,
        weight_decay=5e-4, warmup_epochs=E00_WARMUP,
        warmup_momentum=0.8, warmup_bias_lr=0.1,
        close_mosaic=args.close_mosaic, device=args.device,
    )
    if resume:
        kw["resume"] = True

    t0 = time.time()
    print(f"[student_kd] {'RESUME' if resume else 'START'} {args.name} "
          f"kd={args.kd_mode} student={student_cfg.name} "
          f"epochs={args.epochs} lrf={lrf:.4f}", flush=True)
    model.train(**kw)
    wall = time.time() - t0

    best_val = None
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        import csv

        rows = list(csv.DictReader(csv_path.open()))
        if rows:
            best_val = max(float(r["metrics/mAP50-95(B)"]) for r in rows)

    meta = {
        "name": args.name, "kd_mode": args.kd_mode,
        "student_cfg": str(student_cfg), "student_sha256": sha256(student_cfg),
        "teacher_ckpt": str(teacher_ckpt), "teacher_sha256": sha256(teacher_ckpt),
        "data": str(data_yaml), "rgb_yaml": str(rgb_yaml),
        "epochs_requested": args.epochs, "patience": args.patience,
        "batch": args.batch, "imgsz": args.imgsz, "seed": args.seed,
        "workers": args.workers, "resumed": resume,
        "optimizer": "MuSGD", "lr0": 0.01, "lrf": lrf,
        "momentum": 0.937, "weight_decay": 5e-4,
        "warmup_epochs": E00_WARMUP, "warmup_bias_lr": 0.1,
        "close_mosaic": args.close_mosaic,
        "lambda_pres": args.lambda_pres, "lambda_comp": args.lambda_comp,
        "lambda_direct": args.lambda_direct,
        "kd_warmup_epochs": args.kd_warmup_epochs, "kd_alpha": args.kd_alpha,
        "best_val_mAP50_95": best_val,
        "wall_h": round(wall / 3600, 3),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "ref_meta.json").write_text(json.dumps(meta, indent=1))
    (run_dir / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print(f"[student_kd] DONE {args.name} in {wall/3600:.2f} h; "
          f"best val {best_val}", flush=True)


if __name__ == "__main__":
    main()
