#!/usr/bin/env python3
"""Evaluate a trained YOLO detector on a specific dataset split.

Writes a JSON result with mAP50, mAP50-95, precision, recall and provenance
(weights path + sha256, dataset path, split hash, image count, git commit, job id).

Protocol 3A: evaluate the clear-only baseline on translated adverse test.
Protocol 3B: evaluate the mixed-trained model on translated adverse test.

Usage:
    python eval_detector.py --weights runs/clear_baseline/weights/best.pt \
        --data flir_adras_v2/method3_cyclegan/datasets/cyclegan_fog_3a.yaml \
        --name 3a_fog --split test --imgsz 640
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "03_m3fd_weather"))
import provenance

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Evaluate YOLO detector on FLIR ADAS v2")
    ap.add_argument("--weights", required=True,
                    help="path to .pt weights")
    ap.add_argument("--data", required=True,
                    help="dataset YAML path")
    ap.add_argument("--name", required=True,
                    help="result name")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640,
                    help="must match training")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON path (default: results/<name>.json)")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="confidence threshold for evaluation")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU threshold for evaluation")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing result JSON")
    args = ap.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"FATAL: dataset YAML not found: {data_yaml}")
        return 1

    weights = Path(args.weights)
    if not weights.exists():
        print(f"FATAL: weights not found: {weights}")
        return 1

    # Provenance
    split_hash = provenance.split_hash_for_dataset(data_yaml)
    n_images = provenance.count_images(data_yaml, args.split)
    if n_images == 0:
        print(f"FATAL: {data_yaml}: 0 images visible for split '{args.split}'")
        return 1

    # Check that split doesn't alias another
    eval_dirs = set()
    for d in provenance.dataset_entries(data_yaml, args.split):
        if d.is_dir():
            eval_dirs.add(d.resolve())
    for other in ("train", "val"):
        if other == args.split:
            continue
        for d in provenance.dataset_entries(data_yaml, other):
            if d.is_dir() and d.resolve() in eval_dirs:
                print(f"FATAL: split '{args.split}' resolves to same dir as '{other}'")
                return 1

    # Create runs directory
    runs = REPO / "flir_adras_v2" / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    out_file = Path(args.out) if args.out else (REPO / "flir_adras_v2" / "results" / f"{args.name}.json")
    if out_file.exists() and not args.force:
        print(f"SKIP: result already exists at {out_file} (use --force to recompute)")
        return 0

    model = YOLO(str(weights))
    print(f"Evaluating on {n_images} images (split={args.split}, imgsz={args.imgsz})...")

    m = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        project=str(runs),
        name=f"eval_{args.name}",
        exist_ok=True,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    )

    res = {
        "name": args.name,
        "weights": str(weights.resolve()),
        "weights_sha256": provenance.sha256_file(weights),
        "data": str(data_yaml.resolve()),
        "split": args.split,
        "split_hash": split_hash,
        "n_images": n_images,
        "imgsz": args.imgsz,
        "mAP50_95": round(float(m.box.map), 5),
        "mAP50": round(float(m.box.map50), 5),
        "precision": round(float(m.box.mp), 5),
        "recall": round(float(m.box.mr), 5),
        "per_class_AP": {},
        **provenance.run_record(),
    }

    # Per-class AP
    try:
        names = m.names
        class_ap = m.box.p
        for i, cls_name in enumerate(names):
            if i < len(m.box.maps):
                res["per_class_AP"][cls_name] = {
                    "mAP50": round(float(m.box.p[i].mean()) if len(m.box.p[i]) > 0 else 0.0, 5),
                    "mAP50_95": round(float(m.box.maps[i]) if i < len(m.box.maps) else 0.0, 5),
                }
    except Exception:
        pass

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2))
    print(f"\nResults for {args.name} ({n_images} images, split_hash={split_hash}):")
    print(f"  mAP50-95 = {res['mAP50_95']:.4f}")
    print(f"  mAP50    = {res['mAP50']:.4f}")
    print(f"  precision = {res['precision']:.4f}")
    print(f"  recall    = {res['recall']:.4f}")
    print(f"  -> {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
