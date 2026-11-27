#!/usr/bin/env python3
"""Export an IR-only detector's predictions on a DroneVehicle split as COCO JSON.

Self-contained (no dependence on 09's exporter): image ids are the sorted IR
image list of the split, boxes are original-frame pixel xywh, exactly what
``eval/coco_eval.py`` expects.

    python scripts/export_ir.py --weights runs/NAME/weights/best.pt \\
        --dataset-root ../09_graph_fusion/data/dronevehicle_v2/yolo \\
        --split test --out results/preds/NAME_test.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

EXP10 = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = EXP10.parent / "09_graph_fusion/data/dronevehicle_v2/yolo"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = (EXP10 / weights).resolve()
    if not weights.exists():
        raise SystemExit(f"missing checkpoint: {weights}")

    root = Path(args.dataset_root)
    if not root.is_absolute():
        root = (EXP10 / root).resolve()
    img_dir = root / "ir" / "images" / args.split
    imgs = sorted(
        p for g in ("*.jpg", "*.jpeg", "*.png") for p in img_dir.glob(g)
    )
    imgs = sorted(set(imgs), key=lambda p: p.name)
    if not imgs:
        raise SystemExit(f"no images under {img_dir}")

    model = YOLO(str(weights))
    preds = []
    t0 = time.time()
    for c0 in range(0, len(imgs), args.batch):
        chunk = [str(p) for p in imgs[c0:c0 + args.batch]]
        for off, res in enumerate(model.predict(
            chunk, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            max_det=100, device=args.device, stream=True,
            verbose=False,
        )):
            i = c0 + off
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().tolist()
            for j in range(len(boxes)):
                x1, y1, x2, y2 = xyxy[j]
                preds.append({
                    "image_id": i,
                    "category_id": int(boxes.cls[j].item()) + 1,
                    "bbox": [round(x1, 2), round(y1, 2),
                             round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(float(boxes.conf[j].item()), 6),
                })
        if (c0 // args.batch) % 20 == 0:
            print(f"[export_ir] {c0 + len(chunk)}/{len(imgs)} "
                  f"images, {len(preds)} preds", flush=True)

    out = Path(args.out)
    if not out.is_absolute():
        out = EXP10 / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preds))
    meta = {
        "weights": str(weights), "weights_sha256": sha256(weights),
        "dataset_root": str(root), "split": args.split,
        "conf": args.conf, "iou_nms": args.iou, "imgsz": args.imgsz,
        "batch": args.batch, "n_images": len(imgs), "n_preds": len(preds),
        "wall_s": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[export_ir] {len(preds)} preds over {len(imgs)} images -> {out}")


if __name__ == "__main__":
    main()
