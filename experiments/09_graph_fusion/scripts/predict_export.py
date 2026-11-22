#!/usr/bin/env python3
"""Export one detector's predictions on a DV v2 split as COCO-format JSON.

Predictions are in ORIGINAL-frame pixel coordinates (ultralytics maps its
letterboxed boxes back automatically), image_id per gfyolo.dv_coco. The
output is the exact format src/eval/coco_eval.py::evaluate expects.
Settings are recorded in a sidecar <out>.meta.json (weights sha, conf, iou,
imgsz, batch, timestamp) for the lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from gfyolo.dv_coco import split_images  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir name under runs/, e.g. e00_ir")
    ap.add_argument("--weights", default=None, help="explicit checkpoint path; default runs/<run>/weights/best.pt")
    ap.add_argument("--modality", choices=["ir", "rgb"], required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights) if args.weights else (
        EXP / "runs" / args.run / "weights" / "best.pt")
    if not weights.exists():
        raise SystemExit(f"missing checkpoint: {weights}")
    imgs = split_images(args.modality, args.split)

    model = YOLO(str(weights))
    preds = []
    t0 = time.time()
    # chunked predict: ultralytics warmup stacks the whole source list into
    # ONE batch, so a full-split list OOMs on GPU (verified 2026-11-22).
    # Chunking also keeps each call's warmup small and results identical.
    CH = max(1, args.batch)
    for c0 in range(0, len(imgs), CH):
        chunk = [str(p) for p in imgs[c0:c0 + CH]]
        for off, res in enumerate(model.predict(
                chunk, imgsz=args.imgsz, conf=args.conf,
                iou=args.iou, max_det=100, device=args.device,
                half=False, stream=True, verbose=False)):
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
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preds))
    meta = {
        "run": args.run, "weights": str(weights),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "modality": args.modality, "split": args.split,
        "conf": args.conf, "iou_nms": args.iou, "imgsz": args.imgsz,
        "batch": args.batch, "n_images": len(imgs), "n_preds": len(preds),
        "wall_s": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[predict_export] {args.run} {args.modality}/{args.split}: "
          f"{len(preds)} preds over {len(imgs)} images -> {out}")


if __name__ == "__main__":
    main()
