#!/usr/bin/env python3
"""Export a PAIRED (6-channel [IR|RGB]) model's predictions as COCO JSON.

For E02 (early fusion) and later the dual-stream GF-YOLO models: builds the
paired 6-ch tensors with gfyolo.paired_dataset (no augmentation), runs the
model per batch, and writes predictions in ORIGINAL-frame pixel xywh with
image_id per gfyolo.dv_coco. Output format is exactly what
src/eval/coco_eval.py expects.
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
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--paired-yaml", default="configs/dronevehicle_v2_paired.yaml")
    ap.add_argument("--rgb-yaml", default="configs/dronevehicle_v2_rgb.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    from eval.geometry import unletterbox
    from gfyolo.paired_dataset import make_paired_dataset

    weights = Path(args.weights) if args.weights else (
        EXP / "runs" / args.run / "weights" / "best.pt")
    if not weights.exists():
        raise SystemExit(f"missing checkpoint: {weights}")

    pair_yaml = Path(args.paired_yaml)
    if not pair_yaml.is_absolute():
        pair_yaml = EXP / pair_yaml
    rgb_yaml = Path(args.rgb_yaml)
    if not rgb_yaml.is_absolute():
        rgb_yaml = EXP / rgb_yaml

    # image ids MUST match gfyolo.dv_coco (sorted ir images of the split)
    ids = {p.name: i for i, p in enumerate(split_images("ir", args.split))}

    ds = make_paired_dataset(str(pair_yaml), str(rgb_yaml), args.split,
                             mode="val", imgsz=args.imgsz, batch=args.batch)
    model = YOLO(str(weights))
    preds = []
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for k in range(0, len(ds), args.batch):
            batch = [ds[i] for i in range(k, min(k + args.batch, len(ds)))]
            # the paired dataset letterboxes to a square (640x640, pad 64 top
            # and bottom for 640x512 sources); ultralytics reports TENSOR-
            # source boxes in that padded frame (orig_shape=(640,640)), so
            # they must be un-letterboxed to each sample's ori_shape before
            # writing. Verified 2026-11-22: without this, val AP = 0.002.
            imgs = torch.stack([b["img"] for b in batch])
            shapes = [tuple(b["ori_shape"]) for b in batch]
            names = [Path(b["im_file"]).name for b in batch]
            results = model.predict(imgs, imgsz=args.imgsz, conf=args.conf,
                                    iou=args.iou, max_det=100,
                                    device=args.device, verbose=False)
            for name, res in zip(names, results):
                boxes = res.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                h0, w0 = shapes[0]
                # un-letterbox from the padded square frame to the original
                # frame (all samples in a batch share the source size; the
                # DV v2 crops are uniformly 640x512, verified by dv_coco)
                bxyxy = unletterbox(boxes.xyxy.clone(), h0, w0,
                                    imgsz=args.imgsz).cpu().tolist()
                xyxy = boxes.xyxy.cpu().tolist()  # kept for meta/debug only
                for j in range(len(boxes)):
                    x1, y1, x2, y2 = bxyxy[j]
                    preds.append({
                        "image_id": ids[name],
                        "category_id": int(boxes.cls[j].item()) + 1,
                        "bbox": [round(x1, 2), round(y1, 2),
                                 round(x2 - x1, 2), round(y2 - y1, 2)],
                        "score": round(float(boxes.conf[j].item()), 6),
                    })
                n_done += 1
            if (k // args.batch) % 20 == 0:
                print(f"[paired_export] {k + len(batch)}/{len(ds)} images, "
                      f"{len(preds)} preds", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preds))
    meta = {
        "run": args.run, "weights": str(weights),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "split": args.split, "paired_yaml": str(pair_yaml),
        "rgb_yaml": str(rgb_yaml),
        "conf": args.conf, "iou_nms": args.iou, "imgsz": args.imgsz,
        "batch": args.batch, "n_images": len(ds), "n_preds": len(preds),
        "wall_s": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[paired_export] {args.run} {args.split}: {len(preds)} preds over "
          f"{len(ds)} images -> {out}")


if __name__ == "__main__":
    main()
