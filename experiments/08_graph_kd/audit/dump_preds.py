"""Dump per-image GT and detections for paired bootstrap (audit only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.data.build import build_dataloader
from ultralytics.utils.nms import non_max_suppression

from graphkd import paths
from graphkd.data.dataset import make_dataset
from graphkd.training.evaluate import unletterbox
from graphkd.utils import xywh_to_xyxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=str(paths.CONFIGS / "ir.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="audit/preds")
    args = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.weights).model.to(dev).eval()
    nc = int(getattr(model, "nc", model.model[-1].nc))
    ds = make_dataset(args.data, args.split, mode="val", imgsz=args.imgsz, batch=args.batch)
    dl = build_dataloader(ds, batch=args.batch, workers=2, shuffle=False)

    images, anns, preds = [], [], []
    image_id = 0
    with torch.no_grad():
        for batch in dl:
            img = batch["img"].to(dev).float() / 255.0
            out = model(img)
            y = out[0] if isinstance(out, tuple) else out
            det = non_max_suppression(y, 0.001, 0.7, max_det=300, nc=nc)
            for bi, d in enumerate(det):
                iid = image_id
                h0, w0 = int(batch["ori_shape"][bi][0]), int(batch["ori_shape"][bi][1])
                images.append({"id": iid, "file_name": Path(batch["im_file"][bi]).name,
                               "height": h0, "width": w0})
                sel = batch["batch_idx"] == bi
                gb = batch["bboxes"][sel].cpu()
                gc = batch["cls"][sel].cpu().flatten().long()
                if len(gb):
                    gt_xyxy = unletterbox(xywh_to_xyxy(gb * float(args.imgsz)), h0, w0, args.imgsz)
                    for (x1, y1, x2, y2), c in zip(gt_xyxy.tolist(), gc.tolist()):
                        w, h = x2 - x1, y2 - y1
                        anns.append({"id": len(anns) + 1, "image_id": iid,
                                     "category_id": int(c) + 1,
                                     "bbox": [x1, y1, w, h],
                                     "area": w * h, "iscrowd": 0})
                if d is not None and len(d):
                    boxes = unletterbox(d[:, :4], h0, w0, args.imgsz)
                    for row in torch.cat([boxes, d[:, 4:]], 1).cpu().numpy():
                        x1, y1, x2, y2, score, c = row.tolist()
                        preds.append({"image_id": iid, "category_id": int(c) + 1,
                                      "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                                      "score": float(score)})
                image_id += 1

    nmap = getattr(ds, "data", {}).get("names", {})
    names = [nmap.get(i, str(i)) for i in range(nc)] if isinstance(nmap, dict) else [nmap[i] for i in range(nc)]
    payload = {"name": args.name, "split": args.split, "weights": args.weights,
               "images": images, "annotations": anns, "predictions": preds,
               "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
    out = Path(args.out) / f"{args.name}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print("wrote", out, "images", len(images), "anns", len(anns), "preds", len(preds))


if __name__ == "__main__":
    main()
