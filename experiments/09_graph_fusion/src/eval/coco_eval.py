#!/usr/bin/env python3
"""COCO bbox evaluator for experiment 09.

All models export predictions in ORIGINAL-frame coordinates as top-left xywh:
    [{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}, ...]
Ground truth is COCO JSON.  No model-specific code lives here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate(gt: dict, preds: list[dict], conf: float = 0.001, max_det: int = 100) -> dict:
    preds = [p for p in preds if float(p.get("score", 0.0)) >= conf]
    fd, gt_path = tempfile.mkstemp(suffix="_gf09_gt.json")
    os.close(fd)
    with open(gt_path, "w") as f:
        json.dump(gt, f)
    try:
        cg = COCO(gt_path)
        if not preds:
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "APs": 0.0, "APm": 0.0,
                    "APl": 0.0, "AR1": 0.0, "AR10": 0.0, "AR100": 0.0,
                    "per_class_AP": {}, "n_images": len(gt.get("images", [])),
                    "n_annotations": len(gt.get("annotations", [])),
                    "n_predictions": 0}
        cd = cg.loadRes(preds)
        ev = COCOeval(cg, cd, "bbox")
        ev.params.maxDets = [1, 10, max_det]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        s = ev.stats
        prec = ev.eval["precision"]  # [T,R,K,A,M]
        names = [None] * (max((c["id"] for c in gt["categories"]), default=0) + 1)
        for c in gt["categories"]:
            names[c["id"]] = c["name"]
        per_class = {}
        for k in range(prec.shape[2]):
            v = prec[:, :, k, 0, -1]
            v = v[v > -1]
            name = names[k + 1] if k + 1 < len(names) and names[k + 1] else str(k)
            per_class[name] = float(v.mean()) if v.size else 0.0
        return {
            "AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]),
            "APs": float(s[3]), "APm": float(s[4]), "APl": float(s[5]),
            "AR1": float(s[6]), "AR10": float(s[7]), "AR100": float(s[8]),
            "per_class_AP": per_class,
            "n_images": len(gt.get("images", [])),
            "n_annotations": len(gt.get("annotations", [])),
            "n_predictions": len(preds),
            "settings": {"conf": conf, "maxDet": max_det, "iouType": "bbox"},
        }
    finally:
        try:
            os.unlink(gt_path)
        except OSError:
            pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--max-det", type=int, default=100)
    args = ap.parse_args()
    gt = json.loads(Path(args.gt).read_text())
    preds = json.loads(Path(args.preds).read_text())
    metrics = evaluate(gt, preds, conf=args.conf, max_det=args.max_det)
    metrics.update({"gt": str(Path(args.gt).resolve()), "gt_sha256": sha256(Path(args.gt)),
                    "preds": str(Path(args.preds).resolve()),
                    "preds_sha256": sha256(Path(args.preds))})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
