"""COCO-style evaluation for IR students and the paired fused teacher.

Reports AP, AP50, AP75, AP-small/medium/large, per-class AP, ECE/Brier,
parameters and latency.  Works for 3-channel and paired 6-channel models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from ultralytics.data.build import build_dataloader
from ultralytics.utils.nms import non_max_suppression

from graphkd import paths
from graphkd.data.dataset import make_dataset, make_paired_dataset
from graphkd.utils import box_iou_xywh, xywh_to_xyxy, xyxy_to_xywh


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_device(arg: str, cuda_available: bool | None = None) -> torch.device:
    """Map a CLI device string to a torch device.

    ``--device cpu`` must work on a GPU node (used by the CPU-only audit path);
    the previous expression produced the invalid string ``cuda:cpu``.
    """
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    if str(arg).lower() == "cpu" or not cuda_available:
        return torch.device("cpu")
    return torch.device(f"cuda:{arg}")


def unletterbox(boxes: torch.Tensor, h0: int, w0: int, imgsz: int = 640) -> torch.Tensor:
    gain = min(imgsz / h0, imgsz / w0)
    pad_w = (imgsz - w0 * gain) / 2
    pad_h = (imgsz - h0 * gain) / 2
    b = boxes.clone()
    b[:, [0, 2]] = (b[:, [0, 2]] - pad_w) / gain
    b[:, [1, 3]] = (b[:, [1, 3]] - pad_h) / gain
    return b.clamp(min=0)


def coco_eval(gt: dict, preds: list[dict]) -> dict:
    fd, gt_path = tempfile.mkstemp(suffix="_graphkd_gt.json")
    os.close(fd)
    with open(gt_path, "w") as f:
        json.dump(gt, f)
    try:
        cg = COCO(gt_path)
        if not preds:
            return {k: 0.0 for k in ("AP", "AP50", "AP75", "APs", "APm", "APl")}
        cd = cg.loadRes(preds)
        ev = COCOeval(cg, cd, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        stats = ev.stats
        out = {"AP": float(stats[0]), "AP50": float(stats[1]), "AP75": float(stats[2]),
               "APs": float(stats[3]), "APm": float(stats[4]), "APl": float(stats[5])}
        prec = ev.eval["precision"]  # [T,R,K,A,M]
        per_class = {}
        names = [None] * (max(gt["categories"], key=lambda c: c["id"])["id"] + 1)
        for c in gt["categories"]:
            names[c["id"]] = c["name"]
        for k in range(prec.shape[2]):
            v = prec[:, :, k, 0, -1]
            v = v[v > -1]
            per_class[names[k + 1] if k + 1 < len(names) else str(k)] = float(v.mean()) if v.size else 0.0
        out["per_class_AP"] = per_class
        return out
    finally:
        try:
            os.unlink(gt_path)
        except OSError:
            pass


def calibration(preds: list[dict], gt: dict, iou_thr: float = 0.5) -> dict:
    by_img = {}
    for p in preds:
        by_img.setdefault(p["image_id"], []).append(p)
    gts = {}
    for a in gt["annotations"]:
        gts.setdefault(a["image_id"], []).append(a)
    confs, correct = [], []
    for iid, ps in by_img.items():
        g = gts.get(iid, [])
        if not g:
            continue
        gb = torch.tensor([a["bbox"] for a in g], dtype=torch.float32)
        gc = torch.tensor([a["category_id"] for a in g])
        used = torch.zeros(len(g), dtype=torch.bool)
        ps_sorted = sorted(ps, key=lambda x: -x["score"])
        for p in ps_sorted:
            if p["score"] < 0.05:
                continue
            pb = torch.tensor(p["bbox"]).float()
            iou = box_iou_xywh(pb[None], gb)[0]
            iou[used] = -1
            same = gc == p["category_id"]
            iou[~same] = -1
            j = int(iou.argmax())
            hit = iou[j] >= iou_thr
            if hit:
                used[j] = True
            confs.append(p["score"])
            correct.append(1.0 if hit else 0.0)
    if not confs:
        return {"ECE": 0.0, "Brier": 0.0, "n": 0}
    confs = np.array(confs, dtype=np.float64)
    correct = np.array(correct, dtype=np.float64)
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for i in range(15):
        m = (confs >= bins[i]) & (confs < bins[i + 1])
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - confs[m].mean())
    return {"ECE": float(ece), "Brier": float(((confs - correct) ** 2).mean()), "n": int(len(confs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=str(paths.CONFIGS / "ir.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--paired-yaml", default=None)
    ap.add_argument("--rgb-yaml", default=None)
    args = ap.parse_args()

    dev = resolve_device(args.device)
    model = YOLO(args.weights).model.to(dev).eval()
    nc = int(getattr(model, "nc", model.model[-1].nc))
    if args.paired:
        ds = make_paired_dataset(args.paired_yaml or str(paths.CONFIGS / "paired.yaml"),
                                 args.rgb_yaml or str(paths.CONFIGS / "rgb.yaml"),
                                 args.split, mode="val", imgsz=args.imgsz, batch=args.batch)
    else:
        ds = make_dataset(args.data, args.split, mode="val", imgsz=args.imgsz, batch=args.batch)
    dl = build_dataloader(ds, batch=args.batch, workers=min(8, args.batch), shuffle=False)

    images, anns, preds = [], [], []
    image_id = 0
    name_to_id = {}
    with torch.no_grad():
        for batch in dl:
            img = batch["img"].to(dev).float() / 255.0
            out = model(img)
            y = out[0] if isinstance(out, tuple) else out
            det = non_max_suppression(y, args.conf, args.iou, max_det=args.max_det, nc=nc)
            for bi, d in enumerate(det):
                iid = image_id
                name_to_id[Path(batch["im_file"][bi]).name] = iid
                h0, w0 = int(batch["ori_shape"][bi][0]), int(batch["ori_shape"][bi][1])
                images.append({"id": iid, "file_name": Path(batch["im_file"][bi]).name,
                               "height": h0, "width": w0})
                # GT boxes in the batch are normalised to the letterboxed
                # imgsz x imgsz canvas, not to the original image.  Map them
                # back exactly like the predictions.
                sel = batch["batch_idx"] == bi
                gb = batch["bboxes"][sel].cpu()
                gc = batch["cls"][sel].cpu().flatten().long()
                if len(gb):
                    gt_xyxy = xywh_to_xyxy(gb * float(args.imgsz))
                    gt_xyxy = unletterbox(gt_xyxy, h0, w0, args.imgsz)
                    # COCO bbox is top-left xywh (xyxy_to_xywh returns centre).
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
    if isinstance(nmap, dict):
        names = [nmap.get(i, str(i)) for i in range(nc)]
    else:
        names = [nmap[i] if i < len(nmap) else str(i) for i in range(nc)]
    gt = {"images": images, "annotations": anns,
          "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
    metrics = coco_eval(gt, preds)
    metrics.update(calibration(preds, gt))
    metrics["params"] = int(sum(p.numel() for p in model.parameters()))

    # latency: single-image forward, 20 warmup + 100 timed
    sample = next(iter(dl))["img"][:1].to(dev).float() / 255.0
    def _sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(20):
            model(sample)
        _sync()
        t0 = time.perf_counter()
        for _ in range(100):
            model(sample)
        _sync()
        ms = (time.perf_counter() - t0) * 1000 / 100
    metrics["latency_ms"] = float(ms)
    metrics["fps"] = float(1000 / ms)
    metrics.update({
        "name": args.name, "weights": str(Path(args.weights).resolve()),
        "weights_sha256": sha256(Path(args.weights)), "data": args.data,
        "split": args.split, "n_images": len(images), "n_annotations": len(anns),
        "split_hash": hashlib.sha256(",".join(sorted(name_to_id)).encode()).hexdigest()[:16],
        "paired": bool(args.paired), "imgsz": args.imgsz,
    })
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    out = paths.RESULTS / f"{args.name}_{args.split}.json"
    out.write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
