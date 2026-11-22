#!/usr/bin/env python3
"""Phase 1.2 complementarity oracle (CPU only, no training).

Per GT object on the DV v2 val split (thermal GT): detected by IR-only /
RGB-only / both / neither (IoU >= 0.5, score >= threshold fixed on val by
max-F1). Reported overall and by day/night (RGB mean-luminance threshold:
median of TRAIN luminances, train-only choice) and by COCO size bucket.
Oracle gain = recall(best-of-either) - max(recall_IR, recall_RGB).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from eval.geometry import box_iou_xywh  # noqa: E402

IMG_ROOT = EXP / "data" / "dronevehicle_v2" / "yolo" / "rgb" / "images"


def group(preds):
    by = defaultdict(list)
    for p in preds:
        by[p["image_id"]].append(p)
    return by


def best_f1_threshold(gt_imgs, pr, iou_thr=0.5):
    """Sweep score thresholds; return (thr, f1) maximising per-image F1."""
    best = (0.05, -1.0)
    for thr in [i / 100 for i in range(5, 80, 5)]:
        tp = fp = fn = 0
        for i, gts in gt_imgs.items():
            pred = [p for p in pr.get(i, []) if p["score"] >= thr]
            if not gts:
                fp += len(pred)
                continue
            matched = set()
            for j, g in enumerate(gts):
                for p in pred:
                    if p["category_id"] != g["category_id"]:
                        continue
                    iou = float(box_iou_xywh(
                        torch.tensor([g["bbox"]]), torch.tensor([p["bbox"]]))[0, 0])
                    if iou >= iou_thr:
                        matched.add(j)
                        break
            tp += len(matched)
            fn += len(gts) - len(matched)
            fp += len(pred) - len(matched) if len(pred) > len(gts) else 0
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best[1]:
            best = (thr, f1)
    return best


def detect_flags(gts, pred_list, thr, iou_thr=0.5):
    """Boolean list: GT object matched by >=1 same-class pred (score>=thr)."""
    out = []
    for g in gts:
        ok = False
        for p in pred_list:
            if p["category_id"] != g["category_id"] or p["score"] < thr:
                continue
            iou = float(box_iou_xywh(
                torch.tensor([g["bbox"]]), torch.tensor([p["bbox"]]))[0, 0])
            if iou >= iou_thr:
                ok = True
                break
        out.append(ok)
    return out


def size_bucket(area):
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir-preds", default="results/preds/e00_ir_val.json")
    ap.add_argument("--rgb-preds", default="results/preds/e01_rgb_val.json")
    ap.add_argument("--gt", default="results/gt/dv_v2_ir_val.json")
    ap.add_argument("--out", default="results/oracle_dv_val.json")
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text())
    ir = group(json.loads(Path(args.ir_preds).read_text()))
    rgb = group(json.loads(Path(args.rgb_preds).read_text()))

    gt_imgs = defaultdict(list)
    for a in gt["annotations"]:
        gt_imgs[a["image_id"]].append(a)
    imgs = {im["id"]: im for im in gt["images"]}

    # day/night threshold from TRAIN luminances only (sampled, median)
    train_imgs = sorted((IMG_ROOT / "train").glob("*.jpg"))
    lums = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).mean()
            for p in train_imgs[::10]]
    lum_thr = sorted(lums)[len(lums) // 2]

    thr_ir, f1_ir = best_f1_threshold(gt_imgs, ir)
    thr_rgb, f1_rgb = best_f1_threshold(gt_imgs, rgb)

    cells = defaultdict(lambda: defaultdict(int))
    report = {"thresholds": {"ir": thr_ir, "rgb": thr_rgb, "iou": 0.5,
                             "f1_ir": round(f1_ir, 4), "f1_rgb": round(f1_rgb, 4),
                             "day_night_luminance_thr": round(lum_thr, 1),
                             "day_night_thr_source": "train RGB luminance median"},
              "total": defaultdict(int), "by_day_night": {}, "by_size": {}}
    for i, gts in gt_imgs.items():
        im = imgs[i]
        lum = cv2.imread(str(IMG_ROOT / "val" / im["file_name"]),
                         cv2.IMREAD_GRAYSCALE).mean()
        phase = "day" if lum >= lum_thr else "night"
        d_ir = detect_flags(gts, ir.get(i, []), thr_ir)
        d_rgb = detect_flags(gts, rgb.get(i, []), thr_rgb)
        for g, a, b in zip(gts, d_ir, d_rgb):
            cell = "both" if a and b else ("ir_only" if a else
                                           ("rgb_only" if b else "neither"))
            for key in ("total", phase):
                report.setdefault(key, defaultdict(int))[cell] += 1
            cells[size_bucket(g["area"])][cell] += 1

    def recall_gain(counts):
        n = sum(counts.values())
        r_ir = (counts["ir_only"] + counts["both"]) / n
        r_rgb = (counts["rgb_only"] + counts["both"]) / n
        r_any = (counts["ir_only"] + counts["rgb_only"] + counts["both"]) / n
        return r_ir, r_rgb, r_any, r_any - max(r_ir, r_rgb)

    report["total"] = dict(report["total"])
    report["oracle_gain_overall"] = dict(
        zip(("recall_ir", "recall_rgb", "recall_best_of_either", "oracle_gain"),
            (round(v, 4) for v in recall_gain(report["total"]))))
    for phase in ("day", "night"):
        c = dict(report.get(phase, {}))
        report["by_day_night"][phase] = c
        report[f"oracle_gain_{phase}"] = dict(
            zip(("recall_ir", "recall_rgb", "recall_best_of_either", "oracle_gain"),
                (round(v, 4) for v in recall_gain(c)))) if c else {}
    for sz, c in cells.items():
        report["by_size"][sz] = dict(c)
        report[f"oracle_gain_size_{sz}"] = dict(
            zip(("recall_ir", "recall_rgb", "recall_best_of_either", "oracle_gain"),
                (round(v, 4) for v in recall_gain(c))))
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in
                      ("total", "oracle_gain_overall", "oracle_gain_day",
                       "oracle_gain_night", "oracle_gain_size_small")},
                     indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
