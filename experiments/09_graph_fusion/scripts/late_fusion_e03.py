#!/usr/bin/env python3
"""E03: late fusion (WBF) of E00 (IR) + E01 (RGB) detections.

Calibration (IoU threshold, WBF weights, members-scaling) is chosen on VAL
ONLY, against the IR (thermal) GT. Optional --emit-test applies the chosen
config to TEST predictions and writes the merged JSON; scoring the test set
goes exclusively through scripts/make_lock.py + scripts/final_eval.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from eval.coco_eval import evaluate  # noqa: E402
from gfyolo.wbf import wbf_merge  # noqa: E402


def group_by_image(preds: list[dict]):
    """prediction JSON -> {image_id: (boxes_xyxy, scores, labels)}."""
    out: dict[int, tuple[list, list, list]] = {}
    for p in preds:
        x, y, w, h = p["bbox"]
        b, s, l = out.setdefault(p["image_id"], ([], [], []))
        b.append([x, y, x + w, y + h])
        s.append(float(p["score"]))
        l.append(int(p["category_id"]) - 1)
    return out


def merge_pred_sets(ir_preds, rgb_preds, cfg) -> list[dict]:
    gi, gr = group_by_image(ir_preds), group_by_image(rgb_preds)
    ids = sorted(set(gi) | set(gr))
    merged = []
    for i in ids:
        bi, si, li = gi.get(i, ([], [], []))
        br, sr, lr = gr.get(i, ([], [], []))
        boxes, scores, labels = wbf_merge(
            bi, si, li, br, sr, lr,
            iou_thr=cfg["iou_thr"], wa=cfg["wa"], wb=cfg["wb"],
            scale_members=cfg["scale_members"])
        for k in range(len(scores)):
            x1, y1, x2, y2 = (float(v) for v in boxes[k].tolist())
            merged.append({
                "image_id": i, "category_id": labels[k] + 1,
                "bbox": [round(x1, 2), round(y1, 2),
                         round(x2 - x1, 2), round(y2 - y1, 2)],
                "score": round(scores[k], 6),
            })
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir-preds", required=True, help="E00 val predictions JSON")
    ap.add_argument("--rgb-preds", required=True, help="E01 val predictions JSON")
    ap.add_argument("--gt", required=True, help="VAL GT json (ir/thermal labels)")
    ap.add_argument("--out", default=None, help="val report JSON path")
    ap.add_argument("--emit-test", action="store_true")
    ap.add_argument("--test-ir-preds", default=None)
    ap.add_argument("--test-rgb-preds", default=None)
    ap.add_argument("--test-out", default=None)
    args = ap.parse_args()

    ir = json.loads(Path(args.ir_preds).read_text())
    rgb = json.loads(Path(args.rgb_preds).read_text())
    gt = json.loads(Path(args.gt).read_text())

    report = {
        "experiment": "e03_late_fusion_wbf",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "val_gt": str(Path(args.gt).resolve()),
        "baselines_val": {},
        "grid": [],
    }
    report["baselines_val"]["e00_ir_only"] = evaluate(gt, ir)
    report["baselines_val"]["e01_rgb_only"] = evaluate(gt, rgb)

    best = None
    for iou_thr in (0.5, 0.55, 0.6):
        for wa, wb in ((1.0, 1.0), (1.5, 1.0), (1.0, 1.5)):
            for scale in (False, True):
                cfg = {"iou_thr": iou_thr, "wa": wa, "wb": wb,
                       "scale_members": scale}
                m = evaluate(gt, merge_pred_sets(ir, rgb, cfg))
                row = {"config": cfg, "AP": m["AP"], "AP50": m["AP50"],
                       "AP75": m["AP75"]}
                report["grid"].append(row)
                print(f"[e03] {cfg} AP={m['AP']:.5f}", flush=True)
                if best is None or m["AP"] > best[1]:
                    best = (cfg, m["AP"])
    report["best_config"] = best[0]
    report["best_val"] = {"AP": best[1]}
    out = Path(args.out) if args.out else EXP / "results" / "e03_val_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"[e03] best config {best[0]} val AP={best[1]:.5f} -> {out}")

    if args.emit_test:
        assert args.test_ir_preds and args.test_rgb_preds and args.test_out, \
            "--emit-test requires --test-ir-preds --test-rgb-preds --test-out"
        tir = json.loads(Path(args.test_ir_preds).read_text())
        trgb = json.loads(Path(args.test_rgb_preds).read_text())
        merged = merge_pred_sets(tir, trgb, best[0])
        to = Path(args.test_out)
        to.parent.mkdir(parents=True, exist_ok=True)
        to.write_text(json.dumps(merged))
        meta = {"config": best[0],
                "ir_preds": str(Path(args.test_ir_preds).resolve()),
                "rgb_preds": str(Path(args.test_rgb_preds).resolve()),
                "val_AP": best[1],
                "note": "merge ONLY; scoring via make_lock + final_eval",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        to.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
        print(f"[e03] merged TEST predictions written (NOT scored) -> {to}")


if __name__ == "__main__":
    main()
