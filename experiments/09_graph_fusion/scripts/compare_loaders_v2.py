#!/usr/bin/env python3
"""Compare the IR data path of stock E00, unified V2 and (optionally) broken V1.

For a handful of indices and seeds this prints/records:

* exact tensor equality between the stock ``YOLODataset`` and ``V2 modality=ir``
  (they must be bit-identical -- V2 *is* the stock chain),
* exact equality between the IR half of a V2 *pair* sample and the IR-only
  sample (RNG parity),
* how far the V1 (``gfdata.unified``) sample drifts from the stock one, which
  is the pathology under investigation,
* side-by-side PNGs with the transformed boxes drawn, for visual inspection.

Usage
-----
  python scripts/compare_loaders_v2.py --indices 0,1,7,1234 --out audit/v2_compare
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
for p in (str(EXP / "src"), str(EXP)):
    if p not in sys.path:
        sys.path.insert(0, p)

IMGSZ = 640


def seek(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def draw(img_chw: torch.Tensor, boxes: torch.Tensor) -> np.ndarray:
    """CHW uint8 -> HWC BGR with normalised xywh boxes drawn in red."""
    im = img_chw.detach().cpu().numpy().transpose(1, 2, 0)
    im = np.ascontiguousarray(im)
    h, w = im.shape[:2]
    for b in boxes.detach().cpu().numpy():
        cx, cy, bw, bh = b
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return im


def stats(img: torch.Tensor) -> dict:
    x = img.float()
    return {
        "shape": list(img.shape),
        "dtype": str(img.dtype),
        "min": float(x.min()), "max": float(x.max()),
        "mean": round(float(x.mean()), 4), "std": round(float(x.std()), 4),
        "per_channel_mean": [round(float(x[c].mean()), 4) for c in range(x.shape[0])],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/dronevehicle_v2_ir.yaml")
    ap.add_argument("--rgb-yaml", default="configs/dronevehicle_v2_rgb.yaml")
    ap.add_argument("--indices", default="0,1,7,1234")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out", default="audit/v2_compare")
    ap.add_argument("--skip-v1", action="store_true")
    args = ap.parse_args()

    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.cfg import get_cfg

    from gfdata.unified_v2 import make_dataset_v2

    data = (EXP / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    rgb_yaml = (EXP / args.rgb_yaml) if not Path(args.rgb_yaml).is_absolute() else Path(args.rgb_yaml)
    ir_data = check_det_dataset(str(data))
    rgb_data = check_det_dataset(str(rgb_yaml))

    cfg = get_cfg(overrides=dict(
        task="detect", imgsz=IMGSZ, seed=0, mosaic=1.0, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5,
        degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0, erasing=0.4,
        mixup=0.0, cutmix=0.0, copy_paste=0.0, close_mosaic=0, rect=False,
        cache=False, single_cls=False, fraction=1.0,
    ))

    stock = build_yolo_dataset(cfg=cfg, img_path=ir_data["train"], batch=16,
                               data=ir_data, mode="train", rect=False, stride=32)
    v2_ir = make_dataset_v2(cfg=cfg, img_path=ir_data["train"], batch=16,
                            data=ir_data, mode="train", modality="ir")
    v2_pair = make_dataset_v2(cfg=cfg, img_path=ir_data["train"], batch=16,
                              data=ir_data, mode="train", modality="pair",
                              rgb_data=rgb_data,
                              rgb_img_path=rgb_data["train"])

    v1 = None
    if not args.skip_v1:
        try:
            from gfdata.unified import make_dataset as make_unified
            v1 = make_unified(str(data), "train", modality="ir", mode="train",
                              imgsz=IMGSZ, batch=16)
        except Exception as exc:                       # pragma: no cover
            print(f"[compare] V1 unavailable: {exc}", file=sys.stderr)

    out = Path(args.out)
    if not out.is_absolute():
        out = EXP / out
    (out / "png").mkdir(parents=True, exist_ok=True)

    indices = [int(i) for i in args.indices.split(",") if i]
    seeds = [int(s) for s in args.seeds.split(",") if s]
    rows = []

    for idx in indices:
        for seed in seeds:
            seek(seed)
            a = stock[idx]
            seek(seed)
            b = v2_ir[idx]
            seek(seed)
            c = v2_pair[idx]

            row = {
                "index": idx, "seed": seed,
                "im_file": Path(a["im_file"]).name,
                "stock": stats(a["img"]), "v2_ir": stats(b["img"]),
                "v2_pair_ir_half": stats(c["img"][:3]),
                "n_boxes": int(a["bboxes"].shape[0]),
                "checks": {
                    "v2_ir_equals_stock_img": bool(torch.equal(a["img"], b["img"])),
                    "v2_ir_equals_stock_boxes": bool(torch.equal(a["bboxes"], b["bboxes"])),
                    "pair_ir_half_equals_stock_img": bool(torch.equal(a["img"], c["img"][:3])),
                    "pair_ir_half_equals_stock_boxes": bool(torch.equal(a["bboxes"], c["bboxes"])),
                    "pair_channels": int(c["img"].shape[0]),
                },
            }

            panels = [("stock_E00", draw(a["img"], a["bboxes"])),
                      ("v2_ir", draw(b["img"], b["bboxes"]))]

            if v1 is not None:
                seek(seed)
                d = v1[idx]
                row["v1_ir"] = stats(d["img"])
                row["v1_n_boxes"] = int(d["bboxes"].shape[0])
                if d["img"].shape == a["img"].shape:
                    diff = (d["img"].float() - a["img"].float()).abs()
                    row["checks"]["v1_mean_abs_diff_vs_stock"] = round(float(diff.mean()), 4)
                    row["checks"]["v1_max_abs_diff_vs_stock"] = int(diff.max())
                    row["checks"]["v1_pixels_differ_pct"] = round(
                        float((diff > 0).float().mean() * 100), 2)
                panels.append(("v1_unified", draw(d["img"], d["bboxes"])))

            panels.append(("v2_pair_ir", draw(c["img"][:3], c["bboxes"])))
            panels.append(("v2_pair_rgb", draw(c["img"][3:], c["bboxes"])))

            row["png"] = str((out / "png" / f"idx{idx}_seed{seed}.png").relative_to(EXP))
            cv2.imwrite(str(out / "png" / f"idx{idx}_seed{seed}.png"),
                        np.concatenate([p[1] for p in panels], 1))
            rows.append(row)
            print(json.dumps(row["checks"], indent=None), f"idx={idx} seed={seed}")

    summary = {
        "generated_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "data": str(data), "rgb_yaml": str(rgb_yaml),
        "indices": indices, "seeds": seeds,
        "all_v2_ir_equals_stock": all(
            r["checks"]["v2_ir_equals_stock_img"]
            and r["checks"]["v2_ir_equals_stock_boxes"] for r in rows),
        "all_pair_ir_half_equals_stock": all(
            r["checks"]["pair_ir_half_equals_stock_img"]
            and r["checks"]["pair_ir_half_equals_stock_boxes"] for r in rows),
        "v1_available": v1 is not None,
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[compare] all_v2_ir_equals_stock = {summary['all_v2_ir_equals_stock']}")
    print(f"[compare] all_pair_ir_half_equals_stock = "
          f"{summary['all_pair_ir_half_equals_stock']}")
    print(f"[compare] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
