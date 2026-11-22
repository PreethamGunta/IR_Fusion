"""Shared COCO helpers for the DroneVehicle v2 YOLO trees.

Every script that needs image ids or COCO ground truth must use THIS module
so ids stay consistent between GT JSONs and prediction JSONs:
image_id = index into ``sorted(glob(images/<split>/*.jpg))``.

Categories follow the yaml order: car, truck, bus, van, freight_car
(COCO ids 1..5). Boxes are original-frame pixel xywh (top-left).
"""
from __future__ import annotations

from pathlib import Path

import cv2

EXP = Path(__file__).resolve().parents[2]
DATA = EXP / "data" / "dronevehicle_v2" / "yolo"
CLASSES = ["car", "truck", "bus", "van", "freight_car"]


def split_images(modality: str, split: str) -> list[Path]:
    imgs = sorted((DATA / modality / "images" / split).glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"no images under {DATA / modality / 'images' / split}")
    return imgs


def _probe_size(modality: str, split: str, n: int = 25) -> tuple[int, int]:
    """Verify the crops are a single size; return (height, width)."""
    imgs = split_images(modality, split)
    sizes = set()
    for p in imgs[:n] + imgs[-n:]:
        h, w = cv2.imread(str(p)).shape[:2]
        sizes.add((h, w))
    if len(sizes) != 1:
        raise ValueError(f"mixed image sizes in {modality}/{split}: {sizes}")
    return sizes.pop()


def make_coco_gt(modality: str, split: str) -> dict:
    """COCO GT dict for one modality/split from the YOLO label tree."""
    h, w = _probe_size(modality, split)
    imgs = split_images(modality, split)
    images = [{"id": i, "file_name": p.name, "width": w, "height": h}
              for i, p in enumerate(imgs)]
    annotations = []
    ann_id = 0
    for i, p in enumerate(imgs):
        lbl = DATA / modality / "labels" / split / f"{p.stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c, cx, cy, bw, bh = (float(x) for x in parts[:5])
            bw, bh = bw * w, bh * h
            x, y = cx * w - bw / 2, cy * h - bh / 2
            annotations.append({
                "id": ann_id, "image_id": i, "category_id": int(c) + 1,
                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2), "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1
    categories = [{"id": k + 1, "name": n} for k, n in enumerate(CLASSES)]
    return {"images": images, "annotations": annotations,
            "categories": categories}
