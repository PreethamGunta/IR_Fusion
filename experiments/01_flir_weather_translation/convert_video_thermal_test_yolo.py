#!/usr/bin/env python3
"""Convert FLIR ADAS v2 video_thermal_test COCO annotations to YOLO format.

This script converts the 3749-image video_thermal_test COCO annotations to YOLO
labels and places them in the specified output directory so the gap measurement
script can use them for the adverse weather test set.

Usage:
    python flir_adras_v2/convert_video_thermal_test_yolo.py
    python flir_adras_v2/convert_video_thermal_test_yolo.py --output /path/to/labels
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve()
while not (ROOT / "data" / "flir_adas_v2").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
REPO = ROOT
DATA_ROOT = REPO / "data" / "flir_adas_v2"
COCO_PATH = DATA_ROOT / "FLIR_ADAS_v2" / "video_thermal_test" / "coco.json"

ALL_CLASSES = [
    "person", "bike", "car", "motor", "bus", "train", "truck",
    "light", "hydrant", "sign", "face", "other_vehicle",
    "backpack", "bag", "other"
]


def main():
    ap = argparse.ArgumentParser(description="Convert video_thermal_test COCO to YOLO")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output directory for YOLO labels (default: synthetic/cyclediff/fog/labels/test)")
    args = ap.parse_args()

    # Load COCO
    with open(COCO_PATH) as f:
        coco = json.load(f)

    # Map category IDs to YOLO indices
    cat_id_to_idx = {}
    for i, name in enumerate(ALL_CLASSES):
        for cat in coco["categories"]:
            if cat["name"] == name:
                cat_id_to_idx[cat["id"]] = i

    # Group annotations by image_id
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    # Determine output directory
    if args.output:
        labels_dir = args.output
    else:
        labels_dir = DATA_ROOT / "synthetic" / "cyclediff" / "fog" / "labels" / "test"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Convert video_thermal_test annotations
    converted = 0
    for img in coco["images"]:
        img_id = img["id"]
        anns = ann_by_img.get(img_id, [])
        img_name = Path(img["file_name"]).name
        stem = Path(img_name).stem
        label_path = labels_dir / f"{stem}.txt"

        w, h = img["width"], img["height"]
        lines = []
        for ann in anns:
            cls = cat_id_to_idx.get(ann["category_id"], -1)
            if cls < 0:
                continue
            x, y, bw, bh = ann["bbox"]
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        label_path.write_text("\n".join(lines))
        converted += 1

    print(f"Converted {converted} video_thermal_test labels to YOLO format")
    print(f"Labels saved to: {labels_dir}")
    print(f"Label files: {len(list(labels_dir.glob('*.txt')))}")


if __name__ == "__main__":
    main()
