#!/usr/bin/env python3
"""Blend Cyclediff fog translated images with original clear images at a given strength.

Creates a 75% foggy + 25% original blend (configurable via --strength).
Output images are written as JPG to a new directory for downstream YOLO training/eval.

For each foggy image, finds the matching clear image by stem (frame ID) and blends:
    blended = strength * foggy + (1 - strength) * clear

Usage:
    python blend_cyclediff_fog.py --split train --strength 0.75
    python blend_cyclediff_fog.py --split test  --strength 0.75

Output goes to: data/flir_adas_v2/synthetic/cyclediff_75fog/fog/{train,test}/
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tqdm

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "flir_adas_v2"
CD_FOG_ROOT = DATA / "synthetic" / "cyclediff" / "fog"

FLIR_TRAIN = DATA / "FLIR_ADAS_v2" / "images_thermal_train" / "data"
FLIR_TEST = DATA / "FLIR_ADAS_v2" / "video_thermal_test" / "data"

# Use the split-specific directories to avoid val contamination:
# yolo/thermal/images/train has 9051 images (train split only, excludes 1144 val)
# yolo/thermal/images/test has 1200 images (test split from FLIR ADAS v2)
YOLO_THERMAL = DATA / "yolo" / "thermal"
SPLIT_TRAIN = YOLO_THERMAL / "images" / "train"
SPLIT_VAL = YOLO_THERMAL / "images" / "val"
SPLIT_TEST = YOLO_THERMAL / "images" / "test"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_clear_source(split):
    """Return the correct source directory for the given split.

    For train: use yolo/thermal/images/train (9051 images, no val contamination)
    For test: use FLIR VTT test (3749 images, matches cyclediff fog test)
    For val: use FLIR val (1144 images)
    """
    if split == "train":
        return SPLIT_TRAIN
    elif split == "val":
        return DATA / "FLIR_ADAS_v2" / "images_thermal_val" / "data"
    elif split == "test":
        return FLIR_TEST
    else:
        raise ValueError(f"Unknown split: {split}")


def read_image(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.dtype.itemsize > 1:
        im = (im.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    else:
        im = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2RGB)
    return im


def find_clear_image(stem, clear_dir):
    """Find the clear source image matching the foggy image stem."""
    for ext in IMG_EXTS:
        candidate = clear_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser(description="Blend cyclediff fog with original clear images")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--strength", type=float, default=0.75,
                    help="Foggy contribution (0=clear, 1=foggy only, 0.75=75% foggy)")
    ap.add_argument("--out-root", type=Path, default=None)
    args = ap.parse_args()

    if not (0.0 <= args.strength <= 1.0):
        print(f"ERROR: strength must be in [0,1], got {args.strength}")
        return 1

    foggy_dir = CD_FOG_ROOT / args.split
    if not foggy_dir.is_dir():
        print(f"ERROR: foggy directory not found: {foggy_dir}")
        return 1

    clear_dir = find_clear_source(args.split)
    if not clear_dir.is_dir():
        print(f"ERROR: clear source directory not found: {clear_dir}")
        return 1

    if args.out_root:
        out_dir = args.out_root / args.split
    else:
        blend_root = DATA / "synthetic" / "cyclediff_75fog"
        out_dir = blend_root / "fog" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    foggy_files = sorted(foggy_dir.glob("*"))
    foggy_files = [f for f in foggy_files if f.is_file() and f.suffix.lower() in IMG_EXTS]
    if not foggy_files:
        print(f"ERROR: no images found in {foggy_dir}")
        return 1

    print(f"Blending {len(foggy_files)} images ({args.split}) "
          f"with strength={args.strength} (foggy) + {1-args.strength:.2f} (clear)")
    print(f"  Foggy source: {foggy_dir}")
    print(f"  Clear source: {clear_dir}")
    print(f"  Output:       {out_dir}")

    n_ok = 0
    n_missing = 0
    pbar = tqdm.tqdm(foggy_files, desc=f"blend {args.split}", file=sys.stdout)
    for f in pbar:
        stem = f.stem
        clear_path = find_clear_image(stem, clear_dir)
        if clear_path is None:
            n_missing += 1
            if n_missing <= 5:
                tqdm.tqdm.write(f"  WARN: no clear match for {stem}")
            continue

        foggy = read_image(f)
        clear = read_image(clear_path)
        if foggy is None or clear is None:
            n_missing += 1
            continue

        if foggy.shape != clear.shape:
            foggy = cv2.resize(foggy, (clear.shape[1], clear.shape[0]), interpolation=cv2.INTER_AREA)

        blended = args.strength * foggy.astype(np.float32) + (1.0 - args.strength) * clear.astype(np.float32)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        out = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{stem}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_ok += 1

    pbar.close()
    print(f"\nDone: {n_ok} blended images written, {n_missing} missing clear matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
