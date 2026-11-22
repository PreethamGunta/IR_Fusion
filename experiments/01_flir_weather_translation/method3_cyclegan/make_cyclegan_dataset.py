#!/usr/bin/env python3
"""Set up YOLO-compatible directories for Method 3 CycleGAN translated images.

YOLO discovers labels by replacing /images/ -> /labels/ in an image path.
translate.py writes to translated/<config>/images/<split>/img.jpg, which
would produce 0 labels. This script:

  1. Runs translate.py for each needed split, writing into
     translated/<config>/images/<split>/img.jpg
  2. Symlinks labels from the prepare.py output:
     translated/<config>/labels/<split> -> data/flir_adas_v2/yolo/thermal/labels/<split>/
  3. Writes Protocol 3A and 3B YAML files.

Usage:
    python make_cyclegan_dataset.py [--configs fog rain snow]
    python make_cyclegan_dataset.py --configs fog --splits train val test
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
PY = sys.executable

FLIR_DATA = REPO / "data" / "flir_adas_v2" / "FLIR_ADAS_v2"
YOLO_DIR = REPO / "data" / "flir_adas_v2" / "yolo"

# 15 classes from FLIR ADAS v2
FLIR_NAMES = [
    "person", "bike", "car", "motor", "bus", "train", "truck",
    "light", "hydrant", "sign", "face", "other_vehicle",
    "backpack", "bag", "other"
]
FLIR_NC = 15

# Splits that need translation
ALL_SPLITS = ["train", "val", "test"]
ALL_CONFIGS = ["fog", "rain", "snow"]


def translated_dir(config, split):
    return HERE / "translated" / config / split


def translate_split(config, split, skip=False):
    """Run translate.py for one config/split."""
    out = translated_dir(config, split)
    out.mkdir(parents=True, exist_ok=True)

    # Check if already complete
    n_existing = len(list(out.glob("*.jpg")) + list(out.glob("*.png")))
    if n_existing > 0 and not skip:
        print(f"  [{config}/{split}] already has {n_existing} translated images -- skipping")
        return out

    done_marker = HERE / "runs" / config / "DONE"
    if not done_marker.exists():
        print(f"  WARNING: {config} CycleGAN not done (no DONE marker at {done_marker}).")
        print(f"  Skipping translation for {split}.")
        return out

    weights = HERE / "runs" / config / "G_AB.pt"
    if not weights.exists():
        print(f"  ERROR: no generator weights at {weights}")
        return out

    print(f"  translating {config}/{split} -> {out}")
    cmd = [
        PY, str(HERE / "translate.py"),
        "--config", config,
        "--weights", str(weights),
        "--split", split,
        "--out", str(out),
        "--check-shift",
    ]
    subprocess.run(cmd, check=True)
    return out


def symlink_labels(config, split, modality="thermal"):
    """Symlink labels from the YOLO prepare output.

    Labels are inherited from the clear source images; filenames are identical
    because translate.py preserves them.
    """
    lbl_dir = translated_dir(config, split).parent / "labels" / split
    src_labels = YOLO_DIR / modality / "labels" / split

    if not src_labels.is_dir():
        print(f"  WARNING: source labels missing: {src_labels}")
        return

    if lbl_dir.is_symlink():
        if lbl_dir.resolve() == src_labels.resolve():
            return
        lbl_dir.unlink()
    elif lbl_dir.exists():
        print(f"  WARNING: {lbl_dir} exists but is not a symlink -- leaving as-is")
        return

    lbl_dir.parent.mkdir(parents=True, exist_ok=True)
    lbl_dir.symlink_to(src_labels.resolve())
    print(f"  symlink: {lbl_dir} -> {src_labels.resolve()}")


def count_images(config, split):
    """Count translated images and matching labels."""
    img_dir = translated_dir(config, split)
    lbl_dir = translated_dir(config, split).parent / "labels" / split

    n_imgs = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    n_lbls = 0
    if lbl_dir.is_symlink() or lbl_dir.is_dir():
        n_lbls = len(list(lbl_dir.glob("*.txt")))
    return n_imgs, n_lbls


def write_yaml(path, nc, names, train, val, test):
    """Write a YOLO dataset YAML.

    train/val/test can be strings or lists of strings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"nc: {nc}", "names:"]
    for n in names:
        lines.append(f"  - {n}")
    for key, value in [("train", train), ("val", val), ("test", test)]:
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n")
    print(f"  YAML -> {path}")


def main():
    ap = argparse.ArgumentParser(description="Build YOLO datasets from CycleGAN translations")
    ap.add_argument("--configs", nargs="+", default=ALL_CONFIGS, choices=ALL_CONFIGS,
                    help="which weather conditions to process")
    ap.add_argument("--splits", nargs="+", default=ALL_SPLITS, choices=ALL_SPLITS,
                    help="which splits to translate")
    ap.add_argument("--skip-translate", action="store_true",
                    help="skip running translate.py, only build YAMLs from existing translations")
    args = ap.parse_args()

    yolo_thermal = YOLO_DIR / "thermal"
    yaml_dir = HERE / "datasets"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    for config in args.configs:
        print(f"\n=== {config} ===")

        # Translate each split
        for split in args.splits:
            translate_split(config, split, skip=args.skip_translate)
            symlink_labels(config, split, "thermal")
            n_img, n_lbl = count_images(config, split)
            print(f"  [{config}/{split}] {n_img} images, {n_lbl} labels")

        # Write YAMLs
        def img_dir(split):
            return str(translated_dir(config, split).resolve())

        def lbl_dir(split):
            p = translated_dir(config, split).parent / "labels" / split
            return str(p.resolve())

        # Protocol 3A: eval only, clear baseline on translated test
        write_yaml(
            yaml_dir / f"cyclegan_{config}_3a.yaml",
            nc=FLIR_NC, names=FLIR_NAMES,
            train=str(yolo_thermal / "images" / "train"),
            val=str(yolo_thermal / "images" / "val"),
            test=img_dir("test"),
        )

        # Protocol 3B: mixed clear + translated train/val, clear test for gap
        write_yaml(
            yaml_dir / f"cyclegan_{config}_3b.yaml",
            nc=FLIR_NC, names=FLIR_NAMES,
            train=[str(yolo_thermal / "images" / "train"), img_dir("train")],
            val=[str(yolo_thermal / "images" / "val"), img_dir("val")],
            test=str(yolo_thermal / "images" / "test"),
        )

        # Protocol 3B eval: clear baseline trained on 3B, evaluated on translated test
        write_yaml(
            yaml_dir / f"cyclegan_{config}_3b_adverse_test.yaml",
            nc=FLIR_NC, names=FLIR_NAMES,
            train=str(yolo_thermal / "images" / "train"),
            val=str(yolo_thermal / "images" / "val"),
            test=img_dir("test"),
        )

    print(f"\nDone. YAMLs written to: {yaml_dir}")


if __name__ == "__main__":
    main()
