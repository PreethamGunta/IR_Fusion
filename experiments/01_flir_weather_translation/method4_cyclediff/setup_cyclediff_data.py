#!/usr/bin/env python3
"""Set up CycleDiff-compatible directory structure for FLIR ADAS v2 and DSERT-RoLL.

CycleDiff's Single_dataset and RGB_Unpair_dataset expect:
  data_root / split / datafolder_name / *.png

This script creates those directories with symlinks to the actual image files,
without duplicating storage.
"""

import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "data" / "flir_adas_v2").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
REPO = ROOT

# ---------------------------------------------------------------------------
# FLIR ADAS v2 -> clear domain
# ---------------------------------------------------------------------------
FLIR_ROOT = REPO / "data" / "flir_adas_v2" / "FLIR_ADAS_v2"


def setup_flir_clear():
    print("=== FLIR ADAS v2 -> clear ===")
    train_clear = FLIR_ROOT / "train" / "clear"
    test_clear = FLIR_ROOT / "test" / "clear"
    train_clear.mkdir(parents=True, exist_ok=True)
    test_clear.mkdir(parents=True, exist_ok=True)

    # Training images: from both images_thermal_train and images_thermal_val
    train_src_dirs = [
        FLIR_ROOT / "images_thermal_train" / "data",
        FLIR_ROOT / "images_thermal_val" / "data",
    ]
    train_count = 0
    for src_dir in train_src_dirs:
        if not src_dir.is_dir():
            print(f"  WARNING: {src_dir} not found")
            continue
        for img in sorted(src_dir.iterdir()):
            if img.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            dst = train_clear / img.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(img.resolve())
            train_count += 1
    print(f"  train/clear: {train_count} images")

    # Test images: from video_thermal_test
    test_src = FLIR_ROOT / "video_thermal_test" / "data"
    test_count = 0
    if test_src.is_dir():
        for img in sorted(test_src.iterdir()):
            if img.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            dst = test_clear / img.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(img.resolve())
            test_count += 1
    print(f"  test/clear: {test_count} images")
    return train_count, test_count


# ---------------------------------------------------------------------------
# DSERT-RoLL -> target domains (fog, rain, snow)
# ---------------------------------------------------------------------------
DSERT_ROOT = REPO / "data" / "DSERT-RoLL"


def setup_dsert_domain(condition, datafolder_name, source_dirs,
                       train_ratio=0.9, seed=42):
    """Create train/datafolder_name and test/datafolder_name under DSERT_ROOT.

    If source_dirs have images, symlinks are created with a random split.
    If no images are found, directories are created but left empty.
    """
    train_dir = DSERT_ROOT / "train" / datafolder_name
    test_dir = DSERT_ROOT / "test" / datafolder_name
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing symlinks
    for d in (train_dir, test_dir):
        for item in d.iterdir():
            if item.is_symlink() or item.is_file():
                item.unlink()

    all_images = []
    for sd in source_dirs:
        if sd.is_dir():
            for img in sd.rglob("*"):
                if img.suffix.lower() in ('.png', '.jpg', '.jpeg') and not img.name.startswith('._'):
                    all_images.append(img)

    if not all_images:
        print(f"  {datafolder_name}: {train_dir} and {test_dir} created (EMPTY - no images on disk)")
        return 0, 0

    random.seed(seed)
    random.shuffle(all_images)
    n_train = int(len(all_images) * train_ratio)
    train_imgs = all_images[:n_train]
    test_imgs = all_images[n_train:]

    for img in train_imgs:
        unique_name = f"{img.parent.parent.name}_{img.name}"
        dst = train_dir / unique_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(img.resolve())

    for img in test_imgs:
        unique_name = f"{img.parent.parent.name}_{img.name}"
        dst = test_dir / unique_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(img.resolve())

    print(f"  {datafolder_name}: train={len(train_imgs)}, test={len(test_imgs)}")
    return len(train_imgs), len(test_imgs)


def setup_dsert_snow():
    print("=== DSERT-RoLL -> snow ===")
    snow_source_dirs = []
    for recording in (DSERT_ROOT / "Heavy_Snow").iterdir():
        if recording.is_dir():
            thermal_dir = recording / "rectified_THERMAL_L"
            if thermal_dir.is_dir():
                snow_source_dirs.append(thermal_dir)

    return setup_dsert_domain("snow", "snow", snow_source_dirs)


def setup_dsert_fog():
    print("=== DSERT-RoLL -> fog ===")
    fog_source_dirs = []
    fog_root = DSERT_ROOT / "Fog"
    if fog_root.is_dir():
        for recording in fog_root.iterdir():
            if recording.is_dir():
                thermal_dir = recording / "rectified_THERMAL_L"
                if thermal_dir.is_dir():
                    fog_source_dirs.append(thermal_dir)
    return setup_dsert_domain("fog", "fog", fog_source_dirs)


def setup_dsert_rain():
    print("=== DSERT-RoLL -> rain ===")
    rain_source_dirs = []
    for cond in ["Heavy_Rain", "Light_Rain"]:
        rain_root = DSERT_ROOT / cond
        if rain_root.is_dir():
            for recording in rain_root.iterdir():
                if recording.is_dir():
                    thermal_dir = recording / "rectified_THERMAL_L"
                    if thermal_dir.is_dir():
                        rain_source_dirs.append(thermal_dir)
    return setup_dsert_domain("rain", "rain", rain_source_dirs)


def setup_heavy_snow_split():
    """Create train/snow and test/snow under Heavy_Snow for snow configs that
    point data_root to Heavy_Snow."""
    print("=== DSERT-RoLL/Heavy_Snow -> snow split ===")
    snow_root = DSERT_ROOT / "Heavy_Snow"
    train_dir = snow_root / "train" / "snow"
    test_dir = snow_root / "test" / "snow"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for d in (train_dir, test_dir):
        for item in d.iterdir():
            if item.is_symlink() or item.is_file():
                item.unlink()

    all_images = []
    for recording in snow_root.iterdir():
        if recording.is_dir():
            thermal_dir = recording / "rectified_THERMAL_L"
            if thermal_dir.is_dir():
                for img in thermal_dir.rglob("*"):
                    if img.suffix.lower() in ('.png', '.jpg', '.jpeg') and not img.name.startswith('._'):
                        all_images.append(img)

    if not all_images:
        print(f"  snow (Heavy_Snow): {train_dir} and {test_dir} created (EMPTY)")
        return 0, 0

    random.seed(42)
    random.shuffle(all_images)
    n_train = int(len(all_images) * 0.9)

    for img in all_images[:n_train]:
        unique_name = f"{img.parent.parent.name}_{img.name}"
        dst = train_dir / unique_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(img.resolve())

    for img in all_images[n_train:]:
        unique_name = f"{img.parent.parent.name}_{img.name}"
        dst = test_dir / unique_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(img.resolve())

    print(f"  snow (Heavy_Snow): train={n_train}, test={len(all_images) - n_train}")
    return n_train, len(all_images) - n_train


def main():
    tr_clear, te_clear = setup_flir_clear()
    setup_dsert_fog()        # likely empty - no Fog images on disk
    setup_dsert_rain()       # likely empty - no Rain data on disk
    setup_dsert_snow()       # DSERT-RoLL/train/snow, test/snow
    tr_snow, te_snow = setup_heavy_snow_split()  # DSERT-RoLL/Heavy_Snow/train/snow, test/snow

    print("\n=== Summary ===")
    print(f"  Clear (FLIR):  train={tr_clear}, test={te_clear}")
    print(f"  Snow (DSERT):  train={tr_snow}, test={te_snow}")


if __name__ == "__main__":
    main()
