#!/usr/bin/env python
"""
Set up DSERT-RoLL data symlinks for CycleDiff training.

The main data directory (data/dsert_roll)
has the complete Fog/ and Heavy_Rain/ directories with rectified_THERMAL_L images.
This script creates symlinks in the worktree's data/DSERT-RoLL/train/{fog,rain}/
so that CycleDiff's Single_dataset and RGB_Unpair_dataset can find the data.

Also sets up test directories.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve()
while not (ROOT / "data" / "flir_adas_v2").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
REPO = ROOT
MAIN_REPO = REPO
DSERT_WORKTREE = REPO / "data" / "dsert_roll"
MAIN_DSERT = REPO / "data" / "dsert_roll"

# Mapping: worktree condition name -> main repo DSERT-RoLL directory
COND_MAP = {
    "fog": "Fog",
    "rain": "Heavy_Rain",
    "snow": "Heavy_Snow",
}

def find_thermal_images(cond_dir):
    """Find all thermal images under DSERT-RoLL/<Condition>/<seq>/rectified_THERMAL_L/"""
    images = []
    for seq_dir in sorted(cond_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        thermal_dir = seq_dir / "rectified_THERMAL_L"
        if thermal_dir.is_dir():
            for ext in ("*.png", "*.jpg"):
                images.extend(thermal_dir.glob(ext))
    return images

def setup_condition(cond_name, main_cond_name, split):
    """Create symlinks for a condition in train/ or test/"""
    main_dir = MAIN_DSERT / main_cond_name
    if not main_dir.is_dir():
        print(f"  WARNING: {main_dir} does not exist. Skipping {cond_name}.")
        return 0

    target_dir = DSERT_WORKTREE / split / cond_name
    target_dir.mkdir(parents=True, exist_ok=True)

    images = find_thermal_images(main_dir)
    if not images:
        print(f"  WARNING: No thermal images found in {main_dir}")
        return 0

    # Create symlinks for each image
    count = 0
    for img in images:
        symlink_name = f"{img.parent.parent.name}_{img.name}"
        symlink_path = target_dir / symlink_name
        if not symlink_path.exists():
            os.symlink(img, symlink_path)
        count += 1

    print(f"  {cond_name}/{split}: {count} images symlinked to {target_dir}")
    return count

if __name__ == "__main__":
    print(f"Worktree DSERT: {DSERT_WORKTREE}")
    print(f"Main repo DSERT: {MAIN_DSERT}")

    for cond_name, main_cond_name in COND_MAP.items():
        print(f"\n--- Setting up {cond_name} ---")
        for split in ("train", "test"):
            setup_condition(cond_name, main_cond_name, split)
