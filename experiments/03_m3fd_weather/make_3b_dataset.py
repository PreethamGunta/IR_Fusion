#!/usr/bin/env python3
"""
Create the Protocol 3B combined dataset YAML for Method 2 (depth).

3B training set = clear_train + depth_heavy_train (mixed)
Validation set  = clear_val + depth_heavy_val (mixed)
Test set        = clear_test   (train_m3fd.py scores this after training)

A separate depth_heavy test eval runs explicitly afterwards via eval_m3fd.py, and
the 3B gap is
    gap = (clear test mAP@50-95) - (depth_heavy test mAP@50-95)

Validation is *mixed*, and that is deliberate rather than cosmetic.  An earlier
version validated on clear only, and an earlier version still earlier validated
on weather only; weather-only validation collapsed a 3B model from 0.56 to 0.28
on clear test, because early stopping then optimises the wrong domain.  Mixed
validation makes the selected checkpoint decent on both, which is the only way
the clear-test-minus-weather-test gap measures generalisation rather than which
domain happened to be chosen for us.  Method 3's 3B YAMLs have used mixed
validation since the same bug was found there; this aligns Method 2 with it.

YOLO discovers labels by replacing /images/ -> /labels/ in the path, so both
directories are checked for existence *and* image/label parity before the YAML is
written.  A silent 0-label dataset is the single most expensive failure mode in
this repo: it does not crash, it reports 0.000 mAP.
"""

import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import provenance

CLEAR_IR = HERE / "data/split/IR"
WEATHER = "heavy"
OUT_DIR = HERE / f"data/3b_fog_{WEATHER}"
# The 3B weather dataset lives in its own destination, separate from the 3A
# screening datasets: 3A needs `test` and no `train`, 3B needs `train` and `test`,
# and writing both into one directory made the second overwrite the first YAML.
WEATHER_ROOT = Path(os.environ.get("M3B_DEPTH_ROOT",
                                   HERE / f"data/synthetic_depth_3b"))
WEATHER_IR = WEATHER_ROOT / f"fog_{WEATHER}" / "IR"

NAMES = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]


def check_pair(images: Path, labels: Path, description: str) -> bool:
    """A directory is usable only if it has images AND matching labels."""
    if not images.is_dir():
        print(f"ERROR: {description} images missing: {images}")
        return False
    imgs = [p for p in images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not imgs:
        print(f"ERROR: {description} images dir is empty: {images}")
        return False
    if not labels.is_dir():
        print(f"ERROR: {description} labels missing: {labels}")
        return False
    lbls = [p for p in labels.iterdir() if p.suffix.lower() == ".txt"]
    if len(lbls) != len(imgs):
        print(f"ERROR: {description}: {len(imgs)} images vs {len(lbls)} labels "
              f"-- Ultralytics would silently treat the difference as background")
        return False
    print(f"  OK  {description}: {len(imgs)} images / {len(lbls)} labels")
    return True


def main():
    print("=== make_3b_dataset.py: Method 2 Protocol 3B combined YAML ===")

    ok = True
    ok &= check_pair(CLEAR_IR / "images" / "train", CLEAR_IR / "labels" / "train",
                     "clear IR train")
    ok &= check_pair(CLEAR_IR / "images" / "val", CLEAR_IR / "labels" / "val",
                     "clear IR val")
    ok &= check_pair(CLEAR_IR / "images" / "test", CLEAR_IR / "labels" / "test",
                     "clear IR test")
    ok &= check_pair(WEATHER_IR / "images" / "train", WEATHER_IR / "labels" / "train",
                     f"depth fog_{WEATHER} IR train")
    ok &= check_pair(WEATHER_IR / "images" / "val", WEATHER_IR / "labels" / "val",
                     f"depth fog_{WEATHER} IR val")
    ok &= check_pair(WEATHER_IR / "images" / "test", WEATHER_IR / "labels" / "test",
                     f"depth fog_{WEATHER} IR test")

    if not ok:
        print("\nOne or more required directories are missing or unbalanced.")
        print(f"Run apply_weather_depth.py --dest {WEATHER_ROOT.parent} "
              f"--splits train val test --condition fog --severity {WEATHER} first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "nc": 6,
        "names": NAMES,
        "train": [str(CLEAR_IR / "images" / "train"),
                  str(WEATHER_IR / "images" / "train")],
        "val": [str(CLEAR_IR / "images" / "val"),
                str(WEATHER_IR / "images" / "val")],
        "test": str(CLEAR_IR / "images" / "test"),
    }

    yaml_path = OUT_DIR / "m3fd_3b.yaml"
    if yaml is not None:
        yaml_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
    else:
        lines = ["nc: 6", "names:"] + [f"- {n}" for n in NAMES]
        lines.append("train:")
        lines += [f"- {t}" for t in config["train"]]
        lines.append("val:")
        lines += [f"- {v}" for v in config["val"]]
        lines.append(f"test: {config['test']}")
        yaml_path.write_text("\n".join(lines) + "\n")

    stamp = provenance.stamp_split(yaml_path,
                                   manifest_src=CLEAR_IR.parent / "split_manifest.json")
    if stamp is None:
        raise SystemExit("no split manifest found -- refusing to write a 3B YAML "
                         "whose results could not be traced to a split")

    print(f"\nWrote 3B YAML -> {yaml_path}")
    print(f"  train: clear + depth fog_{WEATHER}")
    print(f"  val:   clear + depth fog_{WEATHER}   (mixed, so early stopping does "
          f"not optimise one domain)")
    print(f"  test:  clear")
    print(f"  split_hash: {stamp}")


if __name__ == "__main__":
    main()