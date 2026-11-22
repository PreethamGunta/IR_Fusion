#!/usr/bin/env python3
"""Set up YOLO-compatible directories for Method 3 CycleGAN translated images.

YOLO discovers labels by replacing /images/ -> /labels/ in an image path.
translate.py writes to `translated/<run>/images/<split>/img.png`, which has no
/images/ segment and would silently produce 0 labels.  This script:

  1. Runs translate.py for each needed split, writing into
       translated/<run>/images/<split>/img.png
     but only after checking that whatever is already there was translated from
     the CURRENT split.  That check is the point: the old `skip if the output
     directory has any PNG` guard meant that re-running this script after a
     resplit kept the pre-split translations -- old-split filenames sitting
     next to a label symlink pointing at the new split's labels.  Most
     translated images then had no matching label file, Ultralytics treated
     them as background, and the evaluation completed with a plausible-looking
     mAP over the wrong data.  Every translation is now stamped with the split
     hash it was produced from, and a mismatch forces a re-translate.

  2. Asserts the translated count equals the source count per split, before any
     YAML is written.  A short translation is the same silent failure one level
     up.

  3. Symlinks labels from the original M3FD split:
       translated/<run>/labels/<split>/ -> data/split/IR/labels/<split>/
     Labels are inherited from the clear source images; filenames are identical
     because translate.py preserves them.  The symlink stores a path, so it
     survives the resplit's atomic directory swap and resolves to the new tree.

  4. Writes six YAML files per weather type and copies the split manifest next
     to them, so every downstream result carries the split it was computed on:
        - cyclegan_{fog,rain}_3a.yaml         eval-only: clear baseline on translated test
        - cyclegan_{fog,rain}_3b.yaml         3B training: mixed clear+translated train/val, clear test
        - cyclegan_{fog,rain}_3b_fogtest.yaml 3B eval: same model on translated test (for gap calc)

Usage:
    python make_cyclegan_dataset.py [--runs thermal_fog thermal_rain]
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                                    # release root
sys.path.insert(0, str(HERE.parent))                       # 03_m3fd_weather/provenance.py
import provenance

M3FD_SPLIT = REPO / "data" / "m3fd" / "split"
M3FD_LABELS = M3FD_SPLIT / "IR" / "labels"
M3FD_CLEAR  = M3FD_SPLIT / "IR"
CLEAR_YAML  = M3FD_SPLIT / "m3fd_clear.yaml"

TRANSLATED_ROOT = HERE / "translated"
RUNS_ROOT       = HERE / "runs"
PY = sys.executable

WEATHER_SPLITS_NEEDED = {
    "thermal_fog":  ("fog",  "Fog"),
    "thermal_rain": ("rain", "Rain"),
}


def source_pngs(split):
    d = M3FD_CLEAR / "images" / split
    return sorted(d.glob("*.png")) if d.is_dir() else []


def stamp_and_verify(translated_run: Path, run: str, split: str, split_hash):
    """Return True if `translated_run` holds this split's translation."""
    out = TRANSLATED_ROOT / run / "images" / split
    marker = TRANSLATED_ROOT / run / "SPLIT_HASH"
    stamped = marker.read_text().strip() if marker.is_file() else None

    if stamped != split_hash:
        if stamped is None and out.is_dir() and any(out.glob("*.png")):
            print(f"  [{run}/{split}] translations exist with no split stamp "
                  f"-- discarding them (provenance unknown)")
        elif stamped is not None:
            print(f"  [{run}/{split}] translations were made from split "
                  f"{stamped}, current split is {split_hash} -- re-translating")
        if out.exists():
            shutil.rmtree(out)
        return False

    n_src = len(source_pngs(split))
    n_out = sum(1 for _ in out.glob("*.png")) if out.is_dir() else 0
    if n_src and n_out != n_src:
        print(f"  [{run}/{split}] stale: {n_out}/{n_src} translated -- re-translating")
        shutil.rmtree(out, ignore_errors=True)
        return False
    print(f"  [{run}/{split}] already translated for this split ({n_out} images)")
    return True


def write_stamp(translated_run: Path, split_hash):
    (translated_run / "SPLIT_HASH").write_text(str(split_hash) + "\n")


def symlink_labels(translated_run: Path, split: str):
    """Create translated/<run>/labels/<split> -> original M3FD labels."""
    lbl_dir = translated_run / "labels" / split
    target = M3FD_LABELS / split
    if not target.is_dir():
        print(f"  WARNING: source label dir missing: {target} -- "
              f"labels will be missing for {split}")
        return
    if lbl_dir.is_symlink():
        # Refresh unconditionally: cheap, and a symlink left pointing at a
        # renamed tree is exactly the kind of thing that quietly yields 0 labels.
        if lbl_dir.resolve() == target.resolve():
            return
        lbl_dir.unlink()
    elif lbl_dir.exists():
        return
    lbl_dir.parent.mkdir(parents=True, exist_ok=True)
    lbl_dir.symlink_to(target)
    print(f"  symlink: {lbl_dir} -> {target}")


def translate_split(run: str, split: str, skip: bool, split_hash) -> Path:
    """Translate <run>/<split> into translated/<run>/images/<split>/."""
    out = TRANSLATED_ROOT / run / "images" / split
    translated_run = TRANSLATED_ROOT / run

    if not skip and stamp_and_verify(translated_run, run, split, split_hash):
        symlink_labels(translated_run, split)
        return out

    if skip:
        print(f"  [{run}/{split}] --skip-translate: leaving {out} as-is")
        return out

    done_marker = RUNS_ROOT / run / "DONE"
    if not done_marker.exists():
        print(f"  WARNING: {run} not done yet (no DONE marker at {done_marker}). "
              f"Skipping translation for {split}.")
        return out

    weights = RUNS_ROOT / run / "G_AB.pt"
    if not weights.exists():
        print(f"  ERROR: no generator weights at {weights}")
        sys.exit(1)

    print(f"  translating {run} / {split} -> {out}")
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, str(HERE / "translate.py"),
        "--run", run,
        "--split", split,
        "--out", str(out),
        "--check-shift",
    ]
    subprocess.run(cmd, check=True)

    n_src = len(source_pngs(split))
    n_out = sum(1 for _ in out.glob("*.png"))
    if n_src and n_out != n_src:
        raise SystemExit(
            f"FATAL: {run}/{split} produced {n_out} translated images for {n_src} "
            f"source frames. A short translation means some evaluation images "
            f"would silently become backgrounds. Not writing YAMLs.")
    write_stamp(translated_run, split_hash)
    symlink_labels(translated_run, split)
    return out


def write_yaml(path: Path, nc: int, names: list, train, val, test: str):
    """Write a minimal YOLO dataset YAML and stamp it with the active split.

    The stamp is not optional. `provenance.split_hash_for_dataset()` resolves a
    dataset's split by walking up from the YAML looking for `split_manifest.json`,
    and `datasets/` sits under method3_cyclegan/ where no manifest exists. Without
    this, every CycleGAN 3A result records `split_hash: null` and every 3B run
    hard-aborts in train_m3fd.py's provenance guard -- which is exactly how job
    8752 died, 41 minutes in, with the manifest sitting one level up in
    data/split/ the whole time.
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

    stamped = provenance.stamp_split(path, manifest_src=M3FD_SPLIT / "split_manifest.json")
    if stamped is None:
        raise SystemExit(
            f"could not stamp {path} with a split manifest. Every result computed "
            f"from this dataset would be untraceable to a split.")
    print(f"  YAML -> {path}  (split_hash {stamped})")


M3FD_NAMES = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]
M3FD_NC    = 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["thermal_fog", "thermal_rain"],
                    choices=list(WEATHER_SPLITS_NEEDED))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--skip-translate", action="store_true",
                    help="skip translate.py even if output is missing")
    args = ap.parse_args()

    if not CLEAR_YAML.exists():
        raise SystemExit(f"{CLEAR_YAML} not found -- run resplit_m3fd.py first")
    split_hash = provenance.split_hash_for_dataset(CLEAR_YAML)
    if split_hash is None:
        raise SystemExit(
            f"no split_hash in {CLEAR_YAML}'s manifest. Every CycleGAN result "
            f"would be untraceable to a split; refusing to build the dataset.")

    out_dir = HERE / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    for run in args.runs:
        short, label = WEATHER_SPLITS_NEEDED[run]
        translated_run = TRANSLATED_ROOT / run

        print(f"\n=== {run} ({label}) ===")

        # --- translate each split we need ---
        for split in args.splits:
            translate_split(run, split, args.skip_translate, split_hash)

        def img(split):
            return str(TRANSLATED_ROOT / run / "images" / split)

        # --- Protocol 3A YAML: eval only, test on CycleGAN fog ---
        # No val or train needed for 3A — we reuse the clear baseline model.
        # Providing dummy clear paths for train/val so the file is a valid YAML
        # (eval_m3fd.py only reads `test`).
        write_yaml(
            out_dir / f"cyclegan_{short}_3a.yaml",
            nc=M3FD_NC, names=M3FD_NAMES,
            train=str(M3FD_CLEAR / "images" / "train"),
            val=str(M3FD_CLEAR / "images" / "val"),
            test=img("test"),
        )

        # --- Protocol 3B YAML: mixed train, mixed val, clear test ---
        # Val is mixed (clear + CycleGAN weather): early stopping optimises
        # for both domains so the model doesn't forget clear while adapting
        # to weather.  This prevents the 3B sabotage bug where weather-only
        # val caused the model to collapse on clear test (mAP 0.56 -> 0.28).
        #
        # Test is clear (not fog): train_m3fd.py runs a test eval after training
        # and the gap is computed as:
        #   gap = clear_test_mAP (from training JSON) - fog_test_mAP (separate eval)
        # If test were fog here, we would get fog mAP from the JSON and would
        # still need a separate clear eval -- doing it this way keeps the
        # training JSON's test number directly comparable to the 3A baseline.
        write_yaml(
            out_dir / f"cyclegan_{short}_3b.yaml",
            nc=M3FD_NC, names=M3FD_NAMES,
            train=[str(M3FD_CLEAR / "images" / "train"), img("train")],
            val=[str(M3FD_CLEAR / "images" / "val"), img("val")],
            test=str(M3FD_CLEAR / "images" / "test"),
        )

        # --- Protocol 3B fog-test YAML: for the separate fog eval after 3B training ---
        write_yaml(
            out_dir / f"cyclegan_{short}_3b_fogtest.yaml",
            nc=M3FD_NC, names=M3FD_NAMES,
            train=str(M3FD_CLEAR / "images" / "train"),   # unused; just valid YAML
            val=str(M3FD_CLEAR / "images" / "val"),       # unused
            test=img("test"),
        )

    # Every result computed from these YAMLs is stamped with this split.
    provenance.stamp_split(CLEAR_YAML, manifest_src=M3FD_SPLIT / "split_manifest.json")
    print("\nDone.  Check the YAML paths above before submitting 3B training.")
    print(f"Split hash: {split_hash}")
    print(f"YAMLs written to: {out_dir}")


if __name__ == "__main__":
    main()
