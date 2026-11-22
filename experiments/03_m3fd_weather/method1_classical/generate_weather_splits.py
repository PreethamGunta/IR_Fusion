#!/usr/bin/env python3
"""Method 1: classical synthetic weather on M3FD thermal frames.

IR ONLY. The project decided against RGB, and this script used to walk
`src/RGB` (which `resplit_m3fd.py` never creates) while merely tolerating a
missing IR file -- the mirror image of the bug in `apply_weather_depth.py`, where
a missing RGB directory meant zero images were generated and a YAML was written
anyway. Both modalities are now gone: thermal in, thermal out, labels copied
verbatim.

The image list came from `os.listdir`, which also returns the sibling `.npy`
disk-cache files Ultralytics writes under `--cache disk`. That is why the live
log for val read `0/648` for a 324-frame split: 324 PNGs plus 324 .npy. The
`imread -> None -> continue` guard hid the miscount instead of preventing it, and
the same `os.listdir` shape would silently halve or double any future iteration.
Globbing `*.png` is the fix, and it matches how every other script in this repo
reads the split.

Labels are copied unchanged. Editing the ground truth of an evaluation set makes
a measured "gap" partly a change of denominator rather than a detection failure.

Severity is not decided in this file. Both conditions take their contrast
retention from `severity.CONTRAST_TRANSFER` and both record the contrast and
high-frequency loss they actually achieved in `generation.json`, so the two arms
are comparable by measurement rather than by two hand-picked constants.
"""

import argparse
import json
import shutil
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import provenance
import severity

M3FD_NAMES = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]

# Rain on thermal. The previous implementation was `GaussianBlur(k)` with
# k = 3/5/7, which is blur and not rain. Measured over 24 val frames against the
# source, it shifted the frame mean by ~0.000 grey levels at every k -- the
# docstring used to claim 0.6-0.8, which is the mean absolute deviation, not the
# mean shift -- while removing 51 / 65 / 76 % of the Laplacian high-frequency
# energy. So the old model was not the no-op the comment described; it simply
# removed texture indiscriminately, which is a blur and carries no information
# about rain. Thermal rain is modelled instead as droplet streaks plus a contrast
# loss, and the contrast loss is the *same* one fog applies at the same severity.
RAIN_STREAKS = {"light": 120, "medium": 320, "heavy": 700}
RAIN_AMPLITUDE = {"light": 12.0, "medium": 22.0, "heavy": 34.0}
RAIN_ANGLE_JITTER = 12.0     # degrees either side of vertical


def _contract_contrast(img, ctr):
    """Scale contrast about the frame mean, preserving the mean exactly.

    This is the single contrast-loss mechanism shared by both conditions: rain
    applies it too, so at equal severity the two arms lose the same global
    contrast and differ only in the structure they remove.
    """
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return mean + ctr * (img.astype(np.float32) - mean)


def fog_ir(img, ctr):
    """Loss of thermal contrast, spatially uniform. Mean-preserving."""
    return np.clip(_contract_contrast(img, ctr), 0, 255).astype(np.uint8)


def _streak_layer(shape, n_streaks, amplitude, seed):
    """Additive droplet layer: short near-vertical segments, lightly blurred."""
    h, w = shape[:2]
    rng = np.random.default_rng(seed)
    layer = np.zeros((h, w), np.float32)
    length = rng.integers(12, 40, n_streaks)
    for i in range(n_streaks):
        y = int(rng.integers(0, h))
        x = int(rng.integers(0, w))
        angle = np.deg2rad(90.0 + rng.normal(0.0, RAIN_ANGLE_JITTER))
        dx, dy = np.cos(angle), -np.sin(angle)
        L = int(length[i])
        seg = np.zeros((h, w), np.float32)
        cv2.line(seg, (x, y),
                 (int(x + dx * L), int(y + dy * L)), 1.0,
                 thickness=int(rng.integers(1, 3)))
        layer += seg * float(rng.uniform(-amplitude, amplitude))
    return cv2.GaussianBlur(layer, (3, 3), 0)


def rain_ir(img, severity_name, seed=0):
    """Thermal rain: the fog contrast loss, then droplet texture on top.

    Order matters. The contrast contraction is applied to the *clean* frame and
    the streak layer is added afterwards, so streaks enter at their physical
    amplitude instead of being scaled down along with everything else -- which is
    what previously cancelled most of the streak layer's contribution to texture
    loss and left the rain arm mechanically a second fog.

    Because the droplet layer is a near-field effect it is not attenuated by the
    scene's own transmission: a droplet 2 m from the camera looks the same
    whether the scene behind it is 5 m or 50 m away. It is therefore added
    uniformly. Depth belongs in the *contrast* term, where it belongs physically,
    and in Method 2 that term is what varies per pixel.

    Seeded per frame so regeneration is bit-reproducible; without that, two runs
    of the pipeline would produce different images from identical inputs and no
    two result files would be comparable.
    """
    ctr = severity.contrast_transfer(severity_name)
    out = _contract_contrast(img, ctr)
    out = out + _streak_layer(out.shape[:2], RAIN_STREAKS[severity_name],
                              RAIN_AMPLITUDE[severity_name], seed)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def generate_condition(src_base: Path, dest_base: Path, condition: str,
                       severity_name: str, splits):
    ctr = severity.contrast_transfer(severity_name)
    dest_base = Path(dest_base)
    img_root, lbl_root = dest_base / "IR" / "images", dest_base / "IR" / "labels"
    for split in splits:
        (img_root / split).mkdir(parents=True, exist_ok=True)
        (lbl_root / split).mkdir(parents=True, exist_ok=True)

    total_instances = 0
    degradation = severity.DegradationLog()
    for split in splits:
        src_img_dir = src_base / "IR" / "images" / split
        src_lbl_dir = src_base / "IR" / "labels" / split
        if not src_img_dir.is_dir():
            raise SystemExit(f"{src_img_dir} missing -- is the split built?")

        # *.png explicitly: os.listdir also returns the sibling .npy files that
        # `ultralytics --cache disk` writes into the split.
        files = sorted(src_img_dir.glob("*.png"))
        if not files:
            raise SystemExit(f"{src_img_dir}: no *.png")

        n_labels = 0
        for f in tqdm(files, desc=f"{condition}_{severity_name}/{split}"):
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"unreadable {f}")
            # Seed from the frame name so the degradation is a pure function of
            # the input: re-running the pipeline must not change the dataset.
            seed = int(f.stem) if f.stem.isdigit() else zlib.crc32(f.stem.encode())
            out = fog_ir(img, ctr) if condition == "fog" \
                else rain_ir(img, severity_name, seed=seed)
            if not cv2.imwrite(str(img_root / split / f.name), out):
                raise SystemExit(f"failed to write {img_root / split / f.name}")
            degradation.add(img, out)

            lbl = src_lbl_dir / f"{f.stem}.txt"
            if not lbl.exists():
                raise SystemExit(f"{lbl} missing -- image/label parity broken")
            shutil.copy2(lbl, lbl_root / split / lbl.name)
            n_labels += 1
            total_instances += sum(1 for line in lbl.read_text().splitlines() if line.strip())

        if n_labels != len(files):
            raise SystemExit(f"{split}: {len(files)} images but {n_labels} labels")
        print(f"  {condition}_{severity_name}/{split}: {len(files)} images, "
              f"{n_labels} labels")

    config = {
        "nc": 6,
        "names": M3FD_NAMES,
        "path": str((dest_base / "IR").absolute()),
        "train": f"images/{splits[0]}",
        "val": f"images/{'val' if 'val' in splits else splits[0]}",
        "test": f"images/{'test' if 'test' in splits else splits[-1]}",
    }
    resolved_test = "test" if "test" in splits else splits[-1]
    yaml_path = dest_base / f"m3fd_{condition}_{severity_name}.yaml"
    yaml_path.write_text(yaml.dump(config, sort_keys=False))

    meta = {
        "method": "method1_classical",
        "condition": condition,
        "severity": severity_name,
        "modality": "IR only",
        "severity_defined_by": "contrast_transfer_ratio",
        "contrast_transfer_target": ctr,
        # Only the parameters this condition actually used. The rain arm used to
        # be described by a sigma dictionary and the fog arm by a different one,
        # which is how the two arms drifted 1.33x apart at equal severity labels.
        # What the ladder actually achieved is under "degradation" below.
        "rain_streaks": RAIN_STREAKS[severity_name] if condition == "rain" else None,
        "rain_amplitude": RAIN_AMPLITUDE[severity_name] if condition == "rain" else None,
        "labels_filtered": False,
        "instances": total_instances,
        "splits": splits,
        # A dataset generated without a `test` split still gets a YAML, and its
        # `test:` then aliases whichever split it does have. eval_m3fd.py refuses
        # to score that; recorded here so the artefact is self-describing.
        "test_split_resolved": resolved_test,
        "test_is_aliased": resolved_test != "test",
        "degradation": degradation.summary(),
    }
    meta.update(provenance.run_record(
        split_hash=provenance.split_hash_for_dataset(src_base / "m3fd_clear.yaml")))
    (dest_base / "generation.json").write_text(json.dumps(meta, indent=2))

    # Hard requirement, not best-effort: without the manifest next to the YAML
    # every result computed from this dataset records split_hash: null, which is
    # how the previous audit found Method 1 untraceable.
    stamped = provenance.stamp_split(
        yaml_path, manifest_src=src_base / "split_manifest.json")
    if not stamped:
        raise SystemExit(
            f"could not stamp {yaml_path} with a split manifest "
            f"({src_base / 'split_manifest.json'} not readable). Every result "
            f"from this dataset would be untraceable to a split.")
    print(f"  YAML -> {yaml_path}  (split_hash {stamped})")


def main():
    ap = argparse.ArgumentParser()
    ROOT = Path(__file__).resolve()
    while not (ROOT / "data" / "flir_adas_v2").exists() and ROOT != ROOT.parent:
        ROOT = ROOT.parent
    ap.add_argument("--src", default=str(ROOT / "data" / "m3fd" / "split"))
    ap.add_argument("--dest", default=str(ROOT / "data" / "m3fd" / "synthetic"))
    ap.add_argument("--conditions", nargs="+", default=["fog", "rain"])
    ap.add_argument("--severities", nargs="+", default=list(severity.SEVERITIES),
                    help=f"any of {severity.SEVERITIES}; the contrast each one "
                         f"implies is fixed by severity.CONTRAST_TRANSFER")
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    help="3A screening needs only the evaluation splits; pass "
                         "`train val test` when building a 3B training set")
    args = ap.parse_args()

    src = Path(args.src)
    for cond in args.conditions:
        for sev in args.severities:
            print(f"\n=== Method 1: {cond}_{sev} "
                  f"(target contrast transfer {severity.contrast_transfer(sev):.2f}) ===")
            generate_condition(src, Path(args.dest) / f"{cond}_{sev}",
                               cond, sev, args.splits)


if __name__ == "__main__":
    main()