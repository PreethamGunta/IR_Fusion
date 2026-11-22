#!/usr/bin/env python3
"""Build a tiny debug subset of a YOLO dataset (2 frames/split) for smoke tests.

Two bugs in the previous version, both of which produce a run that *looks* fine:

* The source label directory was computed as `base_dir.parent / "labels"`, i.e.
  `data/split/labels/train` for a YAML whose `path:` is `data/split/IR`. Labels
  live at `data/split/IR/labels/train`. Every debug subset therefore got zero
  labels, and Ultralytics does not crash on that -- it logs
  "0 images, N backgrounds", trains on nothing, and reports 0.000 mAP.
* `data[split]` may be a list (the mixed clear+weather train entries), and
  `.replace` on a list raises. Mixed 3B datasets are exactly the ones you want
  to smoke-test, so the list form is handled here.
"""
import os
import shutil
import sys
from pathlib import Path

import yaml

IMG_EXTS = (".png", ".jpg", ".jpeg")


def label_dir_for(image_dir: Path) -> Path:
    """YOLO's own rule: replace the trailing /images/ with /labels/."""
    parts = list(image_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    raise SystemExit(f"cannot derive a label dir from {image_dir} (no /images/ in it)")


def resolve(root: Path, value: str) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else root / p


def make_debug(src_yaml_path, n_per_split=4):
    src_yaml = Path(src_yaml_path)
    data = yaml.safe_load(src_yaml.read_text()) or {}

    root = Path(str(data["path"])) if data.get("path") else src_yaml.parent
    if not root.is_absolute():
        root = (src_yaml.parent / root).resolve()
    debug_dir = root.parent / (root.name + "_debug")

    copied = 0
    for split in ("train", "val", "test"):
        raw = data.get(split)
        if not raw:
            continue
        # A list-valued entry (mixed clear + weather training set) becomes a
        # numbered subdirectory so the two domains do not overwrite each other.
        entries = raw if isinstance(raw, list) else [raw]
        for k, entry in enumerate(entries):
            src_img_dir = resolve(root, entry)
            if not src_img_dir.is_dir():
                print(f"  WARNING: {src_img_dir} missing -- skipping {split}")
                continue
            try:
                src_lbl_dir = label_dir_for(src_img_dir)
            except SystemExit as exc:
                print(f"  WARNING: {exc} -- {split} contributes no labels")
                continue

            suffix = "" if len(entries) == 1 else f"_{k}"
            dest_img_dir = debug_dir / f"images{suffix}" / split
            dest_lbl_dir = label_dir_for(dest_img_dir) if (debug_dir / "images" / split).exists() \
                else debug_dir / f"labels{suffix}" / split
            dest_img_dir.mkdir(parents=True, exist_ok=True)
            dest_lbl_dir.mkdir(parents=True, exist_ok=True)

            images = sorted(f for f in os.listdir(src_img_dir)
                            if f.lower().endswith(IMG_EXTS))[:n_per_split]
            for img in images:
                shutil.copy2(src_img_dir / img, dest_img_dir / img)
                lbl = src_lbl_dir / f"{img.rsplit('.', 1)[0]}.txt"
                if lbl.exists():
                    shutil.copy2(lbl, dest_lbl_dir / lbl.name)
                else:
                    # Background image: YOLO still wants the file to exist.
                    (dest_lbl_dir / f"{img.rsplit('.', 1)[0]}.txt").touch()
                copied += 1
            print(f"  {split}{suffix}: {len(images)} images from {src_img_dir}")

    # Every debug image dir must be paired with a sibling labels/ dir, because
    # that pairing is what YOLO looks for.
    for img_dir in sorted(debug_dir.glob("images*/**")):
        if not img_dir.is_dir() or img_dir.name == "labels":
            continue
        try:
            lbl = label_dir_for(img_dir)
        except SystemExit:
            continue
        if not lbl.is_dir():
            lbl.mkdir(parents=True, exist_ok=True)

    debug_yaml = src_yaml.parent / f"{src_yaml.stem}_debug.yaml"
    out = dict(data)
    out["path"] = str(debug_dir.absolute())
    debug_yaml.write_text(yaml.dump(out, sort_keys=False))
    print(f"Created debug subset at {debug_yaml} ({copied} images)")
    if copied == 0:
        raise SystemExit("debug subset is empty -- refusing to write a YAML that "
                         "would train on nothing")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        make_debug(sys.argv[1])
    else:
        raise SystemExit("usage: make_debug_subset.py <dataset.yaml>")