"""Build the matched Clear/Fog datasets for the GATE-IR-on-FLIR experiment.

Design:
  * Every detector is evaluated on the SAME held-out VTT frames:
      - clear: FLIR_ADAS_v2/test/clear          (3,749 frames)
      - fog:   synthetic/cyclegan/fog/images/test (3,749 frames, same stems)
    This fixes the 1,200-vs-3,749 mismatch in the old 3A numbers.
  * Training uses the existing sequence-disjoint clear train/val split.
  * Filtered variants are built by filters.py into data/filtered/.

Datasets:
  b1  : clear train + unfiltered fog train          (mixed baseline)
  b0  : clear train only                            (domain-gap lower bound)
  b2  : unfiltered fog train only                   (fog-only baseline)
  b3  : filtered clear train + filtered fog train   (always-on restoration)
  g1  : clear train + filtered fog train            (gated / filtered-domain)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from .paths import (
    CLASS_NAMES,
    CLEAR_TEST_IMG,
    CLEAR_TEST_LBL,
    CLEAR_TRAIN_IMG,
    CLEAR_TRAIN_LBL,
    CLEAR_VAL_IMG,
    CLEAR_VAL_LBL,
    DATA,
    FOG,
    CONFIGS,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _images(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def link_split(dst_root: Path, split: str, src_img: Path, src_lbl: Path,
               limit: int | None = None, only_labelled: bool = True) -> int:
    """Create file-level symlinks for one split.

    File-level links are used (not directory links) because Ultralytics resolves
    labels by replacing ``/images/`` with ``/labels/``; directory symlinks lose
    that mapping.
    """
    dst_img = dst_root / "images" / split
    dst_lbl = dst_root / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    n = 0
    for img in _images(src_img):
        lbl = src_lbl / f"{img.stem}.txt"
        if only_labelled and not lbl.exists():
            continue
        di = dst_img / img.name
        dl = dst_lbl / f"{img.stem}.txt"
        if not di.exists():
            di.symlink_to(img.resolve())
        if lbl.exists() and not dl.exists():
            dl.symlink_to(lbl.resolve())
        n += 1
        if limit and n >= limit:
            break
    return n


def write_yaml(path: Path, root: Path, train: list[str], val: list[str] | str,
               test: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "path": str(root.resolve()),
        "train": train,
        "val": val,
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    if test:
        content["test"] = test
    path.write_text(yaml.safe_dump(content, sort_keys=False))
    print(f"[yaml] {path} -> {root}")


def _filtered(filter_name: str, source: str, split: str, kind: str = "images") -> Path:
    base = DATA / "filtered" / filter_name / source / split
    return base if kind == "images" else base


def build_one(name: str, train_clear_img: Path | None, train_clear_lbl: Path | None,
              train_fog_img: Path | None, train_fog_lbl: Path | None,
              test_clear_img: Path, test_fog_img: Path, test_fog_lbl: Path,
              limit: int | None = None) -> Path:
    root = DATA / name
    if root.exists():
        shutil.rmtree(root)  # safe: this is a generated symlink tree, not data
    root.mkdir(parents=True)

    if train_clear_img and train_clear_lbl:
        link_split(root, "train_clear", train_clear_img, train_clear_lbl, limit)
    if train_fog_img and train_fog_lbl:
        link_split(root, "train_fog", train_fog_img, train_fog_lbl, limit)

    # Clear val (same as the existing sequence-disjoint split).
    link_split(root, "val", CLEAR_VAL_IMG, CLEAR_VAL_LBL, limit)

    # Matched test sets: same VTT stems, clear vs fog.
    link_split(root, "test_clear", test_clear_img, CLEAR_TEST_LBL, limit)
    link_split(root, "test_fog", test_fog_img, test_fog_lbl, limit)

    train = []
    if train_clear_img:
        train.append("images/train_clear")
    if train_fog_img:
        train.append("images/train_fog")

    write_yaml(CONFIGS / f"{name}_train.yaml", root, train, "images/val", "images/test_fog")
    write_yaml(CONFIGS / f"{name}_eval_clear.yaml", root, train, "images/val", "images/test_clear")
    write_yaml(CONFIGS / f"{name}_eval_fog.yaml", root, train, "images/val", "images/test_fog")
    return root


def build_all(condition: str = "cyclegan", filter_name: str = "gamma_clahe",
              limit: int | None = None) -> None:
    fog = FOG[condition]
    fog_train_img = fog["train_img"]
    fog_train_lbl = fog["train_lbl"]
    fog_test_img = fog["test_img"]
    fog_test_lbl = fog["test_lbl"]

    filt = DATA / "filtered" / filter_name
    f_clear_train = filt / "clear" / "train"
    f_clear_val = filt / "clear" / "val"
    f_fog_train = filt / condition / "train"
    f_fog_test = filt / condition / "test"
    f_clear_test = filt / "clear" / "test"

    for p in (f_clear_train, f_clear_val, f_fog_train, f_fog_test, f_clear_test):
        if not p.exists():
            raise FileNotFoundError(f"filtered directory missing: {p}\nRun the filter step first.")

    # b1: mixed, unfiltered
    build_one("b1", CLEAR_TRAIN_IMG, CLEAR_TRAIN_LBL, fog_train_img, fog_train_lbl,
              CLEAR_TEST_IMG, fog_test_img, fog_test_lbl, limit)
    # b0: clear only
    build_one("b0", CLEAR_TRAIN_IMG, CLEAR_TRAIN_LBL, None, None,
              CLEAR_TEST_IMG, fog_test_img, fog_test_lbl, limit)
    # b2: fog only, unfiltered
    build_one("b2", None, None, fog_train_img, fog_train_lbl,
              CLEAR_TEST_IMG, fog_test_img, fog_test_lbl, limit)
    # b3: always-on filter (filtered clear + filtered fog)
    build_one("b3", f_clear_train, CLEAR_TRAIN_LBL, f_fog_train, fog_train_lbl,
              f_clear_test, f_fog_test, fog_test_lbl, limit)
    # g1: gated / filtered-domain (clear unfiltered + fog filtered)
    build_one("g1", CLEAR_TRAIN_IMG, CLEAR_TRAIN_LBL, f_fog_train, fog_train_lbl,
              CLEAR_TEST_IMG, f_fog_test, fog_test_lbl, limit)
    print("[datasets] built b0, b1, b2, b3, g1")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build matched FLIR clear/fog datasets")
    ap.add_argument("--condition", default="cyclegan", choices=list(FOG))
    ap.add_argument("--filter", dest="filter_name", default="gamma_clahe")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build_all(args.condition, args.filter_name, args.limit)


if __name__ == "__main__":
    main()
