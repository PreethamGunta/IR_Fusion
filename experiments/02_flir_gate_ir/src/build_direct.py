"""Build inode/space-efficient YOLO YAMLs that point directly at the source dirs.

No per-filter symlink trees. Clear train/val/test come from the original FLIR
YOLO tree; filtered fog images get their labels hard-linked next to them (no
data duplication), so YOLO resolves labels in the same directory.
"""

from __future__ import annotations

import argparse
import os
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
    CONFIGS,
    DATA,
    FOG,
    REPO_ROOT,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def init_clear_test() -> None:
    """One shared symlink tree for the matched VTT clear test set."""
    dst_img = DATA / "vtt_clear" / "images" / "test"
    dst_lbl = DATA / "vtt_clear" / "labels" / "test"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in CLEAR_TEST_IMG.iterdir():
        if img.suffix.lower() not in IMG_EXTS:
            continue
        dst = dst_img / img.name
        if not dst.exists():
            dst.symlink_to(img.resolve())
        lbl = CLEAR_TEST_LBL / f"{img.stem}.txt"
        if lbl.exists():
            dl = dst_lbl / f"{img.stem}.txt"
            if not dl.exists():
                dl.symlink_to(lbl.resolve())
        n += 1
    print(f"[direct] vtt_clear: {n} images")


def link_labels(filter_name: str, condition: str) -> None:
    """Hard-link the original labels next to the filtered images."""
    fog = FOG[condition]
    base = DATA / "filtered" / filter_name / condition
    for split, lbl_dir in (("train", fog["train_lbl"]), ("test", fog["test_lbl"])):
        img_dir = base / split
        if not img_dir.exists():
            raise FileNotFoundError(img_dir)
        n = 0
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue
            src = lbl_dir / f"{img.stem}.txt"
            dst = img_dir / f"{img.stem}.txt"
            if dst.exists() or not src.exists():
                continue
            try:
                os.link(src, dst)  # hard link: no new inode, no data copy
            except OSError:
                os.symlink(src.resolve(), dst)
            n += 1
        print(f"[direct] {filter_name}/{condition}/{split}: linked {n} labels")


def _write(path: Path, train: list[str], val: str, test: str) -> None:
    content = {
        "path": str(REPO_ROOT),
        "train": train,
        "val": val,
        "test": test,
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content, sort_keys=False))
    print(f"[direct] {path}")


def build_b1(condition: str, name: str) -> None:
    fog = FOG[condition]
    train = [str(CLEAR_TRAIN_IMG), str(fog["train_img"])]
    _write(CONFIGS / f"{name}_train.yaml", train, str(CLEAR_VAL_IMG), str(fog["test_img"]))
    _write(CONFIGS / f"{name}_eval_clear.yaml", train, str(CLEAR_VAL_IMG),
           str(DATA / "vtt_clear" / "images" / "test"))
    _write(CONFIGS / f"{name}_eval_fog.yaml", train, str(CLEAR_VAL_IMG), str(fog["test_img"]))


def build_g1(filter_name: str, condition: str, name: str) -> None:
    link_labels(filter_name, condition)
    base = DATA / "filtered" / filter_name / condition
    train = [str(CLEAR_TRAIN_IMG), str(base / "train")]
    _write(CONFIGS / f"{name}_train.yaml", train, str(CLEAR_VAL_IMG), str(base / "test"))
    _write(CONFIGS / f"{name}_eval_clear.yaml", train, str(CLEAR_VAL_IMG),
           str(DATA / "vtt_clear" / "images" / "test"))
    _write(CONFIGS / f"{name}_eval_fog.yaml", train, str(CLEAR_VAL_IMG), str(base / "test"))


def link_clear_labels(filter_name: str) -> None:
    """Hard-link clear labels next to the filtered clear images (B3 only)."""
    base = DATA / "filtered" / filter_name / "clear"
    for split, lbl_dir in (("train", CLEAR_TRAIN_LBL), ("val", CLEAR_VAL_LBL),
                           ("test", CLEAR_TEST_LBL)):
        img_dir = base / split
        if not img_dir.exists():
            continue
        n = 0
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue
            src = lbl_dir / f"{img.stem}.txt"
            dst = img_dir / f"{img.stem}.txt"
            if dst.exists() or not src.exists():
                continue
            try:
                os.link(src, dst)
            except OSError:
                os.symlink(src.resolve(), dst)
            n += 1
        print(f"[direct] clear-labels {filter_name}/{split}: linked {n}")


def build_b0(condition: str, name: str) -> None:
    train = [str(CLEAR_TRAIN_IMG)]
    _write(CONFIGS / f"{name}_train.yaml", train, str(CLEAR_VAL_IMG), str(FOG[condition]["test_img"]))
    _write(CONFIGS / f"{name}_eval_clear.yaml", train, str(CLEAR_VAL_IMG),
           str(DATA / "vtt_clear" / "images" / "test"))
    _write(CONFIGS / f"{name}_eval_fog.yaml", train, str(CLEAR_VAL_IMG), str(FOG[condition]["test_img"]))


def build_b2(condition: str, name: str) -> None:
    train = [str(FOG[condition]["train_img"])]
    _write(CONFIGS / f"{name}_train.yaml", train, str(CLEAR_VAL_IMG), str(FOG[condition]["test_img"]))
    _write(CONFIGS / f"{name}_eval_fog.yaml", train, str(CLEAR_VAL_IMG), str(FOG[condition]["test_img"]))


def build_b3(filter_name: str, condition: str, name: str) -> None:
    link_clear_labels(filter_name)
    link_labels(filter_name, condition)
    ct = DATA / "filtered" / filter_name / "clear" / "train"
    cte = DATA / "filtered" / filter_name / "clear" / "test"
    ft = DATA / "filtered" / filter_name / condition / "train"
    fte = DATA / "filtered" / filter_name / condition / "test"
    train = [str(ct), str(ft)]
    _write(CONFIGS / f"{name}_train.yaml", train, str(CLEAR_VAL_IMG), str(fte))
    _write(CONFIGS / f"{name}_eval_clear.yaml", train, str(CLEAR_VAL_IMG), str(cte))
    _write(CONFIGS / f"{name}_eval_fog.yaml", train, str(CLEAR_VAL_IMG), str(fte))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build direct-path YOLO YAMLs")
    ap.add_argument("kind", choices=["init", "b0", "b1", "b2", "b3", "g1"])
    ap.add_argument("--condition", default="cyclegan", choices=list(FOG))
    ap.add_argument("--filter", dest="filter_name", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    default = f"{args.kind}_{args.condition}_direct"
    if args.kind == "init":
        init_clear_test()
    elif args.kind == "b0":
        build_b0(args.condition, args.name or default)
    elif args.kind == "b1":
        build_b1(args.condition, args.name or default)
    elif args.kind == "b2":
        build_b2(args.condition, args.name or default)
    elif args.kind == "b3":
        if not args.filter_name:
            raise SystemExit("--filter required for b3")
        build_b3(args.filter_name, args.condition,
                 args.name or f"b3_{args.filter_name}_{args.condition}_direct")
    else:
        if not args.filter_name:
            raise SystemExit("--filter required for g1")
        build_g1(args.filter_name, args.condition,
                 args.name or f"g1_{args.filter_name}_{args.condition}_direct")


if __name__ == "__main__":
    main()
