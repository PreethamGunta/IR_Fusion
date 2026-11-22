"""Apply the learned gate at inference time and materialise the routed test sets.

Two routings are produced:
  * gated_learned : the trained gate decides clear->identity / fog->filter
  * gated_oracle  : ground truth decides (upper bound on gate quality)

The detector is then evaluated on these pre-routed images, which is equivalent
to running the gate online but keeps the detector harness unchanged. Latency is
measured separately and reported alongside the accuracy table.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from .gate import predict_paths
from .paths import (
    CLASS_NAMES,
    CLEAR_TEST_IMG,
    CLEAR_TEST_LBL,
    CLEAR_VAL_IMG,
    CLEAR_VAL_LBL,
    CONFIGS,
    DATA,
    FOG,
    RESULTS,
    CLEAR_TRAIN_IMG,
    CLEAR_TRAIN_LBL,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def _reset(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)  # generated symlink tree only
    root.mkdir(parents=True)


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.symlink_to(src.resolve())


def _write_yaml(path: Path, root: Path, test_entry: str) -> None:
    content = {
        "path": str(root.resolve()),
        "train": ["images/train_clear", "images/train_fog"],
        "val": "images/val",
        "test": test_entry,
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    path.write_text(yaml.safe_dump(content, sort_keys=False))


def materialise(name: str, condition: str, filter_name: str, use_gate: bool) -> dict:
    clear_imgs = _images(CLEAR_TEST_IMG)
    fog = FOG[condition]
    fog_imgs = _images(fog["test_img"])

    filt = DATA / "filtered" / filter_name
    clear_filtered = filt / "clear" / "test"
    fog_filtered = filt / condition / "test"

    if use_gate:
        clear_pred = predict_paths(clear_imgs, condition)
        fog_pred = predict_paths(fog_imgs, condition)
    else:
        clear_pred = [0] * len(clear_imgs)
        fog_pred = [1] * len(fog_imgs)

    root = DATA / name
    _reset(root)

    # Clear test frames: gate clear -> original, gate fog -> filtered.
    for img, pred in zip(clear_imgs, clear_pred):
        src = (clear_filtered / f"{img.stem}.jpg") if int(pred) == 1 else img
        _link(src, root / "images" / "test_clear" / img.name)
        lbl = CLEAR_TEST_LBL / f"{img.stem}.txt"
        if lbl.exists():
            _link(lbl, root / "labels" / "test_clear" / f"{img.stem}.txt")

    # Fog test frames: gate fog -> filtered, gate clear -> original.
    for img, pred in zip(fog_imgs, fog_pred):
        src = (fog_filtered / f"{img.stem}.jpg") if int(pred) == 1 else img
        _link(src, root / "images" / "test_fog" / img.name)
        lbl = fog["test_lbl"] / f"{img.stem}.txt"
        if lbl.exists():
            _link(lbl, root / "labels" / "test_fog" / f"{img.stem}.txt")

    # Training entries + clear val (labels only; not used by eval).
    for split, src_img, src_lbl in [
        ("train_clear", CLEAR_TRAIN_IMG, CLEAR_TRAIN_LBL),
        ("train_fog", DATA / "filtered" / filter_name / condition / "train", fog["train_lbl"]),
        ("val", CLEAR_VAL_IMG, CLEAR_VAL_LBL),
    ]:
        for img in _images(src_img):
            _link(img, root / "images" / split / img.name)
            lbl = src_lbl / f"{img.stem}.txt"
            if lbl.exists():
                _link(lbl, root / "labels" / split / f"{img.stem}.txt")

    _write_yaml(CONFIGS / f"{name}_eval_clear.yaml", root, "images/test_clear")
    _write_yaml(CONFIGS / f"{name}_eval_fog.yaml", root, "images/test_fog")

    stats = {
        "name": name,
        "condition": condition,
        "filter": filter_name,
        "gate": "learned" if use_gate else "oracle",
        "n_clear": len(clear_imgs),
        "n_fog": len(fog_imgs),
        "clear_routed_to_filter": int(sum(clear_pred)),
        "fog_routed_to_clear": int(len(fog_pred) - sum(fog_pred)),
        "clear_false_positive_rate": float(sum(clear_pred) / max(1, len(clear_pred))),
        "fog_false_negative_rate": float((len(fog_pred) - sum(fog_pred)) / max(1, len(fog_pred))),
    }
    (RESULTS / f"routing_{name}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialise gated test sets")
    ap.add_argument("--condition", default="cyclegan", choices=list(FOG))
    ap.add_argument("--filter", dest="filter_name", default="gamma_clahe")
    ap.add_argument("--name", default="gated_learned")
    ap.add_argument("--oracle", action="store_true", help="use ground-truth routing")
    args = ap.parse_args()
    materialise(args.name, args.condition, args.filter_name, use_gate=not args.oracle)


if __name__ == "__main__":
    main()
