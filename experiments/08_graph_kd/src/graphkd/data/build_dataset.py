"""Build the paired RGB/IR YOLO dataset from the aligned FLIR release.

Input : data/flir_aligned/raw/align  (JPEGImages, Annotations, split lists)
Output: data/flir_aligned/yolo/{ir,rgb}/{images,labels}/{train,val,test}
        data/flir_aligned/rgb_fixed/  (cropped RGB for letterboxed originals)
        configs/{ir,rgb,paired}.yaml
        data/flir_aligned/pair_manifest.json
        results/dataset_audit.json

The aligned release occasionally stores the RGB frame letterboxed inside a
640x512 canvas (13 % of pairs).  Those frames are cropped to their valid
content and resized to 640x512, which recovers the registration.  The split is
the paper's sequence-disjoint train/validation split; validation is further
split by frame id into val/test so the test set stays untouched by model
selection.

Run:  python -m graphkd.data.build_dataset
"""
from __future__ import annotations

import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from graphkd import paths

SPLIT_CUT = 9500  # frames >= cut become test (paper val is frames 8864..10227)


def _frame_id(name: str) -> int:
    return int(re.search(r"(\d+)", name).group(1))


def _parse_xml(path: Path) -> list[tuple[int, float, float, float, float]]:
    root = ET.parse(path).getroot()
    out = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in paths.CLASS_NAMES:
            continue
        if obj.findtext("difficult", "0").strip() == "1":
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((paths.CLASS_NAMES.index(name), x1, y1, x2, y2))
    return out


def _content_bbox(path: Path) -> tuple[int, int, int, int] | None:
    a = np.asarray(Image.open(path).convert("L"))
    ys, xs = np.where(a > 8)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _fixed_rgb(src: Path, dst: Path, bbox) -> bool:
    im = Image.open(src).convert("RGB")
    if bbox is None:
        return False
    cropped = bbox != (0, 0, im.width, im.height)
    if cropped:
        im = im.crop(bbox)
    if im.size != (640, 512):
        im = im.resize((640, 512), Image.BILINEAR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=95)
    return cropped


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(os.path.relpath(src, dst.parent), dst)


def _write_label(objs, dst: Path) -> None:
    lines = []
    for c, x1, y1, x2, y2 in objs:
        cx, cy = (x1 + x2) / 2 / 640, (y1 + y2) / 2 / 512
        w, h = (x2 - x1) / 640, (y2 - y1) / 512
        lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    dst.write_text("\n".join(lines) + ("\n" if lines else ""))


def _audit_alignment(names: list[str], sample: int = 300) -> dict:
    try:
        import cv2
    except Exception:
        return {"available": False}
    rng = np.random.default_rng(0)
    pick = rng.choice(len(names), size=min(sample, len(names)), replace=False)
    shifts, scores = [], []
    for idx in pick:
        n = names[int(idx)]
        stem = n.replace("_PreviewData", "")
        ir = cv2.imread(str(paths.RAW / "JPEGImages" / f"{n}.jpeg"), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(str(paths.FIXED_RGB / f"{stem}_RGB.jpg"), cv2.IMREAD_GRAYSCALE)
        if ir is None or rgb is None:
            continue
        ir_e = cv2.Canny(ir, 50, 150).astype(np.float32) / 255.0
        rg_e = cv2.Canny(rgb, 50, 150).astype(np.float32) / 255.0
        best = (-1, 0, 0)
        for dy in range(-6, 7, 2):
            for dx in range(-6, 7, 2):
                m = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(rg_e, m, (640, 512))
                s = float((ir_e * shifted).sum())
                if s > best[0]:
                    best = (s, dx, dy)
        shifts.append((best[1], best[2]))
        scores.append(best[0])
    shifts = np.array(shifts)
    return {
        "available": True,
        "n_sampled": int(len(shifts)),
        "median_abs_dx": float(np.median(np.abs(shifts[:, 0]))),
        "median_abs_dy": float(np.median(np.abs(shifts[:, 1]))),
        "frac_shift_gt2px": float(np.mean((np.abs(shifts) > 2).any(axis=1))),
        "mean_edge_overlap": float(np.mean(scores)),
    }


def main() -> None:
    paths.ensure_dirs()
    train = (paths.RAW / "align_train.txt").read_text().split()
    val = (paths.RAW / "align_validation.txt").read_text().split()
    if not train or not val:
        raise SystemExit("aligned split lists are empty")

    yolo = paths.YOLO
    for mod in ("ir", "rgb"):
        shutil.rmtree(yolo / mod, ignore_errors=True)
    for mod in ("ir", "rgb"):
        for split in ("train", "val", "test"):
            for sub in ("images", "labels"):
                (yolo / mod / sub / split).mkdir(parents=True, exist_ok=True)
    paths.FIXED_RGB.mkdir(parents=True, exist_ok=True)

    split_of = {}
    for n in train:
        split_of[n] = "train"
    for n in val:
        split_of[n] = "test" if _frame_id(n) >= SPLIT_CUT else "val"

    manifest = {"schema": 1, "source": "Zhang et al. aligned FLIR",
                "class_names": paths.CLASS_NAMES, "split_cut": SPLIT_CUT,
                "pairs": [], "counts": {}, "classes": {}}
    counts = Counter()
    classes = {s: Counter() for s in ("train", "val", "test")}
    cropped = 0

    for n, split in sorted(split_of.items(), key=lambda kv: _frame_id(kv[0])):
        stem = n.replace("_PreviewData", "")
        ir_src = paths.RAW / "JPEGImages" / f"{n}.jpeg"
        rgb_src = paths.RAW / "JPEGImages" / f"{stem}_RGB.jpg"
        if not ir_src.exists() or not rgb_src.exists():
            counts["missing"] += 1
            continue
        objs = _parse_xml(paths.RAW / "Annotations" / f"{n}.xml")
        if not objs:
            counts["empty"] += 1
        dst_rgb = paths.FIXED_RGB / f"{stem}_RGB.jpg"
        did_crop = _fixed_rgb(rgb_src, dst_rgb, _content_bbox(rgb_src))
        cropped += int(did_crop)

        _link(ir_src, yolo / "ir" / "images" / split / ir_src.name)
        _link(dst_rgb, yolo / "rgb" / "images" / split / dst_rgb.name)
        lab_ir = yolo / "ir" / "labels" / split / f"{ir_src.stem}.txt"
        lab_rgb = yolo / "rgb" / "labels" / split / f"{dst_rgb.stem}.txt"
        _write_label(objs, lab_ir)
        _link(lab_ir, lab_rgb)

        counts[split] += 1
        for c, *_ in objs:
            classes[split][paths.CLASS_NAMES[c]] += 1
        manifest["pairs"].append({
            "pair_id": stem, "split": split, "frame": _frame_id(n),
            "ir": ir_src.name, "rgb": dst_rgb.name,
            "rgb_cropped": did_crop, "n_boxes": len(objs),
        })

    for mod in ("ir", "rgb"):
        y = paths.CONFIGS / f"{mod}.yaml"
        y.write_text(
            f"path: {yolo / mod}\n"
            f"train: images/train\nval: images/val\ntest: images/test\n\n"
            f"nc: {len(paths.CLASS_NAMES)}\nnames:\n"
            + "".join(f"  {i}: {c}\n" for i, c in enumerate(paths.CLASS_NAMES))
        )
    (paths.CONFIGS / "paired.yaml").write_text(
        f"path: {yolo / 'ir'}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(paths.CLASS_NAMES)}\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(paths.CLASS_NAMES))
        + "\nchannels: 6\n"
    )

    manifest["counts"] = dict(counts)
    manifest["counts"]["rgb_cropped"] = cropped
    manifest["classes"] = {k: dict(v) for k, v in classes.items()}
    paths.PAIR_MANIFEST.write_text(json.dumps(manifest, indent=1))

    audit = {
        "counts": manifest["counts"],
        "classes": manifest["classes"],
        "alignment": _audit_alignment(train + val),
    }
    (paths.RESULTS / "dataset_audit.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
