#!/usr/bin/env python3
"""Build DroneVehicle v2 for experiment 09.

Policy
------
- IR supervision: ``*labelr`` (thermal annotations) for train, val and test.
- RGB supervision: ``*label`` (visible annotations) for train, val and test.
- Images: 100 px white border removed -> 640x512, as in experiment 08.
- The official split has 17,990 train labels but only 17,962 extractable
  RGB/IR pairs; the 28 missing images are listed, not invented.

Paths are derived from this file; no absolute paths are hardcoded.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "dronevehicle" / "raw" / "extracted"
SRC08 = REPO / "data" / "dronevehicle" / "yolo"  # read-only image source
OUT = EXP / "data" / "dronevehicle_v2"

CLASSES = ["car", "truck", "bus", "van", "freight_car"]
ALIASES = {"feright car": "freight_car", "freight car": "freight_car", "freightcar": "freight_car"}
BORDER = 100

LABELS = {
    "ir": {"train": "trainlabelr", "val": "vallabelr", "test": "testlabelr"},
    "rgb": {"train": "trainlabel", "val": "vallabel", "test": "testlabel"},
}
IMAGES = {"ir": {"train": "trainimgr", "val": "valimgr", "test": "testimgr"},
          "rgb": {"train": "trainimg", "val": "valimg", "test": "testimg"}}


def _name(tag: str) -> str:
    t = tag.strip().lower().replace(" ", "_")
    return ALIASES.get(t.replace("_", " "), t)


def parse_xml(xml_bytes: bytes) -> tuple[list[tuple[int, float, float, float, float]], Counter]:
    root = ET.fromstring(xml_bytes)
    out = []
    stats = Counter()
    for obj in root.findall("object"):
        n = _name(obj.findtext("name", ""))
        if n not in CLASSES:
            stats["unknown_class"] += 1
            continue
        pts = []
        poly = obj.find("polygon")
        if poly is not None:
            for i in range(1, 5):
                x = poly.findtext(f"x{i}")
                y = poly.findtext(f"y{i}")
                if x is not None and y is not None:
                    pts.append((float(x), float(y)))
        box = obj.find("bndbox")
        if not pts and box is not None:
            pts = [(float(box.findtext("xmin")), float(box.findtext("ymin"))),
                   (float(box.findtext("xmax")), float(box.findtext("ymax")))]
        if not pts:
            stats["no_points"] += 1
            continue
        stats[f"points_{len(pts)}"] += 1
        xs = [p[0] - BORDER for p in pts]
        ys = [p[1] - BORDER for p in pts]
        if min(xs) < 0 or min(ys) < 0 or max(xs) > 640 or max(ys) > 512:
            stats["outside_crop"] += 1
        x1, x2 = max(0.0, min(xs)), min(640.0, max(xs))
        y1, y2 = max(0.0, min(ys)), min(512.0, max(ys))
        if x2 - x1 < 2 or y2 - y1 < 2:
            stats["degenerate"] += 1
            continue
        out.append((CLASSES.index(n), x1, y1, x2, y2))
    return out, stats


def norm(objs):
    return [(c, round((x1 + x2) / 2 / 640, 6), round((y1 + y2) / 2 / 512, 6),
             round((x2 - x1) / 640, 6), round((y2 - y1) / 512, 6)) for c, x1, y1, x2, y2 in objs]


def write_labels(path: Path, objs) -> None:
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in norm(objs)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing {OUT}; move/rename it first")
    audit = {"classes": CLASSES, "border": BORDER, "splits": {}, "missing": {}}

    for split in ("train", "val", "test"):
        counts = {m: Counter() for m in ("ir", "rgb")}
        boxes = {m: 0 for m in ("ir", "rgb")}
        frames = 0
        stats_all = Counter()
        for modality in ("ir", "rgb"):
            for sub in ("images", "labels"):
                (OUT / "yolo" / modality / sub / split).mkdir(parents=True, exist_ok=True)
        # The 08 crop is read-only; both modalities are complete pairs there.
        src_img = SRC08 / "ir" / "images" / split
        stems = sorted(p.stem for p in src_img.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if args.limit:
            stems = stems[: args.limit]
        base = RAW / split / split
        for i, stem in enumerate(stems):
            ir_xml = base / LABELS["ir"][split] / f"{stem}.xml"
            rgb_xml = base / LABELS["rgb"][split] / f"{stem}.xml"
            if not ir_xml.exists() or not rgb_xml.exists():
                stats_all["missing_xml"] += 1
                continue
            objs_ir, st_ir = parse_xml(ir_xml.read_bytes())
            objs_rgb, st_rgb = parse_xml(rgb_xml.read_bytes())
            stats_all.update({f"ir_{k}": v for k, v in st_ir.items()})
            stats_all.update({f"rgb_{k}": v for k, v in st_rgb.items()})
            for modality, objs in (("ir", objs_ir), ("rgb", objs_rgb)):
                src = SRC08 / modality / "images" / split / f"{stem}.jpg"
                dst = OUT / "yolo" / modality / "images" / split / f"{stem}.jpg"
                if not src.exists():
                    stats_all[f"{modality}_missing_image"] += 1
                    continue
                link_or_copy(src, dst)
                write_labels(OUT / "yolo" / modality / "labels" / split / f"{stem}.txt", objs)
                for c, *_ in objs:
                    counts[modality][CLASSES[c]] += 1
                boxes[modality] += len(objs)
            frames += 1
            if (i + 1) % 4000 == 0:
                print(f"[{split}] {i+1}/{len(stems)}", flush=True)
        # Assert round-trip: written labels equal the source parse.
        mism = {"ir": 0, "rgb": 0}
        for modality, label_dir in (("ir", LABELS["ir"][split]), ("rgb", LABELS["rgb"][split])):
            for stem in stems:
                xp = base / label_dir / f"{stem}.xml"
                yp = OUT / "yolo" / modality / "labels" / split / f"{stem}.txt"
                if not xp.exists() or not yp.exists():
                    continue
                want = norm(parse_xml(xp.read_bytes())[0])
                got = []
                for line in yp.read_text().splitlines():
                    p = line.split()
                    if len(p) == 5:
                        got.append((int(float(p[0])), round(float(p[1]), 6), round(float(p[2]), 6),
                                    round(float(p[3]), 6), round(float(p[4]), 6)))
                if got != want:
                    mism[modality] += 1
        # Pair integrity: same stems in both modalities.
        ir_stems = {p.stem for p in (OUT / "yolo" / "ir" / "images" / split).glob("*")}
        rgb_stems = {p.stem for p in (OUT / "yolo" / "rgb" / "images" / split).glob("*")}
        audit["splits"][split] = {
            "frames": frames,
            "pairs": len(ir_stems & rgb_stems),
            "ir_only": len(ir_stems - rgb_stems),
            "rgb_only": len(rgb_stems - ir_stems),
            "ir_boxes": boxes["ir"],
            "rgb_boxes": boxes["rgb"],
            "ir_classes": dict(counts["ir"]),
            "rgb_classes": dict(counts["rgb"]),
            "roundtrip_mismatch": mism,
            "parse_stats": dict(stats_all),
        }
        print(split, json.dumps(audit["splits"][split], sort_keys=True), flush=True)

    # Missing 28: labels exist but one image is absent in the raw archive.
    for split in ("train", "val", "test"):
        base = RAW / split / split
        def stems(sub, ext):
            return {p.stem for p in (base / sub).glob(f"*.{ext}")}
        ir = stems(IMAGES["ir"][split], "jpg")
        rgb = stems(IMAGES["rgb"][split], "jpg")
        li = stems(LABELS["ir"][split], "xml")
        audit["missing"][split] = {
            "ir_images_no_rgb": sorted(ir - rgb),
            "rgb_images_no_ir": sorted(rgb - ir),
            "labels_no_ir_image": sorted(li - ir),
            "labels_no_rgb_image": sorted(li - rgb),
        }

    (OUT / "manifest.json").write_text(json.dumps(audit, indent=1))
    (EXP / "audit" / "dv_v2_audit.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
