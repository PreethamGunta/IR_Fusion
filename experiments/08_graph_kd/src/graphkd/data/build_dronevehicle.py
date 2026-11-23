"""Build DroneVehicle IR/RGB YOLO datasets from the McCheng HuggingFace zips.

The release stores 840x712 images with a 100 px white border and four-point
oriented polygons.  We crop the border to 640x512 and convert polygons to
axis-aligned boxes (min/max of the four points), which matches the detector and
the FLIR protocol used in experiment 08.

Run after FLIR (CPU-heavy):
  python -m graphkd.data.build_dronevehicle
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from graphkd import paths

CLASS_NAMES = ["car", "truck", "bus", "van", "freight_car"]
# DroneVehicle ships separate visible (``*label``) and thermal (``*labelr``)
# annotations.  The IR student must be supervised with the thermal labels; the
# RGB branch uses the visible labels.  Using ``testlabel`` for IR test was the
# source of the original invalid test evaluation.
LABEL_DIRS = {
    "ir": {"train": "trainlabelr", "val": "vallabelr", "test": "testlabelr"},
    "rgb": {"train": "trainlabel", "val": "vallabel", "test": "testlabel"},
}
BORDER = 100
OUT = paths.REPO_ROOT / "data" / "dronevehicle"
RAW = OUT / "raw"
EXTRACT = RAW / "extracted"


_ALIASES = {"feright car": "freight_car", "freight car": "freight_car",
            "freightcar": "freight_car"}


def _name(tag: str) -> str:
    t = tag.strip().lower().replace(" ", "_")
    return _ALIASES.get(t.replace("_", " "), t)


def _parse(xml_bytes: bytes) -> list[tuple[int, float, float, float, float]]:
    root = ET.fromstring(xml_bytes)
    out = []
    for obj in root.findall("object"):
        name = _name(obj.findtext("name", ""))
        if name not in CLASS_NAMES:
            continue
        pts = []
        poly = obj.find("polygon")
        if poly is not None:
            for i in range(1, 5):
                x = poly.findtext(f"x{i}")
                y = poly.findtext(f"y{i}")
                if x is not None and y is not None:
                    pts.append((float(x) - BORDER, float(y) - BORDER))
        box = obj.find("bndbox")
        if not pts and box is not None:
            pts = [(float(box.findtext("xmin")) - BORDER, float(box.findtext("ymin")) - BORDER),
                   (float(box.findtext("xmax")) - BORDER, float(box.findtext("ymax")) - BORDER)]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = max(0.0, min(xs)), min(640.0, max(xs))
        y1, y2 = max(0.0, min(ys)), min(512.0, max(ys))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append((CLASS_NAMES.index(name), x1, y1, x2, y2))
    return out


def _extract() -> None:
    EXTRACT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        zpath = RAW / f"{split}.zip"
        if not zpath.exists():
            raise FileNotFoundError(zpath)
        dst = EXTRACT / split
        marker = dst / ".extract_done"
        if marker.exists():
            continue
        skipped = []
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                try:
                    z.extract(info, dst)
                except (zipfile.BadZipFile, OSError) as e:
                    skipped.append({"file": info.filename, "error": str(e)})
                    print(f"[{split}] skipped corrupt member {info.filename}: {e}", flush=True)
        marker.write_text(json.dumps(skipped, indent=1))
        if skipped:
            print(f"[{split}] {len(skipped)} corrupt members skipped", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    _extract()

    for mod in ("ir", "rgb"):
        for split in ("train", "val", "test"):
            for sub in ("images", "labels"):
                (OUT / "yolo" / mod / sub / split).mkdir(parents=True, exist_ok=True)

    manifest = {"classes": CLASS_NAMES, "border": BORDER, "splits": {}}
    for split, ir_dir, rgb_dir in (
        ("train", "trainimgr", "trainimg"),
        ("val", "valimgr", "valimg"),
        ("test", "testimgr", "testimg"),
    ):
        root = EXTRACT / split / split
        lab_ir_dir = root / LABEL_DIRS["ir"][split]
        lab_rgb_dir = root / LABEL_DIRS["rgb"][split]
        imgs = sorted((root / ir_dir).glob("*.jpg"))
        if args.limit:
            imgs = imgs[: args.limit]
        counts = Counter()
        n_boxes = 0
        bad = 0
        for i, ip in enumerate(imgs):
            stem = ip.stem
            lp_ir = lab_ir_dir / f"{stem}.xml"
            lp_rgb = lab_rgb_dir / f"{stem}.xml"
            rp = root / rgb_dir / f"{stem}.jpg"
            if not rp.exists() or not lp_ir.exists() or not lp_rgb.exists():
                bad += 1
                continue
            objs = _parse(lp_ir.read_bytes())
            objs_rgb = _parse(lp_rgb.read_bytes())
            if not objs:
                pass
            ir = cv2.imread(str(ip))
            rgb = cv2.imread(str(rp))
            if ir is None or rgb is None:
                bad += 1
                continue
            ir = ir[BORDER:-BORDER, BORDER:-BORDER]
            rgb = rgb[BORDER:-BORDER, BORDER:-BORDER]
            if ir.shape[:2] != (512, 640):
                ir = cv2.resize(ir, (640, 512))
                rgb = cv2.resize(rgb, (640, 512))
            cv2.imwrite(str(OUT / "yolo" / "ir" / "images" / split / f"{stem}.jpg"), ir,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(OUT / "yolo" / "rgb" / "images" / split / f"{stem}.jpg"), rgb,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            def _lines(items):
                return [f"{c} {(x1+x2)/2/640:.6f} {(y1+y2)/2/512:.6f} {(x2-x1)/640:.6f} {(y2-y1)/512:.6f}"
                        for c, x1, y1, x2, y2 in items]

            lines = _lines(objs)
            txt = "\n".join(lines) + ("\n" if lines else "")
            lines_rgb = _lines(objs_rgb)
            txt_rgb = "\n".join(lines_rgb) + ("\n" if lines_rgb else "")
            (OUT / "yolo" / "ir" / "labels" / split / f"{stem}.txt").write_text(txt)
            (OUT / "yolo" / "rgb" / "labels" / split / f"{stem}.txt").write_text(txt_rgb)
            for c, *_ in objs:
                counts[CLASS_NAMES[c]] += 1
            n_boxes += len(objs)
            if (i + 1) % 1000 == 0:
                print(f"[{split}] {i+1}/{len(imgs)}", flush=True)
        manifest["splits"][split] = {"images": len(imgs) - bad, "boxes": n_boxes,
                                     "classes": dict(counts), "bad_pairs": bad}
        print(split, manifest["splits"][split], flush=True)

    for mod in ("ir", "rgb"):
        (paths.CONFIGS / f"dronevehicle_{mod}.yaml").write_text(
            f"path: {OUT / 'yolo' / mod}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
            f"nc: {len(CLASS_NAMES)}\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASS_NAMES))
        )
    (paths.CONFIGS / "dronevehicle_paired.yaml").write_text(
        f"path: {OUT / 'yolo' / 'ir'}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASS_NAMES))
        + "\nchannels: 6\n"
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
