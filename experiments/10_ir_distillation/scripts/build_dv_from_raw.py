#!/usr/bin/env python3
"""Build the DroneVehicle v2 YOLO tree from the raw release.

Why this exists: ``09_graph_fusion/src/data/build_dv_v2.py`` takes its images
from the experiment-08 YOLO tree (~24 GB), which is not part of the release.
This script reproduces the **same** tree from ``data/dronevehicle/raw`` alone:

- split membership is fixed by the committed manifest
  (``assets/dv_v2_splits.json``, extracted from the audited 09 tree), so train /
  val / test contain exactly the same frames as the development node;
- labels are parsed by the *same* 09 functions (``*labelr`` for IR, ``*label``
  for RGB), so the YOLO label files are byte-identical;
- images are the 100 px border crop saved as PIL JPEG **quality 92**, which was
  verified bit-exact against the 09 tree on 24/24 sampled frames across all
  splits and both modalities.

    python scripts/build_dv_from_raw.py
    python scripts/build_dv_from_raw.py --raw data/dronevehicle/raw \
        --out experiments/09_graph_fusion/data/dronevehicle_v2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

EXP10 = Path(__file__).resolve().parents[1]
REPO = EXP10.parents[1]
EXP09 = REPO / "experiments" / "09_graph_fusion"
for _p in (str(EXP09 / "src"), str(EXP09)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.build_dv_v2 import BORDER, IMAGES, LABELS, parse_xml, write_labels  # noqa: E402

SPLITS = ("train", "val", "test")


def find_raw(root: Path) -> Path:
    """Accept raw/extracted/... or raw/... layouts."""
    if (root / "extracted" / "train" / "train").is_dir():
        return root / "extracted"
    if (root / "train" / "train").is_dir():
        return root
    raise SystemExit(
        f"cannot find <split>/<split> under {root} "
        "(expected data/dronevehicle/raw[/extracted]/train/train)"
    )


def save_jpeg(img: np.ndarray, dst: Path, quality: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(dst, format="JPEG", quality=quality)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(REPO / "data/dronevehicle/raw"))
    ap.add_argument("--out", default=str(EXP09 / "data/dronevehicle_v2"))
    ap.add_argument("--manifest", default=str(EXP10 / "assets/dv_v2_splits.json"))
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N stems/split")
    args = ap.parse_args()

    raw = find_raw(Path(args.raw))
    out = Path(args.out)
    manifest = json.loads(Path(args.manifest).read_text())
    border = int(manifest.get("border", BORDER))
    quality = int(manifest.get("jpeg_quality", 92))

    if (out / "yolo").exists():
        raise SystemExit(f"refusing to overwrite existing {out}/yolo; move/rename it first")

    audit = {"raw": str(raw), "out": str(out), "border": border,
             "jpeg_quality": quality, "splits": {}, "missing": {},
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0 = time.time()

    for split in SPLITS:
        stems = list(manifest["splits"][split])
        if args.limit:
            stems = stems[: args.limit]
        base = raw / split / split
        counts = {}
        missing = []
        for modality in ("ir", "rgb"):
            img_dir = base / IMAGES[modality][split]
            lbl_dir = base / LABELS[modality][split]
            dst_img = out / "yolo" / modality / "images" / split
            dst_lbl = out / "yolo" / modality / "labels" / split
            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)
            n = 0
            for i, stem in enumerate(stems):
                img_path = img_dir / f"{stem}.jpg"
                xml_path = lbl_dir / f"{stem}.xml"
                if not img_path.exists() or not xml_path.exists():
                    missing.append(f"{split}/{modality}/{stem}")
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    missing.append(f"{split}/{modality}/{stem} (unreadable)")
                    continue
                if img.shape[:2] != (512 + 2 * border, 640 + 2 * border):
                    raise SystemExit(
                        f"unexpected raw size {img.shape[:2]} for {img_path}; "
                        "expected 712x840 for border=100"
                    )
                crop = img[border:border + 512, border:border + 640]
                save_jpeg(crop, dst_img / f"{stem}.jpg", quality)
                objs, _ = parse_xml(xml_path.read_bytes())
                write_labels(dst_lbl / f"{stem}.txt", objs)
                n += 1
                if (i + 1) % 2000 == 0:
                    print(f"[{split}/{modality}] {i+1}/{len(stems)}", flush=True)
            counts[modality] = n
        audit["splits"][split] = {"stems": len(stems), "written": counts}
        audit["missing"][split] = missing
        print(f"[build] {split}: {counts} (manifest {len(stems)}, "
              f"missing {len(missing)})", flush=True)

    (out / "manifest.json").write_text(json.dumps(audit, indent=1))
    print(f"[build] done in {(time.time()-t0)/60:.1f} min -> {out}")
    print(f"[build] DV_ROOT={out / 'yolo'}")


if __name__ == "__main__":
    main()
