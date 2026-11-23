"""Reconcile the UNIRI-TID release into one canonical manifest.

The IEEE DataPort archive ships two parallel annotation exports. They are NOT
equivalent and one of them is broken:

    All_In_One_Anot_*      6276 files, 9244 <object>, 9244 <bndbox>   <- clean
    ALL_IN_ONE_RGB_ANOT_*  4370 files, 4370 <object>, 7335 <bndbox>   <- broken

In the RGB export, 1908 files pack several <bndbox> elements inside a single
<object>, which is invalid VOC: every box after the first loses its class
association. Naively parsing it silently drops ~2965 boxes. We therefore use
All_In_One_Anot_* exclusively and ignore the RGB export.

Writes manifest.csv: frame_id, image, label, n_boxes, width, height
"""

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "UNIRI-TID"
RAW = ROOT / "raw"
YOLO = ROOT / "annotations" / "All_In_One_Anot_Yolo"
BROKEN = ROOT / "annotations" / "ALL_IN_ONE_RGB_ANOT_YOLO"
OUT = ROOT / "manifest.csv"

IMG_W, IMG_H = 1280, 960


def stem_for(fid):
    """Filenames are zero-padded to 3 digits below 1000 and 4 digits above."""
    for cand in (f"{fid:04d}", f"{fid:03d}"):
        if (RAW / f"{cand}.jpg").exists():
            return cand
    return None


def main():
    img_ids = sorted(int(p.stem) for p in RAW.glob("*.jpg"))
    ann_ids = sorted(int(p.stem) for p in YOLO.glob("*.txt"))
    ann_set = set(ann_ids)

    print(f"images                    : {len(img_ids)}  (id {img_ids[0]}-{img_ids[-1]})")
    print(f"clean annotations         : {len(ann_ids)}")
    print(f"images without annotation : {len(set(img_ids) - ann_set)}  -> excluded")
    print(f"annotations without image : {len(ann_set - set(img_ids))}")
    if BROKEN.exists():
        print(f"broken RGB export         : {len(list(BROKEN.glob('*.txt')))} files -> IGNORED")

    rows, total_boxes, empty = [], 0, 0
    for fid in img_ids:
        if fid not in ann_set:
            continue
        stem = stem_for(fid)
        if stem is None:
            continue
        label = YOLO / f"{stem}.txt"
        if not label.exists():                       # zero-pad width can differ
            label = next(YOLO.glob(f"*{fid}.txt"), None)
            if label is None:
                continue
        boxes = [l for l in label.read_text().splitlines() if l.strip()]
        total_boxes += len(boxes)
        empty += (len(boxes) == 0)
        rows.append(
            dict(frame_id=fid, image=f"raw/{stem}.jpg",
                 label=f"annotations/All_In_One_Anot_Yolo/{label.name}",
                 n_boxes=len(boxes), width=IMG_W, height=IMG_H)
        )

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nmanifest rows             : {len(rows)}")
    print(f"total boxes               : {total_boxes}")
    print(f"images with zero boxes    : {empty}")
    print(f"mean boxes/image          : {total_boxes / len(rows):.2f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
