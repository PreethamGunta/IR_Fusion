"""Materialise a few-image mirror of the real split for DEBUG runs.

Same directory layout and same yaml keys as the real dataset, so the only thing
that changes between a debug run and a real one is which yaml gets loaded. That
is deliberate: if the debug path used a different structure it would stop
testing the thing we care about.

Frames are drawn per condition per split so every condition stays non-empty and
the per-condition evaluation path is genuinely exercised.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import debugmode

ROOT = Path(__file__).resolve().parents[3] / "data" / "uniri_tid"
SRC = ROOT / "yolo"
DEST = ROOT / "yolo_debug"
CONDITIONS = ("clear", "fog", "rain")


def main():
    n = debugmode.FRAMES_PER_CONDITION
    cond_of = {int(r["frame_id"]): r["condition"]
               for r in csv.DictReader(open(ROOT / "conditions.csv"))}

    rows = list(csv.DictReader(open(ROOT / "splits.csv")))
    picked = defaultdict(list)
    for r in rows:
        fid, split = int(r["frame_id"]), r["split"]
        cond = cond_of[fid]
        if len(picked[(split, cond)]) < n:
            picked[(split, cond)].append(fid)

    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = DEST / sub / split
            if d.exists():
                for p in d.iterdir():
                    p.unlink()
            d.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(int)
    for (split, cond), fids in sorted(picked.items()):
        for fid in fids:
            # resolve the stem through the real split dir so we inherit whatever
            # zero-padding prepare_uniri settled on
            hits = list((SRC / "images" / split).glob(f"*{fid:03d}.jpg")) or \
                   list((SRC / "images" / split).glob(f"*{fid:04d}.jpg"))
            src_img = next((h for h in hits if int(h.stem) == fid), None)
            if src_img is None:
                continue
            stem = src_img.stem
            src_lbl = SRC / "labels" / split / f"{stem}.txt"
            (DEST / "images" / split / f"{stem}.jpg").symlink_to(src_img.resolve())
            (DEST / "labels" / split / f"{stem}.txt").symlink_to(src_lbl.resolve())
            counts[(split, cond)] += 1

    (DEST / "uniri_debug.yaml").write_text(
        f"path: {DEST}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnc: 1\nnames: ['Human']\n"
    )

    print(f"{'split':>6} " + " ".join(f"{c:>7}" for c in CONDITIONS) + f"{'total':>8}")
    for split in ("train", "val", "test"):
        c = [counts[(split, cond)] for cond in CONDITIONS]
        print(f"{split:>6} " + " ".join(f"{x:7d}" for x in c) + f"{sum(c):8d}")
    print(f"\nwrote {DEST}/uniri_debug.yaml")


if __name__ == "__main__":
    main()
