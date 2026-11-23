"""Build labelled contact sheets of UNIRI-TID frames.

The dataset ships as one flat directory (001.jpg ... 7412.jpg) with no weather
condition, no sequence id, and an empty <path/> in every VOC annotation. The
condition labels described in the paper (2663 clear / 1135 fog / 2313 rain) are
simply not distributed.

Recovering them by clustering image statistics would be circular: WeatherGate's
five features *are* image statistics, so labels derived from them would make the
gate ~100% accurate by construction. So we recover the boundaries visually and
keep the label source strictly human/visual.

Usage:
    python contact_sheet.py --stride 40                 # coarse pass
    python contact_sheet.py --start 3000 --end 3600 --stride 5   # refine a boundary
"""

import argparse
import os
from pathlib import Path

from PIL import Image, ImageDraw

RAW = Path(__file__).resolve().parents[2] / "data" / "UNIRI-TID" / "raw"
OUT = Path(__file__).resolve().parents[1] / "contact_sheets"


def frame_ids(start, end):
    ids = sorted(int(p.stem) for p in RAW.glob("*.jpg"))
    return [i for i in ids if start <= i <= end]


def build(ids, cols, thumb_w, thumb_h, per_sheet, tag):
    OUT.mkdir(parents=True, exist_ok=True)
    pad, label_h = 4, 14
    cell_w, cell_h = thumb_w + pad, thumb_h + label_h + pad
    written = []

    for s in range(0, len(ids), per_sheet):
        chunk = ids[s : s + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "black")
        draw = ImageDraw.Draw(sheet)

        for k, fid in enumerate(chunk):
            # filenames are zero-padded to 3 or 4 digits depending on magnitude
            src = RAW / f"{fid:04d}.jpg"
            if not src.exists():
                src = RAW / f"{fid:03d}.jpg"
            if not src.exists():
                continue
            im = Image.open(src).convert("RGB").resize((thumb_w, thumb_h), Image.BILINEAR)
            x, y = (k % cols) * cell_w, (k // cols) * cell_h
            sheet.paste(im, (x, y + label_h))
            draw.rectangle([x, y, x + thumb_w, y + label_h - 1], fill="white")
            draw.text((x + 3, y + 2), str(fid), fill="black")

        name = f"{tag}_{s // per_sheet:02d}_{chunk[0]}-{chunk[-1]}.png"
        sheet.save(OUT / name)
        written.append((name, chunk[0], chunk[-1], len(chunk)))
        print(f"  {name}  ({len(chunk)} frames, id {chunk[0]}-{chunk[-1]})")

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=10**9)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--thumb-w", type=int, default=200)
    ap.add_argument("--thumb-h", type=int, default=150)
    ap.add_argument("--per-sheet", type=int, default=48)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    ids = frame_ids(args.start, args.end)[:: args.stride]
    tag = args.tag or f"s{args.stride}_{args.start}-{min(args.end, 7412)}"
    print(f"{len(ids)} frames sampled (stride {args.stride}) -> {OUT}")
    build(ids, args.cols, args.thumb_w, args.thumb_h, args.per_sheet, tag)


if __name__ == "__main__":
    main()
