"""Read the burned-in camera frame counter from UNIRI-TID frames.

Each frame carries a 6-digit counter burned into the top-left corner by the
camera. It increments within a recording and restarts for each new recording, so
it is the dataset's true sequence marker -- far more reliable than guessing
boundaries from file numbering or image statistics.

The font is a fixed bitmap, so this is template matching rather than real OCR:

    cluster   extract every digit glyph, normalise, and group identical bitmaps
    label     write one representative per cluster to a sheet for a human to read
    decode    apply the labelled templates to all frames

Deliberately uses only the counter overlay, never scene content, so nothing about
weather leaks into the labels this produces.

    python read_counter.py cluster
    python read_counter.py decode --map 0123456789
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2] / "data" / "UNIRI-TID"
RAW = ROOT / "raw"
SHEETS = Path(__file__).resolve().parents[1] / "contact_sheets"
TEMPLATES = ROOT / "counter_templates.json"
OUT = ROOT / "counter.csv"

CROP = (0, 0, 300, 60)     # region containing the counter
THRESH = 200               # counter is near-white on a dark plate
GLYPH = (12, 18)           # normalised glyph size


def stem_for(fid):
    for cand in (f"{fid:04d}", f"{fid:03d}"):
        if (RAW / f"{cand}.jpg").exists():
            return cand
    return None


def glyphs(fid):
    """Return (normalised glyph array, ...) for one frame, or [] if no counter."""
    stem = stem_for(fid)
    if stem is None:
        return []
    a = np.asarray(Image.open(RAW / f"{stem}.jpg").convert("L").crop(CROP))
    mask = a > THRESH
    if mask.sum() < 40:
        return []
    cols = mask.any(axis=0)
    runs, start = [], None
    for x, on in enumerate(cols):
        if on and start is None:
            start = x
        elif not on and start is not None:
            if x - start >= 3:
                runs.append((start, x))
            start = None
    if start is not None and len(cols) - start >= 3:
        runs.append((start, len(cols)))
    if not (5 <= len(runs) <= 7):
        return []

    out = []
    for x0, x1 in runs:
        sub = mask[:, x0:x1]
        rows = np.where(sub.any(axis=1))[0]
        if len(rows) == 0:
            return []
        g = Image.fromarray((sub[rows[0]:rows[-1] + 1] * 255).astype(np.uint8))
        out.append(np.asarray(g.resize(GLYPH, Image.NEAREST)) > 127)
    return out


def key_of(g):
    return "".join("1" if v else "0" for v in g.flatten())


def cmd_cluster(args):
    ids = sorted(int(p.stem) for p in RAW.glob("*.jpg"))
    step = max(1, len(ids) // args.sample)
    clusters, counts = {}, defaultdict(int)
    for fid in ids[::step]:
        for g in glyphs(fid):
            k = key_of(g)
            counts[k] += 1
            clusters.setdefault(k, g)

    order = sorted(counts, key=lambda k: -counts[k])
    print(f"{len(order)} distinct glyph bitmaps from {len(ids[::step])} frames")

    cell_w, cell_h, cols = 40, 60, 10
    rows = (len(order) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "black")
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for i, k in enumerate(order):
        im = Image.fromarray((clusters[k] * 255).astype(np.uint8)).resize((24, 36), Image.NEAREST)
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        sheet.paste(im.convert("RGB"), (x + 8, y + 18))
        d.text((x + 2, y + 2), f"#{i}:{counts[k]}", fill="yellow")
    SHEETS.mkdir(parents=True, exist_ok=True)
    sheet.save(SHEETS / "counter_glyphs.png")
    TEMPLATES.write_text(json.dumps({"order": order}))
    print(f"wrote {SHEETS / 'counter_glyphs.png'}  (read digits left-to-right, top-to-bottom)")
    print(f"then: python read_counter.py decode --map <digits in that order>")


def cmd_decode(args):
    order = json.loads(TEMPLATES.read_text())["order"]
    if len(args.map) != len(order):
        raise SystemExit(f"--map has {len(args.map)} chars but there are {len(order)} clusters")
    lookup = {k: c for k, c in zip(order, args.map)}

    ids = sorted(int(p.stem) for p in RAW.glob("*.jpg"))
    rows, unread = [], 0
    for n, fid in enumerate(ids):
        gs = glyphs(fid)
        val = ""
        for g in gs:
            val += lookup.get(key_of(g), "?")
        ok = bool(gs) and "?" not in val
        if not ok:
            unread += 1
        rows.append(dict(frame_id=fid, counter=int(val) if ok else -1, raw=val))
        if (n + 1) % 1500 == 0:
            print(f"  {n + 1}/{len(ids)}")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "counter", "raw"])
        w.writeheader()
        w.writerows(rows)
    print(f"\ndecoded {len(rows) - unread}/{len(rows)} frames  ({unread} without a readable counter)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cluster"); c.add_argument("--sample", type=int, default=900); c.set_defaults(fn=cmd_cluster)
    d = sub.add_parser("decode"); d.add_argument("--map", required=True); d.set_defaults(fn=cmd_decode)
    a = ap.parse_args()
    a.fn(a)
