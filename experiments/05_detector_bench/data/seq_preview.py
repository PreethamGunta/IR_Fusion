"""Render one representative frame per sequence, for visual condition labelling.

Condition labels (clear / fog / rain) are assigned by looking at these sheets.
They must NOT be derived from image statistics: WeatherGate's five features are
image statistics, so statistically-derived labels would make the gate accurate by
construction and the result meaningless.
"""

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2] / "data" / "UNIRI-TID"
RAW = ROOT / "raw"
OUT = Path(__file__).resolve().parents[1] / "contact_sheets"


def stem_for(fid):
    for cand in (f"{fid:04d}", f"{fid:03d}"):
        if (RAW / f"{cand}.jpg").exists():
            return cand
    return None


def main(per_seq=2, cols=6, tw=300, th=225, per_sheet=24):
    byseq = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "sequence_index.csv")):
        byseq[int(r["seq_id"])].append(int(r["frame_id"]))

    picks = []
    for s in sorted(byseq):
        f = byseq[s]
        idxs = [len(f) // (per_seq + 1) * (k + 1) for k in range(per_seq)]
        for k, i in enumerate(dict.fromkeys(idxs)):
            picks.append((s, f[min(i, len(f) - 1)], len(f)))

    OUT.mkdir(parents=True, exist_ok=True)
    pad, lh = 4, 16
    cw, ch = tw + pad, th + lh + pad
    for s0 in range(0, len(picks), per_sheet):
        chunk = picks[s0:s0 + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cw, rows * ch), "black")
        d = ImageDraw.Draw(sheet)
        for k, (seq, fid, n) in enumerate(chunk):
            stem = stem_for(fid)
            if stem is None:
                continue
            im = Image.open(RAW / f"{stem}.jpg").convert("RGB").resize((tw, th), Image.BILINEAR)
            x, y = (k % cols) * cw, (k // cols) * ch
            sheet.paste(im, (x, y + lh))
            d.rectangle([x, y, x + tw, y + lh - 1], fill="white")
            d.text((x + 3, y + 3), f"seq {seq}  (n={n})  frame {fid}", fill="black")
        name = f"seqs_{chunk[0][0]:02d}-{chunk[-1][0]:02d}.png"
        sheet.save(OUT / name)
        print(f"  {name}  seqs {chunk[0][0]}-{chunk[-1][0]}")


if __name__ == "__main__":
    main()
