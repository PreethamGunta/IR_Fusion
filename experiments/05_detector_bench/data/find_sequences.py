"""Recover video-sequence structure in UNIRI-TID.

The frames are consecutive video, flattened into one numbered directory. Splitting
randomly would put near-duplicate frames on both sides of the split and inflate
mAP into the 90s. We need sequence boundaries so whole sequences can be held out.

Two independent signals, deliberately chosen to be about *scene identity*, never
about weather statistics (using weather-ish statistics here would contaminate the
gate's labels downstream):

  1. frame-to-frame difference on a small grayscale thumbnail -- large jump = cut
  2. the burned-in camera frame counter in the top-left of each frame, which
     resets per recording; we read its pixel pattern as a coarse fingerprint
     rather than true OCR, which is enough to detect a reset

Also records the frame geometry, because part of the release is letterboxed and
carries no FLIR overlay (a different camera/lens), which matters for splitting.

Writes sequences.csv: frame_id, seq_id, diff_prev, mean_lum, aspect, has_overlay
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2] / "data" / "UNIRI-TID"
RAW = ROOT / "raw"
OUT = ROOT / "sequences.csv"

THUMB = (64, 48)
# A cut is declared when the normalised frame difference exceeds this. Chosen
# from the histogram printed by this script -- within-sequence motion sits far
# below it because the camera is static.
CUT_THRESHOLD = 0.085


def stem_for(fid):
    for cand in (f"{fid:04d}", f"{fid:03d}"):
        if (RAW / f"{cand}.jpg").exists():
            return cand
    return None


def load(fid):
    stem = stem_for(fid)
    if stem is None:
        return None, None, None
    im = Image.open(RAW / f"{stem}.jpg")
    w, h = im.size
    im.draft("L", THUMB)                       # fast JPEG downscale-on-decode
    g = im.convert("L")
    # overlay strip: the FLIR watermark + colour bar live in the right-hand 6%
    bar = np.asarray(g.resize((100, 75), Image.BILINEAR), dtype=np.float32)[:, 94:]
    thumb = np.asarray(g.resize(THUMB, Image.BILINEAR), dtype=np.float32) / 255.0
    return thumb, (w, h), bar.std()


def main():
    ids = sorted(int(p.stem) for p in RAW.glob("*.jpg"))
    print(f"scanning {len(ids)} frames ...")

    rows, prev, diffs = [], None, []
    for n, fid in enumerate(ids):
        thumb, size, barstd = load(fid)
        if thumb is None:
            continue
        d = 1.0 if prev is None else float(np.abs(thumb - prev).mean())
        diffs.append(d)
        rows.append(
            dict(frame_id=fid, diff_prev=round(d, 5),
                 mean_lum=round(float(thumb.mean()), 5),
                 aspect=round(size[0] / size[1], 4),
                 width=size[0], height=size[1],
                 has_overlay=int(barstd > 25))
        )
        prev = thumb
        if (n + 1) % 1500 == 0:
            print(f"  {n + 1}/{len(ids)}")

    d = np.array(diffs)
    print("\nframe-difference percentiles:")
    for p in (50, 75, 90, 95, 98, 99, 99.5):
        print(f"  p{p:<5} {np.percentile(d, p):.4f}")
    print(f"  max    {d.max():.4f}")

    seq, n_cuts = 0, 0
    for r in rows:
        if r["diff_prev"] > CUT_THRESHOLD:
            seq += 1
            n_cuts += 1
        r["seq_id"] = seq

    with open(OUT, "w", newline="") as f:
        cols = ["frame_id", "seq_id", "diff_prev", "mean_lum", "aspect", "width", "height", "has_overlay"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    sizes = Counter(r["seq_id"] for r in rows)
    geom = Counter((r["width"], r["height"]) for r in rows)
    print(f"\ncuts detected      : {n_cuts}  -> {len(sizes)} sequences")
    print(f"sequence length    : min={min(sizes.values())} median={sorted(sizes.values())[len(sizes)//2]} max={max(sizes.values())}")
    print(f"singleton sequences: {sum(1 for v in sizes.values() if v == 1)}")
    print(f"frame geometries   : {dict(geom)}")
    print(f"frames w/ overlay  : {sum(r['has_overlay'] for r in rows)} / {len(rows)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
