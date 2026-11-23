"""Assign clear/fog/rain to every UNIRI-TID frame.

The IEEE DataPort release ships no condition labels. They are recovered here from
three INDEPENDENT lines of evidence that agree, none of which uses the image
statistics WeatherGate is later asked to classify (that would be circular):

 1. Archive ordering. The flat file numbering turns out to be grouped by
    scenario. Cutting at the sequence boundaries found by build_sequences.py
    gives 2697 / 1136 / 2507 frames against the paper's published
    2663 / 1135 / 2313 -- the fog block matches to a single frame.

 2. Camera distance (Kristo et al., IEEE Access 2020, Table 2). Fog was shot at
    0-30 m and 50 m, clear at 50-160 m, rain at 30-215 m. This predicts large
    close subjects in fog and small distant ones in rain, which is what the
    contact sheets show for exactly these blocks.

 3. The dog. The paper states the recordings with a dog "were made in dense fog
    only". The dog appears in seq 23-24, inside the fog block.

Residual disagreement with the published counts (clear +34, rain +194) is
consistent with this archive holding 6340 images against the paper's 6111
annotated frames.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "UNIRI-TID"
OUT = ROOT / "conditions.csv"

# sequence id -> condition, from the evidence above (inclusive ranges)
BLOCKS = [(0, 22, "clear"), (23, 31, "fog"), (32, 58, "rain")]
PUBLISHED = {"clear": 2663, "fog": 1135, "rain": 2313}


def condition_of(seq_id):
    for lo, hi, cond in BLOCKS:
        if lo <= seq_id <= hi:
            return cond
    raise ValueError(f"sequence {seq_id} outside every block")


def main():
    rows = []
    for r in csv.DictReader(open(ROOT / "sequence_index.csv")):
        fid, seq = int(r["frame_id"]), int(r["seq_id"])
        rows.append(dict(frame_id=fid, seq_id=seq, condition=condition_of(seq)))

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "seq_id", "condition"])
        w.writeheader()
        w.writerows(rows)

    counts = Counter(r["condition"] for r in rows)
    seqs = defaultdict(set)
    for r in rows:
        seqs[r["condition"]].add(r["seq_id"])

    print(f"{'condition':>9} {'seqs':>5} {'frames':>7} {'published':>10} {'diff':>6}")
    for c in ("clear", "fog", "rain"):
        d = counts[c] - PUBLISHED[c]
        print(f"{c:>9} {len(seqs[c]):5d} {counts[c]:7d} {PUBLISHED[c]:10d} {d:+6d}")
    print(f"{'total':>9} {len(seqs['clear']) + len(seqs['fog']) + len(seqs['rain']):5d} "
          f"{sum(counts.values()):7d} {sum(PUBLISHED.values()):10d} "
          f"{sum(counts.values()) - sum(PUBLISHED.values()):+6d}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
