"""Assign every UNIRI-TID frame to a recording sequence.

Primary signal is the burned-in camera counter (see read_counter.py): it rises
within a recording and resets between recordings, so a decrease or a large jump
is a boundary. Frames whose counter could not be read (the letterboxed frames
carry no overlay) fall back to the thumbnail frame-difference from
find_sequences.py.

Both signals describe *scene/recording identity*, never weather appearance, so
the sequence labels stay independent of anything the weather gate is later asked
to predict.

Writes sequence_index.csv (per frame) and prints a per-sequence summary.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "UNIRI-TID"
OUT = ROOT / "sequence_index.csv"

COUNTER_JUMP = 100      # frames within a recording advance by 1-6, rarely ~20
DIFF_CUT = 0.02         # thumbnail difference indicating a scene change


def load(name, cast):
    out = {}
    for r in csv.DictReader(open(ROOT / name)):
        out[int(r["frame_id"])] = {k: cast(k, v) for k, v in r.items()}
    return out


def main():
    counters = load("counter.csv", lambda k, v: int(v) if k in ("frame_id", "counter") else v)
    diffs = load("sequences.csv", lambda k, v: float(v) if k in ("diff_prev", "mean_lum", "aspect") else int(v) if k in ("frame_id", "width", "height", "has_overlay") else v)

    ids = sorted(counters)
    rows, seq = [], 0
    last_c, gap = None, 0        # last successfully read counter, frames since then

    for i, fid in enumerate(ids):
        c = counters[fid]["counter"]
        readable = c >= 0
        d = diffs.get(fid, {}).get("diff_prev", 0.0)

        if i == 0:
            boundary = False
        elif readable and last_c is not None:
            # An isolated unreadable frame must not split a sequence, so allow the
            # counter to have advanced across the frames we could not read.
            allowed = COUNTER_JUMP * (1 + gap)
            boundary = c < last_c or (c - last_c) > allowed
        else:
            # no counter to go on (e.g. the letterboxed block): fall back to pixels
            boundary = d > DIFF_CUT

        if boundary:
            seq += 1
        rows.append(dict(frame_id=fid, seq_id=seq, counter=c,
                         readable=int(readable), diff_prev=round(d, 5)))
        if readable:
            last_c, gap = c, 0
        else:
            gap += 1

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "seq_id", "counter", "readable", "diff_prev"])
        w.writeheader()
        w.writerows(rows)

    from collections import defaultdict
    byseq = defaultdict(list)
    for r in rows:
        byseq[r["seq_id"]].append(r)

    print(f"{len(byseq)} sequences over {len(rows)} frames\n")
    print(f"{'seq':>4} {'n':>5} {'frame_id range':>16} {'counter range':>16}  readable")
    for s in sorted(byseq):
        g = byseq[s]
        cs = [r["counter"] for r in g if r["readable"]]
        cr = f"{min(cs)}-{max(cs)}" if cs else "-"
        print(f"{s:4d} {len(g):5d} {g[0]['frame_id']:7d}-{g[-1]['frame_id']:<8d} {cr:>16}  {sum(r['readable'] for r in g)}/{len(g)}")

    sizes = sorted(len(v) for v in byseq.values())
    print(f"\nsequence sizes: min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")
    print(f"singletons: {sum(1 for s in sizes if s == 1)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
