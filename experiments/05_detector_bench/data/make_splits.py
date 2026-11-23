"""Build a sequence-disjoint train/val/test split and materialise it for Ultralytics.

Why sequence-disjoint: consecutive frames in UNIRI-TID differ by ~0.1% on median
(measured in find_sequences.py). A random split therefore puts near-duplicate
frames on both sides and inflates mAP into the 90s for free. Whole sequences are
held out instead, so the test set contains recordings the model never saw.

Note the published baseline on this dataset is already 97.93% AP (Kristo et al.,
IEEE Access 2020, Table 3) under their own split, so mAP@50 here is close to
saturated -- mAP@50-95 is the metric with headroom, and detect_eval reports it.

Images are symlinked, not copied, to stay inside the 220 GB quota.
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "UNIRI-TID"
RAW = ROOT / "raw"
YOLO_LABELS = ROOT / "annotations" / "All_In_One_Anot_Yolo"
DEST = ROOT / "yolo"
SPLIT_CSV = ROOT / "splits.csv"


def stem_for(fid):
    for cand in (f"{fid:04d}", f"{fid:03d}"):
        if (RAW / f"{cand}.jpg").exists():
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # frames that actually carry an annotation (see prepare_uniri.py)
    labelled = {int(p.stem) for p in YOLO_LABELS.glob("*.txt")}
    seqs = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "sequence_index.csv")):
        fid, s = int(r["frame_id"]), int(r["seq_id"])
        if fid in labelled:
            seqs[s].append(fid)

    # Each sequence sits entirely in one weather condition (see
    # label_conditions.py), so stratify by allocating within each condition
    # separately -- otherwise fog, the smallest block, lands lopsidedly.
    cond_of_seq = {}
    for r in csv.DictReader(open(ROOT / "conditions.csv")):
        cond_of_seq[int(r["seq_id"])] = r["condition"]

    rng = random.Random(args.seed)
    assigned, have = {}, {"train": 0, "val": 0, "test": 0}
    for cond in ("clear", "fog", "rain"):
        members = [s for s in seqs if cond_of_seq[s] == cond]
        sub = sum(len(seqs[s]) for s in members)
        tgt = {"test": args.test_frac * sub, "val": args.val_frac * sub}
        tgt["train"] = sub - tgt["test"] - tgt["val"]
        got = {"train": 0, "val": 0, "test": 0}

        # largest sequences first, each to whichever split is furthest below
        # its target for THIS condition
        rng.shuffle(members)
        members.sort(key=lambda s: -len(seqs[s]))
        for s in members:
            pick = max(got, key=lambda k: (tgt[k] - got[k]) / max(tgt[k], 1))
            assigned[s] = pick
            got[pick] += len(seqs[s])
            have[pick] += len(seqs[s])

    total = sum(len(v) for v in seqs.values())

    rows = []
    for s, split in assigned.items():
        for fid in seqs[s]:
            rows.append(dict(frame_id=fid, seq_id=s, split=split))
    rows.sort(key=lambda r: r["frame_id"])
    with open(SPLIT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "seq_id", "split"])
        w.writeheader()
        w.writerows(rows)

    # ---- materialise for Ultralytics -------------------------------------
    n_box = defaultdict(int)
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = DEST / sub / split
            # Wipe before relinking. A frame that moves between splits when the
            # split is rebuilt would otherwise keep its stale symlink and end up
            # in two splits at once -- silent train/val contamination that no
            # assertion on the *assignment* can catch.
            if d.exists():
                for p in d.iterdir():
                    p.unlink()
            d.mkdir(parents=True, exist_ok=True)
    for r in rows:
        stem = stem_for(r["frame_id"])
        if stem is None:
            continue
        split = r["split"]
        img_dst = DEST / "images" / split / f"{stem}.jpg"
        lbl_dst = DEST / "labels" / split / f"{stem}.txt"
        for src, dst in ((RAW / f"{stem}.jpg", img_dst), (YOLO_LABELS / f"{stem}.txt", lbl_dst)):
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src.resolve())
        n_box[split] += len([l for l in (YOLO_LABELS / f"{stem}.txt").read_text().splitlines() if l.strip()])

    (DEST / "uniri.yaml").write_text(
        f"path: {DEST}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnc: 1\nnames: ['Human']\n"
    )

    # ---- report + integrity assertions -----------------------------------
    per_split = defaultdict(list)
    for s, sp in assigned.items():
        per_split[sp].append(s)
    print(f"{total} labelled frames in {len(seqs)} sequences\n")
    print(f"{'split':>6} {'seqs':>5} {'frames':>7} {'share':>7} {'boxes':>7}")
    for sp in ("train", "val", "test"):
        n = have[sp]
        print(f"{sp:>6} {len(per_split[sp]):5d} {n:7d} {n / total:6.1%} {n_box[sp]:7d}")

    cond_of = {int(r["frame_id"]): r["condition"] for r in csv.DictReader(open(ROOT / "conditions.csv"))}
    by = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by[r["split"]][cond_of[r["frame_id"]]] += 1
    print(f"\n{'split':>6} {'clear':>7} {'fog':>7} {'rain':>7}")
    for sp in ("train", "val", "test"):
        c = by[sp]
        print(f"{sp:>6} {c['clear']:7d} {c['fog']:7d} {c['rain']:7d}")

    seen = [set(per_split[sp]) for sp in ("train", "val", "test")]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2]), "sequence overlap!"

    # Assert on what actually landed on disk, not just on the assignment: the
    # directories are what Ultralytics reads, and they are what went wrong.
    disk = {sp: {p.name for p in (DEST / "images" / sp).iterdir()} for sp in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        dup = disk[a] & disk[b]
        assert not dup, f"{len(dup)} images materialised into both {a} and {b}, e.g. {sorted(dup)[:3]}"
    n_disk = sum(len(v) for v in disk.values())
    assert n_disk == total, f"materialised {n_disk} images but split has {total} rows"
    for sp in ("train", "val", "test"):
        n_lbl = len(list((DEST / "labels" / sp).iterdir()))
        assert n_lbl == len(disk[sp]), f"{sp}: {len(disk[sp])} images but {n_lbl} labels"

    print("\nOK: no sequence appears in more than one split")
    print(f"OK: {n_disk} images materialised, no image in two split directories, labels match")
    print(f"wrote {SPLIT_CSV} and {DEST}/uniri.yaml")


if __name__ == "__main__":
    main()
