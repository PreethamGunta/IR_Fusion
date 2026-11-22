#!/usr/bin/env python3
"""
Chunk-disjoint temporal resplit for M3FD.

M3FD IR is 4200 sequential frames (00000.png-04199.png) -- one continuous
drive.  Measured here by frame-differencing: the inter-frame difference
distribution is unimodal (median 9.7, p95 32.5, max 73.6 on 48x64 greyscale)
with no bimodal mode, so there is no reliable shot boundary to split on and
scene-level splitting is not available for this dataset.  What is available is
a temporal gap, and that is what this script builds:

  1. Groups frames into contiguous chunks of --chunk-size (default 240 = 8 s
     at 30 fps). 4200/240 = 17 full chunks plus a 120-frame remainder.

  2. Assigns chunks to train/val/test by a stochastic search that holds the
     chunk counts to the exact targets, rejects any assignment where a split
     would end up missing a class, and scores the *post-drop* class
     distributions -- i.e. the frames that actually survive the boundary gap.
     Scoring pre-drop counts (as an earlier version did) optimises a quantity
     that boundary drops then destroy, which is how the realised split came
     out 74.2/8.9/16.9 frames against a claimed 70/10/20, with Truck down 62%
     in val.

  3. Copies frames to data/split/IR/{images,labels}/{train,val,test}, removing
     --drop-frames (default 60 = 2 s @ 30 fps, half from each side) at every
     cross-split chunk transition.  Every split is separated from its temporal
     neighbours by at least --drop-frames frames.

  4. Measures residual object leakage and treats it as a hard failure above
     --max-contamination: the fraction of val/test frames that carry a
     same-class box matching some train frame at IoU > --leak-iou within
     --leak-window frames.  On failure it retries with a proportionally larger
     chunk and gap.

     This is a real fix, not a stronger assertion on a weaker check.  The
     previous check compared one frame pair per boundary, 17 frames apart, and
     only printed a warning -- which it did, with max_iou 0.9582, and the
     warning was then ignored in three separate documents while the split was
     declared "FIXED".  One pair 17 frames apart cannot see that an object
     persists: on this dataset same-class IoU is still 0.88 at a lag of 120
     frames.

     The enforced metric is a *contamination fraction*, not a single worst
     pair.  That is deliberate and measured: on a 4200-frame continuous drive
     some object instance reappears across any boundary after more than a few
     seconds, so no gap size drives the worst-pair IoU to zero.  Measured on
     this dataset, the fraction of eval frames with an IoU>0.7 train match
     within +-240 frames is 32.8% at chunk 120/gap 60, 5.2% at chunk 240/gap
     120 and 0.6% at chunk 360/gap 180 -- which is why the defaults are the
     middle of that ladder.

  5. Writes split_manifest.json with a split_hash, the realised post-drop frame
     ratios, the minimum cross-split temporal gap, and the full leakage report
     -- for downstream provenance verification.

  LIMITATION, stated because no gap size fixes it: within a single continuous
  drive the same object instance (a parked car, a building edge) reappears
  across any boundary after more than a few seconds.  This script bounds that
  leakage and measures it; it cannot eliminate it.  Treat `boundary_leakage` in
  the manifest as a budget to be read, not as a promise that train and test
  share no objects.  M3FD numbers on this split are optimistically biased, and
  the size of that bias is bounded by the reported fractions, not by zero.
"""

import argparse
import bisect
import contextlib
import fcntl
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
M3FD_IR = HERE / "data" / "M3FD" / "ir"
M3FD_LABELS = HERE / "data" / "M3FD" / "labels"
OUT_SPLIT = HERE / "data" / "split"
LOCK_FILE = HERE / "data" / ".split.lock"

CLASS_NAMES = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]
SPLITS = ("train", "val", "test")


def parse_args():
    ap = argparse.ArgumentParser(description="Chunk-disjoint temporal resplit for M3FD")
    ap.add_argument("--chunk-size", type=int, default=240,
                    help="frames per chunk (default 240 -> 17 chunks + remainder)")
    ap.add_argument("--drop-frames", type=int, default=120,
                    help="frames removed at each cross-split transition (half "
                         "from each side); also the minimum temporal gap "
                         "between two different splits")
    ap.add_argument("--val-pct", type=float, default=0.1)
    ap.add_argument("--test-pct", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-trials", type=int, default=4000)
    ap.add_argument("--dry-run", action="store_true",
                    help="report assignment, realised ratios and leakage; copy nothing")
    ap.add_argument("--leak-iou", type=float, default=0.7,
                    help="IoU above which two same-class boxes count as the "
                         "same object instance")
    ap.add_argument("--leak-window", type=int, default=240,
                    help="search +-this many frames around an eval frame for a "
                         "matching train box")
    ap.add_argument("--max-contamination", type=float, default=0.10,
                    help="hard ceiling on the fraction of val+test frames that "
                         "match a train box at IoU > --leak-iou within "
                         "--leak-window. Exceeding it grows the chunk and the "
                         "gap together, then fails the split.")
    ap.add_argument("--max-extra-lag", type=int, default=120,
                    help="how far past the boundary gap the per-boundary "
                         "diagnostic keeps looking")
    ap.add_argument("--max-attempts", type=int, default=3)
    return ap.parse_args()


# -- label I/O -----------------------------------------------------------------

_BOX_CACHE: dict = {}


def count_classes(label_file: Path) -> dict:
    """Object instances per class in a YOLO label file."""
    counts: dict = {}
    if not label_file.exists():
        return counts
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                c = int(parts[0])
                counts[c] = counts.get(c, 0) + 1
    return counts


def parse_boxes(stem: str) -> list:
    """YOLO labels for a frame stem as (cls, cx, cy, w, h); cached."""
    hit = _BOX_CACHE.get(stem)
    if hit is not None:
        return hit
    boxes = []
    label_file = M3FD_LABELS / (stem + ".txt")
    if label_file.exists():
        with open(label_file, "r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    boxes.append((int(p[0]), float(p[1]), float(p[2]),
                                  float(p[3]), float(p[4])))
    _BOX_CACHE[stem] = boxes
    return boxes


def box_iou(b1, b2) -> float:
    inter_w = max(0.0, min(b1[0] + b1[2] / 2, b2[0] + b2[2] / 2)
                  - max(b1[0] - b1[2] / 2, b2[0] - b2[2] / 2))
    inter_h = max(0.0, min(b1[1] + b1[3] / 2, b2[1] + b2[3] / 2)
                  - max(b1[1] - b1[3] / 2, b2[1] - b2[3] / 2))
    inter = inter_w * inter_h
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.0


def max_same_class_iou(boxes_a, boxes_b) -> tuple:
    """Best same-class IoU between two box lists -> (iou, class_id or None)."""
    best, best_cls = 0.0, None
    for cls_a, cx_a, cy_a, w_a, h_a in boxes_a:
        for cls_b, cx_b, cy_b, w_b, h_b in boxes_b:
            if cls_a != cls_b:
                continue
            iou = box_iou((cx_a, cy_a, w_a, h_a), (cx_b, cy_b, w_b, h_b))
            if iou > best:
                best, best_cls = iou, cls_a
    return best, best_cls


# -- chunking ------------------------------------------------------------------

def _head(frames, k):
    return frames[:k] if k else []


def _tail(frames, k):
    # not frames[-k:]: with k == 0 that is the WHOLE list.
    return frames[len(frames) - k:] if k else []


def make_chunks(ir_files, chunk_size, half_drop):
    """Group sorted IR frames into contiguous temporal chunks.

    Per-chunk class counts are precomputed three ways -- whole chunk, head
    slice, tail slice -- because the boundary gap removes exactly those two
    slices and the assignment search needs a post-drop count for every trial
    without re-reading 4200 label files 4000 times.
    """
    if 2 * half_drop >= chunk_size:
        raise SystemExit(
            f"--chunk-size {chunk_size} with --drop-frames {2 * half_drop} would "
            f"empty every interior chunk. Need --chunk-size >= {2 * half_drop + 8}.")

    chunks = []
    for i in range(0, len(ir_files), chunk_size):
        frames = ir_files[i:i + chunk_size]
        if not frames:
            break
        classes, head, tail = defaultdict(int), defaultdict(int), defaultdict(int)
        for f in frames:
            for c, n in count_classes(M3FD_LABELS / (f.stem + ".txt")).items():
                classes[c] += n
        for f in _head(frames, half_drop):
            for c, n in count_classes(M3FD_LABELS / (f.stem + ".txt")).items():
                head[c] += n
        for f in _tail(frames, half_drop):
            for c, n in count_classes(M3FD_LABELS / (f.stem + ".txt")).items():
                tail[c] += n
        chunks.append({
            "id": len(chunks),
            "frames": frames,
            "classes": dict(classes),
            "head_classes": dict(head),
            "tail_classes": dict(tail),
            "length": len(frames),
        })
    return chunks


def kept_frames_for(chunks, chunk_to_split, split_name, half_drop):
    """Frames of `split_name` that survive the boundary gap, chronologically.

    Note the `if half_drop` guards: `frames[:-half_drop]` is `frames[:0]`,
    i.e. an empty list, when half_drop is 0. A --drop-frames 0 run would
    otherwise silently produce an empty split.
    """
    kept = []
    for chunk in sorted(chunks, key=lambda c: c["id"]):
        if chunk_to_split[chunk["id"]] != split_name:
            continue
        frames = list(chunk["frames"])
        if half_drop and chunk_to_split.get(chunk["id"] + 1, split_name) != split_name:
            frames = frames[:len(frames) - half_drop]
        if half_drop and chunk_to_split.get(chunk["id"] - 1, split_name) != split_name:
            frames = frames[half_drop:]
        kept.extend(frames)
    return kept


def post_drop_classes(chunks, chunk_to_split, half_drop):
    """Per-split class instance counts over surviving frames only."""
    out = {}
    for split in SPLITS:
        counts: dict = defaultdict(int)
        for chunk in chunks:
            if chunk_to_split[chunk["id"]] != split:
                continue
            c = defaultdict(int, chunk["classes"])
            if chunk_to_split.get(chunk["id"] + 1, split) != split:
                for k, v in chunk["tail_classes"].items():
                    c[k] -= v
            if chunk_to_split.get(chunk["id"] - 1, split) != split:
                for k, v in chunk["head_classes"].items():
                    c[k] -= v
            for k, v in c.items():
                if v > 0:
                    counts[k] += v
        out[split] = dict(counts)
    return out


# -- assignment search ---------------------------------------------------------

def assign_chunks(chunks, val_pct, test_pct, seed, num_trials, half_drop):
    """Chunk->split assignment that best matches the class distribution.

    Hard constraints: exact chunk counts per split, and every globally-present
    class survives in every split *after* the boundary gap.  An earlier version
    checked this before the drop, where a class can be present only in frames
    the gap then removes.

    Objective: sum over splits and classes of |actual - expected| / expected,
    with actual measured post-drop and expected = global_count * frame_share.
    """
    rng = random.Random(seed)
    n = len(chunks)

    n_val = round(n * val_pct)
    n_test = round(n * test_pct)
    n_train = n - n_val - n_test
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"Bad split: {n} chunks -> {n_train}/{n_val}/{n_test}")

    global_classes: dict = defaultdict(int)
    for c in chunks:
        for cls, count in c["classes"].items():
            global_classes[cls] += count
    global_classes = {k: v for k, v in global_classes.items() if v > 0}
    all_classes = set(global_classes)

    print(f"  target chunks: train={n_train} val={n_val} test={n_test} (of {n})")

    best_assignment, best_classes, best_score = None, None, float("inf")
    n_rejected = 0
    train_set = val_set = None

    for _ in range(num_trials):
        order = list(range(n))
        rng.shuffle(order)
        train_set = set(order[:n_train])
        val_set = set(order[n_train:n_train + n_val])
        assignment = {cid: ("train" if cid in train_set
                             else "val" if cid in val_set else "test")
                      for cid in range(n)}

        counts = post_drop_classes(chunks, assignment, half_drop)
        if any(counts[s].get(c, 0) == 0 for s in SPLITS for c in all_classes):
            n_rejected += 1
            continue

        kept = {s: len(kept_frames_for(chunks, assignment, s, half_drop)) for s in SPLITS}
        total = max(1, sum(kept.values()))
        score = 0.0
        for split in SPLITS:
            share = kept[split] / total
            for cls, gc in global_classes.items():
                expected = gc * share
                if expected > 0:
                    score += abs(counts[split].get(cls, 0) - expected) / expected

        if score < best_score:
            best_score, best_assignment, best_classes = score, dict(assignment), counts

    if best_assignment is None:
        raise RuntimeError(
            f"No valid assignment in {num_trials} trials ({n_rejected} rejected for "
            f"a class missing after the boundary drop). Try a smaller --chunk-size "
            f"or more --num-trials.")

    print(f"  search: {num_trials} trials, {n_rejected} rejected, best score={best_score:.4f}")
    return best_assignment, best_classes


# -- leakage measurement -------------------------------------------------------

def measure_leakage(chunks, chunk_to_split, half_drop, max_extra_lag):
    """Per-boundary object IoU diagnostic.

    For each cross-split boundary, every one of the last `half_drop` kept
    frames on side A is compared against every frame up to `max_extra_lag`
    frames past the gap on side B.  Looking only at the single pair straddling
    the gap -- what the previous check did -- cannot see persistence, which is
    the failure that actually occurs.
    """
    timeline = []
    for split in SPLITS:
        for f in kept_frames_for(chunks, chunk_to_split, split, half_drop):
            timeline.append((int(f.stem), f.stem, split))
    timeline.sort()
    split_of = [t[2] for t in timeline]
    stems = [t[1] for t in timeline]
    frame_no = [t[0] for t in timeline]

    boundaries = [i for i in range(len(timeline) - 1) if split_of[i] != split_of[i + 1]]

    max_iou = 0.0
    n_over = 0
    details = []

    for i in boundaries:
        split_a, split_b = split_of[i], split_of[i + 1]
        a_lo = i
        while a_lo > 0 and split_of[a_lo - 1] == split_a:
            a_lo -= 1
        a_idx = range(max(a_lo, i - half_drop + 1), i + 1) if half_drop else [i]
        b_idx = range(i + 1, min(len(timeline), i + 1 + max_extra_lag))

        worst, worst_cls, worst_pair = 0.0, None, None
        for a in a_idx:
            boxes_a = parse_boxes(stems[a])
            for b in b_idx:
                iou, cls = max_same_class_iou(boxes_a, parse_boxes(stems[b]))
                if iou > 0.3:
                    n_over += 1
                if iou > worst:
                    worst, worst_cls = iou, cls
                    worst_pair = {"left": stems[a], "right": stems[b], "lag": b - i}

        max_iou = max(max_iou, worst)
        details.append({
            "transition": f"{split_a}|{split_b}",
            "boundary": f"{stems[i]}|{stems[i + 1]}",
            "gap_frames": frame_no[i + 1] - frame_no[i],
            "max_iou_over_window": round(worst, 4),
            "class": CLASS_NAMES[worst_cls] if worst_cls is not None else None,
            "pair": worst_pair,
        })

    return max_iou, n_over, details


def measure_contamination(chunks, chunk_to_split, half_drop, window, iou_thr):
    """Fraction of val+test frames matching a train box at IoU > iou_thr.

    This is the number that decides whether the split is usable, because it
    says how much of the evaluation set the detector could have already seen.
    Exits early on the first match per frame, so the cost is proportional to
    how contaminated the split is rather than to the window size.
    """
    train = {int(f.stem): parse_boxes(f.stem)
             for f in kept_frames_for(chunks, chunk_to_split, "train", half_drop)}
    train_nums = sorted(train)
    evals = {s: sorted(int(f.stem) for f in kept_frames_for(chunks, chunk_to_split, s, half_drop))
             for s in ("val", "test")}

    per_split, contaminated_total, n_total = {}, 0, 0
    for split, nums in evals.items():
        hit = 0
        for e in nums:
            boxes = parse_boxes(f"{e:05d}")
            lo = bisect.bisect_left(train_nums, e - window)
            hi = bisect.bisect_right(train_nums, e + window)
            for t in train_nums[lo:hi]:
                if max_same_class_iou(boxes, train[t])[0] > iou_thr:
                    hit += 1
                    break
        per_split[split] = {
            "frames": len(nums),
            "contaminated": hit,
            "fraction": round(hit / len(nums), 4) if nums else None,
        }
        contaminated_total += hit
        n_total += len(nums)

    return {
        "iou_threshold": iou_thr,
        "window_frames": window,
        "per_split": per_split,
        "eval_frames": n_total,
        "contaminated_frames": contaminated_total,
        "fraction": round(contaminated_total / n_total, 4) if n_total else 0.0,
    }


# -- materialisation -----------------------------------------------------------

def write_yaml(out_dir: Path, final_dir: Path) -> Path:
    """Write the dataset YAML inside `out_dir`.

    `path:` names `final_dir`, not `out_dir`: the YAML is written while the tree
    still carries its build-time name and is renamed into place afterwards, so
    pointing `path:` at the build directory would leave every downstream YAML
    and result pointing into data/split.tmp.<pid>, which no longer exists.
    """
    yaml_path = out_dir / "m3fd_clear.yaml"
    yaml_path.write_text(
        f"path: {final_dir.absolute()}/IR\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 6\n"
        "names:\n"
        "  0: People\n  1: Car\n  2: Bus\n  3: Motorcycle\n  4: Lamp\n  5: Truck\n")
    return yaml_path


def materialise(chunks, chunk_to_split, half_drop, out_root: Path) -> dict:
    """Copy surviving frames+labels into `out_root`; return per-split stats."""
    img_root = out_root / "IR" / "images"
    lbl_root = out_root / "IR" / "labels"
    stats = {}
    for split in SPLITS:
        img_dir, lbl_dir = img_root / split, lbl_root / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        kept = kept_frames_for(chunks, chunk_to_split, split, half_drop)
        classes: dict = defaultdict(int)
        for f in kept:
            shutil.copy2(f, img_dir / f.name)
            lbl = M3FD_LABELS / (f.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, lbl_dir / lbl.name)
                for c, n in count_classes(lbl).items():
                    classes[c] += n
            else:
                # YOLO wants an empty label file for a background image.
                (lbl_dir / (f.stem + ".txt")).touch()

        n_imgs = sum(1 for _ in img_dir.glob("*.png"))
        n_lbls = sum(1 for _ in lbl_dir.glob("*.txt"))
        assert n_imgs == n_lbls, f"{split}: {n_imgs} images vs {n_lbls} labels"
        assert n_imgs == len(kept), f"{split}: kept {len(kept)} but found {n_imgs}"

        stats[split] = {"frames": n_imgs, "classes": dict(classes)}
        print(f"  {split:<5}: {n_imgs:>5} frames, {n_lbls:>5} labels, "
              f"{sum(classes.values()):>6} instances")
    return stats


def publish(tmp_root: Path, out_root: Path):
    """Swap `tmp_root` into place, so readers never see a partial split.

    The previous version called `shutil.rmtree(out_root)` and rebuilt in place,
    leaving a window where data/split was half-populated.  That window is not
    theoretical: two archived job logs died with `FileNotFoundError: Image Not
    Found .../data/split/IR/images/train/02279.png` raised from a dataloader
    worker mid-training, and `imread_('.../images/train/00254.png')` inside
    translate.py -- one job's resplit deleting another job's data mid-read.

    Building beside the target and renaming means a reader sees either the old
    tree or the new one, and a kill mid-copy leaves the previous split intact.
    """
    out_root.parent.mkdir(parents=True, exist_ok=True)
    trash = out_root.with_name(out_root.name + ".trash")
    if trash.exists():
        shutil.rmtree(trash)
    if out_root.exists():
        os.rename(out_root, trash)
    try:
        os.rename(tmp_root, out_root)
    except OSError:
        if trash.exists() and not out_root.exists():
            os.rename(trash, out_root)
        raise
    if trash.exists():
        shutil.rmtree(trash)


@contextlib.contextmanager
def split_lock():
    """Serialise resplits against each other.

    build+rename above already makes concurrent *readers* safe, but two
    resplits racing on one output are still a lost update.
    """
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "w") as fh:
        print(f"  waiting for split lock {LOCK_FILE} ...")
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# -- main ----------------------------------------------------------------------

def main():
    args = parse_args()

    ir_files = sorted(M3FD_IR.glob("*.png"))
    if not ir_files:
        print(f"No images found in {M3FD_IR}")
        return 1

    chunk_size, drop_frames = args.chunk_size, args.drop_frames
    attempt = 0
    while True:
        attempt += 1
        half_drop = drop_frames // 2
        chunks = make_chunks(ir_files, chunk_size, half_drop)
        print(f"\n=== attempt {attempt}: chunk={chunk_size} gap={drop_frames} "
              f"({half_drop} each side) -> {len(chunks)} chunks "
              f"(last {chunks[-1]['length']}) ===")

        assignment, classes = assign_chunks(chunks, args.val_pct, args.test_pct,
                                            args.seed, args.num_trials, half_drop)
        kept = {s: len(kept_frames_for(chunks, assignment, s, half_drop)) for s in SPLITS}
        total_kept = sum(kept.values())
        if min(kept.values()) == 0:
            print("  degenerate split (a split kept 0 frames) -- growing chunk/gap")
            chunk_size *= 2
            drop_frames *= 2
            continue

        print("\nRealised frame ratios (post-drop):")
        for s, target in (("train", 1 - args.val_pct - args.test_pct),
                          ("val", args.val_pct), ("test", args.test_pct)):
            print(f"  {s:<5}: {kept[s]:>5} frames ({kept[s] / total_kept * 100:5.2f}%)"
                  f"   [target {target * 100:.0f}%]")

        print("\nPost-drop class distributions:")
        for s in SPLITS:
            print(f"  {s:<5}: {classes[s]}")

        for s in SPLITS:
            missing = sorted(c for c in range(6) if classes[s].get(c, 0) == 0)
            assert not missing, f"{s} would contain no instances of classes {missing}"

        print(f"\nBoundary diagnostic (window +{args.max_extra_lag} frames) ...")
        max_iou, n_over, details = measure_leakage(
            chunks, assignment, half_drop, args.max_extra_lag)
        for d in details:
            print(f"  {d['transition']:<12} {d['boundary']}  gap={d['gap_frames']:>4}"
                  f"  maxIoU={d['max_iou_over_window']:.4f}  {d['class'] or ''}")
        print(f"  worst cross-split same-class IoU: {max_iou:.4f} "
              f"(reported only -- no gap size drives this to 0 on a single drive)")

        print(f"\nContamination (eval frames with a train box at IoU > "
              f"{args.leak_iou} within +-{args.leak_window} frames) ...")
        t0 = time.time()
        contam = measure_contamination(chunks, assignment, half_drop,
                                       args.leak_window, args.leak_iou)
        for s in ("val", "test"):
            ps = contam["per_split"][s]
            print(f"  {s:<5}: {ps['contaminated']:>4}/{ps['frames']:<4} frames "
                  f"({ps['fraction'] * 100:5.2f}%)")
        print(f"  overall: {contam['fraction'] * 100:.2f}%  "
              f"(ceiling {args.max_contamination * 100:.0f}%, "
              f"{time.time() - t0:.0f}s)")

        if contam["fraction"] <= args.max_contamination or attempt >= args.max_attempts:
            break
        # Grow the chunk and the gap together: doubling only the gap would empty
        # every interior chunk (a 240-frame chunk with a 240-frame gap keeps
        # nothing) and hand 100% of the frames to train.
        chunk_size *= 2
        drop_frames *= 2
        print(f"  -> over budget, retrying with --chunk-size {chunk_size} "
              f"--drop-frames {drop_frames}")

    if contam["fraction"] > args.max_contamination:
        print(f"\nERROR: {contam['fraction'] * 100:.2f}% of eval frames match a train "
              f"box at IoU > {args.leak_iou} within +-{args.leak_window} frames, over "
              f"the {args.max_contamination * 100:.0f}% ceiling, after {attempt} "
              f"attempts. M3FD IR is one continuous drive, so this needs a "
              f"sequence-annotated subset (or a different dataset) rather than a "
              f"bigger gap. Refusing to ship.")
        return 2

    if args.dry_run:
        print("\n[DRY RUN] no files written.")
        return 0

    with split_lock():
        tmp_root = OUT_SPLIT.with_name(OUT_SPLIT.name + f".tmp.{os.getpid()}")
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True)
        try:
            print("\nCopying frames ...")
            stats = materialise(chunks, assignment, half_drop, tmp_root)
            write_yaml(tmp_root, OUT_SPLIT)

            hash_input = json.dumps(
                {s: sorted(f.stem for f in kept_frames_for(chunks, assignment, s, half_drop))
                 for s in SPLITS}, sort_keys=True)
            split_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

            gaps = [d["gap_frames"] for d in details]
            manifest = {
                "schema_version": 2,
                "args": {**vars(args), "chunk_size": chunk_size,
                         "drop_frames": drop_frames, "attempts": attempt},
                "chunk_size": chunk_size,
                "splits": {
                    s: {
                        "chunks": sorted(c["id"] for c in chunks if assignment[c["id"]] == s),
                        "frames": stats[s]["frames"],
                        "frame_pct": round(stats[s]["frames"] / total_kept * 100, 2),
                        "classes": stats[s]["classes"],
                    } for s in SPLITS
                },
                "frames_kept": total_kept,
                "frames_available": len(ir_files),
                "min_cross_split_gap_frames": min(gaps) if gaps else None,
                "boundary_leakage": {
                    "contamination": contam,
                    "worst_pair_iou": round(max_iou, 4),
                    "n_boundary_pairs_iou_gt_0.3": n_over,
                    "boundaries_checked": len(details),
                    "details": details,
                    "note": "M3FD IR is a single continuous drive; a persistent "
                            "object can reappear across any gap, so this is a "
                            "measured budget, not a zero. Read contamination."
                },
                "split_hash": split_hash,
            }
            (tmp_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
            publish(tmp_root, OUT_SPLIT)
        finally:
            if tmp_root.exists():
                shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\nSplit hash: {split_hash}")
    print(f"Eval frames contaminated at IoU > {args.leak_iou}: "
          f"{contam['fraction'] * 100:.2f}%")
    print(f"Manifest: {OUT_SPLIT / 'split_manifest.json'}")
    print(f"YAML:     {OUT_SPLIT / 'm3fd_clear.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())