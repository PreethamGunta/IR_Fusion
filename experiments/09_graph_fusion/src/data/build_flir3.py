#!/usr/bin/env python3
"""Build the 3-class FLIR-aligned dataset for experiment 09.

Policy
------
- 3 classes in ICAFusion order: person, car, bicycle.  Dog boxes are dropped.
- Test = the complete paper validation list (1,013 frames), which is the
  literature-comparable test set.
- Validation is a contiguous temporal block carved out of the 4,129 paper-train
  frames with a +/-10-frame buffer.  The old 08 val/test halves are not reused.
- Two alignment policies are generated:
    drop: flagged pairs (|shift| > 2 px) are removed from train/val; test_all
          keeps every frame for comparability and test_clean removes flagged.
    warp: flagged RGB frames are translated by the estimated shift.
- All shifts are estimated by phase correlation on Canny edges.

Paths are derived from this file; no absolute paths are hardcoded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "flir_aligned" / "raw" / "align"
FIXED_RGB = REPO / "data" / "flir_aligned" / "rgb_fixed"
Y08 = REPO / "data" / "flir_aligned" / "yolo"
OUT = EXP / "data" / "flir3"
AUD = EXP / "audit"

NAMES = ["person", "car", "bicycle"]
OLD2NEW = {0: 0, 2: 1, 1: 2}  # 08: person,bicycle,car,dog -> 09: person,car,bicycle
BUFFER = 10
# The 667 letterboxed RGB frames are concentrated in frames 5500-8859, so a
# single tail holdout is 79% corrupted.  Use one clean block and one small
# corrupted block, each isolated by BUFFER frames, to keep the val alignment
# mix close to the dataset (about 16% cropped).
DEFAULT_VAL_BLOCKS = "3900-4400,7000-7120"


def frame_id(name: str) -> int:
    return int(re.search(r"(\d+)", name).group(1))


def stem_of(name: str) -> str:
    return name.replace("_PreviewData", "")


def read_frames(path: Path) -> list[str]:
    return path.read_text().split()


def source_map() -> dict[str, str]:
    """IR image stem -> source split (train/val/test in the 08 dataset)."""
    out = {}
    for split in ("train", "val", "test"):
        for p in (Y08 / "ir" / "images" / split).glob("*"):
            out[p.stem] = split
    return out


def remap_labels(src: Path) -> tuple[list[tuple[int, float, float, float, float]], int]:
    rows = []
    dropped = 0
    for line in src.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c = int(float(p[0]))
        if c not in OLD2NEW:
            dropped += 1
            continue
        rows.append((OLD2NEW[c], float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return rows, dropped


def write_labels(dst: Path, rows) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows))


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    # The 08 image trees are symlinks into the raw release.  Hardlinking the
    # symlink itself would make its relative target resolve from the new
    # directory and break; link the resolved target instead.
    target = src.resolve()
    try:
        os.link(target, dst)
    except OSError:
        shutil.copy2(target, dst)


def edge_shift(ir_path: Path, rgb_path: Path) -> tuple[float, float, float]:
    ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
    if ir is None or rgb is None or ir.shape != rgb.shape:
        return 999.0, 999.0, 0.0
    ir = cv2.Canny(ir, 50, 150).astype(np.float32) / 255.0
    rgb = cv2.Canny(rgb, 50, 150).astype(np.float32) / 255.0
    win = cv2.createHanningWindow((ir.shape[1], ir.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(ir, rgb, win)
    return float(dx), float(dy), float(response)


def dhash(path: Path) -> int:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (im[:, 1:] > im[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def nn_distances(hashes: list[int], pool: list[int], exclude_self: bool = False) -> list[int]:
    h = np.asarray(hashes, dtype=np.uint64)
    p = np.asarray(pool, dtype=np.uint64)
    d = np.bitwise_count(h[:, None] ^ p[None, :])
    if exclude_self:
        # caller passes pool == hashes; mask the diagonal
        n = min(len(h), len(p))
        d[np.arange(n), np.arange(n)] = 64
    return d.min(axis=1).tolist()


def build_policy(policy: str, splits: dict[str, list[str]], shifts: dict[str, tuple[float, float, float]],
                 srcdir: dict[str, str]) -> dict:
    root = OUT / policy
    if root.exists():
        raise SystemExit(f"refusing to overwrite {root}")
    stats = {"policy": policy, "splits": {}, "dog_dropped": 0}
    for split, frames in splits.items():
        # For test, keep a clean subset as an extra split.
        variants = {"test": frames} if split == "test" else {split: frames}
        if split == "test":
            variants["test_clean"] = [n for n in frames if abs(shifts[n][0]) <= 2 and abs(shifts[n][1]) <= 2]
        for out_split, names in variants.items():
            for modality in ("ir", "rgb"):
                for sub in ("images", "labels"):
                    (root / modality / sub / out_split).mkdir(parents=True, exist_ok=True)
            kept = 0
            for n in names:
                if policy == "drop" and out_split in ("train", "val") and (
                        abs(shifts[n][0]) > 2 or abs(shifts[n][1]) > 2):
                    continue
                src_split = srcdir[n]
                ir_src = Y08 / "ir" / "images" / src_split / f"{n}.jpeg"
                rgb_src = Y08 / "rgb" / "images" / src_split / f"{stem_of(n)}_RGB.jpg"
                lab_src = Y08 / "ir" / "labels" / src_split / f"{n}.txt"
                if not ir_src.exists() or not rgb_src.exists() or not lab_src.exists():
                    stats.setdefault("missing", []).append(
                        {"frame": n, "ir": ir_src.exists(), "rgb": rgb_src.exists(), "label": lab_src.exists()})
                    continue
                ir_dst = root / "ir" / "images" / out_split / f"{n}.jpeg"
                rgb_dst = root / "rgb" / "images" / out_split / f"{stem_of(n)}_RGB.jpg"
                link_or_copy(ir_src, ir_dst)
                rows, dropped = remap_labels(lab_src)
                stats["dog_dropped"] += dropped
                write_labels(root / "ir" / "labels" / out_split / f"{n}.txt", rows)
                write_labels(root / "rgb" / "labels" / out_split / f"{stem_of(n)}_RGB.txt", rows)
                if policy == "warp" and (abs(shifts[n][0]) > 2 or abs(shifts[n][1]) > 2):
                    im = cv2.imread(str(rgb_src))
                    dx, dy, _ = shifts[n]
                    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
                    im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]),
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    cv2.imwrite(str(rgb_dst.with_suffix(".jpg")), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    link_or_copy(rgb_src, rgb_dst)
                kept += 1
            stats["splits"][out_split] = kept
    # configs
    for modality in ("ir", "rgb"):
        (EXP / "configs" / f"flir3_{policy}_{modality}.yaml").write_text(
            f"path: {root / modality}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
            f"nc: 3\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
        )
    (EXP / "configs" / f"flir3_{policy}_paired.yaml").write_text(
        f"path: {root / 'ir'}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: 3\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
        + "\nchannels: 6\n"
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["shifts", "drop", "warp", "all"], default="shifts")
    ap.add_argument("--val-blocks", default=DEFAULT_VAL_BLOCKS,
                    help="comma-separated frame-id ranges, e.g. 3900-4400,7000-7120")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    AUD.mkdir(parents=True, exist_ok=True)

    train_all = read_frames(RAW / "align_train.txt")
    test = read_frames(RAW / "align_validation.txt")
    ids = sorted(train_all, key=frame_id)
    blocks = []
    for part in args.val_blocks.split(","):
        a, b = part.split("-")
        blocks.append((int(a), int(b)))
    def in_block(x, pad=0):
        return any(a - pad <= x <= b + pad for a, b in blocks)
    val = [n for n in ids if in_block(frame_id(n))]
    train = [n for n in ids if not in_block(frame_id(n), BUFFER)]
    srcdir = source_map()
    splits = {"train": train, "val": val, "test": test}

    # Frame-id disjointness.
    id_sets = {k: {frame_id(n) for n in v} for k, v in splits.items()}
    disjoint = {f"{a}_{b}": len(id_sets[a] & id_sets[b]) for a, b in
                (("train", "val"), ("train", "test"), ("val", "test"))}

    shift_csv = AUD / "flir3_shifts_robust.csv"
    if not shift_csv.exists():
        shift_csv = AUD / "flir3_shifts.csv"
    shifts = {}
    if not shift_csv.exists():
        with shift_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "split", "dx", "dy", "response", "flagged"])
            for split, names in splits.items():
                for i, n in enumerate(names):
                    dx, dy, r = edge_shift(RAW / "JPEGImages" / f"{n}.jpeg",
                                           FIXED_RGB / f"{stem_of(n)}_RGB.jpg")
                    flagged = int(abs(dx) > 2 or abs(dy) > 2)
                    w.writerow([n, split, f"{dx:.4f}", f"{dy:.4f}", f"{r:.5f}", flagged])
                    if (i + 1) % 500 == 0:
                        print(split, i + 1, "/", len(names), flush=True)
    for row in csv.DictReader(shift_csv.open()):
        shifts[row["frame"]] = (float(row["dx"]), float(row["dy"]), float(row["response"]))

    by_split = {}
    for split, names in splits.items():
        vals = np.array([[abs(shifts[n][0]), abs(shifts[n][1])] for n in names])
        flagged = np.array([abs(shifts[n][0]) > 2 or abs(shifts[n][1]) > 2 for n in names])
        by_split[split] = {
            "n": len(names),
            "median_abs_dx": float(np.median(vals[:, 0])),
            "median_abs_dy": float(np.median(vals[:, 1])),
            "frac_flagged": float(flagged.mean()),
        }

    audit = {
        "names": NAMES,
        "old2new": {str(k): v for k, v in OLD2NEW.items()},
        "buffer": BUFFER,
        "val_blocks": blocks,
        "frame_counts": {k: len(v) for k, v in splits.items()},
        "frame_id_overlap": disjoint,
        "shift_stats": by_split,
        "shift_csv": str(shift_csv.relative_to(EXP)),
    }
    (AUD / "flir3_split_audit.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps(audit, indent=1))

    if args.policy in ("drop", "warp", "all"):
        policies = ["drop", "warp"] if args.policy == "all" else [args.policy]
        for pol in policies:
            st = build_policy(pol, splits, shifts, srcdir)
            (AUD / f"flir3_{pol}_build.json").write_text(json.dumps(st, indent=1))
            print(pol, json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
