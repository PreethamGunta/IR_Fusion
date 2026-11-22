#!/usr/bin/env python3
"""Leakage, split-integrity and normalization audit for the FLIR 3-class build.

Reads only the generated data/flir3/drop tree.  Paths are derived from this file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[2]
AUD = EXP / "audit"
SPLITS = ("train", "val", "test")


def dhash(path: Path) -> int:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (im[:, 1:] > im[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def nn(h: list[int], pool: list[int], exclude_self: bool = False) -> list[int]:
    a = np.asarray(h, dtype=np.uint64)
    b = np.asarray(pool, dtype=np.uint64)
    d = np.bitwise_count(a[:, None] ^ b[None, :])
    if exclude_self:
        n = min(len(a), len(b))
        d[np.arange(n), np.arange(n)] = 64
    return d.min(axis=1).tolist()


def norm_stats(split: str, modality: str, step: int = 4) -> dict:
    files = sorted((ROOT / modality / "images" / split).glob("*"))[::step]
    n = 0
    s = np.zeros(3, dtype=np.float64)
    ss = np.zeros(3, dtype=np.float64)
    for p in files:
        im = cv2.imread(str(p))
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        s += im.reshape(-1, 3).sum(0)
        ss += (im.reshape(-1, 3) ** 2).sum(0)
        n += im.shape[0] * im.shape[1]
    mean = (s / n).tolist()
    std = np.sqrt(ss / n - (s / n) ** 2).tolist()
    return {"images": len(files), "pixels": int(n), "mean": mean, "std": std}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["drop", "warp"], default="drop")
    args = ap.parse_args()
    global ROOT
    ROOT = EXP / "data" / "flir3" / args.policy
    audit = {"policy": args.policy, "splits": {}, "pair_integrity": {}, "leakage": {}, "normalization": {}}
    hashes = {}
    for split in SPLITS:
        ir = sorted((ROOT / "ir" / "images" / split).glob("*"))
        rgb = sorted((ROOT / "rgb" / "images" / split).glob("*"))
        base_ir = {p.stem.replace("_PreviewData", "") for p in ir}
        base_rgb = {p.stem.replace("_RGB", "") for p in rgb}
        hashes[split] = [dhash(p) for p in ir]
        audit["splits"][split] = {"ir_images": len(ir), "rgb_images": len(rgb)}
        audit["pair_integrity"][split] = {
            "base_equal": base_ir == base_rgb,
            "ir_only": len(base_ir - base_rgb),
            "rgb_only": len(base_rgb - base_ir),
        }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ex = len(set(hashes[a]) & set(hashes[b]))
        audit["leakage"][f"{a}_{b}"] = {"exact_dhash_duplicates": ex}
    for label, a, b, excl in (("test_to_train_nn", "test", "train", False),
                              ("val_to_train_nn", "val", "train", False),
                              ("train_to_train_nn", "train", "train", True)):
        d = np.asarray(nn(hashes[a], hashes[b], exclude_self=excl))
        audit["leakage"][label] = {"min": int(d.min()), "median": float(np.median(d)),
                                   "p05": float(np.percentile(d, 5)),
                                   "p95": float(np.percentile(d, 95))}
    for modality in ("ir", "rgb"):
        audit["normalization"][modality] = norm_stats("train", modality)
    # Hash the stats so every config can cite the exact normalization.
    for modality, st in audit["normalization"].items():
        st["sha256"] = hashlib.sha256(json.dumps(st, sort_keys=True).encode()).hexdigest()
    (AUD / f"flir3_leakage_{args.policy}.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps(audit, indent=1))


if __name__ == "__main__":
    main()
