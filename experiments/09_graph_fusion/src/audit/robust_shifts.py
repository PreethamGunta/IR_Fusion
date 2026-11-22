#!/usr/bin/env python3
"""Robust per-pair registration shifts for FLIR-aligned.

Raw phase correlation fails on low-texture/night frames (negative response,
spurious 20+ px shifts).  This script:
- keeps raw phase-correlation values;
- marks a measurement valid when response >= 0.05 and |shift| <= 10 px;
- fills invalid frames from a +/-7-frame local median of valid shifts;
- applies a 5-frame rolling median;
- records the 08 ``rgb_cropped`` flag for diagnosis.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[2]
AUD = EXP / "audit"
MAN = REPO / "data" / "flir_aligned" / "pair_manifest.json"


def fid(name: str) -> int:
    return int(re.search(r"(\d+)", name).group(1))


def main() -> None:
    rows = list(csv.DictReader((AUD / "flir3_shifts.csv").open()))
    cropped = {p["ir"].replace(".jpeg", ""): bool(p["rgb_cropped"]) for p in json.loads(MAN.read_text())["pairs"]}
    raw = REPO / "data" / "flir_aligned" / "raw" / "align"
    train_set = set((raw / "align_train.txt").read_text().split())
    test_set = set((raw / "align_validation.txt").read_text().split())
    # Group by the paper split, not the (old) split column: the new val blocks
    # live inside the paper-train list and need local smoothing across it.
    by = {"train": [], "test": []}
    for r in rows:
        by["train" if r["frame"] in train_set else "test"].append(r)
    out = []
    summary = {}
    for split, rs in by.items():
        rs.sort(key=lambda r: fid(r["frame"]))
        dx = np.array([float(r["dx"]) for r in rs])
        dy = np.array([float(r["dy"]) for r in rs])
        resp = np.array([float(r["response"]) for r in rs])
        valid = (resp >= 0.05) & (np.abs(dx) <= 10) & (np.abs(dy) <= 10)
        ndx, ndy = dx.copy(), dy.copy()
        for i in range(len(rs)):
            if valid[i]:
                continue
            lo, hi = max(0, i - 7), min(len(rs), i + 8)
            v = valid[lo:hi]
            if v.sum() >= 3:
                ndx[i] = np.median(dx[lo:hi][v])
                ndy[i] = np.median(dy[lo:hi][v])
            elif valid.sum():
                ndx[i] = np.median(dx[valid])
                ndy[i] = np.median(dy[valid])
            else:
                ndx[i], ndy[i] = 0.0, 0.0
        # 5-frame rolling median
        for i in range(len(rs)):
            lo, hi = max(0, i - 2), min(len(rs), i + 3)
            ndx[i] = np.median(ndx[lo:hi])
            ndy[i] = np.median(ndy[lo:hi])
        flagged = (np.abs(ndx) > 2) | (np.abs(ndy) > 2)
        summary[split] = {
            "n": len(rs), "valid_raw": int(valid.sum()),
            "median_abs_dx": float(np.median(np.abs(ndx))),
            "median_abs_dy": float(np.median(np.abs(ndy))),
            "frac_flagged": float(flagged.mean()),
            "cropped_n": int(sum(cropped.get(r["frame"], False) for r in rs)),
            "cropped_frac_flagged": float(flagged[[cropped.get(r["frame"], False) for r in rs]].mean()) if any(cropped.get(r["frame"], False) for r in rs) else 0.0,
        }
        for i, r in enumerate(rs):
            out.append({"frame": r["frame"], "split": split,
                        "raw_dx": r["dx"], "raw_dy": r["dy"], "response": r["response"],
                        "dx": f"{ndx[i]:.4f}", "dy": f"{ndy[i]:.4f}",
                        "valid_raw": int(valid[i]),
                        "cropped": int(cropped.get(r["frame"], False)),
                        "flagged": int(flagged[i])})
    with (AUD / "flir3_shifts_robust.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    (AUD / "flir3_shift_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
