#!/usr/bin/env python3
"""Print a run's best training-val mAP50-95 (one number, for shell gates).

    python scripts/best_ap.py e00_ir_unified
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def main() -> None:
    name = sys.argv[1]
    f = EXP / "runs" / name / "results.csv"
    if not f.exists():
        print("0")
        return
    rows = list(csv.DictReader(f.open()))
    if not rows:
        print("0")
        return
    print(f"{max(float(r['metrics/mAP50-95(B)']) for r in rows):.4f}")


if __name__ == "__main__":
    main()
