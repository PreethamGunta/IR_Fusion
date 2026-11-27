#!/usr/bin/env python3
"""Best training-val mAP50-95 of a student run (one number, for shell gates)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

EXP10 = Path(__file__).resolve().parents[1]


def main() -> None:
    path = Path(sys.argv[1])
    if not path.is_absolute():
        path = EXP10 / path
    if path.is_dir():
        path = path / "results.csv"
    if not path.exists():
        print("0")
        return
    rows = list(csv.DictReader(path.open()))
    if not rows:
        print("0")
        return
    print(f"{max(float(r['metrics/mAP50-95(B)']) for r in rows):.4f}")


if __name__ == "__main__":
    main()
