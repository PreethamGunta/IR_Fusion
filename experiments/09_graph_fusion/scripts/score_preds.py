#!/usr/bin/env python3
"""Score a prediction JSON against a COCO GT JSON with the 09 evaluator.

Sanity/scoring helper (CPU, lightweight). Official locked test numbers still
come from scripts/final_eval.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from eval.coco_eval import evaluate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--min-ap", type=float, default=None,
                    help="exit nonzero if AP < this (sanity gate for chained commands)")
    args = ap.parse_args()
    gt = json.loads(Path(args.gt).read_text())
    preds = json.loads(Path(args.preds).read_text())
    m = evaluate(gt, preds)
    print(json.dumps({k: round(m[k], 5) for k in
                      ("AP", "AP50", "AP75", "APs", "APm", "APl")}))
    print("per_class_AP:", json.dumps(
        {k: round(v, 5) for k, v in m["per_class_AP"].items()}))
    print("n_preds:", m["n_predictions"], "n_images:", m["n_images"])
    if args.min_ap is not None and m["AP"] < args.min_ap:
        raise SystemExit(f"SANITY GATE FAILED: AP {m['AP']:.5f} < {args.min_ap}")


if __name__ == "__main__":
    main()
