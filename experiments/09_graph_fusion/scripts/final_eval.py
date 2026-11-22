#!/usr/bin/env python3
"""Locked test evaluation. Refuses to run without results/LOCK_<exp>.json."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from eval.coco_eval import evaluate, sha256  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--override", action="store_true")
    ap.add_argument("--results-dir", default=str(EXP / "results"))
    args = ap.parse_args()
    results = Path(args.results_dir)
    lock = results / f"LOCK_{args.exp}.json"
    if not lock.exists():
        raise SystemExit(f"missing test lock {lock}; run scripts/make_lock.py first")
    out = Path(args.out)
    if out.exists() and not args.override:
        raise SystemExit(f"output exists: {out}; pass --override to replace it")
    if args.override:
        with (results / f"LOCK_{args.exp}_overrides.jsonl").open("a") as f:
            f.write(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "out": str(out)}) + "\n")
    gt = json.loads(Path(args.gt).read_text())
    preds = json.loads(Path(args.preds).read_text())
    metrics = evaluate(gt, preds)
    metrics.update({
        "experiment": args.exp,
        "lock": json.loads(lock.read_text()),
        "gt": str(Path(args.gt).resolve()), "gt_sha256": sha256(Path(args.gt)),
        "preds": str(Path(args.preds).resolve()), "preds_sha256": sha256(Path(args.preds)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
