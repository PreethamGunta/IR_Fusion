#!/usr/bin/env python3
"""Create the immutable test lock for one experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"  # default; --results-dir overrides


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-metric", type=float, required=True)
    ap.add_argument("--threshold", default="AP50-95")
    ap.add_argument("--results-dir", default=str(RESULTS))
    args = ap.parse_args()
    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    out = results / f"LOCK_{args.exp}.json"
    if out.exists():
        raise SystemExit(f"lock already exists: {out}; refusing to overwrite")
    payload = {
        "experiment": args.exp,
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha(Path(args.checkpoint)),
        "selection_split": "val",
        "selection_metric": args.threshold,
        "val_metric": args.val_metric,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out.write_text(json.dumps(payload, indent=1))
    print(out)


if __name__ == "__main__":
    main()
