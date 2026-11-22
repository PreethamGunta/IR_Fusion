#!/usr/bin/env python3
"""Write COCO ground-truth JSONs for the DV v2 splits (val/test/train)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from gfyolo.dv_coco import make_coco_gt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["ir", "rgb"], required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    gt = make_coco_gt(args.modality, args.split)
    out = Path(args.out) if args.out else (
        EXP / "results" / "gt" / f"dv_v2_{args.modality}_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gt))
    print(f"[make_gt_dv] {args.modality}/{args.split}: "
          f"{len(gt['images'])} images, {len(gt['annotations'])} annotations -> {out}")
    print(f"[make_gt_dv] sha256-file written alongside for provenance")
    import hashlib
    (out.with_suffix(".sha256")).write_text(
        hashlib.sha256(out.read_bytes()).hexdigest() + "\n")
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


if __name__ == "__main__":
    main()
