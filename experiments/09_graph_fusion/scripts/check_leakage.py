#!/usr/bin/env python3
"""Pre-launch leakage / split / label-policy checks for experiment 09."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
AUD = EXP / "audit"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["drop", "warp"], default="warp")
    args = ap.parse_args()
    failures = []

    split = json.loads((AUD / "flir3_split_audit.json").read_text())
    leak = json.loads((AUD / f"flir3_leakage_{args.policy}.json").read_text())
    if any(v != 0 for v in split["frame_id_overlap"].values()):
        failures.append(f"FLIR frame-id overlap: {split['frame_id_overlap']}")
    for s, v in leak["pair_integrity"].items():
        if not v["base_equal"] or v["ir_only"] or v["rgb_only"]:
            failures.append(f"FLIR pair integrity {s}: {v}")
    for k, v in leak["leakage"].items():
        if isinstance(v, dict) and v.get("exact_dhash_duplicates", 0) != 0:
            failures.append(f"FLIR exact hash duplicates {k}: {v}")
    if "sha256" not in leak["normalization"]["ir"] or "sha256" not in leak["normalization"]["rgb"]:
        failures.append("FLIR normalization stats missing hash")

    dv = json.loads((AUD / "dv_v2_audit.json").read_text())
    for s, v in dv["splits"].items():
        if v["ir_only"] or v["rgb_only"] or v["roundtrip_mismatch"]["ir"] or v["roundtrip_mismatch"]["rgb"]:
            failures.append(f"DV pair/label integrity {s}: {v}")
    if dv["splits"]["train"]["pairs"] != 17962:
        failures.append(f"DV train pair count changed: {dv['splits']['train']['pairs']}")

    # No test path may appear in a train/val field of a training config.
    for cfg in (EXP / "configs").glob("*.yaml"):
        text = cfg.read_text()
        if "images/test" in text and "train: images/test" in text:
            failures.append(f"test path used as train in {cfg.name}")

    if failures:
        print("FAIL")
        for f in failures:
            print(" -", f)
        return 1
    print("PASS: frame-id disjoint, pair integrity, no exact hash duplicates, normalization hashed, DV labels round-trip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
