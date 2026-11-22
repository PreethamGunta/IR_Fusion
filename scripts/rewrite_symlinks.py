#!/usr/bin/env python3
"""Rewrite absolute symlinks inside the release to relative symlinks.

The pre-reorganisation tree used absolute symlinks into
``/home2/home/preetham_g/gate_ir_release/...``.  Those break as soon as the project is
moved or shared.  This script maps each old prefix to its new location and
replaces the link with a relative one, so the whole tree becomes relocatable.

Run from anywhere:

    python3 scripts/rewrite_symlinks.py [--dry-run]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = Path("/home2/home/preetham_g/Gate_IR")

# Longest prefixes first: the first match wins.
PREFIX_MAP: list[tuple[Path, Path]] = [
    (OLD_ROOT / ".kilo" / "worktrees" / "recover-session-12" / "data" / "DSERT-RoLL",
     PROJECT_ROOT / "data" / "dsert_roll"),
    (OLD_ROOT / "m3fd_experiments" / "method3_cyclegan" / "data" / "DSERT-RoLL",
     PROJECT_ROOT / "data" / "dsert_roll"),
    (OLD_ROOT / ".kilo" / "worktrees" / "recover-session-12",
     PROJECT_ROOT),
    (OLD_ROOT / "flir_adas_experiments" / "data" / "flir_adas_v2",
     PROJECT_ROOT / "data" / "flir_adas_v2"),
    (OLD_ROOT / "flir_adas_experiments" / "flir_adras_v2",
     PROJECT_ROOT / "experiments" / "01_flir_weather_translation"),
    (OLD_ROOT / "flir_adas_experiments",
     PROJECT_ROOT),
    (OLD_ROOT / "flir_gate_experiments",
     PROJECT_ROOT / "experiments" / "02_flir_gate_ir"),
    (OLD_ROOT / "m3fd_experiments",
     PROJECT_ROOT / "experiments" / "03_m3fd_weather"),
    (OLD_ROOT / "uniri_gate_experiments",
     PROJECT_ROOT / "experiments" / "04_uniri_gate"),
    (OLD_ROOT / "bench",
     PROJECT_ROOT / "experiments" / "05_detector_bench"),
    (OLD_ROOT / "GATE-IR",
     PROJECT_ROOT / "experiments" / "06_gate_ir_scaffolding"),
    (OLD_ROOT / "weights",
     PROJECT_ROOT / "models" / "pretrained"),
    (OLD_ROOT / "datasets" / "coco8",
     PROJECT_ROOT / "data" / "coco8"),
    (OLD_ROOT / "data" / "UNIRI-TID",
     PROJECT_ROOT / "data" / "uniri_tid"),
    (OLD_ROOT / "data",
     PROJECT_ROOT / "data"),
    (OLD_ROOT, PROJECT_ROOT),
]


def map_target(target: str) -> Path | None:
    p = Path(target)
    for old, new in PREFIX_MAP:
        try:
            rel = p.relative_to(old)
        except ValueError:
            continue
        return new / rel
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seen = rewritten = broken = 0
    for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
        # Do not descend into the excluded caches if they exist in-tree.
        dirnames[:] = [d for d in dirnames if d not in {".git", ".kilo", "__pycache__"}]
        for name in filenames + dirnames:
            link = Path(dirpath) / name
            if not link.is_symlink():
                continue
            seen += 1
            target = os.readlink(link)
            if not os.path.isabs(target):
                continue
            new_target = map_target(target)
            if new_target is None:
                broken += 1
                continue
            rel = os.path.relpath(new_target, link.parent)
            if not args.dry_run:
                link.unlink()
                link.symlink_to(rel)
            rewritten += 1

    print(f"symlinks seen: {seen}")
    print(f"rewritten to relative: {rewritten}")
    print(f"absolute but unmapped (left as-is): {broken}")
    if args.dry_run:
        print("(dry run)")


if __name__ == "__main__":
    main()
