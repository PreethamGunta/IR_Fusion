#!/usr/bin/env python3
"""Static validation of the packaged GATE-IR release.

Checks (no training, no GPU, no network):
  * expected top-level directories exist
  * every symlink resolves and none is absolute
  * no machine-specific paths remain in source code / launch scripts
  * every dataset YAML's path/train/val/test entries exist

Known, documented issues (`docs/KNOWN_ISSUES.md`) are reported as warnings and
do not make the check fail:

  * 282 pre-existing dangling symlinks in the CycleDiff train split
  * generated YAMLs that contain absolute paths
  * generated YAMLs whose `data/filtered/...` targets are regenerated on demand
  * the archived legacy synthetic copy

Usage:
    python3 scripts/verify_release.py [--strict]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ["data", "experiments", "models", "results", "docs", "environment", "scripts", "logs", "configs"]
SOURCE_SUFFIXES = {".py", ".sh", ".bash", ".sbatch", ".slurm"}
CONFIG_SUFFIXES = {".yaml", ".yml"}
MACHINE_PREFIXES = ("/home2/home/preetham_g",)
# These contain the mapping constants / the checker itself, by design.
SOURCE_SKIP = {"scripts/relocate_paths.py", "scripts/rewrite_symlinks.py", "scripts/verify_release.py"}
GENERATED_YAML_HINTS = ("/runs/",)
KNOWN_YAML_TARGETS = (
    "data/filtered",
    "experiments/02_flir_gate_ir/data/",
    "experiments/03_m3fd_weather/method3_cyclegan/translated",
    "data/m3fd/synthetic_depth_3b",
    "data/flir_adras_v2",
)


def check_dirs() -> list[str]:
    return [f"missing dir: {d}" for d in EXPECTED if not (ROOT / d).is_dir()]


def check_symlinks() -> tuple[int, int, int, list[str]]:
    total = broken = absolute = 0
    broken_paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        for name in filenames + dirnames:
            p = Path(dirpath) / name
            if not p.is_symlink():
                continue
            total += 1
            if os.path.isabs(os.readlink(p)):
                absolute += 1
            if not p.exists():
                broken += 1
                broken_paths.append(str(p.relative_to(ROOT)))
    return total, broken, absolute, broken_paths


def check_machine_paths() -> list[str]:
    hits: list[str] = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in SOURCE_SUFFIXES:
            continue
        rel = p.relative_to(ROOT).as_posix()
        if any(part in {".git", "__pycache__", "archive", "_old", "pre_audit", "legacy_flir_synthetic_diff"} for part in p.parts):
            continue
        if rel in SOURCE_SKIP:
            continue
        try:
            text = p.read_text(errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        if any(pref in text for pref in MACHINE_PREFIXES):
            hits.append(rel)
    return hits


def check_dataset_yamls() -> tuple[list[str], list[str]]:
    try:
        import yaml
    except ImportError:
        return ["PyYAML not available: dataset YAML check skipped"], []
    errors: list[str] = []
    warnings: list[str] = []
    for p in ROOT.rglob("*.yaml"):
        if any(part in {".git", "__pycache__", "archive", "_old", "pre_audit", "legacy_flir_synthetic_diff"} for part in p.parts):
            continue
        rel = p.relative_to(ROOT).as_posix()
        if any(h in f"/{rel}" for h in GENERATED_YAML_HINTS):
            # Ultralytics run args.yaml: `val: True` and absolute paths are normal.
            continue
        try:
            data = yaml.safe_load(p.read_text())
        except Exception:
            continue
        if not isinstance(data, dict) or not ({"train", "val", "test"} & data.keys()):
            continue
        base = Path(str(data.get("path", p.parent)))
        if not base.is_absolute():
            base = (p.parent / base).resolve()
        for key in ("train", "val", "test"):
            if key not in data:
                continue
            values = data[key] if isinstance(data[key], list) else [data[key]]
            for v in values:
                if not isinstance(v, str) or v.lower() in {"true", "false"}:
                    continue
                q = Path(v)
                if not q.is_absolute():
                    q = base / q
                if glob.glob(str(q)):
                    continue
                msg = f"{rel}: {key}={v} -> {q} (missing)"
                if any(k in str(q) for k in KNOWN_YAML_TARGETS):
                    warnings.append(msg)
                else:
                    errors.append(msg)
    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="treat known warnings as failures")
    args = ap.parse_args()

    errors = check_dirs()
    errors += [f"machine-specific path in {m}" for m in check_machine_paths()]
    yaml_errors, yaml_warnings = check_dataset_yamls()
    errors += yaml_errors

    total, broken, absolute, broken_paths = check_symlinks()
    if absolute:
        errors.append(f"{absolute} absolute symlinks")
    if broken:
        # all 282 are in the one pre-existing CycleDiff split
        roots = {p.split("/fog/images/train/")[0] for p in broken_paths}
        if roots == {"data/flir_adas_v2/synthetic/cyclediff"}:
            yaml_warnings.append(f"{broken} pre-existing dangling symlinks in the CycleDiff train split")
        else:
            errors.append(f"{broken} broken symlinks (see output)")

    print(f"root: {ROOT}")
    print(f"symlinks: {total} (broken {broken}, absolute {absolute})")
    print(f"errors: {len(errors)}")
    for msg in errors[:40]:
        print("  ERROR:", msg)
    if len(errors) > 40:
        print(f"  ... and {len(errors) - 40} more errors")
    print(f"known warnings: {len(yaml_warnings)}")
    for msg in yaml_warnings[:10]:
        print("  WARN:", msg)
    if len(yaml_warnings) > 10:
        print(f"  ... and {len(yaml_warnings) - 10} more warnings")

    if errors:
        return 1
    if args.strict and yaml_warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
