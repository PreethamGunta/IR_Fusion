#!/usr/bin/env python3
"""Rewrite machine-specific absolute paths inside the release.

The project used to live at ``/home2/home/preetham_g/Gate_IR``.  Every text
artefact now lives under this release root.  This script maps the old prefixes
to the new locations (longest prefix first) and removes the remaining
machine-specific conda paths.

Run after moving the release folder:

    python3 scripts/relocate_paths.py --root /path/to/release
    python3 scripts/relocate_paths.py --dry-run
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = Path("/home2/home/preetham_g/Gate_IR")
OLD_HOME = Path("/home2/home/preetham_g")

# (old_prefix, new_prefix_template) — applied in order; {root} is substituted.
REPLACEMENTS: list[tuple[str, str]] = [
    # DSERT data consolidated into data/dsert_roll
    (f"{OLD_ROOT}/.kilo/worktrees/recover-session-12/data/DSERT-RoLL", "{root}/data/dsert_roll"),
    (f"{OLD_ROOT}/m3fd_experiments/method3_cyclegan/data/DSERT-RoLL", "{root}/data/dsert_roll"),
    # legacy worktrees referenced by historical scripts
    (f"{OLD_ROOT}/.kilo/worktrees/recover-session-12/flir_adras_v2",
     "{root}/experiments/01_flir_weather_translation"),
    (f"{OLD_ROOT}/.kilo/worktrees/recover-session-12/data/flir_adas_v2",
     "{root}/data/flir_adas_v2"),
    (f"{OLD_ROOT}/.kilo/worktrees/recover-session-12/m3fd_experiments",
     "{root}/experiments/03_m3fd_weather"),
    (f"{OLD_ROOT}/.kilo/worktrees/recover-session-12", "{root}"),
    (f"{OLD_ROOT}/.kilo/worktrees/tide-street", "{root}"),
    # old tree -> new tree
    (f"{OLD_ROOT}/flir_adas_experiments/data/flir_adras_v2/synthetic",
     "{root}/data/legacy_flir_synthetic"),
    (f"{OLD_ROOT}/flir_adas_experiments/data/flir_adas_v2",
     "{root}/data/flir_adas_v2"),
    (f"{OLD_ROOT}/flir_adas_experiments/flir_adras_v2",
     "{root}/experiments/01_flir_weather_translation"),
    (f"{OLD_ROOT}/flir_adas_experiments", "{root}"),
    (f"{OLD_ROOT}/flir_gate_experiments", "{root}/experiments/02_flir_gate_ir"),
    (f"{OLD_ROOT}/m3fd_experiments/data", "{root}/data/m3fd"),
    (f"{OLD_ROOT}/m3fd_experiments", "{root}/experiments/03_m3fd_weather"),
    (f"{OLD_ROOT}/uniri_gate_experiments", "{root}/experiments/04_uniri_gate"),
    (f"{OLD_ROOT}/bench", "{root}/experiments/05_detector_bench"),
    (f"{OLD_ROOT}/GATE-IR", "{root}/experiments/06_gate_ir_scaffolding"),
    (f"{OLD_ROOT}/weights", "{root}/models/pretrained"),
    (f"{OLD_ROOT}/data/UNIRI-TID", "{root}/data/uniri_tid"),
    (f"{OLD_ROOT}/datasets/coco8", "{root}/data/coco8"),
    (f"{OLD_ROOT}", "{root}"),
    # machine-specific interpreter / conda
    (f"{OLD_HOME}/.conda/envs/gateir/bin/python", "python"),
    (f"{OLD_HOME}/.conda/etc/profile.d/conda.sh",
     "$(conda info --base)/etc/profile.d/conda.sh"),
    (f"{OLD_HOME}/.conda", "${HOME}/.conda"),
    # second pass: dataset dirs that were separated from their experiment dir
    ("{root}/experiments/03_m3fd_weather/data", "{root}/data/m3fd"),
    ("{root}/experiments/01_flir_weather_translation/data/flir_adas_v2",
     "{root}/data/flir_adas_v2"),
    ("{root}/experiments/01_flir_weather_translation/data/DSERT-RoLL",
     "{root}/data/dsert_roll"),
]

TEXT_SUFFIXES = {
    ".py", ".sh", ".bash", ".sbatch", ".slurm", ".yaml", ".yml",
    ".json", ".cfg", ".ini", ".toml", ".md", ".txt", ".tex",
}
SKIP_FILES = {
    "docs/README_HISTORICAL.md",  # historical snapshot: keep old paths verbatim
    "docs/REORG_PLAN.md",
    "docs/MOVE_LOG.txt",
    "docs/KNOWN_ISSUES.md",
    "scripts/relocate_paths.py",   # do not rewrite this tool itself
    "scripts/rewrite_symlinks.py",
}
MAX_BYTES = 5_000_000


def rewrite_text(text: str, root: Path, extra: list[tuple[str, str]] | None = None) -> tuple[str, int]:
    n = 0
    for old, new_t in (extra or []) + REPLACEMENTS:
        old_s = old.replace("{root}", str(root))
        new = new_t.replace("{root}", str(root))
        if old_s in text:
            n += text.count(old_s)
            text = text.replace(old_s, new)
    return text, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT))
    ap.add_argument("--from", dest="from_path", default=None,
                    help="old release root (default: this packaged checkout)")
    ap.add_argument("--to", dest="to_path", default=None,
                    help="new release root (default: --root)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    to_path = Path(args.to_path).resolve() if args.to_path else root
    extra: list[tuple[str, str]] = []
    if args.from_path:
        extra = [(str(Path(args.from_path).resolve()), str(to_path))]

    changed_files = changed_occurrences = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", ".kilo", "__pycache__"}]
        for name in filenames:
            p = Path(dirpath) / name
            rel = p.relative_to(root).as_posix()
            if rel in SKIP_FILES:
                continue
            if p.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                if p.stat().st_size > MAX_BYTES:
                    continue
                text = p.read_text(errors="strict")
            except (UnicodeDecodeError, OSError):
                continue
            new_text, n = rewrite_text(text, root, extra)
            if n:
                changed_files += 1
                changed_occurrences += n
                if not args.dry_run:
                    p.write_text(new_text)

    print(f"files changed: {changed_files}")
    print(f"occurrences rewritten: {changed_occurrences}")
    if args.dry_run:
        print("(dry run)")


if __name__ == "__main__":
    main()
