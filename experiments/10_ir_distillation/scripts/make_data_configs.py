#!/usr/bin/env python3
"""Generate DroneVehicle v2 data yamls with paths derived from THIS node.

The 09 yamls contain absolute paths from the development node. Experiment 10
must run on a different machine, so it regenerates its three yamls from the
dataset root at run time (no hardcoded /home2/... paths in code).

    python scripts/make_data_configs.py
    python scripts/make_data_configs.py --root /data/dronevehicle_v2/yolo
"""
from __future__ import annotations

import argparse
from pathlib import Path

EXP10 = Path(__file__).resolve().parents[1]
REPO = EXP10.parents[1]
DEFAULT_ROOT = REPO / "experiments/09_graph_fusion/data/dronevehicle_v2/yolo"

NAMES = """nc: 5
names:
  0: car
  1: truck
  2: bus
  3: van
  4: freight_car
"""


def write(path: Path, root: Path, channels: int | None) -> None:
    text = (
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        + NAMES
    )
    if channels:
        text += f"\nchannels: {channels}\n"
    path.write_text(text)
    print(f"[cfg] {path} -> {root}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="DroneVehicle v2 yolo root (contains ir/ and rgb/)")
    ap.add_argument("--out", default=str(EXP10 / "configs"))
    args = ap.parse_args()

    root = Path(args.root) if args.root else DEFAULT_ROOT
    root = root.resolve()
    if not (root / "ir/images/train").exists():
        raise SystemExit(f"dataset root not found: {root}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    write(out / "dv_ir.yaml", root / "ir", None)
    write(out / "dv_rgb.yaml", root / "rgb", None)
    write(out / "dv_paired.yaml", root / "ir", 6)


if __name__ == "__main__":
    main()
