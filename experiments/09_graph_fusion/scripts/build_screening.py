#!/usr/bin/env python3
"""Build the DV screening subset: ~1/3 of train in contiguous blocks with gaps.

Derived artifact only (Shared Dataset Rule): a path-list txt + screening
yamls that point at the shared image trees; no images are copied. Selection
for screening runs stays on the official val. Deterministic: sorted frame
order, ON=400 / OFF=800 blocks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from gfyolo.dv_coco import split_images  # noqa: E402

ON, OFF = 400, 800


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", type=int, default=ON)
    ap.add_argument("--off", type=int, default=OFF)
    args = ap.parse_args()

    ir = split_images("ir", "train")
    rgb = split_images("rgb", "train")
    assert [p.name for p in ir] == [p.name for p in rgb], "pair order mismatch"

    # contiguous blocks with gaps over the sorted train order
    keep, blocks = [], []
    start = None
    for i, p in enumerate(ir):
        if (i % (args.on + args.off)) < args.on:
            keep.append(p)
            if start is None:
                start = i
        elif start is not None:
            blocks.append((start, i - 1))
            start = None
    if start is not None:
        blocks.append((start, len(ir) - 1))

    out_dir = EXP / "data" / "screening"
    out_dir.mkdir(parents=True, exist_ok=True)
    root = EXP / "data" / "dronevehicle_v2" / "yolo"
    # absolute per-modality lists: ultralytics resolves txt entries against
    # the txt location, not the yaml `path` root, so absolute paths are the
    # unambiguous form (generated files may contain resolved absolute paths
    # per the Shared Dataset Rule; the generator derives them from the repo
    # root and nothing is copied).
    txt_ir = out_dir / f"dv_v2_train_on{args.on}_off{args.off}_ir.txt"
    txt_rgb = out_dir / f"dv_v2_train_on{args.on}_off{args.off}_rgb.txt"
    txt_ir.write_text("\n".join(str(root / "ir" / "images" / "train" / p.name)
                                 for p in keep) + "\n")
    txt_rgb.write_text("\n".join(str(root / "rgb" / "images" / "train" / p.name)
                                  for p in keep) + "\n")

    screening = {
        "n_train_total": len(ir), "n_screening": len(keep),
        "fraction": round(len(keep) / len(ir), 4),
        "block_pattern": f"ON={args.on} OFF={args.off} over sorted order",
        "blocks": [f"{a}-{b}" for a, b in blocks],
        "list_ir": str(txt_ir), "list_rgb": str(txt_rgb),
        "pair_check": "ir/rgb stems identical",
        "val_test": "untouched; selection stays on official val",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    audit = out_dir / "screening_audit.json"
    audit.write_text(json.dumps(screening, indent=1))

    yamls = {
        "dronevehicle_v2_ir_screening.yaml": (root / "ir", txt_ir),
        "dronevehicle_v2_rgb_screening.yaml": (root / "rgb", txt_rgb),
        "dronevehicle_v2_paired_screening.yaml": (root / "ir", txt_ir),
    }
    for name, (mod_root, txt) in yamls.items():
        body = (f"path: {mod_root}\n"
                f"train: {txt}\n"
                f"val: images/val\n\n"
                f"nc: 5\nnames:\n  0: car\n  1: truck\n  2: bus\n  3: van\n  4: freight_car\n")
        if "paired" in name:
            body += "\nchannels: 6\n"
        (EXP / "configs" / name).write_text(body)
    print(f"[screening] kept {len(keep)}/{len(ir)} train frames "
          f"({screening['fraction']*100:.1f}%) in {len(blocks)} blocks -> {txt_ir}")


if __name__ == "__main__":
    main()
