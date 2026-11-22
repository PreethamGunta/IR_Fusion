#!/usr/bin/env python3
"""Apply a trained CycleGAN generator to translate clear FLIR ADAS thermal images.

Adapted from m3fd_experiments/method3_cyclegan/translate.py.

Writes translated images into translated/<config>/<split>/ preserving filenames,
so inherited YOLO labels stay aligned. Includes an edge-displacement check
that flags when the generator has moved structure (which would make inherited
labels unreliable).

Usage:
    python translate.py --config fog --split train
    python translate.py --config fog --split val
    python translate.py --config fog --split test
    python translate.py --config rain --limit 10 --check-shift
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "experiments" / "03_m3fd_weather" / "method3_cyclegan"))
from models import define_G

FLIR_DATA = REPO / "data" / "flir_adas_v2" / "FLIR_ADAS_v2"
SHIFT_SEARCH = 3  # +/- px window for edge-displacement search


def edge_shift(a, b):
    """Mean L2 shift of gradient magnitude between two images, in pixels.

    A crude but effective proxy for 'did the generator move structure?'.
    Real fog reduces edge contrast; it should not move edges.
    """
    ga = cv2.Laplacian(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    gb = cv2.Laplacian(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    best = None
    for dy in range(-SHIFT_SEARCH, SHIFT_SEARCH + 1):
        for dx in range(-SHIFT_SEARCH, SHIFT_SEARCH + 1):
            shifted = np.roll(np.roll(gb, dy, axis=0), dx, axis=1)
            d = float(np.abs(ga - shifted).mean())
            if best is None or d < best[0]:
                best = (d, dy, dx)
    return best


def load_source_files(split):
    """Load source image filenames for a given split.

    FLIR ADAS v2 has:
      images_thermal_train/data/*.jpg  (train)
      images_thermal_val/data/*.jpg    (val)
      video_thermal_test/*.jpg         (test)
    """
    if split == "train":
        d = FLIR_DATA / "images_thermal_train" / "data"
    elif split == "val":
        d = FLIR_DATA / "images_thermal_val" / "data"
    elif split == "test":
        d = FLIR_DATA / "video_thermal_test" / "data"
    else:
        raise ValueError(f"Unknown split: {split}")

    if not d.is_dir():
        raise FileNotFoundError(f"Source dir missing for split {split}: {d}")

    files = sorted(d.glob("*.jpg"))
    if not files:
        files = sorted(d.glob("*.png"))
    return files


def read_image(path):
    """Read image as RGB float32 tensor in [-1, 1]."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(f"cv2 could not decode {path}")

    if raw.dtype.itemsize > 1:
        raw = (raw.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)

    if raw.ndim == 2:
        img = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)

    arr = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(arr * 2.0 - 1.0)


def main():
    global FLIR_DATA
    ap = argparse.ArgumentParser(description="Translate FLIR ADAS images through CycleGAN")
    ap.add_argument("--config", required=True, choices=["fog", "rain", "snow"],
                    help="weather condition (matches training config)")
    ap.add_argument("--weights", type=Path, default=None,
                    help="generator .pt (default: runs/<config>/G_AB.pt)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="limit images (for testing)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--norm", default="instance", choices=("instance", "batch"))
    ap.add_argument("--n-blocks", type=int, default=9)
    ap.add_argument("--source-root", type=Path, default=FLIR_DATA,
                    help="FLIR ADAS v2 root")
    ap.add_argument("--check-shift", action="store_true",
                    help="report mean edge displacement input vs output")
    args = ap.parse_args()

    if args.source_root.name == "FLIR_ADAS_v2":
        FLIR_DATA = args.source_root
    else:
        FLIR_DATA = args.source_root / "FLIR_ADAS_v2"

    w = args.weights
    if w is None:
        w = HERE / "runs" / args.config / "G_AB.pt"
    if not w.exists():
        print(f"ERROR: no generator at {w}\n"
              f"Train it first:  python train_cyclegan.py --config {args.config}")
        return 1

    out_dir = args.out
    if out_dir is None:
        out_dir = HERE / "translated" / args.config / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = define_G(3, 3, 64, args.n_blocks, args.norm).to(dev)
    G.load_state_dict(torch.load(w, map_location=dev, weights_only=False))
    G.eval()
    print(f"loaded {w}  ({sum(p.numel() for p in G.parameters()) / 1e6:.1f}M params)  device={dev}")

    files = load_source_files(args.split)
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"ERROR: no images found for split '{args.split}'")
        return 1

    print(f"{len(files):,d} images from split '{args.split}'\n-> {out_dir}")

    shifts, sizes = [], []
    with torch.no_grad():
        for i in range(0, len(files), args.batch):
            chunk = files[i:i + args.batch]
            arrs = []
            for f in chunk:
                im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if im is None:
                    print(f"  WARN: could not read {f}")
                    arrs.append(None)
                    continue
                if im.dtype.itemsize > 1:
                    im = (im.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
                if im.ndim == 2:
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
                else:
                    im = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2RGB)
                arrs.append(im)

            valid_arrs = [(f, a) for f, a in zip(chunk, arrs) if a is not None]
            if not valid_arrs:
                continue

            batch = torch.from_numpy(
                np.stack([a.transpose(2, 0, 1) for _, a in valid_arrs]).astype(np.float32) / 127.5 - 1.0
            ).to(dev)

            fake = G(batch).clamp(-1, 1).cpu().numpy()
            for (f, a), g in zip(valid_arrs, fake):
                out = ((g.transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
                if out.shape != a.shape:
                    print(f"  SHAPE MISMATCH {f.name}: {a.shape} -> {out.shape} "
                          f"(not resizing -- labels would no longer line up)")
                    continue
                out_path = out_dir / f.name
                if out_path.exists():
                    sizes.append(out.shape)
                    continue
                cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                sizes.append(out.shape)
                if args.check_shift:
                    d, dy, dx = edge_shift(a, out)
                    shifts.append((d, dy, dx))

            if len(files) > 50 and (i // args.batch) % 50 == 0:
                print(f"  {i + len(chunk):6,d}/{len(files):,d}", flush=True)

    n_out = len(list(out_dir.glob("*.png")) + list(out_dir.glob("*.jpg")))
    print(f"\nwrote {n_out:,d} images -> {out_dir}")
    if sizes:
        print(f"output geometry: {sizes[0][1]}x{sizes[0][0]}")

    if args.check_shift and shifts:
        d = np.array([s[0] for s in shifts])
        dy = np.array([s[1] for s in shifts])
        dx = np.array([s[2] for s in shifts])
        print(f"\nedge check over {len(shifts)} images:")
        print(f"  residual   mean={d.mean():.2f}  max={d.max():.2f}")
        print(f"  best shift dy={dy.mean():+.2f}px  dx={dx.mean():+.2f}px")
        at_edge = max(abs(dy.mean()), abs(dx.mean())) >= SHIFT_SEARCH
        if at_edge:
            print(f"  WARNING: best shift is at the edge of the +-{SHIFT_SEARCH}px "
                  f"search window. The edge check is UNINFORMATIVE here.")
        else:
            print("  (real weather lowers edge contrast without moving it; "
                  "a large shift means inherited labels are unreliable)")
        (out_dir / "edge_check.json").write_text(json.dumps(dict(
            config=args.config, split=args.split, source_dir=str(files[0].parent),
            n=len(shifts), residual_mean=float(d.mean()),
            residual_max=float(d.max()), dy_mean=float(dy.mean()),
            dx_mean=float(dx.mean()),
            at_search_boundary=bool(at_edge),
            informative=not bool(at_edge)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
