"""Apply a trained generator to a directory of images.

Kept separate from training on purpose. This script produces the actual test
sets that Protocol 3A and 3B will consume, and it is where two things that
training cannot check have to be checked:

  * the generator really is fully convolutional, so a model trained on
    512x512 crops produces a usable 1024x768 image with no resampling step
    in between. If it needed resampling, the resize would perturb object
    scale and YOLO would score that as a weather gap.

  * the output geometry still lines up with the YOLO labels. CycleGAN can
    hallucinate content and move edges, and M3FD labels are inherited from
    the clear image rather than re-drawn. That is a validity question for
    the evaluation stage, not something to silently assume here -- so
    --check-shift reports the mean displacement of edge energy between input
    and output, and a large value is a reason to distrust the labels, not a
    reason to silently resize anything.

Nothing here overwrites an input image.

    python translate.py --run thermal_fog --split test --out ../../data/gan_fog
    python translate.py --run thermal_fog --limit 8          # just look at some
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import data as data_mod
from models import define_G

RUNS = {
    "thermal_fog": {"modality": "thermal", "target_conditions": ["Fog"]},
    "thermal_rain": {"modality": "thermal",
                     "target_conditions": ["Heavy_Rain", "Light_Rain"]},
}
M3FD_SPLITS = {
    "thermal": HERE.parent / "data" / "split" / "IR" / "images",
}
SHIFT_SEARCH = 3   # +/- px window for the edge-displacement search


def edge_shift(a, b):
    """Mean L2 shift of gradient magnitude between two images, in pixels.

    A crude but effective proxy for "did the generator move structure?".
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(RUNS))
    ap.add_argument("--weights", type=Path, default=None,
                    help="generator .pt (default: runs/<run>/G_AB.pt)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--norm", default="instance", choices=("instance", "batch"))
    ap.add_argument("--n-blocks", type=int, default=9)
    ap.add_argument("--canonical", action="store_true",
                    help="resample input to M3FD native geometry before "
                         "translating. Off by default: test images must keep "
                         "the exact geometry of the labels they inherit.")
    ap.add_argument("--check-shift", action="store_true",
                    help="report mean edge displacement input vs output")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = RUNS[args.run]
    w = args.weights or (HERE / "runs" / args.run / "G_AB.pt")
    if not w.exists():
        print(f"ERROR: no generator at {w}\n"
              f"Train it first:  python train_cyclegan.py --run {args.run}")
        return 1
    out_dir = args.out or (HERE / "translated" / args.run / args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = define_G(3, 3, 64, args.n_blocks, args.norm).to(dev)
    G.load_state_dict(torch.load(w, map_location=dev, weights_only=False))
    G.eval()
    print(f"loaded {w}  ({sum(p.numel() for p in G.parameters()) / 1e6:.1f}M params)"
          f"  device={dev}")

    src_dir = M3FD_SPLITS[run["modality"]] / args.split
    # *.png only: the split dirs carry a sibling .npy per PNG (Ultralytics'
    # --cache disk output), which is not a decodable image.
    files = sorted(src_dir.glob("*.png"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"ERROR: no *.png under {src_dir}")
        return 1
    print(f"{len(files):,d} images from {src_dir}\n-> {out_dir}")

    shifts, sizes = [], []
    with torch.no_grad():
        for i in range(0, len(files), args.batch):
            chunk = files[i:i + args.batch]
            arrs = []
            for f in chunk:
                im = cv2.imread(str(f), cv2.IMREAD_COLOR)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                arrs.append(im)
            if args.canonical:
                arrs = [data_mod.to_canonical(a) for a in arrs]

            batch = torch.from_numpy(
                np.stack([a.transpose(2, 0, 1) for a in arrs]).astype(np.float32) / 127.5 - 1.0
            ).to(dev)

            fake = G(batch).clamp(-1, 1).cpu().numpy()
            for f, a, g in zip(chunk, arrs, fake):
                out = ((g.transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
                if out.shape != a.shape:
                    # The generator is fully convolutional, so this should be
                    # impossible. Report it loudly rather than resizing: a
                    # silent resize here would decouple geometry from labels.
                    print(f"  SHAPE MISMATCH {f.name}: {a.shape} -> {out.shape} "
                          f"(not resizing -- labels would no longer line up)")
                    continue
                cv2.imwrite(str(out_dir / f.name),
                            cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                sizes.append(out.shape)
                if args.check_shift:
                    d, dy, dx = edge_shift(a, out)
                    shifts.append((d, dy, dx))

            if (i // args.batch) % 50 == 0:
                print(f"  {i + len(chunk):6,d}/{len(files):,d}", flush=True)

    n_out = len(list(out_dir.glob("*.png")))
    print(f"\nwrote {n_out:,d} images -> {out_dir}")
    if sizes:
        print(f"output geometry: {sizes[0][1]}x{sizes[0][0]} "
              f"(matches input: {sizes[0][1] == cv2.imread(str(files[0])).shape[1]})")

    if args.check_shift and shifts:
        d = np.array([s[0] for s in shifts])
        dy = np.array([s[1] for s in shifts])
        dx = np.array([s[2] for s in shifts])
        print(f"\nedge check over {len(shifts)} images:")
        print(f"  residual   mean={d.mean():.2f}  max={d.max():.2f}")
        print(f"  best shift dy={dy.mean():+.2f}px  dx={dx.mean():+.2f}px")
        # If the best shift sits on the edge of the search window the search
        # did not find a true optimum, which means the metric is
        # uninformative -- typically an undertrained generator producing
        # noise. Reporting it as a clean number would be misleading.
        at_edge = max(abs(dy.mean()), abs(dx.mean())) >= SHIFT_SEARCH
        if at_edge:
            print(f"  WARNING: best shift is at the edge of the +-{SHIFT_SEARCH}px "
                  f"search window.")
            print("           The edge check is UNINFORMATIVE here and should not "
                  "be read")
            print("           as a real displacement. Usual cause: the generator "
                  "is")
            print("           undertrained (or is the smoke-test model) and is "
                  "emitting")
            print("           near-noise, so no alignment exists to find.")
        else:
            print("  (real fog lowers edge contrast without moving it; a large or")
            print("   systematically non-zero shift means the labels inherited from")
            print("   the clear image are no longer reliable -- check the box-overlap")
            print("   assertion before trusting any mAP from this split)")
        # Into out_dir, not out_dir.parent: each split writes its own file, and
        # the parent (translated/<run>/images/) is shared, so the previous
        # location had every split overwrite the one before it.
        (out_dir / "edge_check.json").write_text(json.dumps(dict(
            run=args.run, split=args.split, source_dir=str(src_dir),
            n=len(shifts), residual_mean=float(d.mean()),
            residual_max=float(d.max()), dy_mean=float(dy.mean()),
            dx_mean=float(dx.mean()),
            at_search_boundary=bool(at_edge),
            informative=not bool(at_edge)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
