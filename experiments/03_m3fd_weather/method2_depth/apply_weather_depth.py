#!/usr/bin/env python3
"""Method 2: depth-aware synthetic weather for M3FD thermal frames.

IR ONLY.  The project decided against RGB: M3FD evaluation is thermal-only, all
M3FD eval YAMLs point at IR image paths, and no RGB number is on the books.  An
earlier version of this script therefore had nothing to do -- it read its
source images from `data/split/RGB`, which `resplit_m3fd.py` never creates, so
`if not rgb_img_dir.exists(): continue` fired for every split, it generated
**zero** images, it still wrote a YAML, and the pipeline then died on
`FileNotFoundError: .../data/synthetic_depth/fog_medium/IR/images/test` while
the whole downstream half of the run (Methods 2 and 3, the gap table) was
silently lost. Depth is therefore estimated from the thermal frame itself.

Three other fixes ride along, because each one made the method's "gaps"
uninterpretable:

* GROUND TRUTH WAS BEING EDITED.  `process_labels` dropped every box whose mean
  transmission was below 0.1, so Protocol 3A compared the clear test set scored
  against full labels with the fog test set scored against a *filtered* label
  set. Part of any measured "fog gap" was then objects removed from the
  denominator rather than objects missed. Filtered labels are now opt-in
  (`--drop-obscured`) and the evaluation sets never use them; if you want to know
  how many objects fog makes undetectable, ask for that separately instead of
  folding it into the gap.

* DEPTH WAS NORMALISED PER IMAGE.  `depth = (depth - depth.min()) / (depth.max()
  - depth.min())` makes the nearest and farthest pixel of *every* frame 0 and 1
  no matter what the scene contains, so "fog density" was not comparable between
  a frame of a distant skyline and a frame taken from under a bridge -- the
  severity knob meant something different for each image. Normalisation is now
  global, from percentiles measured once on a calibration sample of the split
  and recorded in the run's YAML, so a given `beta` is the same physical thing
  everywhere.

* THE MODEL WAS PULLED AT RUNTIME.  `depth-anything/Depth-Anything-V2-Small-hf`
  was fetched from the Hub inside a SLURM job with no revision pinned, so a moved
  branch or a network outage either fails the job or silently changes the depth
  estimator between two runs. The revision is pinned below and the model must
  already be in the local cache (`--allow-download` to override).

Physical model
--------------
Koschmieder for fog: I(x) = t(x) I(x) + (1 - t(x)) A, with t = exp(-beta * d),
d = normalised *distance* (1 - relative depth) and A the ambient the frame washes
out to.

A is the frame's own mean radiance, not a constant. It used to be a hardcoded
128, against a scene whose mean is 86 -- so the model pulled every pixel toward a
level above the scene and turned fog into an exposure change: at heavy beta the
expected output mean was ~19 grey levels brighter than the input, over half a
frame standard deviation, while Method 1's fog is exactly mean-preserving. The two
methods' fog arms therefore differed by a constant pedestal before depth was even
considered, and any Method-1-vs-Method-2 fog comparison was confounded by it.
With A = frame mean, Koschmieder reduces to `mean + t * (I - mean)`: the same
contrast collapse Method 1 applies, except the retention varies per pixel with
depth instead of being uniform. That is the comparison the experiment is for.
Pass `--airlight N` to force an explicit ambient instead; the pedestal that
introduces is measured and recorded.

`beta_ir = 0.3 * beta` is gone. Extinction is a property of the medium, not the
sensor, and `d` here is in units of relative depth rather than metres, so a fixed
0.3 ratio had no quantity to act on and no citation. Severity is now set by
`severity.contrast_transfer` and enforced per frame by measurement
(`severity.retention_for_target_ctr`), which is what makes beta a *shape*
parameter -- how spatially varying the contrast loss is -- rather than a severity
knob.

Rain: visible-domain rain streaks do not exist in thermal, so IR rain is modelled
as depth-dependent scattering -- the longer the path between camera and object, the
more the contrast is washed out -- plus a near-field droplet layer. The droplet
layer is added uniformly. It used to be multiplied by the scene's transmission,
which is not defensible: a droplet 2 m from the camera is attenuated over its own
path and looks the same whether the scene behind it is 5 m or 50 m away, so the
scene's `t` has no business scaling it. Depth stays in the contrast term, where it
belongs.

The previous IR path was a plain `GaussianBlur(k)` with no depth term at all,
which made Method 2's rain IR *byte-identical to Method 1's* and Method 2 not a
depth method on the thermal channel.
"""

import argparse
import json
import os
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    from transformers import pipeline
except ImportError:
    print("transformers not installed: pip install transformers", file=sys.stderr)
    sys.exit(1)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import provenance
import severity

M3FD_NAMES = ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]

# Pinned so two runs cannot silently use two different depth estimators. This is
# the revision already present in the local HF cache.
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"


# ── physics ───────────────────────────────────────────────────────────────────

def koschmieder(img, t, airlight):
    """I(x) = t(x) I(x) + (1 - t(x)) A, applied per pixel."""
    t3 = np.expand_dims(t, axis=-1)
    return np.clip(img.astype(np.float32) * t3 + airlight * (1 - t3), 0, 255).astype(np.uint8)


def ambient_of(img, airlight):
    """Airlight A. `auto` uses the frame's own mean radiance.

    With A = mean the Koschmieder pedestal vanishes and the model becomes a pure
    contrast collapse, which is what makes Method 2 comparable with Method 1.
    """
    if airlight == "auto":
        return float(np.mean(img))
    return float(airlight)


def transmission(depth, beta):
    """t(x) = exp(-beta * distance); higher relative depth == closer."""
    if depth.ndim == 3:
        # Depth-Anything may hand back a 3-channel map; the physics is per-pixel.
        depth = depth[..., 0]
    return np.exp(-beta * (1.0 - depth))


# Beta ladder. This is a *shape* parameter, not a severity knob: it sets how much
# the contrast loss varies between near and far pixels, and severity itself is set
# by severity.CONTRAST_TRANSFER and then enforced per frame by measurement. Kept
# identical for fog and rain so the two conditions are not additionally confounded
# by different extinction.
BETA = {"light": 1.5, "medium": 3.0, "heavy": 4.5}

# Rain severity ladder. Same streak counts and amplitudes as Method 1, so the two
# arms differ only in the depth modulation -- the thing under test.
RAIN_STREAKS = {"light": 120, "medium": 320, "heavy": 700}
RAIN_AMPLITUDE = {"light": 12.0, "medium": 22.0, "heavy": 34.0}
RAIN_ANGLE_JITTER = 12.0     # degrees either side of vertical


def _streak_layer(shape, n_streaks, amplitude, seed):
    """Additive droplet layer: short near-vertical segments, lightly blurred."""
    h, w = shape[:2]
    rng = np.random.default_rng(seed)
    layer = np.zeros((h, w), np.float32)
    length = rng.integers(12, 40, n_streaks)
    for i in range(n_streaks):
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        angle = np.deg2rad(90.0 + rng.normal(0.0, RAIN_ANGLE_JITTER))
        dx, dy = np.cos(angle), -np.sin(angle)
        seg = np.zeros((h, w), np.float32)
        cv2.line(seg, (x, y), (int(x + dx * int(length[i])), int(y + dy * int(length[i]))),
                 1.0, thickness=int(rng.integers(1, 3)))
        layer += seg * float(rng.uniform(-amplitude, amplitude))
    return cv2.GaussianBlur(layer, (3, 3), 0)


def rain_ir_depth_aware(img, depth, severity_name, seed=0, target_ctr=None):
    """Depth-scaled thermal rain: depth-varying contrast loss plus droplet texture.

    Two separable terms.

    The contrast term is the Koschmieder retention `t = exp(-beta * distance)`,
    so near objects keep their contrast and distant ones lose more. Its scale is
    solved per frame so the frame lands on the shared contrast-transfer target;
    that keeps the *shape* (depth-varying) while making the *amount* comparable
    with fog and with Method 1. It used to be rescaled by `sigma / t.mean()`, a
    scene-independent fudge that could not hold across conditions.

    The droplet layer is added uniformly and afterwards. Droplets between camera
    and scene are a near-field effect: their radiance attenuates over their own
    path and does not depend on the scene's distance. Multiplying the layer by the
    scene's `t` -- as this used to -- both dimmed rain in front of exactly the
    objects the atmosphere had already washed out, and contradicted the comment
    above it. Depth belongs in the contrast term, and that is where it now is.
    """
    t = np.clip(transmission(depth, BETA[severity_name]), 0.0, 1.0)
    out = img.astype(np.float32)
    if target_ctr is not None:
        t = severity.retention_for_target_ctr(out, t, float(out.std()), target_ctr)
    mean = np.mean(out, axis=(0, 1), keepdims=True)
    spread = (lambda m: m[..., None]) if out.ndim == 3 else (lambda m: m)
    out = mean + spread(t) * (out - mean)
    out = out + spread(_streak_layer(out.shape[:2], RAIN_STREAKS[severity_name],
                                    RAIN_AMPLITUDE[severity_name], seed))
    return np.clip(out, 0, 255).astype(np.uint8)


def filter_labels(src_label: Path, dest_label: Path, t_map, threshold):
    """Copy labels, optionally dropping boxes fog has made undetectable.

    Only ever called with --drop-obscured. The default path is a plain copy so
    that the degraded evaluation set is scored against the same ground truth as
    the clear one.

    A missing source label is fatal, not an empty file. Method 1 raises on the
    same condition; writing a zero-byte label here instead deleted that frame's
    ground truth while still satisfying the image/label parity check below, which
    made the parity check agree with a corrupted dataset.
    """
    if not src_label.exists():
        raise SystemExit(
            f"{src_label} missing -- image/label parity broken. Refusing to "
            f"publish a dataset that silently deletes ground truth.")
    lines = src_label.read_text().splitlines(keepends=True)
    if t_map is None:
        dest_label.parent.mkdir(parents=True, exist_ok=True)
        dest_label.write_text("".join(lines))
        return len(lines), 0

    h, w = t_map.shape
    kept, dropped = [], 0
    for line in lines:
        p = line.strip().split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        x1, x2 = max(0, int((cx - bw / 2) * w)), min(w, int((cx + bw / 2) * w))
        y1, y2 = max(0, int((cy - bh / 2) * h)), min(h, int((cy + bh / 2) * h))
        box_t = float(np.mean(t_map[y1:y2, x1:x2])) if (x2 > x1 and y2 > y1) else 0.0
        if box_t >= threshold:
            kept.append(line)
        else:
            dropped += 1
    dest_label.parent.mkdir(parents=True, exist_ok=True)
    dest_label.write_text("".join(kept))
    return len(kept), dropped


# ── depth ─────────────────────────────────────────────────────────────────────

def load_estimator(device, allow_download):
    print(f"Loading {DEPTH_MODEL} @ {DEPTH_REVISION[:12]} on device {device} ...")
    try:
        return pipeline("depth-estimation", model=DEPTH_MODEL,
                        revision=DEPTH_REVISION, device=device)
    except Exception as exc:
        if allow_download:
            raise
        raise SystemExit(
            f"could not load the pinned depth model offline ({type(exc).__name__}: "
            f"{exc}). Pre-fetch it on a networked node or pass --allow-download.")


def depth_of(estimator, img_bgr):
    """Relative depth for one BGR uint8 frame, higher == closer."""
    from PIL import Image
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out = estimator(Image.fromarray(rgb))["depth"]
    return np.asarray(out, dtype=np.float32)


# A depth distribution fitted from too few frames of a continuous drive is not a
# depth distribution. Four frames were accepted silently once and the resulting
# (lo, hi) defined the depth scale for a whole dataset.
MIN_CALIB_FRAMES = 8


def calibrate(estimator, files, n_frames, lo_pct, hi_pct):
    """One global depth range, fitted over every split being generated.

    Fitted once over the union of the requested splits and reused for all of them.
    It used to be called inside the per-split loop, so train, val and test each got
    their own `(lo, hi)` -- which contradicts the normalisation rationale in this
    module's docstring and means a given `beta` was a different physical thing on
    each split. A 3B model trained on train-normalised frames and evaluated on
    test-normalised frames was exactly that case.

    Pixel stride keeps the per-frame memory cost down, but `pool` still grows
    linearly in `n_frames`; only the per-frame term is independent of it.
    """
    if n_frames < MIN_CALIB_FRAMES:
        raise SystemExit(
            f"--calib-frames {n_frames} is below the minimum of {MIN_CALIB_FRAMES}. "
            f"A global depth range fitted from that few frames of a continuous "
            f"drive is not a depth range.")
    stride = max(1, len(files) // n_frames)
    sample = files[::stride][:n_frames]
    pool = []
    print(f"  calibrating on {len(sample)} frames ...")
    for f in tqdm(sample, desc="calibration"):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        d = depth_of(estimator, img)[::4, ::4]
        pool.append(d.ravel())
    if not pool:
        raise SystemExit("depth calibration produced no samples -- unreadable images?")
    joined = np.concatenate(pool)
    lo, hi = float(np.percentile(joined, lo_pct)), float(np.percentile(joined, hi_pct))
    if hi <= lo:
        raise SystemExit(f"degenerate depth calibration: lo={lo} hi={hi}")
    print(f"  global depth percentiles: p{lo_pct}={lo:.1f} p{hi_pct}={hi:.1f}")
    return lo, hi, len(sample)


def normalise(depth, lo, hi):
    """Clip to the global range and map to [0, 1] -- same scale for every frame."""
    return np.clip((depth - lo) / (hi - lo), 0.0, 1.0)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ROOT = Path(__file__).resolve()
    while not (ROOT / "data" / "flir_adas_v2").exists() and ROOT != ROOT.parent:
        ROOT = ROOT.parent
    ap.add_argument("--src", default=str(ROOT / "data" / "m3fd" / "split" / "IR"))
    ap.add_argument("--dest", default=str(ROOT / "data" / "m3fd" / "synthetic_depth"))
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--severity", default="heavy", choices=["light", "medium", "heavy"])
    ap.add_argument("--condition", default="fog", choices=["fog", "rain"])
    ap.add_argument("--calib-frames", type=int, default=32)
    ap.add_argument("--calib-lo-pct", type=float, default=1.0)
    ap.add_argument("--calib-hi-pct", type=float, default=99.0)
    ap.add_argument("--airlight", default="auto",
                    help="Koschmieder ambient. 'auto' (default) uses each frame's "
                         "own mean radiance, which makes the model mean-preserving "
                         "and comparable with Method 1. A number forces an explicit "
                         "ambient and reintroduces a brightness pedestal; the "
                         "pedestal is measured and recorded either way.")
    ap.add_argument("--drop-obscured", action="store_true",
                    help="drop GT boxes whose mean transmission is below "
                         "--threshold. OFF by default: filtering the labels of a "
                         "degraded *evaluation* set makes the resulting gap "
                         "partly a change of denominator.")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--allow-download", action="store_true",
                    help="permit fetching the pinned model from the Hub")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="smoke-test only: process at most N frames per split. "
                         "The resulting YAML will not match a full split, so "
                         "nothing downstream should consume it.")
    args = ap.parse_args()

    src = Path(args.src)
    dest_base = Path(args.dest) / f"{args.condition}_{args.severity}"
    beta = BETA[args.severity]
    target_ctr = severity.contrast_transfer(args.severity)
    auto_airlight = str(args.airlight).lower() == "auto"
    if not auto_airlight:
        try:
            args.airlight = float(args.airlight)
        except ValueError:
            raise SystemExit(f"--airlight must be 'auto' or a number, got "
                             f"{args.airlight!r}")

    if not (src / "images").is_dir():
        raise SystemExit(f"no IR images under {src / 'images'} -- pass --src "
                         f"data/m3fd/split/IR")

    # Set before the pipeline is built: with the revision pinned, an offline run
    # should use the local cache rather than reaching for the Hub and hanging
    # until the job's wall clock runs out.
    os.environ.setdefault("HF_HUB_OFFLINE", "0" if args.allow_download else "1")

    device = 0 if torch.cuda.is_available() else -1
    estimator = load_estimator(device, args.allow_download)

    # Collect every split's files first, so one calibration covers all of them.
    per_split = {}
    for split in args.splits:
        img_dir = src / "images" / split
        files = sorted(img_dir.glob("*.png"))
        if not files:
            raise SystemExit(f"{img_dir}: no *.png (the split is not built?)")
        per_split[split] = (files, src / "labels" / split)

    degradation = severity.DegradationLog()
    try:
        # Calibrate on the *full* splits, not the --max-frames prefix, and once for
        # all of them: the global depth range is what makes "fog density"
        # comparable between frames and between splits.
        calib_files = [f for files, _ in per_split.values() for f in files]
        lo, hi, n_cal = calibrate(estimator, calib_files, args.calib_frames,
                                  args.calib_lo_pct, args.calib_hi_pct)

        for split, (files, lbl_dir) in per_split.items():
            if args.max_frames:
                files = files[:args.max_frames]

            dest_img = dest_base / "IR" / "images" / split
            dest_lbl = dest_base / "IR" / "labels" / split
            dest_img.mkdir(parents=True, exist_ok=True)
            dest_lbl.mkdir(parents=True, exist_ok=True)

            print(f"Processing {split}: {len(files)} frames -> {dest_img}")
            n_written = n_boxes = n_dropped = 0
            for img_path in tqdm(files, desc=f"{args.condition}_{args.severity}/{split}"):
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    raise SystemExit(f"unreadable source image {img_path}")
                depth = normalise(depth_of(estimator, img), lo, hi)

                if args.condition == "fog":
                    t_ir = np.clip(transmission(depth, beta), 0.0, 1.0)
                    if auto_airlight:
                        # A = frame mean, so Koschmieder collapses to a contrast
                        # contraction about the mean. Scale `t` so the frame lands
                        # on the shared contrast target and keep its depth shape.
                        t_ir = severity.retention_for_target_ctr(
                            img.astype(np.float32), t_ir,
                            float(img.astype(np.float32).std()), target_ctr)
                        m = np.mean(img, axis=(0, 1), keepdims=True)
                        out = np.clip(m + t_ir[..., None]
                                      * (img.astype(np.float32) - m),
                                      0, 255).astype(np.uint8)
                    else:
                        out = koschmieder(img, t_ir, args.airlight)
                else:
                    t_ir = None
                    seed = int(img_path.stem) if img_path.stem.isdigit() \
                        else zlib.crc32(img_path.stem.encode())
                    out = rain_ir_depth_aware(img, depth, args.severity, seed=seed,
                                              target_ctr=target_ctr)

                if not cv2.imwrite(str(dest_img / img_path.name), out):
                    raise SystemExit(f"failed to write {dest_img / img_path.name}")
                degradation.add(img, out)
                n_written += 1

                kept, dropped = filter_labels(
                    lbl_dir / f"{img_path.stem}.txt",
                    dest_lbl / f"{img_path.stem}.txt",
                    t_ir if args.drop_obscured else None,
                    args.threshold)
                n_boxes += kept
                n_dropped += dropped

            n_written_labels = len(list(dest_lbl.glob("*.txt")))
            if n_written != len(files) or n_written_labels != len(files):
                raise SystemExit(
                    f"{split}: wrote {n_written} images and {n_written_labels} "
                    f"labels for {len(files)} source frames -- refusing to "
                    f"publish a dataset whose image/label counts disagree.")
            print(f"  {split}: {n_written} images, {n_boxes} boxes"
                  + (f", {n_dropped} dropped as obscured" if args.drop_obscured else ""))
    finally:
        del estimator

    # Dataset YAML, built explicitly rather than by string-replacing the clear
    # split's YAML -- which silently inherits whatever train/val the source had.
    splits_done = args.splits
    resolved_test = "test" if "test" in splits_done else splits_done[-1]
    yaml_lines = ["nc: 6", "names:"]
    yaml_lines += [f"  - {n}" for n in M3FD_NAMES]
    yaml_lines.append(f"train: {dest_base.absolute()}/IR/images/"
                      f"{'train' if 'train' in splits_done else splits_done[0]}")
    yaml_lines.append(f"val: {dest_base.absolute()}/IR/images/"
                      f"{'val' if 'val' in splits_done else splits_done[0]}")
    yaml_lines.append(f"test: {dest_base.absolute()}/IR/images/{resolved_test}")
    yaml_dest = dest_base / f"m3fd_{args.condition}_{args.severity}.yaml"
    yaml_dest.parent.mkdir(parents=True, exist_ok=True)
    yaml_lines.append(f"# test: aliases '{resolved_test}' -- this dataset was "
                      f"generated with --splits {' '.join(splits_done)}")
    yaml_dest.write_text("\n".join(yaml_lines) + "\n")

    meta = {
        "method": "method2_depth",
        "condition": args.condition,
        "severity": args.severity,
        "modality": "IR only",
        "source": str(src),
        "depth_model": DEPTH_MODEL,
        "depth_revision": DEPTH_REVISION,
        "severity_defined_by": "contrast_transfer_ratio",
        "contrast_transfer_target": target_ctr,
        # beta shapes *how much the retention varies with depth*; the contrast
        # actually lost is set by the target above and enforced per frame.
        "beta": beta,
        "airlight": "auto (frame mean)" if auto_airlight else args.airlight,
        "depth_range": {"lo": round(lo, 3), "hi": round(hi, 3),
                        "calibration_frames": n_cal,
                        "percentiles": [args.calib_lo_pct, args.calib_hi_pct],
                        "scope": "global: fitted once over all requested splits"},
        "rain_streaks": RAIN_STREAKS[args.severity] if args.condition == "rain" else None,
        "rain_amplitude": RAIN_AMPLITUDE[args.severity] if args.condition == "rain" else None,
        "labels_filtered": bool(args.drop_obscured),
        "label_threshold": args.threshold if args.drop_obscured else None,
        "splits": splits_done,
        "test_split_resolved": resolved_test,
        "test_is_aliased": resolved_test != "test",
        "frames_processed": degradation.count,
        "degradation": degradation.summary(),
    }
    meta.update(provenance.run_record(
        split_hash=provenance.split_hash_for_dataset(src.parent / "m3fd_clear.yaml")))
    (dest_base / "generation.json").write_text(json.dumps(meta, indent=2))

    stamped = provenance.stamp_split(
        yaml_dest, manifest_src=src.parent / "split_manifest.json")
    if not stamped:
        raise SystemExit(
            f"could not stamp {yaml_dest} with a split manifest. Every result "
            f"from this dataset would be untraceable to a split.")

    print(f"\nGenerated YAML: {yaml_dest}  (split_hash {stamped})")
    print(f"Generation metadata: {dest_base / 'generation.json'}")


if __name__ == "__main__":
    main()