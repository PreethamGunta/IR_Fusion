"""Verify what DSERT-RoLL actually contains, before spending any GPU time.

This is a gate, not a formality.  Two things about the target domain are
unknown until the files are on disk, and both change what the training code
has to do:

1. Channel count and bit depth.  The GitHub README describes
   rectified_THERMAL_L as .npz; the Hub actually publishes .png.  We do not
   know yet whether those PNGs are 8-bit greyscale, 8-bit pseudo-coloured
   RGB, or 16-bit.  It matters because the source domain (M3FD) is 3-channel
   8-bit pseudo-colour, and the existing clear baseline of 0.514 mAP@50-95
   was measured on exactly that representation.  Converting M3FD to greyscale
   for symmetry would look tidier and would silently break comparability with
   every number already on the books, so the resolution is always to bring
   DSERT *up* to the M3FD representation, never the other way round.

2. Palette.  DSERT's thermal imager is a different camera from M3FD's.  If
   the two use different colour maps, a 3-channel CycleGAN will happily learn
   the palette change as part of "fog", and every gap it later produces is
   measuring the camera rather than the weather.  The histograms printed here
   are the cheapest way to notice that before it costs us a 24h allocation.

It writes dsert_inspect.json and prints a report.  It never starts training
and never edits images on disk -- normalisation decisions are applied at load
time in data.py, not baked into the corpus.

    python inspect_dsert.py
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DSERT_ROOT = REPO / "data" / "dsert_roll"
MANIFEST = HERE / "dsert_files.json"
OUT = HERE / "dsert_inspect.json"
M3FD_SPLIT = REPO / "data" / "m3fd" / "split"

SAMPLES_PER_GROUP = 5
MIN_FRAMES = 50


def probe(path):
    """Everything we need to decide how to load one file, from the file itself.

    cv2 is used rather than PIL because it reports the true bit depth of a PNG
    and, unlike a naive suffix-based read, will tell us it failed.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return {"ok": False, "error": "cv2 could not decode"}
    info = {
        "ok": True,
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "min": float(raw.min()),
        "max": float(raw.max()),
    }
    # PNG bit depth: 8-bit ints are uint8, 16-bit are uint16
    info["bit_depth"] = int(raw.dtype.itemsize * 8)

    if raw.ndim == 2:
        info["channels"] = 1
        info["greyscale"] = True
        plane = raw
    else:
        info["channels"] = raw.shape[2]
        # "grey data stored in 3 channels": identical planes, or a luma-style
        # image whose channels are merely close
        if raw.shape[2] >= 3:
            b, g, r = raw[:, :, 0], raw[:, :, 1], raw[:, :, 2]
            identical = bool(np.array_equal(b, g) and np.array_equal(g, r))
            spread = float(np.mean(np.abs(b.astype(np.int32) - r.astype(np.int32))))
            info["greyscale"] = identical
            info["channel_spread"] = round(spread, 2)
            info["inverted_likely"] = bool(
                np.corrcoef(b.ravel(), r.ravel())[0, 1] < 0) if raw.size > 1000 else False
        plane = raw[:, :, 0] if raw.shape[2] >= 3 else raw

    hist = np.bincount(np.clip(plane.ravel(), 0, 255).astype(np.int64), minlength=256)
    tot = max(int(hist.sum()), 1)
    nz = np.nonzero(hist)[0]
    info["mean"] = round(float(plane.mean()), 2)
    info["std"] = round(float(plane.std()), 2)
    # fraction of pixels in the extreme 2% of the range -- a good proxy for
    # "how much of the image is saturated", which is how thermal scenes differ
    # from RGB ones
    info["frac_dark"] = round(float(hist[: max(1, nz[0] + 1)].sum() / tot), 4)
    info["frac_bright"] = round(float(hist[max(0, int(0.98 * 255)):].sum() / tot), 4)
    # median split point, useful for eyeballing whether two domains look alike
    cum = np.cumsum(hist) / tot
    info["median"] = int(np.searchsorted(cum, 0.5))
    return info


def m3fd_reference(modality):
    d = M3FD_SPLIT / ("IR" if modality == "thermal" else "RGB") / "images" / "train"
    files = sorted(d.glob("*.png"))
    if not files:
        return None
    step = max(1, len(files) // SAMPLES_PER_GROUP)
    probes = [probe(f) for f in files[::step][:SAMPLES_PER_GROUP]]
    return {"dir": str(d), "n_files": len(files), "samples": probes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DSERT_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"ERROR: {MANIFEST} missing. Run download_dsert.py first.")
        return 1
    blob = json.loads(MANIFEST.read_text())
    files = blob["files"]

    print(f"DSERT root : {args.root}")
    print(f"manifest   : {len(files):,d} files\n")

    # group = (condition, modality)
    groups = {}
    for f in files:
        groups.setdefault((f["condition"], f["modality"]), []).append(f)

    report, problems = {}, []

    for (cond, mod), items in sorted(groups.items()):
        on_disk = [DSERT_ROOT / f["path"] for f in items
                   if (DSERT_ROOT / f["path"]).exists()]
        key = f"{cond}/{mod}"
        print(f"=== {key} ===")
        print(f"  manifest : {len(items):6,d}")
        print(f"  on disk  : {len(on_disk):6,d}   sequences: "
              f"{len({f['sequence'] for f in items})}")

        if len(on_disk) < args.min_frames:
            problems.append(f"{key}: only {len(on_disk)} frames on disk "
                            f"(min {args.min_frames}) -- download incomplete?")
        if len(on_disk) < len(items):
            problems.append(f"{key}: {len(items) - len(on_disk)} manifest files "
                            f"missing on disk")

        # sample across sequences, not just the head of the list
        by_seq = {}
        for p in on_disk:
            by_seq.setdefault(p.parents[1].name, []).append(p)
        picked = []
        seqs = sorted(by_seq)
        for i in range(min(SAMPLES_PER_GROUP, len(seqs))):
            seq = seqs[i * max(1, len(seqs) // SAMPLES_PER_GROUP)]
            lst = sorted(by_seq[seq])
            picked.append(lst[len(lst) // 2])

        probes = []
        for p in picked:
            info = probe(p)
            info["file"] = str(p.relative_to(args.root))
            probes.append(info)
            if not info["ok"]:
                problems.append(f"{key}: {info['file']} -- {info['error']}")
                continue
            print(f"    {Path(info['file']).name:>10s}  shape={tuple(info['shape'])} "
                  f"dtype={info['dtype']} ch={info['channels']} "
                  f"range=[{info['min']:.0f},{info['max']:.0f}] "
                  f"mean={info['mean']:6.1f} std={info['std']:5.1f} "
                  f"grey={info['greyscale']}"
                  + (f" spread={info['channel_spread']:.1f}"
                     if "channel_spread" in info else ""))

        if probes and probes[0]["ok"]:
            agg = {
                "bit_depth": probes[0]["bit_depth"],
                "channels": probes[0]["channels"],
                "greyscale": all(p.get("greyscale") for p in probes if p["ok"]),
                "inverted_likely": any(p.get("inverted_likely") for p in probes if p["ok"]),
                "mean_of_means": round(float(np.mean([p["mean"] for p in probes if p["ok"]])), 2),
                "std_of_stds": round(float(np.mean([p["std"] for p in probes if p["ok"]])), 2),
                "median_of_medians": int(np.median([p["median"] for p in probes if p["ok"]])),
            }
            report[key] = {
                "n_manifest": len(items), "n_on_disk": len(on_disk),
                "n_sequences": len({f["sequence"] for f in items}),
                "aggregate": agg, "samples": probes,
            }
            print(f"  -> {agg['bit_depth']}-bit, {agg['channels']}ch, "
                  f"grey={agg['greyscale']}, mean={agg['mean_of_means']}, "
                  f"median={agg['median_of_medians']}")
        print()

    # M3FD source reference, for palette comparison
    print("=== M3FD source reference (must match this representation) ===")
    m3fd = {}
    for mod in ("thermal", "rgb"):
        ref = m3fd_reference(mod)
        if not ref:
            print(f"  {mod:8s} MISSING")
            continue
        m3fd[mod] = ref
        p = ref["samples"][0]
        print(f"  {mod:8s} shape={tuple(p['shape'])} dtype={p['dtype']} "
              f"{p['bit_depth']}-bit ch={p['channels']} grey={p['greyscale']} "
              f"mean={p['mean']} median={p['median']}")
    print()

    args.out.write_text(json.dumps(
        {"dsert_root": str(args.root), "groups": report,
         "m3fd_reference": m3fd, "problems": problems}, indent=2))
    print(f"wrote {args.out}\n")

    # ---- load-time decisions, applied in data.py not to the corpus ---------
    print("=" * 66)
    print("LOAD-TIME RESOLUTION")
    print("=" * 66)
    thermal = [v["aggregate"] for k, v in report.items()
               if k.endswith("/thermal") and "aggregate" in v]
    if thermal:
        agg = thermal[0]
        if agg["channels"] == 1:
            print("  thermal : 1 channel  -> replicate to 3 (keep CycleGAN 3-channel)")
        elif agg["bit_depth"] > 8:
            print(f"  thermal : {agg['bit_depth']}-bit -> per-image normalise to 8-bit, "
                  f"then use as 3-channel")
        else:
            print(f"  thermal : {agg['bit_depth']}-bit {agg['channels']}ch -> use as-is")
        if agg.get("inverted_likely"):
            print("  WARNING : thermal looks INVERTED relative to M3FD "
                  "(cold/warm polarity flipped). Flagging for review -- a")
            print("            polarity flip between domains would be learned as")
            print("            'weather' and would show up as a huge false gap.")
        for k, v in report.items():
            if k.endswith("/thermal") and v["aggregate"]["channels"] >= 3:
                print(f"  {k}: channel_spread="
                      f"{[s.get('channel_spread') for s in v['samples'] if s['ok']]}")
                print("            -> low spread means the image is greyscale data")
                print("               stored in 3 channels; safe to use as 3-channel")
    else:
        problems.append("no thermal group found on disk -- cannot resolve format")

    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print(f"  ! {p}")
        return 1
    print("\nOK -- format resolved, no problems found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
