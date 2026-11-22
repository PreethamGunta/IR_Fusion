"""Generate one adverse condition over the clear FLIR splits.

Output layout (regenerable, gitignored via ``experiments/*/data/``)::

    data/generated/<condition>/<severity>/<split>/images/<stem>.jpg
    data/generated/<condition>/<severity>/<split>/labels/<stem>.txt
    data/generated/<condition>/<severity>/generation.json

Run it inside a live SLURM allocation (see ``scripts/generate.sh``).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import conditions, labels, paths, severity

cv2.setNumThreads(1)


def _seed_for(base: int, *parts: str) -> int:
    return (int(base) + zlib.crc32("|".join(parts).encode("utf-8"))) % (2 ** 32)


def _run_job(job: dict) -> dict:
    cv2.setNumThreads(1)
    src_img = Path(job["src_img"])
    src_lbl = Path(job["src_lbl"])
    dst_img = Path(job["dst_img"])
    dst_lbl = Path(job["dst_lbl"])

    img = cv2.imread(str(src_img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"ok": False, "name": src_img.name, "error": "imread returned None"}

    boxes = labels.read_yolo(src_lbl)
    opts = dict(job["opts"])
    opts["_stem"] = src_img.stem
    rng = np.random.default_rng(int(job["seed"]))
    try:
        out, out_boxes, meta = conditions.apply_condition(
            job["condition"], img, boxes, job["severity"], rng, opts,
        )
    except Exception as exc:  # a bad frame must not kill the split
        return {"ok": False, "name": src_img.name, "error": f"{type(exc).__name__}: {exc}"}

    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst_img), out, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        return {"ok": False, "name": src_img.name, "error": "imwrite failed"}
    labels.write_yolo(dst_lbl, out_boxes)

    stats = severity.measure(img, out)
    stats.update(meta)
    stats["boxes_in"] = len(boxes)
    stats["boxes_out"] = len(out_boxes)
    return {"ok": True, "name": src_img.name, "stats": stats}


def _aggregate(rows: list[dict]) -> dict:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {k: round(sums[k] / counts[k], 4) for k in sorted(sums)}


def _expand(value: str, allowed) -> list[str]:
    if value == "all":
        return list(allowed)
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in items if v not in allowed]
    if bad:
        raise SystemExit(f"unknown value(s) {bad}; expected 'all' or a subset of {list(allowed)}")
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate adverse FLIR conditions")
    ap.add_argument("--condition", default="all",
                    help="condition name, comma-separated list, or 'all'")
    ap.add_argument("--severity", default="medium",
                    help="light, medium, heavy, comma-separated, or 'all'")
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="max images per split (0 = all)")
    ap.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--overwrite", action="store_true", help="regenerate existing images")
    ap.add_argument("--variant", default="fog", help="fog variant: fog|mist|humidity")
    ap.add_argument("--atmospheric", type=float, default=None,
                    help="fog path radiance (default: frame mean)")
    ap.add_argument("--depth-dir", default=None, help="optional per-frame depth sidecar dir")
    ap.add_argument("--gamma", type=float, default=None, help="crossover blend factor")
    ap.add_argument("--psf", default="gaussian", choices=["gaussian", "disk"])
    ap.add_argument("--sigma", type=float, default=None, help="defocus Gaussian sigma")
    ap.add_argument("--radius", type=int, default=None, help="defocus disk radius")
    ap.add_argument("--pixels", type=int, default=None, help="small-target k x k pixels")
    ap.add_argument("--occluder", default="mixed", choices=["fence", "foliage", "mixed", "masks"])
    ap.add_argument("--mask-dir", default=None, help="optional per-frame occluder masks")
    ap.add_argument("--min-visible", type=float, default=severity.OCCLUSION_MIN_VISIBLE)
    ap.add_argument("--occluder-value", type=float, default=None)
    ap.add_argument("--fpn-seed", type=int, default=20261110)
    ap.add_argument("--smudge-seed", type=int, default=7)
    args = ap.parse_args()

    conds = _expand(args.condition, conditions.CONDITION_NAMES)
    sevs = _expand(args.severity, severity.SEVERITIES)
    paths.ensure_dirs()

    opts = {
        "variant": args.variant,
        "atmospheric": args.atmospheric,
        "depth_dir": args.depth_dir,
        "inhomogeneous": True,
        "gamma": args.gamma,
        "psf": args.psf,
        "sigma": args.sigma,
        "radius": args.radius,
        "pixels": args.pixels,
        "occluder": args.occluder,
        "mask_dir": args.mask_dir,
        "min_visible": args.min_visible,
        "occluder_value": args.occluder_value,
        "fpn_seed": args.fpn_seed,
        "smudge_seed": args.smudge_seed,
    }

    for condition in conds:
        for sev in sevs:
            written, failed, skipped = 0, [], 0
            split_counts: dict[str, dict] = {}
            rows: list[dict] = []
            for split in args.splits:
                src_img_dir = paths.CLEAR_IMG[split]
                src_lbl_dir = paths.CLEAR_LBL[split]
                if not src_img_dir.is_dir():
                    failed.append({"split": split, "error": f"missing {src_img_dir}"})
                    continue
                images = sorted(
                    p for p in src_img_dir.iterdir() if p.suffix.lower() in paths.IMG_EXTS
                )
                if args.limit:
                    images = images[: args.limit]
                out_dir = paths.condition_dir(condition, sev, split)
                jobs = []
                for src in images:
                    dst_img = out_dir / "images" / f"{src.stem}.jpg"
                    dst_lbl = out_dir / "labels" / f"{src.stem}.txt"
                    if not args.overwrite and dst_img.exists() and dst_lbl.exists():
                        skipped += 1
                        continue
                    jobs.append({
                        "condition": condition,
                        "severity": sev,
                        "src_img": str(src),
                        "src_lbl": str(src_lbl_dir / f"{src.stem}.txt"),
                        "dst_img": str(dst_img),
                        "dst_lbl": str(dst_lbl),
                        "seed": _seed_for(args.seed, condition, sev, split, src.stem),
                        "opts": opts,
                    })
                split_counts[split] = {"available": len(images), "queued": len(jobs)}
                if not jobs:
                    continue
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = [pool.submit(_run_job, job) for job in jobs]
                    for fut in tqdm(as_completed(futures), total=len(futures),
                                    desc=f"{condition}/{sev}/{split}"):
                        result = fut.result()
                        if result["ok"]:
                            written += 1
                            rows.append(result["stats"])
                        else:
                            failed.append({"split": split, **result})

            meta = {
                "condition": condition,
                "severity": sev,
                "spec": conditions.REGISTRY[condition].__dict__,
                "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in opts.items()},
                "splits": split_counts,
                "images_written": written,
                "images_skipped_existing": skipped,
                "images_failed": len(failed),
                "failures": failed[:50],
                "aggregate": _aggregate(rows),
                "seed_base": args.seed,
                "n_images_measured": len(rows),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            }
            out_json = paths.condition_dir(condition, sev) / "generation.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(meta, indent=2))
            print(f"[generate] {condition}/{sev}: wrote {written}, skipped {skipped}, "
                  f"failed {len(failed)} -> {out_json}")
            if failed:
                print(f"[generate]   first failure: {failed[0]}")


if __name__ == "__main__":
    main()
