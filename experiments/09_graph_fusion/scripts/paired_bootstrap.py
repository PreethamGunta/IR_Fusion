#!/usr/bin/env python3
"""Paired image-level bootstrap for the locked test set (experiment 09).

Why this exists: a full pycocotools re-evaluation of 8,980 images costs ~65 s,
so a naive bootstrap (thousands of full re-evaluations) is hours of compute.
This script pays that cost exactly ONCE per model (the per-image matching is
cached from COCOeval.evalImgs), then recomputes the 101-point COCO AP from the
cached per-image TP flags for every resample. That is mathematically the same
computation pycocotools performs in `accumulate()`; the script asserts this
against the full-set pycocotools AP before bootstrapping (fails loudly if the
re-implementation drifts).

Paired = the SAME resampled image indices are used for every model, so the
delta distribution (arm - reference) cancels the image-level difficulty.

Usage:
  python scripts/paired_bootstrap.py \
    --gt results/gt/dv_v2_ir_test.json \
    --model e00_ir=results/preds/e00_ir_test.json \
    --model e10_v2_100ep=results/preds/e10_v2_100ep_test.json \
    --model e30_v2_100ep=results/preds/e30_v2_100ep_test.json \
    --model e35_v2_100ep=results/preds/e35_v2_100ep_test.json \
    --ref e00_ir --n 500 --seed 0 --out results/bootstrap_flagships_vs_e00.json
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

EXP = Path(__file__).resolve().parents[1]


def load_coco(gt: dict) -> tuple[COCO, str]:
    fd, path = tempfile.mkstemp(suffix="_boot_gt.json")
    os.close(fd)
    Path(path).write_text(json.dumps(gt))
    return COCO(path), path


def cache_model(cocoGt: COCO, preds: list[dict], max_det: int = 100) -> dict:
    """Run pycocotools once; return per-(class,image) TP flags + scores."""
    cd = cocoGt.loadRes(preds)
    ev = COCOeval(cocoGt, cd, "bbox")
    ev.params.maxDets = [1, 10, max_det]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()  # fills ev.stats; accumulate() alone leaves it empty
    full_ap = float(ev.stats[0])
    full_ap50 = float(ev.stats[1])

    cat_ids = list(ev.params.catIds)
    img_ids = list(ev.params.imgIds)
    img_pos = {i: k for k, i in enumerate(img_ids)}
    all_rng = ev.params.areaRng[0]
    # layout: evalImgs is [for catId][for areaRng][for imgId]
    per_cat: dict[int, dict] = {c: {"scores": [], "tp": [], "ign": [], "img": [], "npos": np.zeros(len(img_ids), dtype=np.int64)}
                                for c in cat_ids}
    n_entries = 0
    for e in ev.evalImgs:
        if e is None:
            continue
        if e["maxDet"] != max_det:
            continue
        e_rng = e.get("aRng", e.get("areaRng"))
        if e_rng is None or list(e_rng) != list(all_rng):
            continue
        k = img_pos[e["image_id"]]
        c = e["category_id"]
        gt_ignore = np.asarray(e["gtIgnore"])
        per_cat[c]["npos"][k] = int((gt_ignore == 0).sum())
        if e["dtScores"] is None or len(e["dtScores"]) == 0:
            continue
        scores = np.asarray(e["dtScores"], dtype=np.float64)
        tp = np.asarray(e["dtMatches"]) > 0  # [T, D]
        ign = (np.asarray(e["dtIgnore"], dtype=bool)
               if e.get("dtIgnore") is not None
               else np.zeros_like(tp))
        per_cat[c]["scores"].append(scores)
        per_cat[c]["tp"].append(tp.T)  # store [D, T]
        per_cat[c]["ign"].append(ign.T)  # [D, T]
        per_cat[c]["img"].append(np.full(len(scores), k, dtype=np.int64))
        n_entries += 1
    for c in cat_ids:
        d = per_cat[c]
        d["scores"] = np.concatenate(d["scores"]) if d["scores"] else np.zeros(0)
        d["tp"] = np.concatenate(d["tp"]) if d["tp"] else np.zeros((0, 10), dtype=bool)
        d["ign"] = np.concatenate(d["ign"]) if d["ign"] else np.zeros((0, 10), dtype=bool)
        d["img"] = np.concatenate(d["img"]) if d["img"] else np.zeros(0, dtype=np.int64)
    return {"per_cat": per_cat, "cat_ids": cat_ids, "img_ids": img_ids,
            "n_entries": n_entries, "full_ap": full_ap, "full_ap50": full_ap50}


def ap_from_cache(cache: dict, counts: np.ndarray) -> tuple[float, float, dict]:
    """101-point COCO AP from cached per-image TP flags under image multiplicities."""
    per_class = {}
    aps, ap50s = [], []
    for c in cache["cat_ids"]:
        d = cache["per_cat"][c]
        npos = float((counts * d["npos"]).sum())
        if npos <= 0:
            continue
        mult = counts[d["img"]]
        keep = mult > 0
        if not keep.any():
            continue
        scores = np.repeat(d["scores"][keep], mult[keep])
        tp = np.repeat(d["tp"][keep], mult[keep], axis=0)
        ign = np.repeat(d["ign"][keep], mult[keep], axis=0)
        order = np.argsort(-scores, kind="mergesort")
        tp = tp[order]
        ign = ign[order]
        # COCO excludes ignored detections from BOTH tp and fp.
        valid = ~ign
        tp = tp & valid
        tp_cum = np.cumsum(tp, axis=0).astype(np.float64)      # [n, T]
        fp_cum = np.cumsum((~tp) & valid, axis=0).astype(np.float64)
        rec = tp_cum / npos
        prec = tp_cum / np.maximum(tp_cum + fp_cum, np.spacing(1.0))
        prec = np.maximum.accumulate(prec[::-1], axis=0)[::-1]  # precision envelope
        q = np.linspace(0.0, 1.0, 101)
        ap_t = np.zeros(tp.shape[1])
        for t in range(tp.shape[1]):
            idx = np.searchsorted(rec[:, t], q, side="left")
            vals = np.where(idx < len(rec), prec[np.minimum(idx, len(rec) - 1), t], 0.0)
            vals = np.where(idx < len(rec), vals, 0.0)
            ap_t[t] = vals.mean()
        per_class[c] = float(ap_t.mean())
        aps.append(ap_t.mean())
        ap50s.append(float(ap_t[0]))
    return float(np.mean(aps)), float(np.mean(ap50s)), per_class


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--model", action="append", required=True,
                    help="name=preds.json (repeatable)")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text())
    names, preds = [], {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        names.append(name)
        preds[name] = json.loads(Path(path).read_text())
    if args.ref not in preds:
        raise SystemExit(f"--ref {args.ref} not among --model entries")

    cocoGt, gt_path = load_coco(gt)
    caches = {}
    for name in names:
        t0 = time.time()
        caches[name] = cache_model(cocoGt, preds[name])
        print(f"[bootstrap] cached {name}: pycocotools AP={caches[name]['full_ap']:.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    os.unlink(gt_path)

    n_img = len(caches[names[0]]["img_ids"])
    ones = np.ones(n_img, dtype=np.int64)

    # --- self-check: re-implementation must reproduce pycocotools on the full set
    report = {"gt": str(Path(args.gt).resolve()), "n_images": n_img,
              "n_resamples": args.n, "seed": args.seed, "models": {}, "deltas": {}}
    for name in names:
        full_ap, full_ap50, per_class = ap_from_cache(caches[name], ones)
        ref_ap = caches[name]["full_ap"]
        drift = abs(full_ap - ref_ap)
        print(f"[bootstrap] {name}: reimpl AP={full_ap:.5f} vs pycocotools {ref_ap:.5f} "
              f"(drift {drift:.2e})", flush=True)
        if drift > 5e-3:
            raise SystemExit(f"AP re-implementation drifted for {name}: {drift:.2e}")
        report["models"][name] = {"ap": full_ap, "ap50": full_ap50,
                                  "ap_pycocotools": ref_ap,
                                  "per_class_ap": {str(k): v for k, v in per_class.items()}}

    # --- paired bootstrap
    rng = np.random.default_rng(args.seed)
    ref = args.ref
    deltas = {name: [] for name in names if name != ref}
    for b in range(args.n):
        sampled = rng.integers(0, n_img, size=n_img)
        counts = np.bincount(sampled, minlength=n_img).astype(np.int64)
        ap_ref, _, _ = ap_from_cache(caches[ref], counts)
        for name in deltas:
            ap_m, _, _ = ap_from_cache(caches[name], counts)
            deltas[name].append(ap_m - ap_ref)
        if (b + 1) % 50 == 0:
            print(f"[bootstrap] {b + 1}/{args.n} resamples", flush=True)

    for name, ds in deltas.items():
        d = np.asarray(ds)
        report["deltas"][f"{name}_minus_{ref}"] = {
            "mean": float(d.mean()),
            "sd": float(d.std(ddof=1)),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_gt0": float((d > 0).mean()),
            "n": int(len(d)),
        }
        print(f"[bootstrap] {name} - {ref}: mean {d.mean():+.5f} "
              f"ci95 [{np.percentile(d, 2.5):+.5f}, {np.percentile(d, 97.5):+.5f}] "
              f"P(>0)={ (d > 0).mean():.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"[bootstrap] wrote {out}")


if __name__ == "__main__":
    main()
