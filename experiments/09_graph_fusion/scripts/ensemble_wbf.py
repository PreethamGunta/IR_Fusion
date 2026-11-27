#!/usr/bin/env python3
"""Weighted-box-fusion ensemble of K per-image prediction sets (CPU).

Vectorised per (image, class) greedy WBF: boxes are visited in descending
score order; a box joins the first cluster that (a) shares its label,
(b) does not already contain a box from the same source, and (c) has cluster-
mean IoU >= iou_thr. Cluster boxes/scores are score-weighted means; the
label is the highest-score member's. This is the one-box-per-source rule of
WBF (Solovyev et al. 2019) generalised to K sources.

Hyperparameters are FIXED from the WBF paper defaults (iou 0.55, equal source
weights, no member scaling) — deliberately not tuned on the test set.

Usage:
  python scripts/ensemble_wbf.py --gt results/gt/dv_v2_ir_test.json \
    --preds results/preds/e10_v2_100ep_test.json \
    --preds results/preds/e30_v2_100ep_test.json \
    --preds results/preds/e35_v2_100ep_test.json \
    --out results/preds/ensemble_test.json --iou 0.55
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between [Na,4] and [Nb,4] xyxy arrays."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def fuse_group(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
               sources: np.ndarray, iou_thr: float):
    """Greedy one-box-per-source WBF for one (image, class) group."""
    order = np.argsort(-scores, kind="mergesort")
    boxes, scores, labels, sources = boxes[order], scores[order], labels[order], sources[order]
    clusters: list[dict] = []  # mean_box, wsum, score_sum, members, srcs
    for k in range(len(scores)):
        b, s, src = boxes[k], float(scores[k]), int(sources[k])
        placed = False
        if clusters:
            means = np.stack([c["mean"] for c in clusters])
            ious = iou_matrix(b[None, :], means)[0]
            for ci in np.argsort(-ious):
                c = clusters[ci]
                if ious[ci] < iou_thr:
                    break
                if src in c["srcs"]:
                    continue
                n = c["n"] + 1
                c["mean"] = (c["mean"] * c["n"] + b) / n
                c["n"] = n
                c["score_sum"] += s
                c["srcs"].add(src)
                c["members"].append((s, k))
                placed = True
                break
        if not placed:
            clusters.append({"mean": b.astype(np.float64), "n": 1,
                             "score_sum": s, "srcs": {src}, "members": [(s, k)]})
    out = []
    for c in clusters:
        score = c["score_sum"] / c["n"]
        best_k = max(c["members"], key=lambda m: m[0])[1]
        out.append((c["mean"], score, int(labels[best_k])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iou", type=float, default=0.55)
    args = ap.parse_args()

    groups: dict[tuple, list] = defaultdict(list)
    sources = []
    for sid, path in enumerate(args.preds):
        preds = json.loads(Path(path).read_text())
        sources.append(Path(path).name)
        for p in preds:
            x, y, w, h = p["bbox"]
            groups[(p["image_id"], p["category_id"])].append(
                (x, y, x + w, y + h, float(p["score"]), sid))
        print(f"[wbf] loaded {len(preds)} preds from {Path(path).name}", flush=True)

    fused = []
    t0 = time.time()
    for n, ((img, cat), rows) in enumerate(groups.items(), 1):
        arr = np.asarray(rows, dtype=np.float64)
        res = fuse_group(arr[:, :4], arr[:, 4], np.full(len(arr), cat),
                         arr[:, 5].astype(int), args.iou)
        for box, score, label in res:
            x1, y1, x2, y2 = box
            fused.append({"image_id": int(img), "category_id": int(label),
                          "bbox": [round(float(x1), 2), round(float(y1), 2),
                                   round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                          "score": round(float(score), 6)})
        if n % 20000 == 0:
            print(f"[wbf] {n}/{len(groups)} groups, {len(fused)} fused preds "
                  f"({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fused))
    meta = {"sources": sources, "iou_thr": args.iou, "scale_members": False,
            "n_groups": len(groups), "n_fused": len(fused),
            "wall_s": round(time.time() - t0, 1)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[wbf] wrote {out}: {len(fused)} fused preds over {len(groups)} groups "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
