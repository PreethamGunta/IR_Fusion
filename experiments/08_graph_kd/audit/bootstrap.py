"""Paired image bootstrap for AP differences (audit only).

Fast COCO-style AP: per-image greedy matching, top-100 detections per image
(COCO maxDets=100), IoU 0.50:0.05:0.95, 101-point interpolation. The point AP
is validated against pycocotools (see audit/bootstrap_flir.log).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

THRESHOLDS = np.linspace(0.0, 1.0, 101)
IOUS = np.arange(0.5, 1.0, 0.05)
NC = 4


def xywh_to_xyxy(b):
    b = np.asarray(b, dtype=np.float64)
    out = np.empty_like(b)
    out[:, 0] = b[:, 0]
    out[:, 1] = b[:, 1]
    out[:, 2] = b[:, 0] + b[:, 2]
    out[:, 3] = b[:, 1] + b[:, 3]
    return out


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-12, None)


def load(path: Path):
    d = json.loads(path.read_text())
    images = d["images"]
    by_img_pred = {im["id"]: [] for im in images}
    for p in d["predictions"]:
        by_img_pred[p["image_id"]].append(p)
    by_img_gt = {im["id"]: [] for im in images}
    for a in d["annotations"]:
        by_img_gt[a["image_id"]].append(a)
    return d, images, by_img_pred, by_img_gt


def prep(d, images, by_img_pred, by_img_gt):
    n = len(images)
    per_img = [[None] * (NC + 1) for _ in range(n)]
    ngt = np.zeros((n, NC + 1), dtype=np.int64)
    for i, im in enumerate(images):
        iid = im["id"]
        gts = by_img_gt[iid]
        preds = sorted(by_img_pred[iid], key=lambda p: -p["score"])[:100]
        for c in range(1, NC + 1):
            g = [a for a in gts if a["category_id"] == c]
            ngt[i, c] = len(g)
            gb = xywh_to_xyxy([a["bbox"] for a in g]) if g else np.zeros((0, 4))
            ps = [p for p in preds if p["category_id"] == c]
            scores = np.asarray([p["score"] for p in ps], dtype=np.float64)
            pb = xywh_to_xyxy([p["bbox"] for p in ps]) if ps else np.zeros((0, 4))
            per_th = {}
            for th in IOUS:
                used = np.zeros(len(g), dtype=bool)
                tps = np.zeros(len(ps), dtype=np.int64)
                for j in range(len(ps)):
                    if len(g) == 0:
                        continue
                    ious = iou_matrix(pb[j:j + 1], gb)[0]
                    ious[used] = -1
                    k = int(np.argmax(ious))
                    if ious[k] >= th:
                        used[k] = True
                        tps[j] = 1
                per_th[round(float(th), 2)] = (scores, tps)
            per_img[i][c] = per_th
    return per_img, ngt


class Arm:
    def __init__(self, per_img, ngt, n):
        self.n = n
        self.ngt = ngt
        self.data = {}
        for c in range(1, NC + 1):
            for th in IOUS:
                S, T, I = [], [], []
                for i in range(n):
                    s, t = per_img[i][c][round(float(th), 2)]
                    if len(s):
                        S.append(s)
                        T.append(t)
                        I.append(np.full(len(s), i, dtype=np.int32))
                if S:
                    self.data[(c, round(float(th), 2))] = (
                        np.concatenate(S), np.concatenate(T), np.concatenate(I))
                else:
                    self.data[(c, round(float(th), 2))] = (
                        np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32))

    def ap(self, idx):
        counts = np.bincount(idx, minlength=self.n)
        aps = []
        for c in range(1, NC + 1):
            gts = int((counts * self.ngt[:, c]).sum())
            if gts == 0:
                continue
            for th in IOUS:
                S, T, I = self.data[(c, round(float(th), 2))]
                dc = counts[I]
                if dc.sum() == 0:
                    aps.append(0.0)
                    continue
                keep = dc > 0
                s, t, dc = S[keep], T[keep], dc[keep]
                rep = np.repeat(np.arange(len(s)), dc)
                s = s[rep]
                t = t[rep]
                order = np.argsort(-s, kind="stable")
                t = t[order]
                tp = np.cumsum(t)
                fp = np.cumsum(1 - t)
                rec = tp / gts
                prec = tp / np.clip(tp + fp, 1, None)
                for k in range(len(prec) - 2, -1, -1):
                    prec[k] = max(prec[k], prec[k + 1])
                q = np.zeros_like(THRESHOLDS)
                for k, r in enumerate(THRESHOLDS):
                    m = rec >= r
                    q[k] = prec[m].max() if m.any() else 0.0
                aps.append(q.mean())
        return float(np.mean(aps)) if aps else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="audit/preds")
    ap.add_argument("--arms", nargs="+", default=["a0_gt", "a1_pred_ir", "a2_ir_control", "a3_node_edge_ir", "a5_fgw_ir"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    arms = {}
    base = None
    for name in args.arms:
        p = Path(args.preds) / f"{name}_{args.split}.json"
        d, images, bp, bg = load(p)
        names = [im["file_name"] for im in images]
        if base is None:
            base = names
        assert names == base, name
        arms[name] = Arm(*prep(d, images, bp, bg), len(images))
    n = len(base)
    idx_all = np.arange(n)
    point = {name: arms[name].ap(idx_all) for name in args.arms}
    print("point AP", {k: round(v, 6) for k, v in point.items()}, flush=True)
    if args.n_boot <= 0:
        return

    rng = np.random.default_rng(args.seed)
    boot = {name: np.empty(args.n_boot) for name in args.arms}
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        for name in args.arms:
            boot[name][b] = arms[name].ap(idx)
        if (b + 1) % 200 == 0:
            print("boot", b + 1, flush=True)

    print(f"bootstrap n={args.n_boot} images={n}")
    for name in args.arms:
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        print(f"{name:18s} AP={point[name]:.4f} boot_mean={boot[name].mean():.4f} CI95=[{lo:.4f},{hi:.4f}]")
    ref = args.arms[0]
    for name in args.arms[1:]:
        d = boot[name] - boot[ref]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"DELTA {name} - {ref}: point={point[name]-point[ref]:+.4f} "
              f"boot_mean={d.mean():+.4f} CI95=[{lo:+.4f},{hi:+.4f}] "
              f"P(>0)={float((d>0).mean()):.3f} P(<0)={float((d<0).mean()):.3f}")


if __name__ == "__main__":
    main()
