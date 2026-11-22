"""Weighted box fusion (WBF) of two per-image prediction sets.

Re-implementation of the WBF idea (Solovyev et al., arXiv 1910.13302) for
exactly two sources, no external dependency:
- boxes are clustered greedily by IoU; a cluster takes AT MOST ONE box per
  source (the higher-scored one wins);
- the fused box is the score-weighted mean of the member coordinates;
- the fused score is the score-weighted mean of member scores; with
  ``scale_members=True`` it is multiplied by (members / 2), the classic
  both-models bonus that downweights single-model detections;
- the fused label comes from the highest-weight member (clusters are
  label-conservative: a box only joins a cluster of the same label unless
  ``class_agnostic=True``).
"""
from __future__ import annotations

import torch

from eval.geometry import box_iou_xywh


def _iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> float:
    """IoU of two xyxy boxes (tensors of shape [4])."""
    def to_xywh(x):
        return torch.tensor([[x[0], x[1], x[2] - x[0], x[3] - x[1]]])
    iou = box_iou_xywh(to_xywh(a), to_xywh(b))
    return float(iou.reshape(-1)[0])


def wbf_merge(boxes_a, scores_a, labels_a,
              boxes_b, scores_b, labels_b,
              iou_thr: float = 0.55, wa: float = 1.0, wb: float = 1.0,
              scale_members: bool = False,
              class_agnostic: bool = False):
    """Fuse two per-image prediction sets.

    boxes_*: [N,4] xyxy (list/tensor); scores_*: [N]; labels_*: [N] ints.
    Returns (boxes [M,4] xyxy tensor, scores [M], labels [M] int list),
    sorted by score descending.
    """
    entries = []
    for src, (b, s, l, w) in enumerate(((boxes_a, scores_a, labels_a, wa),
                                        (boxes_b, scores_b, labels_b, wb))):
        for k in range(len(s)):
            entries.append({
                "src": src, "box": torch.tensor(b[k], dtype=torch.float32),
                "score": float(s[k]), "label": int(l[k]), "w": float(w),
            })
    entries.sort(key=lambda e: -e["score"])

    clusters: list[dict] = []  # {"members": [entry], "srcs": set}
    for e in entries:
        placed = False
        for c in clusters:
            if e["src"] in c["srcs"]:
                continue
            if not class_agnostic and any(m["label"] != e["label"] for m in c["members"]):
                continue
            mean_box = torch.stack([m["box"] for m in c["members"]]).mean(0)
            if _iou_xyxy(e["box"], mean_box) >= iou_thr:
                c["members"].append(e)
                c["srcs"].add(e["src"])
                placed = True
                break
        if not placed:
            clusters.append({"members": [e], "srcs": {e["src"]}})

    out_boxes, out_scores, out_labels = [], [], []
    for c in clusters:
        ms = c["members"]
        wsum = sum(m["w"] for m in ms)
        box = sum(m["box"] * m["w"] for m in ms) / wsum
        score = sum(m["score"] * m["w"] for m in ms) / wsum
        if scale_members:
            score *= len(ms) / 2.0
        best = max(ms, key=lambda m: m["w"] * m["score"])
        out_boxes.append(box.tolist())
        out_scores.append(score)
        out_labels.append(best["label"])
    order = sorted(range(len(out_scores)), key=lambda i: -out_scores[i])
    return (torch.tensor([out_boxes[i] for i in order], dtype=torch.float32)
            if order else torch.zeros(0, 4),
            [out_scores[i] for i in order],
            [out_labels[i] for i in order])
