"""Teacher-graph condensation: heuristic node selection (not learned)."""
from __future__ import annotations

import torch

from graphkd.utils import box_iou_xywh, relation_features


def select(nodes: dict, strategy: str, k: int,
           gt_boxes: torch.Tensor | None = None,
           gt_cls: torch.Tensor | None = None) -> torch.Tensor:
    n = nodes["boxes"].shape[0]
    if n == 0 or k >= n:
        return torch.arange(n, device=nodes["boxes"].device)
    if strategy == "topk":
        score = nodes["obj"]
    elif strategy == "gt_assoc" and gt_boxes is not None and len(gt_boxes):
        iou = box_iou_xywh(gt_boxes.float(), nodes["boxes"])
        same = gt_cls[:, None] == nodes["cls"].argmax(-1)[None, :]
        assoc = ((iou >= 0.3) & same).any(0).float()
        score = assoc * 2 + nodes["obj"]
    elif strategy == "centrality":
        rel, mask = relation_features(nodes["boxes"], nodes["obj"])
        score = mask.float().sum(1) * nodes["obj"]
    elif strategy == "uncertainty_importance":
        p = nodes["cls"].clamp_min(1e-8)
        ent = -(p * p.log()).sum(-1)
        score = nodes["obj"] * ent
    else:
        score = nodes["obj"]
    return score.topk(k).indices


def subset(nodes: dict, idx: torch.Tensor) -> dict:
    return {k: (v[idx] if torch.is_tensor(v) and v.shape[0] == len(idx) else v)
            for k, v in nodes.items()}


def coverage(nodes: dict, gt_boxes: torch.Tensor, gt_cls: torch.Tensor,
             idx: torch.Tensor) -> float:
    if len(idx) == 0 or len(gt_boxes) == 0:
        return 0.0
    sub = subset(nodes, idx)
    iou = box_iou_xywh(gt_boxes.float(), sub["boxes"])
    same = gt_cls[:, None] == sub["cls"].argmax(-1)[None, :]
    hit = ((iou >= 0.3) & same).any(1).float().mean()
    return float(hit)
