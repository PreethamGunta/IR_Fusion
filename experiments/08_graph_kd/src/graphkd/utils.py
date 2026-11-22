"""Small geometry / tensor helpers shared by graph construction and KD."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def xywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    x, y, w, h = b.unbind(-1)
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], -1)


def xyxy_to_xywh(b: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], -1)


def box_iou_xywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between two sets of xywh boxes (normalized or pixel, same units)."""
    a, b = xywh_to_xyxy(a), xywh_to_xyxy(b)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-7)


def giou_xywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Generalized IoU between two sets of xywh boxes -> [N, M] in [-1, 1]."""
    a, b = xywh_to_xyxy(a), xywh_to_xyxy(b)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / union.clamp(min=1e-7)
    lt_c = torch.min(a[:, None, :2], b[None, :, :2])
    rb_c = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]
    return iou - (area_c - union) / area_c.clamp(min=1e-7)


def dfl_decode(box_logits: torch.Tensor, reg_max: int) -> torch.Tensor:
    """box_logits [B, 4*reg_max, A] -> distances [B, 4, A]."""
    b, _, a = box_logits.shape
    x = box_logits.view(b, 4, reg_max, a).softmax(2)
    proj = torch.arange(reg_max, device=x.device, dtype=x.dtype)
    return (x * proj.view(1, 1, -1, 1)).sum(2)


def dist2bbox(distance: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """distance [B,4,A] (l,t,r,b) + anchors [A,2] -> xywh [B,4,A]."""
    lt, rb = distance.split(2, dim=1)
    x1y1 = anchors.transpose(0, 1)[None] - lt
    x2y2 = anchors.transpose(0, 1)[None] + rb
    ctr = (x1y1 + x2y2) / 2
    return torch.cat([ctr, x2y2 - x1y1], dim=1)


def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(dim)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between rows of p and q -> [N, M]."""
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)
    m = (p[:, None, :] + q[None, :, :]) / 2
    kl_pm = (p[:, None, :] * (p[:, None, :] / m).log()).sum(-1)
    kl_qm = (q[None, :, :] * (q[None, :, :] / m).log()).sum(-1)
    return (kl_pm + kl_qm) / 2


def normalize_cost(x: torch.Tensor) -> torch.Tensor:
    mx = x.max().clamp_min(1e-6)
    return x / mx


def relation_features(boxes: torch.Tensor, scores: torch.Tensor | None = None,
                      k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairwise relation features and edge mask for K nodes.

    boxes [K,4] xywh normalized.  Returns rel [K,K,6], mask [K,K].
    Relation: dx, dy, log(w_j/w_i), log(h_j/h_i), 1-IoU, 1-|score_i-score_j|.
    """
    kk = boxes.shape[0]
    if kk == 0:
        return boxes.new_zeros((0, 0, 6)), boxes.new_zeros((0, 0))
    c = boxes[:, :2]
    wh = boxes[:, 2:].clamp_min(1e-4)
    d = c[:, None, :] - c[None, :, :]
    lw = (wh[None, :, 0] / wh[:, None, 0]).log()
    lh = (wh[None, :, 1] / wh[:, None, 1]).log()
    iou = box_iou_xywh(boxes, boxes)
    if scores is None:
        ds = torch.zeros_like(iou)
    else:
        ds = (scores[:, None] - scores[None, :]).abs()
    rel = torch.stack([d[..., 0], d[..., 1], lw, lh, 1 - iou, ds], -1)
    # sparse: top-k nearest centres + any overlapping pair
    dist = d.pow(2).sum(-1)
    kk_eff = min(k, kk)
    idx = dist.topk(kk_eff, dim=1, largest=False).indices
    mask = torch.zeros((kk, kk), dtype=torch.bool, device=boxes.device)
    mask.scatter_(1, idx, True)
    mask |= iou > 0.05
    mask.fill_diagonal_(False)
    return rel, mask


def structure_matrix(boxes: torch.Tensor) -> torch.Tensor:
    """Scalar relational structure used by FGW: normalized geometry distance."""
    if boxes.shape[0] == 0:
        return boxes.new_zeros((0, 0))
    c = boxes[:, :2]
    wh = boxes[:, 2:].clamp_min(1e-4).log()
    dc = torch.cdist(c, c)
    dw = torch.cdist(wh, wh) / math.sqrt(2)
    a = torch.sqrt(dc.pow(2) + dw.pow(2))
    return a / a.max().clamp_min(1e-6)


def seed_all(seed: int = 0) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
