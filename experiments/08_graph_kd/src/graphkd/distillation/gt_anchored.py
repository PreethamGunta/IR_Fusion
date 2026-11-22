"""Ground-truth anchored heterogeneous graph KD.

Ground truth is available only during training; at inference the student is a
plain IR detector.  Predictions are associated to GT anchors by class + IoU;
teacher/student aggregates at the same anchor are then distilled.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from graphkd.utils import box_iou_xywh, giou_xywh


def _assign(nodes: dict, gt_boxes: torch.Tensor, gt_cls: torch.Tensor,
            iou_thr: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-GT (weights [G,N], best box class probs) for one image."""
    g, n = gt_boxes.shape[0], nodes["boxes"].shape[0]
    if g == 0 or n == 0:
        return nodes["boxes"].new_zeros((g, n)), nodes["boxes"].new_zeros((g, 4))
    iou = box_iou_xywh(gt_boxes, nodes["boxes"])
    same = gt_cls[:, None] == nodes["cls"].argmax(-1)[None, :]
    valid = (iou >= iou_thr) & same
    w = valid.float() * nodes["obj"][None, :] * iou
    return w, nodes["boxes"]


def gt_anchored_kd(s: dict, t: dict, gt_boxes: torch.Tensor, gt_cls: torch.Tensor,
                   cfg: dict) -> tuple[torch.Tensor, dict]:
    gt_boxes = gt_boxes.float()
    gt_cls = gt_cls.long()
    wt, _ = _assign(t, gt_boxes, gt_cls, cfg.get("gt_iou", 0.3))
    ws, _ = _assign(s, gt_boxes, gt_cls, cfg.get("gt_iou", 0.3))
    support = (wt.sum(1) > 0) & (ws.sum(1) > 0)
    z = s["boxes"].new_zeros(())
    if support.sum() == 0:
        return z, {"kd_gt": z}
    wt = wt[support]
    ws = ws[support]
    gt_b = gt_boxes[support]
    wt = wt / wt.sum(1, keepdim=True).clamp_min(1e-6)
    ws = ws / ws.sum(1, keepdim=True).clamp_min(1e-6)
    t_cls = wt @ t["cls"]
    s_cls = ws @ s["cls"]
    t_box = wt @ t["boxes"]
    s_box = ws @ s["boxes"]
    temp = float(cfg.get("temp", 2.0))
    kl = F.kl_div(F.log_softmax(s_cls / temp, dim=-1), t_cls, reduction="none").sum(-1) * temp * temp
    giou = giou_xywh(t_box, s_box)
    loss = kl.mean() + ((1 - giou) / 2).mean()
    return loss, {"kd_gt": loss.detach(), "kd_gt_n": support.sum().float()}
