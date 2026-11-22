"""Turn raw YOLO head outputs into prediction-graph nodes and dense maps.

The Ultralytics 8.4 YOLO12 head returns a dict:
  boxes  [B, 4*reg_max, A]  (DFL logits, concatenated scales)
  scores [B, nc, A]         (class logits, concatenated scales)
  feats  list of neck feature maps (also captured by a forward hook)
This module also accepts the older list-of-per-scale-tensors format.
"""
from __future__ import annotations

from typing import List

import torch
from torchvision.ops import roi_align

from graphkd.utils import dfl_decode, dist2bbox, xywh_to_xyxy


def _boxes_scores(raw, nc: int):
    if isinstance(raw, dict):
        return raw["boxes"], raw["scores"]
    boxes = torch.cat([p[:, : 4 * 16].reshape(p.shape[0], 4, 16, -1) for p in raw], 2)
    reg_max = (raw[0].shape[1] - nc) // 4
    boxes = torch.cat([p[:, : 4 * reg_max].reshape(p.shape[0], 4, reg_max, -1) for p in raw], 2)
    scores = torch.cat([p[:, 4 * reg_max :].reshape(p.shape[0], nc, -1) for p in raw], 2)
    return boxes, scores


def decode_predictions(
    preds,
    feats: List[torch.Tensor],
    strides: torch.Tensor,
    nc: int,
    k: int = 100,
    conf: float = 0.02,
    dense: bool = True,
) -> tuple[list[dict], dict | None]:
    """Decode pre-NMS candidates.

    Returns (per-image node dicts, dense maps or None).  Dense maps use the
    concatenated anchor order of `feats` (same order for all YOLO12 variants).
    """
    device = feats[0].device
    b = feats[0].shape[0]
    boxes_logits, cls_logits = _boxes_scores(preds, nc)
    reg_max = boxes_logits.shape[1] // 4
    d = dfl_decode(boxes_logits, reg_max)  # [B,4,A]

    anchors = []
    for f, s in zip(feats, strides.tolist()):
        h, w = f.shape[-2:]
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * s
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * s
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        anchors.append(torch.stack([gx.reshape(-1), gy.reshape(-1)], 1))
    anchors = torch.cat(anchors, 0)  # [A,2]
    boxes = dist2bbox(d, anchors)  # [B,4,A] pixel xywh
    img_h = feats[0].shape[-2] * float(strides[0])
    img_w = feats[0].shape[-1] * float(strides[0])
    scale = torch.tensor([img_w, img_h, img_w, img_h], device=device)
    boxes_n = boxes / scale.view(1, 4, 1)
    cls = cls_logits.sigmoid()  # [B,nc,A]
    obj = cls.max(1).values  # [B,A]

    out_nodes: list[dict] = []
    for bi in range(b):
        score = obj[bi]
        kk = min(k, score.numel())
        idx = score.topk(kk).indices
        idx = idx[score[idx] >= conf]
        if idx.numel() == 0:
            idx = score.topk(min(3, kk)).indices
        node_boxes = boxes_n[bi, :, idx].transpose(0, 1).float()  # [N,4]
        node_cls = cls[bi, :, idx].transpose(0, 1).float()
        node_obj = score[idx].float()
        rois = boxes[bi, :, idx].transpose(0, 1).float()
        rois_xyxy = xywh_to_xyxy(rois)
        embeds = []
        for f, s in zip(feats, strides.tolist()):
            rf = roi_align(
                f[bi : bi + 1].float(), [rois_xyxy], output_size=1,
                spatial_scale=1.0 / float(s), sampling_ratio=2, aligned=True,
            )
            embeds.append(rf.flatten(1))
        feat = torch.cat(embeds, 1) if embeds else node_cls.new_zeros((idx.numel(), 0))
        out_nodes.append({
            "boxes": node_boxes, "cls": node_cls, "obj": node_obj,
            "feat": feat, "anchor": idx,
        })

    dense_out = None
    if dense:
        # Keep the graph on the autograd path: student dense maps must carry
        # gradients for the dense prediction-KD baseline.  Teacher caches are
        # produced inside torch.no_grad(), so they are detached by construction.
        dense_out = {"obj": obj, "cls": cls.permute(0, 2, 1),
                     "box": boxes_n.permute(0, 2, 1)}
    return out_nodes, dense_out


def node_reliability(node: dict, nc: int) -> torch.Tensor:
    p = node["cls"].clamp_min(1e-8)
    ent = -(p * p.log()).sum(-1) / torch.log(torch.tensor(float(nc), device=p.device))
    return (node["obj"] * (1 - ent)).clamp(0, 1)
