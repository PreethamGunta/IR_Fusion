"""Geometry helpers used by evaluator adapters."""
from __future__ import annotations

import torch


def xywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    out = b.clone()
    out[..., 0] = b[..., 0]
    out[..., 1] = b[..., 1]
    out[..., 2] = b[..., 0] + b[..., 2]
    out[..., 3] = b[..., 1] + b[..., 3]
    return out


def unletterbox(boxes: torch.Tensor, h0: int, w0: int, imgsz: int = 640) -> torch.Tensor:
    gain = min(imgsz / h0, imgsz / w0)
    pad_w = (imgsz - w0 * gain) / 2
    pad_h = (imgsz - h0 * gain) / 2
    b = boxes.clone()
    b[:, [0, 2]] = (b[:, [0, 2]] - pad_w) / gain
    b[:, [1, 3]] = (b[:, [1, 3]] - pad_h) / gain
    return b.clamp(min=0)


def box_iou_xywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = xywh_to_xyxy(a)
    b = xywh_to_xyxy(b)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    bb = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    return inter / (aa[:, None] + bb[None, :] - inter).clamp(min=1e-9)
