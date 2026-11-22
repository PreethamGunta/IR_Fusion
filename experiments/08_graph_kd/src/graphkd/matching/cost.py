"""Node matching cost shared by all graph correspondences."""
from __future__ import annotations

import torch

from graphkd.utils import giou_xywh, js_divergence, normalize_cost


def node_cost(s: dict, t: dict, alpha=(1.0, 1.0, 0.5, 0.5)) -> torch.Tensor:
    """Cost matrix [T, S] between teacher and student nodes."""
    ac, ab, af, ao = alpha
    ct = t["cls"].float()
    cs = s["cls"].float()
    c_cls = js_divergence(ct, cs) if ct.numel() and cs.numel() else ct.new_zeros((len(ct), len(cs)))
    if len(ct) == 0 or len(cs) == 0:
        return c_cls
    c_box = (1.0 - giou_xywh(t["boxes"].float(), s["boxes"].float())) / 2.0
    if t["feat"].numel() and s["feat"].numel() and t["feat"].shape[1] == s["feat"].shape[1]:
        ft = torch.nn.functional.normalize(t["feat"].float(), dim=1)
        fs = torch.nn.functional.normalize(s["feat"].float(), dim=1)
        c_feat = (1.0 - ft @ fs.t()).clamp(0, 2) / 2.0
    else:
        c_feat = torch.zeros_like(c_cls)
    c_obj = (t["obj"][:, None] - s["obj"][None, :]).abs()
    return (
        ac * normalize_cost(c_cls)
        + ab * normalize_cost(c_box)
        + af * normalize_cost(c_feat)
        + ao * normalize_cost(c_obj)
    )
