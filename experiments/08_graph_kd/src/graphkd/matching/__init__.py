"""Matching dispatch."""
from __future__ import annotations

import torch

from graphkd.matching.fused_gw import fused_gw
from graphkd.matching.hungarian import hungarian
from graphkd.matching.sinkhorn import partial_sinkhorn, sinkhorn
from graphkd.utils import structure_matrix


def match(kind: str, s: dict, t: dict, cost: torch.Tensor, cfg: dict) -> torch.Tensor:
    kind = (kind or "hungarian").lower()
    if kind == "hungarian":
        return hungarian(cost)
    if kind == "sinkhorn":
        return sinkhorn(cost, eps=cfg.get("sinkhorn_eps", 0.05), iters=cfg.get("sinkhorn_iters", 40))
    if kind in ("partial", "partial_ot", "dustbin"):
        return partial_sinkhorn(cost, eps=cfg.get("sinkhorn_eps", 0.05),
                                iters=cfg.get("sinkhorn_iters", 40),
                                dustbin_frac=cfg.get("dustbin_frac", 0.25))
    if kind in ("fgw", "fused_gw"):
        return fused_gw(cost, structure_matrix(t["boxes"]), structure_matrix(s["boxes"]),
                        eta=cfg.get("fgw_eta", 0.5), eps=cfg.get("sinkhorn_eps", 0.05),
                        outer=cfg.get("fgw_outer", 12), inner=cfg.get("sinkhorn_iters", 30))
    raise ValueError(f"unknown matcher {kind}")
