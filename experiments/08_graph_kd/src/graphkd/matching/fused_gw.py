"""Fused Gromov-Wasserstein correspondence (entropic, proximal gradient).

The coupling is detached: it selects correspondences, and the KD losses are
then differentiated through the student features/boxes, not through the
transport solve.
"""
from __future__ import annotations

import torch

from graphkd.matching.sinkhorn import sinkhorn


def _fgw_objective(p, cost, at, a_s, eta):
    pt = p.sum(1)   # teacher marginal [T]
    ps = p.sum(0)   # student marginal [S]
    t1 = ((at ** 2) * torch.outer(pt, pt)).sum()
    t2 = ((a_s ** 2) * torch.outer(ps, ps)).sum()
    t3 = (at * (p @ a_s @ p.t())).sum()
    return (1 - eta) * (cost * p).sum() + eta * (t1 + t2 - 2 * t3)


@torch.no_grad()
def fused_gw(cost: torch.Tensor, at: torch.Tensor, a_s: torch.Tensor,
             eta: float = 0.5, eps: float = 0.05, outer: int = 12,
             inner: int = 30) -> torch.Tensor:
    t, s = cost.shape
    if t == 0 or s == 0:
        return cost.new_zeros((t, s))
    if at.shape != (t, t) or a_s.shape != (s, s):
        return sinkhorn(cost, eps=eps, iters=inner)
    p = torch.outer(cost.new_full((t,), 1.0 / t), cost.new_full((s,), 1.0 / s))
    for _ in range(outer):
        p = p.detach().requires_grad_(True)
        with torch.enable_grad():
            obj = _fgw_objective(p, cost, at, a_s, eta)
            grad, = torch.autograd.grad(obj, p)
        if not torch.isfinite(grad).all():
            return sinkhorn(cost, eps=eps, iters=inner)
        p = sinkhorn(grad.detach(), eps=eps, iters=inner)
    return p.detach()
