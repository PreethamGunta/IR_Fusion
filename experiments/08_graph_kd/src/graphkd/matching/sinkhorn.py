"""Balanced Sinkhorn optimal transport (log-domain)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sinkhorn(cost: torch.Tensor, eps: float = 0.05, iters: int = 40,
             a: torch.Tensor | None = None, b: torch.Tensor | None = None) -> torch.Tensor:
    """Return a doubly-stochastic-ish transport plan P [T,S] for `cost`."""
    t, s = cost.shape
    if t == 0 or s == 0:
        return cost.new_zeros((t, s))
    if a is None:
        a = cost.new_full((t,), 1.0 / t)
    if b is None:
        b = cost.new_full((s,), 1.0 / s)
    log_a = a.clamp_min(1e-12).log()
    log_b = b.clamp_min(1e-12).log()
    f = cost.new_zeros(t)
    g = cost.new_zeros(s)
    for _ in range(iters):
        f = eps * (log_a - torch.logsumexp((g[None, :] - cost) / eps, dim=1))
        g = eps * (log_b - torch.logsumexp((f[:, None] - cost) / eps, dim=0))
    p = torch.exp((f[:, None] + g[None, :] - cost) / eps)
    return p / p.sum().clamp_min(1e-12)


def partial_sinkhorn(cost: torch.Tensor, eps: float = 0.05, iters: int = 40,
                     dustbin_frac: float = 0.25) -> torch.Tensor:
    """Sinkhorn with a dustbin so unmatched nodes do not force correspondences."""
    t, s = cost.shape
    if t == 0 or s == 0:
        return cost.new_zeros((t, s))
    mx = cost.max().clamp_min(1e-6)
    dust = (0.5 * mx).to(cost)
    aug = cost.new_full((t + 1, s + 1), dust)
    aug[:t, :s] = cost
    aug[t, s] = 0.0
    # keep the total mass of the augmented problem normalised to 1
    mass = 1.0 + dustbin_frac * max(t, s)
    a = torch.cat([cost.new_full((t,), 1.0 / t), cost.new_full((1,), dustbin_frac/max(1, t))])
    b = torch.cat([cost.new_full((s,), 1.0 / s), cost.new_full((1,), dustbin_frac/max(1, s))])
    a = a / a.sum()
    b = b / b.sum()
    p = sinkhorn(aug, eps=eps, iters=iters, a=a, b=b)
    return p[:t, :s]
