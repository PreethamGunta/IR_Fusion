"""One-to-one Hungarian assignment baseline."""
from __future__ import annotations

import numpy as np
import torch


def hungarian(cost: torch.Tensor) -> torch.Tensor:
    """Return a 0/1 transport matrix [T,S]."""
    t, s = cost.shape
    if t == 0 or s == 0:
        return cost.new_zeros((t, s))
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost.detach().cpu().numpy())
    except Exception:
        r = np.arange(min(t, s))
        c = np.arange(min(t, s))
    p = torch.zeros((t, s), dtype=cost.dtype, device=cost.device)
    keep = cost.new_zeros(len(r))
    p[r, c] = 1.0
    return p
