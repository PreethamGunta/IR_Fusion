"""Reliability weighting and calibration helpers."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def temperature_scale(probs: torch.Tensor, labels: torch.Tensor, iters: int = 200) -> float:
    """Fit a single temperature on validation probabilities by NLL."""
    if probs.numel() == 0:
        return 1.0
    logits = probs.clamp_min(1e-8).log()
    t = torch.ones((), requires_grad=True, device=probs.device)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / t.clamp_min(0.05), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.detach().clamp(0.05, 10.0))


def load_temperature(path: Path | None) -> float:
    if path and Path(path).exists():
        try:
            return float(json.loads(Path(path).read_text()).get("temperature", 1.0))
        except Exception:
            return 1.0
    return 1.0


def reliability(node: dict, nc: int, temperature: float = 1.0,
                quality: torch.Tensor | None = None) -> torch.Tensor:
    """Calibrated-confidence x sharpness x optional image-quality score."""
    p = node["cls"].clamp_min(1e-8)
    if temperature != 1.0:
        p = p.pow(1.0 / temperature)
        p = p / p.sum(-1, keepdim=True)
    conf = p.max(-1).values
    ent = -(p * p.log()).sum(-1) / torch.log(torch.tensor(float(nc), device=p.device))
    r = conf * (1 - ent).clamp_min(0)
    if quality is not None:
        r = r * quality
    return r.clamp(0, 1)
