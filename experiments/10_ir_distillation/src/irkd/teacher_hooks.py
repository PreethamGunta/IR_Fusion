"""Frozen E30 (WG-Fusion) teacher access for feature distillation.

The teacher is the dual-stream GF-YOLO from experiment 09. For every backbone
level l in {P3, P4, P5} the fuse module guarantees

    F_fused_l = F_ir_l + Delta_l

with ``Delta_l`` stored (detached) as ``FuseGATLocal.last_delta``. This module
extracts all three tensors for a 6-channel [IR|RGB] input without gradients.

The teacher is never part of the student checkpoint: it lives on the trainer and
is only used inside ``torch.no_grad()``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

EXP10 = Path(__file__).resolve().parents[2]
REPO = EXP10.parents[1]
EXP09 = REPO / "experiments" / "09_graph_fusion"
for _p in (str(EXP09 / "src"), str(EXP09)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class E30Teacher:
    """Load a frozen E30 checkpoint and expose per-level IR / residual / fused maps."""

    def __init__(self, ckpt: str | Path, device: str | torch.device = "cpu") -> None:
        from ultralytics import YOLO

        ckpt = Path(ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"teacher checkpoint not found: {ckpt}")

        yolo = YOLO(str(ckpt))
        model = yolo.model.float()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)
        self.model = model
        self.device = device

        self.fuse_modules = [m for m in model.model if getattr(m, "is_fuse", False)]
        if len(self.fuse_modules) != 3:
            raise RuntimeError(
                f"expected 3 fuse modules (P3/P4/P5), found {len(self.fuse_modules)}; "
                "is this really an E30 checkpoint?"
            )
        # fmod.f == [IR anchor index, RGB anchor index]
        self.anchor_idx = [int(m.f[0]) for m in self.fuse_modules]
        self.channels = [int(m.C) for m in self.fuse_modules]

        self._feats: dict[int, torch.Tensor] = {}
        self._hooks = [
            model.model[i].register_forward_hook(self._make_hook(i))
            for i in self.anchor_idx
        ]

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            self._feats[idx] = output

        return hook

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    @torch.no_grad()
    def __call__(self, paired: torch.Tensor):
        """paired: float [B, 6, H, W] in [0, 1] (same geometry as the student input).

        Returns ``(ir, delta, fused)``, each a list of three tensors [B, C, h, w].
        """
        if paired.ndim != 4 or paired.shape[1] != 6:
            raise ValueError(f"teacher expects [B,6,H,W], got {tuple(paired.shape)}")
        self._feats.clear()
        self.model._predict_once(paired)

        ir = []
        for i in self.anchor_idx:
            if i not in self._feats:
                raise RuntimeError(f"teacher anchor {i} did not fire")
            ir.append(self._feats[i])

        delta = []
        for m in self.fuse_modules:
            if m.last_delta is None:
                raise RuntimeError("teacher fuse module did not store last_delta")
            delta.append(m.last_delta)

        fused = [a + d for a, d in zip(ir, delta)]
        return ir, delta, fused


def student_anchor_indices(model) -> list[int]:
    """P3/P4/P5 backbone indices of a stock Ultralytics detection model.

    Same derivation as GFYOLO: the neck's Concat/Detect ``from`` lists reference
    the backbone anchors, so the sorted set is the anchor list (4, 6, 8 for
    yolo12s / yolo12n).
    """
    n_bk = len(model.yaml["backbone"])
    anchors = sorted(
        {
            j
            for m in model.model[n_bk:]
            for j in (m.f if isinstance(m.f, list) else [m.f])
            if isinstance(j, int) and 0 <= j < n_bk
        }
    )
    if len(anchors) != 3:
        raise RuntimeError(f"expected 3 backbone anchors, got {anchors}")
    return anchors


@torch.no_grad()
def probe_channels(model, anchors: list[int], size: int = 320) -> list[int]:
    """Channel count at each anchor, measured with an eval-mode dummy forward."""
    was_training = model.training
    model.eval()
    feats: dict[int, int] = {}
    hooks = [
        model.model[i].register_forward_hook(
            lambda _m, _i, o, idx=i: feats.__setitem__(idx, int(o.shape[1]))
        )
        for i in anchors
    ]
    try:
        model(torch.zeros(1, 3, size, size, device=next(model.parameters()).device))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    missing = [a for a in anchors if a not in feats]
    if missing:
        raise RuntimeError(f"anchor(s) {missing} did not fire during probe")
    return [feats[a] for a in anchors]
