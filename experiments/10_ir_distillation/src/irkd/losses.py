"""Training-only projectors and knowledge-distillation losses (P3/P4/P5).

All projectors are 1x1 convolutions that lift the student's IR feature map to
the teacher's channel width. They live on the trainer, never on the student
model, so the exported checkpoint stays a stock IR-only detector.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=1, eps=1e-6)


def cos_huber(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """0.5 * (1 - cos) + 0.5 * SmoothL1 on channel-normalised maps."""
    p, t = _norm(pred), _norm(target)
    cos = 1.0 - (p * t).sum(dim=1, keepdim=True)          # [B,1,H,W]
    huber = F.smooth_l1_loss(p, t, reduction="none").mean(dim=1, keepdim=True)
    return 0.5 * cos.mean() + 0.5 * huber.mean()


class DirectKD(nn.Module):
    """Align the student directly to the teacher's fused feature."""

    loss_keys = ("kd_direct",)

    def __init__(self, student_ch: list[int], teacher_ch: list[int]) -> None:
        super().__init__()
        if not (len(student_ch) == len(teacher_ch) == 3):
            raise ValueError(f"expected 3 levels, got {student_ch} vs {teacher_ch}")
        self.proj = nn.ModuleList(
            [nn.Conv2d(int(s), int(t), 1, bias=True) for s, t in zip(student_ch, teacher_ch)]
        )
        for m in self.proj:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
            nn.init.zeros_(m.bias)

    def forward(self, s_feats, t_ir, t_delta, t_fused):
        losses = [
            cos_huber(p(s.float()), tf)
            for p, s, tf in zip(self.proj, s_feats, t_fused)
        ]
        loss = torch.stack(losses).mean()
        return {"kd_direct": loss}, {"kd_direct": loss.detach()}


class DecomposedKD(nn.Module):
    """Preserve the teacher's IR feature and predict its cross-modal residual.

    ``Proj_P`` aligns to ``T_ir`` (the part an IR-only model can represent);
    ``Proj_C`` aligns to ``Delta`` (the correction the graph added), with a
    per-location weight that grows where ``||Delta||`` is large.
    """

    loss_keys = ("kd_pres", "kd_comp")

    def __init__(self, student_ch: list[int], teacher_ch: list[int],
                 alpha: float = 1.0) -> None:
        super().__init__()
        if not (len(student_ch) == len(teacher_ch) == 3):
            raise ValueError(f"expected 3 levels, got {student_ch} vs {teacher_ch}")
        self.alpha = float(alpha)
        self.proj_p = nn.ModuleList(
            [nn.Conv2d(int(s), int(t), 1, bias=True) for s, t in zip(student_ch, teacher_ch)]
        )
        self.proj_c = nn.ModuleList(
            [nn.Conv2d(int(s), int(t), 1, bias=True) for s, t in zip(student_ch, teacher_ch)]
        )
        for m in list(self.proj_p) + list(self.proj_c):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
            nn.init.zeros_(m.bias)

    def forward(self, s_feats, t_ir, t_delta, t_fused=None):
        pres_l, comp_l = [], []
        for fp, fc, s, ti, td in zip(
            self.proj_p, self.proj_c, s_feats, t_ir, t_delta
        ):
            s = s.float()
            pres_l.append(cos_huber(fp(s), ti))

            d = td.float()
            mag = d.pow(2).sum(dim=1, keepdim=True).sqrt()          # [B,1,H,W]
            w = 1.0 + self.alpha * mag / (mag.mean(dim=(2, 3), keepdim=True) + 1e-6)

            p, t = _norm(fc(s)), _norm(d)
            cos = 1.0 - (p * t).sum(dim=1, keepdim=True)
            huber = F.smooth_l1_loss(p, t, reduction="none").mean(dim=1, keepdim=True)
            num = 0.5 * (w * cos).sum() + 0.5 * (w * huber).sum()
            comp_l.append(num / (w.sum() + 1e-6))

        pres = torch.stack(pres_l).mean()
        comp = torch.stack(comp_l).mean()
        # tensor values: the Ultralytics validator builds zeros_like() from them
        return {"kd_pres": pres, "kd_comp": comp}, {
            "kd_pres": pres.detach(),
            "kd_comp": comp.detach(),
        }


def build_kd(mode: str, student_ch: list[int], teacher_ch: list[int],
             alpha: float = 1.0) -> nn.Module | None:
    if mode == "none":
        return None
    if mode == "direct":
        return DirectKD(student_ch, teacher_ch)
    if mode == "decomposed":
        return DecomposedKD(student_ch, teacher_ch, alpha=alpha)
    raise ValueError(f"unknown kd mode: {mode!r}")
