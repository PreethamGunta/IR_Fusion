"""Node, edge, dense-prediction, structure and feature distillation terms."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from graphkd.matching.cost import node_cost
from graphkd.matching.fused_gw import _fgw_objective
from graphkd.utils import giou_xywh, relation_features, structure_matrix


def _npairs(p: torch.Tensor) -> torch.Tensor:
    return p.sum().clamp_min(1e-6)


def node_kd(s: dict, t: dict, p: torch.Tensor, cfg: dict,
            node_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    if s["boxes"].numel() == 0 or t["boxes"].numel() == 0:
        z = s["boxes"].new_zeros(())
        return z, {"kd_cls": z, "kd_box": z, "kd_obj": z, "kd_feat": z}
    temp = float(cfg.get("temp", 2.0))
    logp = F.log_softmax(s["cls"].float() / temp, dim=-1)  # [S,nc]
    tgt = t["cls"].float()  # [T,nc]
    kl = (tgt[:, None, :] * (tgt[:, None, :].clamp_min(1e-8).log() - logp[None, :, :])).sum(-1)
    kl = kl.clamp_min(0) * temp * temp  # [T,S]
    w = p
    if node_weight is not None:
        w = w * node_weight[:, None]
    denom = w.sum().clamp_min(1e-6)
    cls_loss = (w * kl[None, :]).sum() / denom
    giou = giou_xywh(t["boxes"].float(), s["boxes"].float())
    box_loss = (w * ((1 - giou) / 2)).sum() / denom
    obj_loss = (w * (t["obj"][:, None] - s["obj"][None, :]).pow(2)).sum() / denom
    if s["feat"].numel() and t["feat"].numel() and s["feat"].shape[1] == t["feat"].shape[1]:
        fs = F.normalize(s["feat"].float(), dim=1)
        ft = F.normalize(t["feat"].float(), dim=1)
        cos = fs @ ft.t()  # [S,T]
        feat_loss = (w * (1 - cos.t())).sum() / denom
    else:
        feat_loss = cls_loss.new_zeros(())
    lam = cfg.get("node_weights", {"cls": 1.0, "box": 1.0, "obj": 0.5, "feat": 0.1})
    loss = (lam["cls"] * cls_loss + lam["box"] * box_loss
            + lam["obj"] * obj_loss + lam.get("feat", 0.0) * feat_loss)
    return loss, {
        "kd_cls": cls_loss.detach(), "kd_box": box_loss.detach(),
        "kd_obj": obj_loss.detach(), "kd_feat": feat_loss.detach(),
    }


def edge_kd(s: dict, t: dict, p: torch.Tensor, cfg: dict,
            node_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    if s["boxes"].shape[0] < 2 or t["boxes"].shape[0] < 2:
        z = s["boxes"].new_zeros(())
        return z, {"kd_edge": z}
    rt, mt = relation_features(t["boxes"].float(), t["obj"].float())
    rs, ms = relation_features(s["boxes"].float(), s["obj"].float())
    w = p
    if node_weight is not None:
        w = w * node_weight[:, None]
    ti, si = torch.nonzero(w > 1e-4, as_tuple=True)
    if ti.numel() < 2:
        z = s["boxes"].new_zeros(())
        return z, {"kd_edge": z}
    wij = w[ti, si].clamp_min(1e-6)
    # all matched-pair pairs
    i1, i2 = ti[:, None], ti[None, :]
    j1, j2 = si[:, None], si[None, :]
    m = mt[i1, i2] & ms[j1, j2]
    if m.sum() == 0:
        z = s["boxes"].new_zeros(())
        return z, {"kd_edge": z}
    diff = (rt[i1, i2] - rs[j1, j2]).abs().mean(-1)
    ww = (wij[:, None] * wij[None, :]) * m
    loss = (ww * diff).sum() / ww.sum().clamp_min(1e-6)
    return loss, {"kd_edge": loss.detach()}


def pred_kd(sdense: dict, tdense: dict, cfg: dict) -> tuple[torch.Tensor, dict]:
    to = tdense["obj"].float()
    mask = to > float(cfg.get("pred_thr", 0.05))
    if mask.sum() == 0:
        z = to.new_zeros(())
        return z, {"kd_pred": z}
    temp = float(cfg.get("temp", 2.0))
    logp = F.log_softmax(sdense["cls"].float() / temp, dim=-1)
    kl = F.kl_div(logp, tdense["cls"].float(), reduction="none").sum(-1) * temp * temp
    box = (sdense["box"].float() - tdense["box"].float()).abs().mean(-1)
    obj = (sdense["obj"].float() - to).pow(2)
    w = to * mask
    denom = w.sum().clamp_min(1e-6)
    loss = (w * kl).sum() / denom + (w * box).sum() / denom + 0.5 * (w * obj).sum() / denom
    return loss, {"kd_pred": loss.detach()}


def struct_kd(s: dict, t: dict, p: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict]:
    if s["boxes"].numel() == 0 or t["boxes"].numel() == 0:
        z = s["boxes"].new_zeros(())
        return z, {"kd_struct": z}
    cost = node_cost(s, t, alpha=tuple(cfg.get("cost_alpha", (1.0, 1.0, 0.5, 0.5))))
    obj = _fgw_objective(p.detach(), cost, structure_matrix(t["boxes"].detach()),
                         structure_matrix(s["boxes"]), float(cfg.get("fgw_eta", 0.5)))
    obj = obj / max(1, s["boxes"].shape[0])
    return obj, {"kd_struct": obj.detach()}


def feature_kd(s: dict, t: dict, p: torch.Tensor, cfg: dict,
               proj_s=None, proj_t=None) -> tuple[torch.Tensor, dict]:
    if s["feat"].numel() == 0 or t["feat"].numel() == 0:
        z = s["boxes"].new_zeros(())
        return z, {"kd_if": z}
    fs, ft = s["feat"].float(), t["feat"].float()
    if proj_s is not None:
        fs = proj_s(fs)
    if proj_t is not None:
        ft = proj_t(ft)
    fs = F.normalize(fs, dim=1)
    ft = F.normalize(ft, dim=1)
    cos = fs @ ft.t()  # [S,T]
    w = p
    loss = (w * (1 - cos.t())).sum() / _npairs(w)
    return loss, {"kd_if": loss.detach()}
