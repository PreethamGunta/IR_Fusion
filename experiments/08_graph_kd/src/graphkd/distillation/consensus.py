"""Reliability-aware multi-teacher consensus in the student's node space."""
from __future__ import annotations

import torch


def consensus(s: dict, teachers: list[tuple[dict, torch.Tensor, torch.Tensor | None]]) -> tuple[dict, torch.Tensor]:
    """Combine teacher graphs aligned to student nodes.

    teachers: list of (teacher_nodes, transport [T,S], reliability [T] or None).
    Returns a pseudo-teacher node dict in student space and a per-student weight.
    The transport columns are normalised per student node, so a teacher that has
    no candidate near a student node contributes nothing there.
    """
    ns = s["boxes"].shape[0]
    if ns == 0 or not teachers:
        return s, s["obj"].new_zeros(ns)
    nc = teachers[0][0]["cls"].shape[-1]
    cls_acc = s["cls"].new_zeros((ns, nc))
    box_acc = s["boxes"].new_zeros((ns, 4))
    obj_acc = s["obj"].new_zeros(ns)
    mass = s["obj"].new_zeros(ns)
    for tnodes, p, rel in teachers:
        if tnodes["boxes"].shape[0] == 0:
            continue
        w = p
        if rel is not None:
            w = w * rel[:, None]
        w = w / w.sum(0, keepdim=True).clamp_min(1e-6)
        cls_i = w.t() @ tnodes["cls"]
        box_i = w.t() @ tnodes["boxes"]
        obj_i = w.t() @ tnodes["obj"][:, None]
        m = p.sum(0).clamp(0, 1)
        cls_acc = cls_acc + cls_i * m[:, None]
        box_acc = box_acc + box_i * m[:, None]
        obj_acc = obj_acc + obj_i.squeeze(-1) * m
        mass = mass + m
    valid = mass > 1e-3
    cls = cls_acc / mass.clamp_min(1e-6)[:, None]
    boxes = box_acc / mass.clamp_min(1e-6)[:, None]
    objs = obj_acc / mass.clamp_min(1e-6)
    # sharpen the consensus distribution
    cls = cls.clamp_min(1e-8)
    cls = cls / cls.sum(-1, keepdim=True)
    pseudo = {"boxes": boxes, "cls": cls, "obj": objs, "feat": s["feat"],
              "anchor": s.get("anchor")}
    return pseudo, valid.float()


def agreement_weights(s: dict, teachers, consensus_nodes: dict) -> list[torch.Tensor]:
    """Per-teacher reliability multiplied by agreement with the consensus."""
    out = []
    cn = consensus_nodes["cls"].clamp_min(1e-8)
    for tnodes, p, rel in teachers:
        if tnodes["cls"].numel() == 0:
            out.append(None)
            continue
        w = p / p.sum(0, keepdim=True).clamp_min(1e-6)
        cls_i = (w.t() @ tnodes["cls"]).clamp_min(1e-8)
        cos = (cls_i * cn).sum(-1) / (cls_i.norm(dim=-1) * cn.norm(dim=-1)).clamp_min(1e-6)
        agree = ((cos + 1) / 2).clamp(0, 1)
        base = rel if rel is not None else torch.ones_like(tnodes["obj"])
        # per-teacher-node weight = node reliability; multiplied by agreement in
        # the student space when aggregating.
        out.append(base * (w * agree[None, :]).sum(1))
    return out
