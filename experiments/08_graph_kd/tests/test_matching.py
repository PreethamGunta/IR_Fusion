import torch

from graphkd.matching import match
from graphkd.matching.cost import node_cost
from graphkd.matching.fused_gw import fused_gw
from graphkd.matching.hungarian import hungarian
from graphkd.matching.sinkhorn import partial_sinkhorn, sinkhorn
from graphkd.utils import relation_features, structure_matrix


def _nodes(n, seed=0, nc=4):
    g = torch.Generator().manual_seed(seed)
    boxes = torch.rand(n, 4, generator=g) * 0.5 + 0.25
    cls = torch.softmax(torch.randn(n, nc, generator=g), -1)
    return {"boxes": boxes, "cls": cls, "obj": cls.max(-1).values,
            "feat": torch.randn(n, 16, generator=g)}


def test_hungarian_identity():
    t = _nodes(6, 1)
    s = {k: v.clone() for k, v in t.items()}
    p = hungarian(node_cost(s, t))
    assert p.shape == (6, 6)
    assert torch.allclose(p.sum(1), torch.ones(6))
    assert torch.allclose(p.sum(0), torch.ones(6))


def test_sinkhorn_marginals():
    c = torch.rand(5, 7)
    p = sinkhorn(c, eps=0.05, iters=200)
    assert torch.allclose(p.sum(1), torch.ones(5) / 5, atol=1e-3)
    assert torch.allclose(p.sum(0), torch.ones(7) / 7, atol=1e-3)


def test_partial_mass():
    c = torch.rand(5, 7)
    p = partial_sinkhorn(c)
    assert p.shape == (5, 7)
    assert 0 < p.sum() <= 1.01
    assert torch.isfinite(p).all()


def test_fgw_finite():
    t = _nodes(6, 2)
    s = _nodes(8, 3)
    c = node_cost(s, t)
    p = fused_gw(c, structure_matrix(t["boxes"]), structure_matrix(s["boxes"]), outer=3, inner=10)
    assert p.shape == (6, 8)
    assert torch.isfinite(p).all()
    assert p.min() >= 0


def test_match_dispatch():
    t = _nodes(5, 4)
    s = _nodes(6, 5)
    c = node_cost(s, t)
    for kind in ("hungarian", "sinkhorn", "fgw", "partial"):
        p = match(kind, s, t, c, {"fgw_outer": 2, "sinkhorn_iters": 10})
        assert p.shape == (5, 6)
        assert torch.isfinite(p).all()


def test_relations():
    n = _nodes(5, 6)
    rel, mask = relation_features(n["boxes"], n["obj"], k=3)
    assert rel.shape == (5, 5, 6)
    assert mask.shape == (5, 5)
    assert not mask.diagonal().any()


def test_kd_losses_grad():
    from graphkd.distillation.losses import edge_kd, node_kd, pred_kd, struct_kd

    t = _nodes(5, 7)
    s = _nodes(6, 8)
    s["boxes"].requires_grad_(True)
    s["cls"].requires_grad_(True)
    c = node_cost(s, t)
    p = hungarian(c)
    l1, _ = node_kd(s, t, p, {})
    l2, _ = edge_kd(s, t, p, {})
    l3, _ = struct_kd(s, t, p, {})
    (l1 + l2 + l3).backward()
    assert s["boxes"].grad is not None and torch.isfinite(s["boxes"].grad).all()
    sd = {"obj": torch.rand(20), "cls": torch.softmax(torch.randn(20, 4), -1),
          "box": torch.rand(20, 4)}
    td = {"obj": torch.rand(20), "cls": torch.softmax(torch.randn(20, 4), -1),
          "box": torch.rand(20, 4)}
    l4, _ = pred_kd(sd, td, {})
    assert torch.isfinite(l4)
