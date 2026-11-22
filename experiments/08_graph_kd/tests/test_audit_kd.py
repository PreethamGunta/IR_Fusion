"""Audit tests: KD gradients, dense-map autograd, cache detachment, matchers."""
import torch

from graphkd import paths
from graphkd.distillation.consensus import consensus
from graphkd.distillation.gt_anchored import gt_anchored_kd
from graphkd.distillation.losses import (edge_kd, feature_kd, node_kd, pred_kd,
                                         struct_kd)
from graphkd.graph.cache import TeacherCache
from graphkd.graph.candidates import decode_predictions
from graphkd.matching import match
from graphkd.matching.cost import node_cost
from graphkd.matching.fused_gw import fused_gw
from graphkd.matching.hungarian import hungarian
from graphkd.matching.sinkhorn import partial_sinkhorn, sinkhorn
from graphkd.utils import structure_matrix


def _nodes(n, seed=0, nc=4, feat=16):
    g = torch.Generator().manual_seed(seed)
    boxes = torch.rand(n, 4, generator=g) * 0.5 + 0.25
    cls = torch.softmax(torch.randn(n, nc, generator=g), -1)
    return {"boxes": boxes, "cls": cls, "obj": cls.max(-1).values,
            "feat": torch.randn(n, feat, generator=g)}


def _finite_grad(x):
    return x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def _student(seed=10, n=6):
    s = _nodes(n, seed)
    for k in ("boxes", "cls", "obj", "feat"):
        s[k].requires_grad_(True)
    return s


def test_node_edge_struct_losses_carry_gradients():
    t = _nodes(5, 1)
    s = _student(seed=2)
    p = hungarian(node_cost(s, t))
    l1, _ = node_kd(s, t, p, {})
    l2, _ = edge_kd(s, t, p, {})
    l3, _ = struct_kd(s, t, p, {})
    total = l1 + l2 + l3
    assert total.requires_grad
    total.backward()
    for k in ("boxes", "cls", "obj", "feat"):
        assert _finite_grad(s[k]), f"no gradient through student {k}"


def test_feature_kd_carries_gradient():
    t = _nodes(5, 3)
    s = _student(seed=4)
    p = hungarian(node_cost(s, t))
    l, _ = feature_kd(s, t, p, {})
    assert l.requires_grad
    l.backward()
    assert _finite_grad(s["feat"])


def test_gt_anchored_carries_gradient():
    t = _nodes(5, 5)
    s = _student(seed=6)
    gt = torch.tensor([[0.3, 0.3, 0.6, 0.6], [0.1, 0.1, 0.2, 0.2]])
    gc = torch.tensor([0, 1])
    l, terms = gt_anchored_kd(s, t, gt, gc, {"gt_iou": 0.0})
    assert l.requires_grad
    l.backward()
    assert _finite_grad(s["boxes"]) and _finite_grad(s["cls"])


def test_consensus_carries_gradient():
    s = _student(seed=7, n=7)
    t1, t2 = _nodes(5, 8), _nodes(6, 9)
    p1 = hungarian(node_cost(s, t1))
    p2 = hungarian(node_cost(s, t2))
    pseudo, valid = consensus(s, [(t1, p1, None), (t2, p2, None)])
    assert valid.shape == (7,)
    eye = torch.eye(7)
    l, _ = node_kd(s, pseudo, eye, {})
    assert l.requires_grad
    l.backward()
    assert _finite_grad(s["boxes"]) and _finite_grad(s["cls"])


def test_dense_maps_keep_autograd():
    reg_max, nc, b = 16, 4, 1
    feats = [torch.randn(b, c, s, s, requires_grad=True) for c, s in
             ((32, 80), (64, 40), (128, 20))]
    a = sum(f.shape[-2] * f.shape[-1] for f in feats)
    boxes = torch.randn(b, 4 * reg_max, a, requires_grad=True)
    scores = torch.randn(b, nc, a, requires_grad=True)
    preds = {"boxes": boxes, "scores": scores}
    nodes, dense = decode_predictions(preds, feats, torch.tensor([8, 16, 32]), nc, k=10, conf=0.0)
    assert dense["obj"].requires_grad and dense["box"].requires_grad and dense["cls"].requires_grad
    loss = dense["obj"].sum() + dense["box"].sum() + dense["cls"].sum()
    loss = loss + nodes[0]["feat"].sum() + nodes[0]["boxes"].sum() + nodes[0]["cls"].sum()
    loss.backward()
    for f in feats:
        assert f.grad is not None and torch.isfinite(f.grad).all()
    assert torch.isfinite(boxes.grad).all() and torch.isfinite(scores.grad).all()


def test_teacher_cache_is_detached():
    cache = TeacherCache(paths.CACHE / "smoke")
    stems = [s for shard in cache.index.values() for s in shard]
    rec = cache.get(stems[0])
    for k, v in rec.items():
        if torch.is_tensor(v):
            assert not v.requires_grad, k


def test_matcher_empty_and_ties():
    for shape in ((0, 3), (3, 0), (0, 0)):
        c = torch.zeros(*shape)
        assert hungarian(c).shape == shape
        assert sinkhorn(c).shape == shape
        assert partial_sinkhorn(c).shape == shape
        assert fused_gw(c, torch.zeros(shape[0], shape[0]), torch.zeros(shape[1], shape[1])).shape == shape
    c = torch.zeros(4, 5)
    p1, p2 = hungarian(c), hungarian(c)
    assert torch.equal(p1, p2)
    assert torch.allclose(p1.sum(1), torch.ones(4))   # every teacher node matched
    assert p1.sum(0).max() <= 1.0                     # one-to-one on the student side
    assert p1.sum() == 4.0


def test_sinkhorn_and_partial_marginals_finite():
    # The matcher is fed normalized costs in [0, 1] (node_cost normalizes).
    c = torch.rand(7, 9)
    p = sinkhorn(c, eps=0.05, iters=200)
    assert torch.isfinite(p).all()
    assert torch.allclose(p.sum(1), torch.ones(7) / 7, atol=2e-3)
    assert torch.allclose(p.sum(0), torch.ones(9) / 9, atol=2e-3)
    # Peaked/large costs are finite and conserve mass, but marginals are only
    # approximate; this is expected for low-entropy entropic OT.
    big = sinkhorn(torch.rand(7, 9) * 50, eps=0.05, iters=200)
    assert torch.isfinite(big).all()
    assert abs(float(big.sum()) - 1.0) < 1e-4
    q = partial_sinkhorn(c)
    assert torch.isfinite(q).all() and q.min() >= 0 and q.sum() <= 1.01


def test_fgw_structure_mismatch_falls_back():
    c = torch.rand(4, 6)
    p = fused_gw(c, torch.rand(3, 3), torch.rand(2, 2), outer=2, inner=5)
    assert p.shape == (4, 6) and torch.isfinite(p).all()


def test_node_cost_empty_graphs():
    s = _nodes(0, 11)
    t = _nodes(3, 12)
    assert node_cost(s, t).shape == (3, 0)
    assert node_cost(t, s).shape == (0, 3)
    assert structure_matrix(torch.zeros(0, 4)).shape == (0, 0)
