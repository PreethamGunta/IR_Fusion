"""CPU-only acceptance tests for the GF-YOLO fusion testbed (experiment 09).

Run from this directory with::

    PYTHONPATH=. /home2/home/preetham_g/.conda/envs/gateir/bin/python \
        -m pytest test_fuse_modules.py -q

The suite follows ``EXPERIMENT_PLAN.md`` sec. 10 (implementation bug
checklist).  It covers, for all five rungs (E10/E14/E21/E30/E35):

* the ``[B, C, H, W] -> [B, C, H, W]`` channel/shape contract;
* E14's negative-bias gate identity at initialisation (``< 1e-6``) and the
  zero-RGB/zero-IR behaviour at init;
* attention rows summing to one over **valid** neighbours only;
* neighbour-table bounds, border counts and absence of unintended
  same-modality self-loops;
* semantic-kNN candidates staying inside their radius and reading RGB only;
* fp16 forward being NaN-free (the modules raise on non-finite values);
* one backward pass giving every parameter a finite, non-zero gradient;
* fixed neighbour tables and deterministic repeated forwards;
* parameter counts matching the documented formulas.

All tests are CPU-only and seed ``torch.manual_seed(0)``.  Tests are small
(``B=2, C=8, H=W=6`` unless stated) so the whole file runs in seconds.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import fuse_modules
import neighbors
from fuse_modules import (
    FuseBiCrossAttention,
    FuseConcat1x1,
    FuseGATLocal,
    FuseGATSemantic,
    FuseGatedResidual,
    count_parameters,
)
from neighbors import (
    border_class,
    border_counts,
    flat_to_yx,
    grid8_index,
    ir_graph_index,
    node_coords,
    semantic_candidates,
    semantic_knn_index,
    window_index,
    yx_to_flat,
)

MODULE_NAMES = ["E10", "E14", "E21", "E30", "E35"]
RESIDUAL_NAMES = {"E14", "E21", "E30", "E35"}
DEFAULT_D = 128
EDGE_DIM = 5

# Global default seed for anything not individually seeded below.
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build(name: str, C: int = 8, **kwargs) -> torch.nn.Module:
    """Construct one rung with the plan's default width ``d=128``."""
    kwargs = dict(kwargs)
    if name == "E10":
        return FuseConcat1x1(C)
    if name == "E14":
        return FuseGatedResidual(C, **kwargs)
    if name == "E21":
        return FuseBiCrossAttention(C, d=kwargs.pop("d", DEFAULT_D), **kwargs)
    if name == "E30":
        return FuseGATLocal(C, d=kwargs.pop("d", DEFAULT_D), **kwargs)
    if name == "E35":
        return FuseGATSemantic(C, d=kwargs.pop("d", DEFAULT_D), **kwargs)
    raise ValueError(f"unknown rung {name}")


def rand_pair(
    B: int = 2,
    C: int = 8,
    H: int = 6,
    W: int = 6,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
):
    """Reproducible IR/RGB pair on CPU."""
    torch.manual_seed(seed)
    f_ir = torch.randn(B, C, H, W, dtype=dtype)
    f_rgb = torch.randn(B, C, H, W, dtype=dtype)
    return f_ir, f_rgb


def expected_params(name: str, C: int = 8, d: int = DEFAULT_D, E: int = EDGE_DIM) -> int:
    """Closed-form parameter counts documented in the module docstrings."""
    if name == "E10":
        return 2 * C * C + C
    if name == "E14":
        return 4 * C * C + 2 * C
    if name == "E21":
        return 4 * C * d + 3 * d * d + 5 * d + 4 * C * C + 4 * C
    if name in {"E30", "E35"}:
        return 3 * C * d + 5 * d * d + 6 * d + E * E + 2 * E + 2 * C + 1
    raise ValueError(name)


# ---------------------------------------------------------------------------
# 1. Channel / shape contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_shape_contract(name):
    """All rungs preserve ``[B, C, H, W]`` and residual rungs expose Delta."""
    m = build(name, C=8)
    f_ir, f_rgb = rand_pair()
    out = m(f_ir, f_rgb)
    assert out.shape == f_ir.shape
    assert out.dtype == f_ir.dtype
    assert torch.isfinite(out).all()

    if name in RESIDUAL_NAMES:
        assert hasattr(m, "last_delta")
        assert m.last_delta is not None
        assert m.last_delta.shape == f_ir.shape
        # Value identity: f_fused == f_ir + last_delta (stored detached).
        assert torch.equal(out, f_ir + m.last_delta)


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_shape_contract_nondefault_channels_and_aspect(name):
    """C=3 and a non-square 5x7 map: nothing is hardcoded to 6 channels/square."""
    m = build(name, C=3)
    f_ir, f_rgb = rand_pair(B=1, C=3, H=5, W=7)
    out = m(f_ir, f_rgb)
    assert out.shape == (1, 3, 5, 7)


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_rejects_bad_shapes(name):
    """The contract is validated rather than silently broadcast."""
    m = build(name, C=8)
    f_ir, _ = rand_pair()
    with pytest.raises(ValueError):
        m(f_ir, torch.randn_like(f_ir[:, :4]))
    with pytest.raises(ValueError):
        m(f_ir[0], f_ir[0])


def test_e10_concat_order_is_ir_first():
    """E10's first C input channels are the IR stream (no silent IR/RGB swap).

    The hand-set 1x1 weight extracts the mean of the first C channels, so the
    output can only equal the IR mean if the concatenation is ``[IR | RGB]``.
    """
    C = 3
    m = FuseConcat1x1(C)
    with torch.no_grad():
        m.conv.weight.zero_()
        m.conv.weight[:, :C] = 1.0 / C
        m.conv.bias.zero_()
    f_ir, f_rgb = rand_pair(B=1, C=C, H=4, W=5)
    out = m(f_ir, f_rgb)
    expected = f_ir.mean(dim=1, keepdim=True).expand_as(f_ir)
    assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. E14 gate initialisation and zero-modality probes
# ---------------------------------------------------------------------------


def test_e14_gate_bias_is_minus_four():
    """The gate convolution's bias is exactly the specified -4.0."""
    torch.manual_seed(0)
    m = FuseGatedResidual(8)
    assert m.gate.bias is not None
    assert torch.equal(m.gate.bias, torch.full_like(m.gate.bias, -4.0))


@pytest.mark.parametrize("C", [8, 32, 64, 128, 256])
def test_e14_gate_init_identity(C):
    """At step 0 E14 returns f_ir within 1e-6 (residual starts at the IR anchor)."""
    torch.manual_seed(0)
    m = FuseGatedResidual(C)
    f_ir, f_rgb = rand_pair(C=C)
    out = m(f_ir, f_rgb)
    err = (out - f_ir).abs().max().item()
    assert err < 1e-6, f"E14 init identity failed: max|out-f_ir|={err:.3e}"
    assert m.last_delta is not None
    assert m.last_delta.abs().max().item() < 1e-6


def test_e14_zero_rgb_at_init_returns_f_ir():
    """Zeroing the RGB stream at init leaves the output exactly f_ir."""
    torch.manual_seed(0)
    m = FuseGatedResidual(8)
    f_ir, _ = rand_pair()
    out = m(f_ir, torch.zeros_like(f_ir))
    assert (out - f_ir).abs().max().item() < 1e-6
    # The zeroed stream contributes zero correction at init.
    assert m.last_delta.abs().max().item() < 1e-6


def test_e14_zero_ir_at_init_contributes_zero():
    """Zeroing IR instead also yields a (near-)zero correction at init."""
    torch.manual_seed(0)
    m = FuseGatedResidual(8)
    _, f_rgb = rand_pair()
    out = m(torch.zeros_like(f_rgb), f_rgb)
    assert out.abs().max().item() < 1e-6
    assert m.last_delta.abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# 3. Attention: rows sum to one over valid neighbours only
# ---------------------------------------------------------------------------


def test_e21_attention_rows_are_distributions():
    """Dense bidirectional attention has one distribution per query token."""
    torch.manual_seed(0)
    m = FuseBiCrossAttention(8, d=DEFAULT_D)
    m.collect_attn = True
    f_ir, f_rgb = rand_pair()
    _ = m(f_ir, f_rgb)
    assert m.last_attn is not None
    N = f_ir.shape[-1] * f_ir.shape[-2]
    for key in ("ir", "rgb"):
        attn = m.last_attn[key]
        assert attn.shape == (f_ir.shape[0], N, N)
        assert torch.isfinite(attn).all()
        assert (attn >= 0).all()
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("E30", {}),
        # R=1 with k=6 makes corner nodes candidate-starved -> invalid slots.
        ("E35", {"R": 1, "k": 6}),
        ("E35", {"R": 2, "k": 6}),
    ],
)
def test_graph_attention_rows_are_distributions(name, kwargs):
    """Sparse GAT softmax sums to 1 over valid slots; invalid slots are zero."""
    torch.manual_seed(0)
    m = build(name, C=8, **kwargs)
    m.collect_attn = True
    f_ir, f_rgb = rand_pair()
    _ = m(f_ir, f_rgb)
    assert m.last_attn is not None
    weights = m.last_attn["weights"]
    valid = m.last_attn["valid"]
    assert weights.shape == valid.shape
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    # Softmax really was restricted to valid neighbours.
    assert torch.equal(weights.masked_select(~valid), torch.zeros_like(weights.masked_select(~valid)))
    assert bool(valid.any(dim=-1).all())
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Neighbour tables: bounds, border counts, self-loops, determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("H,W", [(5, 7), (4, 6), (3, 3), (8, 5), (2, 2)])
def test_grid8_bounds_counts_and_no_self_loops(H, W):
    """Corner 3, edge 5, interior 8; raw ids are in bounds and never self."""
    t = grid8_index(H, W)
    N = H * W
    assert t.index.shape == (N, 8)
    assert int(t.index.min()) >= 0 and int(t.index.max()) < N
    # Even invalid fillers are a different node, so the raw table is clean.
    target = torch.arange(N).unsqueeze(1)
    assert not bool((t.index == target).any())

    cls = border_class(H, W)
    counts = t.valid.sum(dim=1)
    if H >= 3 and W >= 3:
        assert torch.equal(counts[cls == 0], torch.full_like(counts[cls == 0], 3))
        assert torch.equal(counts[cls == 1], torch.full_like(counts[cls == 1], 5))
        assert torch.equal(counts[cls == 2], torch.full_like(counts[cls == 2], 8))
    # The dy/dx offsets match the source coordinates for every valid slot.
    ys, xs = node_coords(H, W)
    sy, sx = flat_to_yx(t.index, W)
    assert torch.equal((sy - ys.unsqueeze(1))[t.valid], t.dy[t.valid])
    assert torch.equal((sx - xs.unsqueeze(1))[t.valid], t.dx[t.valid])


@pytest.mark.parametrize("r", [1, 2])
def test_window_border_counts(r):
    """Window counts follow the separable in-bounds row/column counts.

    Expected count for node ``(y, x)`` is
    ``#valid dy * #valid dx`` with ``dy, dx in [-r, r]``.  This is computed
    independently here, so it catches a wrong clamp or a swapped axis.
    """
    H, W = 5, 7
    t = window_index(H, W, r, include_center=True)
    N = H * W
    assert int(t.index.min()) >= 0 and int(t.index.max()) < N
    ys, xs = node_coords(H, W)
    rows = torch.minimum(ys + r, torch.tensor(H - 1)) - torch.maximum(ys - r, torch.tensor(0)) + 1
    cols = torch.minimum(xs + r, torch.tensor(W - 1)) - torch.maximum(xs - r, torch.tensor(0)) + 1
    expected = rows * cols
    assert torch.equal(t.valid.sum(dim=1), expected)

    # Explicit corner / edge / interior spot checks for the reader.
    corner = int(expected[0])
    interior = int(expected[H // 2 * W + W // 2])
    top_mid = int(expected[W // 2])
    left_mid = int(expected[(H // 2) * W])
    assert corner == (r + 1) ** 2
    assert interior == (2 * r + 1) ** 2
    assert top_mid == (r + 1) * (2 * r + 1)
    assert left_mid == (2 * r + 1) * (r + 1)

    t_no = window_index(H, W, r, include_center=False)
    assert torch.equal(t_no.valid.sum(dim=1), expected - 1)
    # Without the centre offset there is no self id at all.
    target = torch.arange(N).unsqueeze(1)
    assert not bool((t_no.index == target).any())


def test_ir_graph_table_bounds_and_composition():
    """Combined E30 table: in bounds, correct modality offsets, no IR self-loop."""
    H, W = 5, 7
    N = H * W
    t = ir_graph_index(H, W, r=1)
    assert t.num_slots == 8 + 9
    assert int(t.src.min()) >= 0 and int(t.src.max()) < 2 * N
    assert torch.equal(t.valid.sum(dim=1), t.grid_count + t.window_count)
    assert torch.equal(t.grid_count, grid8_index(H, W).valid.sum(dim=1))
    assert torch.equal(t.window_count, window_index(H, W, 1, include_center=True).valid.sum(dim=1))

    # 7 / 11 / 17 valid slots for corner / edge / interior in the union.
    assert torch.equal(
        border_counts(t.valid, H, W),
        torch.tensor([4 * 7, ((2 * H + 2 * W) - 8) * 11, (H - 2) * (W - 2) * 17]),
    )

    target = torch.arange(N).unsqueeze(1)
    same_modality = ~t.is_cross
    bad = t.valid & same_modality & (t.src == target)
    assert not bool(bad.any()), "IR grid edges must not contain self-loops"
    # Every valid cross edge points into the RGB half.
    assert bool((t.src[t.valid & t.is_cross] >= N).all())


def test_tables_are_deterministic_across_calls():
    """Repeated table builds (and semantic kNN) are bit-identical."""
    checks = [
        lambda: grid8_index(5, 7),
        lambda: window_index(5, 7, 2, include_center=False),
        lambda: semantic_candidates(5, 7, 2),
    ]
    for fn in checks:
        a, b = fn(), fn()
        assert torch.equal(a.index, b.index)
        assert torch.equal(a.valid, b.valid)
        assert torch.equal(a.dy, b.dy)
        assert torch.equal(a.dx, b.dx)

    f_ir, f_rgb = rand_pair(B=2, C=1, H=5, W=7)
    feats_ir, feats_rgb = f_ir.flatten(2).transpose(1, 2), f_rgb.flatten(2).transpose(1, 2)
    idx1, val1 = semantic_knn_index(feats_ir, feats_rgb, 5, 7, radius=2, k=6)
    idx2, val2 = semantic_knn_index(feats_ir, feats_rgb, 5, 7, radius=2, k=6)
    idx3, val3 = semantic_knn_index(feats_ir, feats_rgb, 5, 7, radius=2, k=6, chunk=7)
    assert torch.equal(idx1, idx2)
    assert torch.equal(val1, val2)
    # Chunking is an implementation detail; it must not change the selection.
    assert torch.equal(idx1, idx3)
    assert torch.equal(val1, val3)


def test_token_index_roundtrip():
    """``(y, x) <-> flat`` round-trips at every tested scale."""
    for H, W in [(5, 7), (20, 16), (40, 32), (80, 64)]:
        flat = torch.arange(H * W)
        y, x = flat_to_yx(flat, W)
        assert torch.equal(yx_to_flat(y, x, W), flat)
        assert int(y.min()) == 0 and int(y.max()) == H - 1
        assert int(x.min()) == 0 and int(x.max()) == W - 1


# ---------------------------------------------------------------------------
# 5. Semantic kNN (E35): radius, modality, top-k correctness, border shortage
# ---------------------------------------------------------------------------


def test_semantic_candidates_stay_inside_radius():
    """Candidate slots are exactly the in-bounds Chebyshev-R window."""
    H, W, R = 5, 7, 2
    t = semantic_candidates(H, W, R)
    assert torch.equal(t.dy.abs() <= R, torch.ones_like(t.valid))
    assert torch.equal(t.dx.abs() <= R, torch.ones_like(t.valid))
    ys, xs = node_coords(H, W)
    sy, sx = flat_to_yx(t.index, W)
    assert torch.equal((sy - ys.unsqueeze(1))[t.valid], t.dy[t.valid])
    assert torch.equal((sx - xs.unsqueeze(1))[t.valid], t.dx[t.valid])


def test_semantic_knn_radius_and_topk_correctness():
    """Selected ids are in-radius RGB candidates with the top-k cosine scores."""
    H, W, R, k = 5, 5, 2, 6
    B, N, d = 2, H * W, 16
    torch.manual_seed(0)
    feat_ir = F.normalize(torch.randn(B, N, d), dim=-1)
    feat_rgb = F.normalize(torch.randn(B, N, d), dim=-1)
    idx, valid = semantic_knn_index(feat_ir, feat_rgb, H, W, radius=R, k=k)
    assert idx.shape == (B, N, k)
    assert valid.shape == (B, N, k)
    assert int(idx.min()) >= 0 and int(idx.max()) < N

    ys, xs = node_coords(H, W)
    sy, sx = flat_to_yx(idx, W)
    dy = sy - ys.view(1, N, 1)
    dx = sx - xs.view(1, N, 1)
    assert bool((dy[valid].abs() <= R).all())
    assert bool((dx[valid].abs() <= R).all())

    cand = semantic_candidates(H, W, R)
    for b in range(B):
        for i in range(N):
            ci = cand.index[i][cand.valid[i]]
            scores = feat_ir[b, i] @ feat_rgb[b, ci].T
            best = torch.sort(scores, descending=True).values
            sel = idx[b, i][valid[b, i]]
            assert bool(torch.isin(sel, ci).all()), "kNN picked a non-candidate"
            sel_scores = feat_ir[b, i] @ feat_rgb[b, sel].T
            sel_sorted = torch.sort(sel_scores, descending=True).values
            # Every selected edge is among the best `len(sel)` candidates.
            assert torch.allclose(sel_sorted, best[: sel_sorted.numel()], atol=1e-5)


def test_semantic_knn_border_shortage_is_masked_not_invented():
    """Small R + large k: border nodes get fewer valid edges, never fake ones."""
    H = W = 4
    N = H * W
    k = 6
    feat_ir = torch.randn(1, N, 8)
    feat_rgb = torch.randn(1, N, 8)
    idx, valid = semantic_knn_index(feat_ir, feat_rgb, H, W, radius=1, k=k)
    assert int(idx.min()) >= 0 and int(idx.max()) < N
    counts = valid.sum(dim=-1).squeeze(0)
    cls = border_class(H, W)
    # R=1 including the centre: 4 corner, 6 edge, 9 interior candidates.
    assert torch.equal(counts[cls == 0], torch.full_like(counts[cls == 0], 4))
    assert torch.equal(counts[cls == 1], torch.full_like(counts[cls == 1], 6))
    assert torch.equal(counts[cls == 2], torch.full_like(counts[cls == 2], k))
    # Invalid slots are still safe in-bounds ids (gatherable, just masked).
    assert int(idx.min()) >= 0 and int(idx.max()) < N


def test_semantic_knn_reads_rgb_only():
    """Candidate ids are consistent with RGB content, and the IR query is used.

    Two checks: (a) changing the RGB tensor changes the selection (it is the
    candidate pool); (b) a brute-force top-k over RGB candidates for a fixed
    IR query matches the implementation.
    """
    H = W = 5
    N, d = H * W, 12
    torch.manual_seed(0)
    feat_ir = torch.randn(1, N, d)
    feat_rgb = torch.randn(1, N, d)
    idx_a, _ = semantic_knn_index(feat_ir, feat_rgb, H, W, radius=2, k=5)
    idx_b, _ = semantic_knn_index(feat_ir, torch.randn_like(feat_rgb), H, W, radius=2, k=5)
    assert not torch.equal(idx_a, idx_b)

    cand = semantic_candidates(H, W, 2)
    i = N // 2
    ci = cand.index[i][cand.valid[i]]
    scores = F.normalize(feat_ir[0, i], dim=-1) @ F.normalize(feat_rgb[0, ci], dim=-1).T
    order = torch.argsort(scores, descending=True, stable=True)[:5]
    expected = ci[order].sort().values
    got = idx_a[0, i].sort().values
    assert torch.equal(expected, got)


# ---------------------------------------------------------------------------
# 6. Module-level graph edges (bounds, modality, radius, self-loops)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,kwargs,K0",
    [("E30", {}, 8 + 9), ("E35", {"R": 1, "k": 6}, 8 + 9)],
)
def test_module_edges_are_in_bounds_and_correctly_typed(name, kwargs, K0):
    """Assembled graph: in bounds, cross edges from RGB, no IR self-loops."""
    torch.manual_seed(0)
    m = build(name, C=8, **kwargs)
    m.collect_attn = True
    H = W = 6
    N = H * W
    f_ir, f_rgb = rand_pair(H=H, W=W)
    _ = m(f_ir, f_rgb)
    assert m.last_edges is not None
    src = m.last_edges["src"]
    valid = m.last_edges["valid"]
    is_cross = m.last_edges["is_cross"]
    dy = m.last_edges["dy"]
    dx = m.last_edges["dx"]
    expected_K = K0 + (int(kwargs.get("k", 0)) if name == "E35" else 0)
    assert src.shape[-1] == expected_K
    assert src.shape[:2] == (f_ir.shape[0], N)
    assert int(src[valid].min()) >= 0
    assert int(src[valid].max()) < 2 * N
    assert torch.equal(is_cross, src >= N)

    target = torch.arange(N).unsqueeze(0).unsqueeze(-1)
    bad = valid & (~is_cross) & (src == target)
    assert not bool(bad.any()), "same-modality self-loop in the IR graph"
    # Every cross edge (window or semantic) comes from the RGB half.
    assert bool((src[valid & is_cross] >= N).all())

    if name == "E35":
        sem_valid = valid[..., K0:]
        assert bool((src[..., K0:][sem_valid] >= N).all())
        assert bool((dy[..., K0:][sem_valid].abs() <= 1).all())
        assert bool((dx[..., K0:][sem_valid].abs() <= 1).all())


# ---------------------------------------------------------------------------
# 7. fp16 forward is NaN-free; modules raise instead of sanitising
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_fp16_forward_is_finite(name):
    """Half-precision CPU forward returns finite values (softmax runs in fp32)."""
    m = build(name, C=8).half()
    f_ir, f_rgb = rand_pair(dtype=torch.float16)
    out = m(f_ir, f_rgb)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all(), f"{name} produced non-finite fp16 output"
    if name in RESIDUAL_NAMES:
        assert torch.isfinite(m.last_delta).all()


def test_no_silent_nan_sanitising():
    """The sources never *call* ``nan_to_num`` nor import forbidden libraries."""
    forbidden_roots = ("ultralytics", "torch_geometric", "dgl")
    for mod in (fuse_modules, neighbors):
        tree = ast.parse(Path(mod.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            for nm in names:
                assert not nm.split(".")[0] in forbidden_roots, f"{mod.__name__} imports {nm}"
            if isinstance(node, ast.Call):
                func = node.func
                called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                assert called != "nan_to_num", f"{mod.__name__} calls nan_to_num"


# ---------------------------------------------------------------------------
# 8. Backward: every parameter gets a finite, non-zero gradient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_backward_all_params_finite_nonzero(name):
    """After one backward every trainable parameter has grad != 0 and finite."""
    torch.manual_seed(0)
    m = build(name, C=8)
    f_ir, f_rgb = rand_pair()
    loss = m(f_ir, f_rgb).square().sum()
    loss.backward()

    missing, nonfinite, zero = [], [], []
    for pname, p in m.named_parameters():
        if p.grad is None:
            missing.append(pname)
        elif not torch.isfinite(p.grad).all():
            nonfinite.append(pname)
        elif p.grad.abs().max().item() == 0.0:
            zero.append(pname)
    assert not missing, f"{name} missing grads: {missing}"
    assert not nonfinite, f"{name} non-finite grads: {nonfinite}"
    assert not zero, f"{name} zero grads: {zero}"


# ---------------------------------------------------------------------------
# 9. Determinism and architecture checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODULE_NAMES)
def test_repeated_forward_is_deterministic(name):
    """Fixed tables + stable top-k => repeated eval forwards are identical."""
    torch.manual_seed(0)
    m = build(name, C=8).eval()
    f_ir, f_rgb = rand_pair()
    with torch.no_grad():
        a = m(f_ir, f_rgb)
        b = m(f_ir, f_rgb)
    assert torch.equal(a, b)


def test_e21_uses_block_shared_qkv():
    """E21 has exactly one q, k and v module, shared by both directions."""
    m = FuseBiCrossAttention(8)
    names = [n for n, _ in m.named_modules()]
    assert names.count("q") == 1
    assert names.count("k") == 1
    assert names.count("v") == 1
    assert not any(n.startswith(("q_", "k_", "v_")) for n in names)
    # The same parameter tensor appears once in the parameter list.
    assert sum(p is m.q.weight for p in m.parameters()) == 1
    assert sum(p is m.k.weight for p in m.parameters()) == 1
    assert sum(p is m.v.weight for p in m.parameters()) == 1

    # Functional check: each shared projection is invoked once per direction.
    counts = {"q": 0, "k": 0, "v": 0}
    handles = []
    for nm in counts:
        def _hook(_mod, _inp, _out, nm=nm):
            counts[nm] += 1
        handles.append(getattr(m, nm).register_forward_hook(_hook))
    f_ir, f_rgb = rand_pair()
    with torch.no_grad():
        m(f_ir, f_rgb)
    for h in handles:
        h.remove()
    assert counts == {"q": 2, "k": 2, "v": 2}


# ---------------------------------------------------------------------------
# 10. Parameter counts match the documented formulas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODULE_NAMES)
@pytest.mark.parametrize("C", [3, 8, 16])
def test_param_count_matches_formula(name, C):
    """Measured parameter count equals the closed-form formula and is > 0."""
    torch.manual_seed(0)
    m = build(name, C=C)
    n = count_parameters(m)
    exp = expected_params(name, C=C, d=DEFAULT_D, E=EDGE_DIM)
    assert n > 0
    assert n == exp, f"{name}(C={C}): {n} params, formula says {exp}"
    # Also report it so the pytest log carries the numbers.
    print(f"{name}(C={C}, d={DEFAULT_D}): {n} params")


def test_e35_has_same_params_as_e30():
    """The semantic edges add no parameters (E35 = E30 + edges)."""
    torch.manual_seed(0)
    e30 = FuseGATLocal(8, d=DEFAULT_D)
    e35 = FuseGATSemantic(8, d=DEFAULT_D)
    assert count_parameters(e30) == count_parameters(e35)
