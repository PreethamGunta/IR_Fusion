"""Neighbour-index builders for the GF-YOLO graph-fusion testbed (experiment 09).

This module is the *geometry* half of the E30/E35 graph rungs.  It knows
nothing about networks: it only turns an ``(H, W)`` feature map into fixed,
in-bounds gather tables that the GAT modules in :mod:`fuse_modules` consume.

Why hand-rolled tables?
-----------------------
The experiment plan (``EXPERIMENT_PLAN.md`` sec. 6/10) requires fixed
neighbour-index tensors plus ``gather`` and explicitly forbids depending on a
graph library.  Sparse, precomputable indices also make the border behaviour
testable: every node's valid-neighbour count is fixed by its ``(y, x)``
position, so a wrong clamp or axis swap shows up immediately as a bad count
or an unintended self-loop.

Graph layout used by E30/E35
----------------------------
Nodes are the ``H*W`` positions of the **IR** feature map, flattened in
row-major order (``flat = y * W + x``).  Each IR node ``i`` receives messages
from

* ``grid``  -- its 8 grid neighbours *within the IR map* (no self edge), and
* ``window`` -- the RGB nodes in a ``(2r+1)^2`` square window centred on
  ``i`` (the co-located RGB node at ``(0, 0)`` is a legitimate cross-modal
  edge, so it is included by default).

E35 additionally receives ``semantic`` edges: the top-``k`` RGB nodes by
cosine similarity inside a Chebyshev radius ``R``.  Those indices depend on
the features, so they are built per forward pass, but the candidate window is
still the fixed table produced here and selection is a deterministic,
stable-sort gather.

Conventions
-----------
* ``index`` tensors contain *global* node ids only in :func:`ir_graph_index`;
  :func:`grid8_index`, :func:`window_index` and :func:`semantic_candidates`
  return local ids in ``[0, H*W)``.
* In the combined table, IR ids are ``0 .. N-1`` and RGB ids are
  ``N .. 2N-1`` (the RGB node ``j`` has global id ``j + N``).
* Offset convention: ``dy = y_source - y_target`` and
  ``dx = x_source - x_target`` (source minus target).
* Invalid slots are filled with a *different* in-bounds node id (never the
  target itself, for ``N > 1``; a 1x1 map has no alternative) so that even the
  raw, unmasked table has no accidental self-loop.  Callers must still respect
  the ``valid`` mask.

All tables are deterministic and cached per ``(H, W, r/R)``.  The tensors are
shared between callers: treat them as read-only and move them with ``.to()``
(which copies) rather than editing them in place.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Table container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeighborTable:
    """A fixed neighbour table for ``H*W`` target nodes.

    Attributes
    ----------
    index:
        ``LongTensor`` of shape ``[N, K]`` with source node ids in
        ``[0, N)``.  Invalid slots contain a harmless in-bounds filler.
    valid:
        ``BoolTensor`` of shape ``[N, K]``; ``True`` where the slot is a real
        neighbour of the target.  Softmax must be restricted to these slots.
    dy, dx:
        ``LongTensor`` of shape ``[N, K]``; integer cell offsets
        ``source - target`` (row, column).
    """

    index: torch.Tensor
    valid: torch.Tensor
    dy: torch.Tensor
    dx: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.index.shape[0])

    @property
    def num_slots(self) -> int:
        return int(self.index.shape[1])

    def to(self, device: torch.device | str) -> "NeighborTable":
        """Return a copy of the table on ``device``."""
        return NeighborTable(
            index=self.index.to(device),
            valid=self.valid.to(device),
            dy=self.dy.to(device),
            dx=self.dx.to(device),
        )


@dataclass(frozen=True)
class IRGraphTable:
    """Combined IR-grid + cross-modal-window table used by E30/E35.

    ``K0 = 8 + (2r+1)^2`` slots are laid out as
    ``[grid slots (8) | window slots ((2r+1)^2)]``.
    """

    src: torch.Tensor  # [N, K0] global ids in [0, 2N)
    valid: torch.Tensor  # [N, K0]
    is_cross: torch.Tensor  # [N, K0]
    dy: torch.Tensor  # [N, K0]
    dx: torch.Tensor  # [N, K0]
    grid_count: torch.Tensor  # [N]
    window_count: torch.Tensor  # [N]

    @property
    def num_slots(self) -> int:
        return int(self.src.shape[1])

    def to(self, device: torch.device | str) -> "IRGraphTable":
        return IRGraphTable(
            src=self.src.to(device),
            valid=self.valid.to(device),
            is_cross=self.is_cross.to(device),
            dy=self.dy.to(device),
            dx=self.dx.to(device),
            grid_count=self.grid_count.to(device),
            window_count=self.window_count.to(device),
        )


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def node_coords(H: int, W: int, device: torch.device | str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(ys, xs)`` of shape ``[H*W]`` for row-major flattened nodes."""
    ys = torch.arange(H, device=device, dtype=torch.long).repeat_interleave(W)
    xs = torch.arange(W, device=device, dtype=torch.long).repeat(H)
    return ys, xs


def flat_to_yx(flat: torch.Tensor, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map flat node ids to ``(y, x)``.  Inverse of :func:`yx_to_flat`."""
    return torch.div(flat, W, rounding_mode="floor"), torch.remainder(flat, W)


def yx_to_flat(y: torch.Tensor, x: torch.Tensor, W: int) -> torch.Tensor:
    """Map ``(y, x)`` coordinates to flat node ids.  Inverse of :func:`flat_to_yx`."""
    return y * W + x


def border_class(H: int, W: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Classify each node: ``0`` corner, ``1`` edge (non-corner), ``2`` interior."""
    ys, xs = node_coords(H, W, device=device)
    on_edge = (ys == 0) | (ys == H - 1) | (xs == 0) | (xs == W - 1)
    on_corner = ((ys == 0) | (ys == H - 1)) & ((xs == 0) | (xs == W - 1))
    out = torch.full_like(ys, 2)
    out[on_edge] = 1
    out[on_corner] = 0
    return out


def _offsets(r: int, include_center: bool) -> List[Tuple[int, int]]:
    """Square-window offsets in deterministic row-major order."""
    if r < 0:
        raise ValueError(f"radius must be >= 0, got {r}")
    offs: List[Tuple[int, int]] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if not include_center and dy == 0 and dx == 0:
                continue
            offs.append((dy, dx))
    return offs


def _build_table(H: int, W: int, offsets: Sequence[Tuple[int, int]]) -> NeighborTable:
    """Build a table for a fixed list of offsets over an ``H x W`` grid."""
    if H <= 0 or W <= 0:
        raise ValueError(f"H and W must be positive, got {H}x{W}")
    N = H * W
    ys, xs = node_coords(H, W)
    target = torch.arange(N, dtype=torch.long)
    # Filler for out-of-bounds slots: the cyclic successor, which is a
    # different node whenever N > 1 and is always in bounds.
    filler = torch.remainder(target + 1, N)

    idx_cols: List[torch.Tensor] = []
    valid_cols: List[torch.Tensor] = []
    dy_cols: List[torch.Tensor] = []
    dx_cols: List[torch.Tensor] = []
    for dy, dx in offsets:
        sy = ys + dy
        sx = xs + dx
        ok = (sy >= 0) & (sy < H) & (sx >= 0) & (sx < W)
        flat = yx_to_flat(sy.clamp(0, H - 1), sx.clamp(0, W - 1), W)
        flat = torch.where(ok, flat, filler)
        idx_cols.append(flat)
        valid_cols.append(ok)
        dy_cols.append(torch.full((N,), dy, dtype=torch.long))
        dx_cols.append(torch.full((N,), dx, dtype=torch.long))

    table = NeighborTable(
        index=torch.stack(idx_cols, dim=1),
        valid=torch.stack(valid_cols, dim=1),
        dy=torch.stack(dy_cols, dim=1),
        dx=torch.stack(dx_cols, dim=1),
    )
    assert_indices_in_bounds(table.index, N, "table.index")
    return table


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=128)
def grid8_index(H: int, W: int) -> NeighborTable:
    """8-neighbour grid table (no centre / self offset).

    Slot order is row-major over ``dy, dx in {-1, 0, 1}`` with ``(0, 0)``
    removed: ``(-1,-1), (-1,0), (-1,+1), (0,-1), (0,+1), (+1,-1), (+1,0),
    (+1,+1)``.  Valid-neighbour counts are 3 at corners, 5 on edges and 8 in
    the interior (for ``H, W >= 3``).
    """
    return _build_table(H, W, _offsets(1, include_center=False))


@functools.lru_cache(maxsize=128)
def window_index(H: int, W: int, r: int, include_center: bool = True) -> NeighborTable:
    """Window table with ``(2r+1)^2`` slots (optionally minus the centre).

    The centre offset is a legitimate cross-modal edge (RGB node at the same
    location), hence ``include_center=True`` by default.  It is *not* a
    self-loop in the graph because the source lives in the other modality.
    """
    return _build_table(H, W, _offsets(r, include_center=include_center))


@functools.lru_cache(maxsize=128)
def semantic_candidates(H: int, W: int, R: int, include_center: bool = True) -> NeighborTable:
    """Fixed RGB candidate window of Chebyshev radius ``R`` for semantic kNN.

    This is just :func:`window_index`; the distinct name documents intent and
    lets callers override caching separately.  Every returned id is an RGB
    node id in ``[0, H*W)`` -- the caller adds the modality offset ``N``.
    """
    return window_index(H, W, R, include_center=include_center)


@functools.lru_cache(maxsize=64)
def ir_graph_index(H: int, W: int, r: int = 1) -> IRGraphTable:
    """Combined IR graph table for E30 (and the base of E35).

    Slot layout: 8 IR-grid slots followed by ``(2r+1)^2`` cross-modal window
    slots.  * ``src`` uses global ids: IR ``0..N-1``, RGB ``N..2N-1``.
    * ``grid_count`` / ``window_count`` are the per-node valid counts.
    """
    grid = grid8_index(H, W)
    win = window_index(H, W, r, include_center=True)
    N = H * W
    zeros = torch.zeros_like(grid.valid)
    ones = torch.ones_like(win.valid)
    src = torch.cat([grid.index, win.index + N], dim=1)
    assert_indices_in_bounds(src, 2 * N, "ir_graph.src")
    return IRGraphTable(
        src=src,
        valid=torch.cat([grid.valid, win.valid], dim=1),
        is_cross=torch.cat([zeros, ones], dim=1),
        dy=torch.cat([grid.dy, win.dy], dim=1),
        dx=torch.cat([grid.dx, win.dx], dim=1),
        grid_count=grid.valid.sum(dim=1),
        window_count=win.valid.sum(dim=1),
    )


# ---------------------------------------------------------------------------
# Semantic kNN (E35)
# ---------------------------------------------------------------------------


def semantic_knn_index(
    feat_ir: torch.Tensor,
    feat_rgb: torch.Tensor,
    H: int,
    W: int,
    radius: int = 4,
    k: int = 6,
    include_center: bool = True,
    chunk: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-``k`` RGB neighbours per IR node by cosine similarity in radius ``R``.

    Parameters
    ----------
    feat_ir, feat_rgb:
        Projected node features, shape ``[B, N, d]`` with ``N = H*W``.  Only
        ``feat_rgb`` is searched -- the function never reads IR features as
        candidates, which is the "wrong modality" failure mode the plan warns
        about.
    H, W:
        Spatial size used to build the fixed candidate window.
    radius:
        Chebyshev radius ``R`` in cells; candidates are the RGB nodes in
        ``dy, dx in [-R, R]``.
    k:
        Number of neighbours per IR node.  Must be ``<=`` candidate slots.
    include_center:
        Keep the co-located RGB node as a candidate (default).  Set to
        ``False`` for purely displacement-aware semantic edges.
    chunk:
        Query-node chunk size; a fixed, data-independent Python loop keeps
        peak memory at ``O(chunk * M * d)`` instead of ``O(N * M * d)``.

    Returns
    -------
    index:
        ``LongTensor [B, N, k]`` with RGB node ids in ``[0, N)``.
    valid:
        ``BoolTensor [B, N, k]``; ``False`` when a border node has fewer
        than ``k`` candidates in the window.  Invalid slots still contain
        in-bounds ids (safe to gather, must be masked).

    Determinism
    -----------
    Scores are computed in float32, cosine similarities are normalised with a
    fixed epsilon, and ties are broken by a *stable* descending sort, so
    repeated calls on the same tensors return bit-identical indices.
    """
    if feat_ir.shape != feat_rgb.shape:
        raise ValueError(f"feature shapes differ: {tuple(feat_ir.shape)} vs {tuple(feat_rgb.shape)}")
    if feat_ir.dim() != 3:
        raise ValueError(f"expected [B, N, d] features, got {tuple(feat_ir.shape)}")
    B, N, _ = feat_ir.shape
    if N != H * W:
        raise ValueError(f"N={N} does not match H*W={H * W}")
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    cand = semantic_candidates(H, W, radius, include_center=include_center)
    M = cand.num_slots
    if k < 1 or k > M:
        raise ValueError(f"k must be in [1, {M}] for radius {radius}, got {k}")

    device = feat_ir.device
    cand_idx = cand.index.to(device)  # [N, M]
    cand_valid = cand.valid.to(device)
    irn = F.normalize(feat_ir.float(), dim=-1, eps=1e-8)
    rgbn = F.normalize(feat_rgb.float(), dim=-1, eps=1e-8)

    idx_chunks: List[torch.Tensor] = []
    val_chunks: List[torch.Tensor] = []
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        ci = cand_idx[i0:i1]  # [n, M]
        cv = cand_valid[i0:i1]
        # [B, n, M, d] -- gather only RGB features, enforces the modality.
        cand_feat = rgbn[:, ci]
        scores = torch.einsum("bnd,bnmd->bnm", irn[:, i0:i1], cand_feat)
        scores = scores.masked_fill(~cv.unsqueeze(0), float("-inf"))
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k]
        ci_b = ci.unsqueeze(0).expand(B, -1, -1)
        cv_b = cv.unsqueeze(0).expand(B, -1, -1)
        idx_chunks.append(torch.gather(ci_b, 2, order))
        val_chunks.append(torch.gather(cv_b, 2, order))

    index = torch.cat(idx_chunks, dim=1)
    valid = torch.cat(val_chunks, dim=1)
    assert_indices_in_bounds(index, N, "semantic_knn.index")
    return index, valid


# ---------------------------------------------------------------------------
# Verification helpers (used by tests and by defensive module code)
# ---------------------------------------------------------------------------


def assert_indices_in_bounds(index: torch.Tensor, num_nodes: int, name: str = "index") -> None:
    """Raise ``ValueError`` if any id is outside ``[0, num_nodes)``."""
    if index.numel() == 0:
        return
    lo = int(index.min())
    hi = int(index.max())
    if lo < 0 or hi >= num_nodes:
        raise ValueError(f"{name}: ids must be in [0, {num_nodes}), got [{lo}, {hi}]")


def assert_no_self_loops(index: torch.Tensor, valid: torch.Tensor, name: str = "edges") -> None:
    """Raise ``ValueError`` if a *valid* slot points at its own target."""
    N, K = index.shape
    target = torch.arange(N, device=index.device).unsqueeze(1).expand(N, K)
    bad = valid & (index == target)
    if bool(bad.any()):
        n_bad = int(bad.sum())
        first = tuple(int(v) for v in bad.nonzero()[0].tolist())
        raise ValueError(f"{name}: {n_bad} unintended self-loop(s), first at {first}")


def border_counts(valid: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Return ``[count_corner, count_edge, count_interior]`` sums of valid slots.

    With a 3-class node partition this gives a compact border regression test:
    for the 8-neighbour grid the expected vector is
    ``[4*3, (2H+2W-8)*5, (H-2)*(W-2)*8]``.
    """
    counts = valid.sum(dim=1)
    cls = border_class(H, W, device=valid.device)
    return torch.stack([counts[cls == c].sum() for c in range(3)])


def gather_node_features(nodes: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Gather ``[B, 2N, d]`` node features with ``[B, N, K]`` ids.

    Convenience for the GAT modules; ``src`` must already contain modality
    offsets and be in bounds.
    """
    B, _num_nodes, d = nodes.shape
    N, K = src.shape[1], src.shape[2]
    flat = src.reshape(B, N * K).unsqueeze(-1).expand(-1, -1, d)
    return nodes.gather(1, flat).reshape(B, N, K, d)
