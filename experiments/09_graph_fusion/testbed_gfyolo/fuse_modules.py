"""Pure-PyTorch fusion modules for the GF-YOLO testbed (experiment 09).

Five rungs of the experiment-09 ladder, implemented without ultralytics, PyG
or any experiment-08 import.  Each module implements the common contract:

    module = FuseXxx(C, ...)
    f_fused = module(f_ir, f_rgb)

with ``f_ir``, ``f_rgb`` of shape ``[B, C, H, W]`` and ``f_fused`` the same
shape.  Channel 0 in the detector stem is IR and channels 3.. are RGB; at this
level the module receives pre-split feature maps, so it only cares that the
first argument is the IR stream and the second is the RGB stream.

Residual modules (E14, E21, E30, E35) additionally expose
``self.last_delta`` with shape ``[B, C, H, W]`` such that

    f_fused == f_ir + self.last_delta

value-wise.  ``last_delta`` is stored **detached** so that keeping it around
for probes/hooks does not pin the autograd graph; recompute the forward (or
re-run with gradients) if a differentiable Delta is needed.

Design notes
------------
* *Gate initialisation.*  The plan asks for a negative gate bias so the
  residual model starts close to the IR-only reference.  A bias of ``-4.0``
  alone gives ``sigmoid(-4) = 0.018``, which is **not** within ``1e-6`` of
  ``f_ir``.  To satisfy the strict "identity at step 0" test while keeping
  every parameter's gradient non-zero, the final output projection is
  initialised with a very small random scale (``1e-7`` for E14, ``1e-6`` for
  E21).  The gate bias remains exactly ``-4.0`` as specified.  E30/E35 use a
  less extreme ``1e-3`` output scale -- they only require a negative-bias
  gate, not exact identity.
* *Numerical safety.*  Softmax logits are computed in float32 (even when the
  module runs in fp16) and every forward checks the fused output for NaNs and
  raises ``FloatingPointError``.  There is deliberately no ``nan_to_num``.
* *Determinism.*  Neighbour tables come from :mod:`neighbors` (cached,
  fixed), semantic top-k uses a stable sort, and there are no data-dependent
  Python loops over the batch.

Parameter-count formulas below use ``C`` = channel count, ``d`` = graph /
attention width (default 128), ``E = 5`` edge features (five for E30/E35).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # package import (``import testbed_gfyolo``)
    from .neighbors import (
        gather_node_features,
        ir_graph_index,
        semantic_knn_index,
    )
except ImportError:  # direct import (``PYTHONPATH=. pytest test_fuse_modules.py``)
    from neighbors import (
        gather_node_features,
        ir_graph_index,
        semantic_knn_index,
    )

__all__ = [
    "FuseConcat1x1",
    "FuseGatedResidual",
    "FuseBiCrossAttention",
    "FuseGATLocal",
    "FuseGATSemantic",
    "FuseGATTwoHop",
    "count_parameters",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Number of scalar parameters in ``module`` (optionally trainable only)."""
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def _check_pair(f_ir: torch.Tensor, f_rgb: torch.Tensor) -> Tuple[int, int, int, int]:
    """Validate the ``[B, C, H, W]`` contract and return its sizes."""
    if f_ir.dim() != 4 or f_rgb.dim() != 4:
        raise ValueError(f"expected 4-D [B,C,H,W] inputs, got {tuple(f_ir.shape)} and {tuple(f_rgb.shape)}")
    if f_ir.shape != f_rgb.shape:
        raise ValueError(f"IR/RGB shapes differ: {tuple(f_ir.shape)} vs {tuple(f_rgb.shape)}")
    B, C, H, W = f_ir.shape
    if B < 1 or C < 1 or H < 1 or W < 1:
        raise ValueError(f"empty input {tuple(f_ir.shape)}")
    if f_ir.device != f_rgb.device:
        raise ValueError(f"IR/RGB live on different devices: {f_ir.device} vs {f_rgb.device}")
    if f_ir.dtype != f_rgb.dtype:
        raise ValueError(f"IR/RGB dtypes differ: {f_ir.dtype} vs {f_rgb.dtype}")
    return B, C, H, W


def _raise_on_nonfinite(tensor: torch.Tensor, name: str) -> None:
    """Log and raise if ``tensor`` has a NaN/Inf; never silent sanitising."""
    if not bool(torch.isfinite(tensor).all()):
        n_bad = int((~torch.isfinite(tensor)).sum())
        msg = f"{name} contains {n_bad} non-finite value(s); refusing to continue"
        logger.error(msg)
        raise FloatingPointError(msg)


def _masked_softmax(logits: torch.Tensor, valid: torch.Tensor, name: str) -> torch.Tensor:
    """Softmax over valid neighbours only.

    ``logits`` is ``[..., K]`` float32, ``valid`` is ``[..., K]`` bool.  A
    row with no valid neighbour would make softmax NaN, so it is a hard error
    rather than something to clamp.
    """
    if not bool(valid.any(dim=-1).all()):
        msg = f"{name}: at least one target node has no valid neighbour"
        logger.error(msg)
        raise ValueError(msg)
    masked = logits.masked_fill(~valid, float("-inf"))
    attn = torch.softmax(masked, dim=-1)
    if not bool(torch.isfinite(attn).all()):
        msg = f"{name}: non-finite attention weights after masked softmax"
        logger.error(msg)
        raise FloatingPointError(msg)
    return attn


def _small_init(linear_or_conv: nn.Module, std: float) -> None:
    """Zero-bias, small-normal-weight init for a 1x1 conv or linear output head."""
    nn.init.normal_(linear_or_conv.weight, mean=0.0, std=std)
    if getattr(linear_or_conv, "bias", None) is not None:
        nn.init.zeros_(linear_or_conv.bias)


# ---------------------------------------------------------------------------
# E10 -- concat + 1x1
# ---------------------------------------------------------------------------


class FuseConcat1x1(nn.Module):
    """E10: ``f_fused = conv1x1([f_ir, f_rgb])``.

    Hypothesis
        A learned linear mixing of the two streams is enough to beat the
        IR-only reference when the modalities are complementary.
    Parent rung
        None (ladder base; E00/E01 are the single-stream references).
    The one change
        Concatenate channels and project back with one 1x1 convolution.

    Parameters
        ``P = 2*C^2 + C`` (weights ``2C x C x 1 x 1`` plus ``C`` biases).
    """

    def __init__(self, C: int) -> None:
        super().__init__()
        if C < 1:
            raise ValueError(f"C must be >= 1, got {C}")
        self.C = int(C)
        self.conv = nn.Conv2d(2 * C, C, kernel_size=1, bias=True)

    def forward(self, f_ir: torch.Tensor, f_rgb: torch.Tensor) -> torch.Tensor:
        _check_pair(f_ir, f_rgb)
        return self.conv(torch.cat([f_ir, f_rgb], dim=1))


# ---------------------------------------------------------------------------
# E14 -- IR-anchored gated residual
# ---------------------------------------------------------------------------


class FuseGatedResidual(nn.Module):
    """E14: IR-anchored residual with a negative-bias gate.

    Hypothesis
        Letting the network learn a small, gated correction on top of the IR
        features is a safer inductive bias than replacing them; at step 0 the
        model is the IR-only reference.
    Parent rung
        E10 (concat + 1x1).
    The one change
        The 1x1 projection becomes a gated residual
        ``f_ir + sigmoid(g([f_ir, f_rgb])) * conv([f_ir, f_rgb])`` with the
        gate-conv bias initialised to ``-4.0``.

    Parameters
        ``P = 4*C^2 + 2*C`` (two ``2C x C x 1 x 1`` convolutions).

    Notes
        The output convolution uses a small random init (``1e-7``) so the
        correction is ``< 1e-6`` in magnitude at step 0; see the module
        docstring at the top of the file for why the bias alone cannot do
        that.  ``self.last_delta`` holds the detached correction.
    """

    def __init__(self, C: int, gate_init_bias: float = -4.0, output_init_std: float = 1e-7) -> None:
        super().__init__()
        if C < 1:
            raise ValueError(f"C must be >= 1, got {C}")
        self.C = int(C)
        self.gate_init_bias = float(gate_init_bias)
        self.gate = nn.Conv2d(2 * C, C, kernel_size=1, bias=True)
        self.out = nn.Conv2d(2 * C, C, kernel_size=1, bias=True)
        nn.init.constant_(self.gate.bias, self.gate_init_bias)
        _small_init(self.out, output_init_std)
        self.last_delta: Optional[torch.Tensor] = None

    def forward(self, f_ir: torch.Tensor, f_rgb: torch.Tensor) -> torch.Tensor:
        _check_pair(f_ir, f_rgb)
        z = torch.cat([f_ir, f_rgb], dim=1)
        delta = torch.sigmoid(self.gate(z)) * self.out(z)
        fused = f_ir + delta
        _raise_on_nonfinite(delta, "E14 delta")
        _raise_on_nonfinite(fused, "E14 fused output")
        self.last_delta = delta.detach()
        return fused


# ---------------------------------------------------------------------------
# E21 -- bidirectional block-shared cross-attention
# ---------------------------------------------------------------------------


class FuseBiCrossAttention(nn.Module):
    """E21: bidirectional IR<->RGB cross-attention (DMFF-style, own code).

    Hypothesis
        Dense, content-based exchange lets each modality query the other
        rather than mixing fixed channels; sharing Q/K/V across directions
        keeps the capacity honest and matches the DMFF block structure.
    Parent rung
        E14.
    The one change
        Replace the concat path with single-head scaled dot-product
        cross-attention over the flattened ``H*W`` tokens, applied in both
        directions with the **same** ``q``, ``k``, ``v`` modules, followed by
        the E14 gated residual in C channels.

    Parameters
        ``P = 4*C*d + 3*d^2 + 4*C^2 + 5*d + 4*C`` for width ``d``:
        two input projections ``C->d``, shared ``q/k/v`` ``d->d``, two output
        projections ``d->C``, and the E14 gate/output pair.

    Notes
        Each modality is updated residually in the ``d``-space
        (``h + attention(h, other)``) and projected back to ``C`` before the
        gate.  Attention is computed in float32 and chunked over queries, so
        the dense ``N x N`` map is never materialised for training.  Set
        ``collect_attn = True`` to store the (detached) maps for tests/probes.
    """

    def __init__(
        self,
        C: int,
        d: int = 128,
        gate_init_bias: float = -4.0,
        output_init_std: float = 1e-6,
        attn_chunk: int = 512,
    ) -> None:
        super().__init__()
        if C < 1 or d < 1:
            raise ValueError(f"C and d must be positive, got C={C}, d={d}")
        if attn_chunk < 1:
            raise ValueError(f"attn_chunk must be >= 1, got {attn_chunk}")
        self.C = int(C)
        self.d = int(d)
        self.attn_chunk = int(attn_chunk)
        self.scale = float(d) ** -0.5

        self.proj_ir = nn.Conv2d(C, d, kernel_size=1, bias=True)
        self.proj_rgb = nn.Conv2d(C, d, kernel_size=1, bias=True)
        # Block-shared projections: the *same* modules serve both directions.
        self.q = nn.Linear(d, d, bias=True)
        self.k = nn.Linear(d, d, bias=True)
        self.v = nn.Linear(d, d, bias=True)
        self.to_c_ir = nn.Linear(d, C, bias=True)
        self.to_c_rgb = nn.Linear(d, C, bias=True)

        self.gate = nn.Conv2d(2 * C, C, kernel_size=1, bias=True)
        self.out = nn.Conv2d(2 * C, C, kernel_size=1, bias=True)
        nn.init.constant_(self.gate.bias, float(gate_init_bias))
        _small_init(self.out, output_init_std)

        self.collect_attn: bool = False
        self.last_attn: Optional[Dict[str, torch.Tensor]] = None
        self.last_delta: Optional[torch.Tensor] = None

    def _attend(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Single-head scaled dot-product attention, chunked over queries.

        ``q`` is ``[B, Nq, d]``; ``k``, ``v`` are ``[B, Nk, d]``.  Returns the
        update plus the optional full attention map (only when
        ``self.collect_attn``).
        """
        Nq = q.shape[1]
        kf = k.float()
        vf = v.float()
        outs = []
        maps = [] if self.collect_attn else None
        for i0 in range(0, Nq, self.attn_chunk):
            i1 = min(i0 + self.attn_chunk, Nq)
            scores = torch.matmul(q[:, i0:i1].float(), kf.transpose(-1, -2)) * self.scale
            attn = torch.softmax(scores, dim=-1)
            if maps is not None:
                maps.append(attn)
            outs.append(torch.matmul(attn, vf))
        out = torch.cat(outs, dim=1).to(q.dtype)
        full = torch.cat(maps, dim=1) if maps is not None else None
        return out, full

    def forward(self, f_ir: torch.Tensor, f_rgb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = _check_pair(f_ir, f_rgb)
        ir = self.proj_ir(f_ir).flatten(2).transpose(1, 2)  # [B, N, d]
        rgb = self.proj_rgb(f_rgb).flatten(2).transpose(1, 2)

        q_ir, k_ir, v_ir = self.q(ir), self.k(ir), self.v(ir)
        q_rgb, k_rgb, v_rgb = self.q(rgb), self.k(rgb), self.v(rgb)

        upd_ir, attn_ir = self._attend(q_ir, k_rgb, v_rgb)
        upd_rgb, attn_rgb = self._attend(q_rgb, k_ir, v_ir)

        ir_upd = self.to_c_ir(ir + upd_ir).transpose(1, 2).reshape(B, C, H, W)
        rgb_upd = self.to_c_rgb(rgb + upd_rgb).transpose(1, 2).reshape(B, C, H, W)

        z = torch.cat([ir_upd, rgb_upd], dim=1)
        delta = torch.sigmoid(self.gate(z)) * self.out(z)
        fused = f_ir + delta
        _raise_on_nonfinite(delta, "E21 delta")
        _raise_on_nonfinite(fused, "E21 fused output")

        self.last_delta = delta.detach()
        if self.collect_attn:
            assert attn_ir is not None and attn_rgb is not None
            self.last_attn = {"ir": attn_ir.detach(), "rgb": attn_rgb.detach()}
        else:
            self.last_attn = None
        return fused


# ---------------------------------------------------------------------------
# E30 -- local graph attention (IR grid + cross-modal window)
# ---------------------------------------------------------------------------


class FuseGATLocal(nn.Module):
    """E30: GAT over IR nodes with grid and cross-modal window edges.

    Hypothesis
        Explicit neighbour-to-neighbour messages (instead of dense attention)
        capture local geometry and cross-modal correspondence more cheaply,
        and are the right substrate for adding non-local semantic edges.
    Parent rung
        E14.
    The one change
        The ``conv([f_ir, f_rgb])`` correction is produced by one hand-rolled
        GAT layer on projected nodes (``d=128``): 8-neighbour grid edges
        inside the IR map plus ``(2r+1)^2`` RGB->IR window edges (default
        ``r=1``), with edge features ``(dx, dy, cos, q_i, q_j)``.

    Parameters
        With ``E = 5``, ``P = 3*C*d + 5*d^2 + 6*d + E^2 + 2*E + 2*C + 1``:
        two ``C->d`` projections, one shared message transform ``W``,
        ``W_v``, one ``(2d+E)->1`` attention vector, ``E->E`` edge projection,
        a two-layer gate MLP, a ``C->1`` quality head and a ``d->C`` output
        projection.

    Message / update
        ``logit_ij = LeakyReLU(a^T [W h_i || W h_j || W_e e_ij])``,
        softmax over **valid** neighbours only,
        ``h_i' = h_i + g_i * sum_j alpha_ij W_v h_j``,
        ``g_i = sigmoid(MLP([h_i, agg_i]))`` with the last MLP bias at
        ``-4.0``, and ``f_fused = f_ir + out(h')``.

    Notes
        Only IR nodes are updated in E30; RGB nodes are sources.  Edge
        offsets are normalised by ``max(r, 1)``.  ``self.last_delta`` is the
        detached correction; set ``collect_attn = True`` to store attention
        weights, the validity mask and the global source ids.
    """

    edge_dim: int = 5

    def __init__(
        self,
        C: int,
        d: int = 128,
        r: int = 1,
        gate_init_bias: float = -4.0,
        output_init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        if C < 1 or d < 1:
            raise ValueError(f"C and d must be positive, got C={C}, d={d}")
        if r < 0:
            raise ValueError(f"r must be >= 0, got {r}")
        self.C = int(C)
        self.d = int(d)
        self.r = int(r)
        self.pos_scale = float(max(r, 1))

        # Node projections (one per modality: features are not interchangeable).
        self.proj_ir = nn.Conv2d(C, d, kernel_size=1, bias=True)
        self.proj_rgb = nn.Conv2d(C, d, kernel_size=1, bias=True)

        # Message transform W, edge transform W_e, value transform W_v.
        self.W = nn.Linear(d, d, bias=False)
        self.edge_proj = nn.Linear(self.edge_dim, self.edge_dim, bias=True)
        self.attn = nn.Linear(2 * d + self.edge_dim, 1, bias=False)
        nn.init.normal_(self.attn.weight, mean=0.0, std=0.1)
        self.leaky = nn.LeakyReLU(0.2)
        self.W_v = nn.Linear(d, d, bias=False)

        # Gate g_i = sigmoid(MLP([h_i, agg_i])), negative-bias init.
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d, d, bias=True),
            nn.GELU(),
            nn.Linear(d, d, bias=True),
        )
        nn.init.constant_(self.gate_mlp[-1].bias, float(gate_init_bias))

        # Small per-node quality head q_i in (0, 1), shared by both modalities.
        self.quality = nn.Conv2d(C, 1, kernel_size=1, bias=True)

        # Back to detector channels; small init keeps Delta modest at step 0.
        self.out_proj = nn.Linear(d, C, bias=True)
        _small_init(self.out_proj, output_init_std)

        self.collect_attn: bool = False
        self.last_attn: Optional[Dict[str, torch.Tensor]] = None
        self.last_edges: Optional[Dict[str, torch.Tensor]] = None
        self.last_delta: Optional[torch.Tensor] = None
        self._table_cache: Dict[Tuple[int, int, torch.device], dict] = {}

    def __getstate__(self):
        """Drop runtime caches from pickles (checkpoints).

        The table cache is keyed by ``(H, W, device)``. Pickling it into a
        checkpoint and reloading with ``map_location='cpu'`` moves the cached
        *tensors* to CPU while the key still says ``cuda:0`` -- the next
        forward then cache-hits and mixes CPU tables with CUDA features
        (E35 exit-time eval, 2026-11-23). Runtime caches must never be
        serialized.
        """
        state = self.__dict__.copy()
        for k in ("_table_cache", "last_attn", "last_edges", "last_delta"):
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._table_cache = {}
        self.last_attn = None
        self.last_edges = None
        self.last_delta = None

    # -- table management -------------------------------------------------

    def _tables(self, H: int, W: int, device: torch.device) -> dict:
        """Move/cache the static E30 graph table for ``(H, W, device)``."""
        key = (H, W, device)
        cached = self._table_cache.get(key)
        if cached is not None and cached["src"].device != torch.device(device):
            cached = None  # stale cache (e.g. loaded from a cpu-mapped ckpt)
        if cached is None:
            tbl = ir_graph_index(H, W, self.r).to(device)
            cached = {
                "src": tbl.src,
                "valid": tbl.valid,
                "dy": tbl.dy,
                "dx": tbl.dx,
                "grid_count": tbl.grid_count,
                "window_count": tbl.window_count,
            }
            self._table_cache[key] = cached
        return cached

    # -- edge construction ------------------------------------------------

    def _semantic_edges(
        self, h_ir: torch.Tensor, h_rgb: torch.Tensor, H: int, W: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Hook for E35: return ``(rgb_index [B,N,k], valid [B,N,k])`` or None."""
        return None

    def _build_edges(
        self, h_ir: torch.Tensor, h_rgb: torch.Tensor, H: int, W: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble global source ids, validity, and integer offsets.

        Returns ``src [B,N,K]`` (IR ids ``<N``, RGB ids ``>=N``),
        ``valid [B,N,K]``, ``dy [B,N,K]`` and ``dx [B,N,K]``.
        """
        B, N, _ = h_ir.shape
        tbl = self._tables(H, W, h_ir.device)
        src = tbl["src"].unsqueeze(0).expand(B, -1, -1)
        valid = tbl["valid"].unsqueeze(0).expand(B, -1, -1)
        dy = tbl["dy"].unsqueeze(0).expand(B, -1, -1)
        dx = tbl["dx"].unsqueeze(0).expand(B, -1, -1)

        sem = self._semantic_edges(h_ir, h_rgb, H, W)
        if sem is not None:
            sem_idx, sem_valid = sem
            if sem_idx.shape[0] != B or sem_idx.shape[1] != N:
                raise ValueError(f"semantic index shape {tuple(sem_idx.shape)} != [B={B}, N={N}, k]")
            ys = torch.arange(N, device=h_ir.device, dtype=torch.long) // W
            xs = torch.arange(N, device=h_ir.device, dtype=torch.long) % W
            sy = torch.div(sem_idx, W, rounding_mode="floor")
            sx = torch.remainder(sem_idx, W)
            src = torch.cat([src, sem_idx + N], dim=2)
            valid = torch.cat([valid, sem_valid], dim=2)
            dy = torch.cat([dy, sy - ys.view(1, N, 1)], dim=2)
            dx = torch.cat([dx, sx - xs.view(1, N, 1)], dim=2)
        return src, valid, dy, dx

    # -- forward ----------------------------------------------------------

    def forward(self, f_ir: torch.Tensor, f_rgb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = _check_pair(f_ir, f_rgb)
        N = H * W

        h_ir = self.proj_ir(f_ir).flatten(2).transpose(1, 2)  # [B, N, d]
        h_rgb = self.proj_rgb(f_rgb).flatten(2).transpose(1, 2)

        # Per-node quality q in (0,1) from a shared small conv head.
        q_ir = torch.sigmoid(self.quality(f_ir)).flatten(2).transpose(1, 2).squeeze(-1)
        q_rgb = torch.sigmoid(self.quality(f_rgb)).flatten(2).transpose(1, 2).squeeze(-1)

        src, valid, dy, dx = self._build_edges(h_ir, h_rgb, H, W)
        K = src.shape[2]

        nodes = torch.cat([h_ir, h_rgb], dim=1)  # [B, 2N, d]
        q_nodes = torch.cat([q_ir, q_rgb], dim=1)  # [B, 2N]

        w_nodes = self.W(nodes)
        w_i = w_nodes[:, :N]  # [B, N, d]
        w_j = gather_node_features(w_nodes, src)  # [B, N, K, d]
        v_j = gather_node_features(self.W_v(nodes), src)
        q_j = torch.gather(q_nodes, 1, src.reshape(B, N * K)).reshape(B, N, K)
        q_i = q_ir.unsqueeze(-1).expand(B, N, K)

        # Edge features: (dx, dy, cosine similarity of W h, q_i, q_j).
        # Cosine is computed in float32 for stability under fp16.
        wi_n = F.normalize(w_i.float(), dim=-1, eps=1e-6).unsqueeze(2)
        wj_n = F.normalize(w_j.float(), dim=-1, eps=1e-6)
        cos = torch.einsum("bnid,bnkd->bnk", wi_n, wj_n).to(h_ir.dtype)
        e = torch.stack(
            [
                dx.to(h_ir.dtype) / self.pos_scale,
                dy.to(h_ir.dtype) / self.pos_scale,
                cos,
                q_i.to(h_ir.dtype),
                q_j.to(h_ir.dtype),
            ],
            dim=-1,
        )  # [B, N, K, E]

        attn_in = torch.cat(
            [
                w_i.unsqueeze(2).expand(B, N, K, self.d),
                w_j,
                F.linear(e.float(), self.edge_proj.weight.float(), self.edge_proj.bias.float()).to(h_ir.dtype),
            ],
            dim=-1,
        )
        logits = F.linear(attn_in.float(), self.attn.weight.float(), None).squeeze(-1)
        logits = self.leaky(logits)
        attn = _masked_softmax(logits, valid, "E30 attention")

        agg = torch.einsum("bnk,bnkd->bnd", attn, v_j.float()).to(h_ir.dtype)
        gate = torch.sigmoid(self.gate_mlp(torch.cat([h_ir, agg], dim=-1)))
        h_new = h_ir + gate * agg

        delta = self.out_proj(h_new).transpose(1, 2).reshape(B, C, H, W)
        fused = f_ir + delta
        _raise_on_nonfinite(delta, "E30 delta")
        _raise_on_nonfinite(fused, "E30 fused output")

        self.last_delta = delta.detach()
        if self.collect_attn:
            self.last_attn = {
                "weights": attn.detach(),
                "valid": valid.detach(),
                "src": src.detach(),
                "is_cross": (src >= N).detach(),
            }
            self.last_edges = {
                "src": src.detach(),
                "valid": valid.detach(),
                "is_cross": (src >= N).detach(),
                "dy": dy.detach(),
                "dx": dx.detach(),
            }
        else:
            self.last_attn = None
            self.last_edges = None
        return fused


# ---------------------------------------------------------------------------
# E35 -- GAT + semantic-kNN cross-modal edges
# ---------------------------------------------------------------------------


class FuseGATSemantic(FuseGATLocal):
    """E35: E30 plus semantic-kNN RGB->IR edges inside a radius.

    Hypothesis
        Same-location/window edges are fragile under misregistration; linking
        an IR node to the most similar RGB content within a radius ``R``
        makes the graph robust to small shifts and occlusion.
    Parent rung
        E30.
    The one change
        Add ``k`` semantic edges per IR node: the top-``k`` RGB nodes by
        cosine similarity of the projected features, searched only in a
        Chebyshev radius ``R`` (default ``R=4`` at P3, ``k=6``).  The edges
        reuse the E30 message/gate weights, so the parameter count is
        identical to E30 (no new parameters).

    Parameters
        Same as E30: ``P = 3*C*d + 5*d^2 + 6*d + E^2 + 2*E + 2*C + 1``.

    Notes
        Only RGB nodes are candidates -- :func:`neighbors.semantic_knn_index`
        gathers from the RGB feature tensor exclusively.  The candidate window
        is cached per ``(H, W, R)``; selection uses a stable sort so repeated
        forwards are deterministic.
    """

    def __init__(
        self,
        C: int,
        d: int = 128,
        r: int = 1,
        R: int = 4,
        k: int = 6,
        include_center: bool = True,
        gate_init_bias: float = -4.0,
        output_init_std: float = 1e-3,
    ) -> None:
        super().__init__(C=C, d=d, r=r, gate_init_bias=gate_init_bias, output_init_std=output_init_std)
        if R < 0:
            raise ValueError(f"R must be >= 0, got {R}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.R = int(R)
        self.k = int(k)
        self.include_center = bool(include_center)

    def _semantic_edges(
        self, h_ir: torch.Tensor, h_rgb: torch.Tensor, H: int, W: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return semantic_knn_index(
            h_ir,
            h_rgb,
            H=H,
            W=W,
            radius=self.R,
            k=self.k,
            include_center=self.include_center,
        )


# ---------------------------------------------------------------------------
# E41 -- two stacked E30 graph hops (deeper relational reasoning)
# ---------------------------------------------------------------------------


class FuseGATTwoHop(nn.Module):
    """E41: two stacked E30 gated GAT hops over the same fixed edge geometry.

    Hypothesis
        One GAT hop can only compose *one* neighbour exchange; the DV oracle
        analysis says most of the remaining headroom is on small objects, whose
        evidence is spread over several neighbours and scales. A second hop
        lets a node reason about its neighbours' neighbours (2-hop receptive
        field along the grid + window graph) without changing the edge set or
        the cost class of the neighbourhood search.

    Parent rung
        E30.

    The one change
        Compose two :class:`FuseGATLocal` layers, each with its own parameters.
        Hop 1 is bit-identical to E30; hop 2 takes the hop-1 fused output as
        its anchor stream and the same RGB features as its source. Kept as
        composition on purpose: the single-hop path stays the validated E30
        code, so a regression in E41 can only come from the extra depth.

    Parameters
        Exactly 2x the E30 module at the same ``d``.

    Notes
        ``last_delta`` follows the E30 contract: ``f_fused == f_ir + delta``.
        The gate bias of both hops is the standard -4.0, so at step 0 the
        module is a near-identity residual on the IR anchor.
    """

    def __init__(self, C: int, d: int = 128, r: int = 1,
                 gate_init_bias: float = -4.0,
                 output_init_std: float = 1e-3) -> None:
        super().__init__()
        self.hop1 = FuseGATLocal(C, d=d, r=r,
                                 gate_init_bias=gate_init_bias,
                                 output_init_std=output_init_std)
        self.hop2 = FuseGATLocal(C, d=d, r=r,
                                 gate_init_bias=gate_init_bias,
                                 output_init_std=output_init_std)
        self.last_delta: Optional[torch.Tensor] = None

    def forward(self, f_ir: torch.Tensor, f_rgb: torch.Tensor) -> torch.Tensor:
        _check_pair(f_ir, f_rgb)
        h = self.hop1(f_ir, f_rgb)
        out = self.hop2(h, f_rgb)
        delta = out - f_ir
        _raise_on_nonfinite(delta, "E41 delta")
        self.last_delta = delta.detach()
        return out
