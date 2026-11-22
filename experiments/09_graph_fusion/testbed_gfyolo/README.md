# GF-YOLO fusion testbed (experiment 09)

Pure-PyTorch fusion modules and neighbour machinery for the graph-based
RGB-IR fusion ladder. No `ultralytics`, no `torch_geometric`/PyG, no
experiment-08 imports. The testbed implements the five rungs that the
minimal first pass actually trains (E10, E14, E21, E30, E35) exactly at the
`Fuse(f_ir, f_rgb) -> f_fused` boundary described in
`../EXPERIMENT_PLAN.md` §5–§6.

## Contract

```python
m = FuseGATLocal(C=64, d=128, r=1)       # or any other rung
f_fused = m(f_ir, f_rgb)                  # both [B, C, H, W]
assert f_fused.shape == f_ir.shape
```

* `f_ir` is the IR stream (detector channels `0..2` after the fixed E02 stem),
  `f_rgb` the visible stream (channels `3..5`). At this boundary the features
  are pre-split, so "IR first, RGB second" is the only channel convention.
* Residual rungs (E14, E21, E30, E35) also expose
  `m.last_delta` of shape `[B, C, H, W]` with
  `f_fused == f_ir + m.last_delta`.
  `last_delta` is **detached** on purpose: it is for probes/hooks and must not
  pin the autograd graph. Recompute if a differentiable Delta is needed.
* Every forward raises `FloatingPointError` if the correction or fused output
  contains NaN/Inf. There is no silent `nan_to_num`.
* Softmax logits are computed in float32 even when the module runs in fp16, so
  half-precision CPU/AMP forwards are NaN-free.

## Rungs

| Rung | Class | Parent | Hypothesis | The one change | Parameters |
|---|---|---|---|---|---|
| E10 | `FuseConcat1x1` | — (ladder base) | a learned linear channel mix captures complementarity | `conv1x1([f_ir, f_rgb])` | `2C² + C` |
| E14 | `FuseGatedResidual` | E10 | a small gated correction on the IR anchor is safer than replacing it; starts as E00 | `f_ir + sigmoid(g(z))·conv(z)`, `g` bias `-4.0` | `4C² + 2C` |
| E21 | `FuseBiCrossAttention` | E14 | dense content-based exchange in both directions beats fixed mixing | single-head bidirectional cross-attention over `H·W` tokens, shared Q/K/V, `d=128`, then the E14 gate | `4Cd + 3d² + 5d + 4C² + 4C` |
| E30 | `FuseGATLocal` | E14 | explicit neighbour messages capture local geometry and cross-modal correspondence more cheaply | one hand-rolled GAT layer: 8-neighbour IR grid + `(2r+1)²` RGB→IR window edges, edge features `(dx,dy,cos,q_i,q_j)` | `3Cd + 5d² + 6d + E² + 2E + 2C + 1`, `E=5` |
| E35 | `FuseGATSemantic` | E30 | window edges are fragile under misregistration; similarity edges are robust | adds top-`k` RGB→IR semantic-kNN edges inside Chebyshev radius `R` (defaults `R=4`, `k=6`); reuses E30 weights | same as E30 |

Measured parameter counts (defaults `d=128`, `E=5`; E35 == E30):

| `C` | E10 | E14 | E21 | E30 / E35 |
|---:|---:|---:|---:|---:|
| 8 | 136 | 272 | 54,176 | 85,812 |
| 32 | 2,080 | 4,160 | 70,400 | 95,076 |
| 64 | 8,256 | 16,512 | 99,200 | 107,428 |
| 128 | 32,896 | 65,792 | 181,376 | 132,132 |
| 256 | 131,328 | 262,656 | 444,032 | 181,540 |

### E21 details

Two `C→d` projections, one shared `q/k/v` set (`self.q`, `self.k`, `self.v`
are used in both directions), a residual update in `d`-space
(`h + attention(h, other)`), two `d→C` projections, then the E14 gate.
Attention is chunked over queries (`attn_chunk=512` by default), so the dense
`N×N` map is not materialised during training. Set `collect_attn = True` to
keep the detached maps (`last_attn["ir"]`, `last_attn["rgb"]`) for tests and
probes; leave it off for real runs.

### E30/E35 details

* Nodes are the IR positions; RGB nodes are sources only (task spec; the
  wider plan's `E_rr` graph edges are a later rung).
* Edge slots = 8 grid + `(2r+1)²` window cross-modal slots, plus `k`
  semantic slots for E35. Window edges include the co-located RGB node
  (different modality, not a self-loop).
* Attention: `LeakyReLU(aᵀ[W h_i ‖ W h_j ‖ W_e e_ij])`, softmax over valid
  neighbours only; update `h_i' = h_i + g_i Σ_j α_ij W_v h_j`,
  `g_i = sigmoid(MLP([h_i, agg_i]))` with the last MLP bias at `-4.0`;
  `f_fused = f_ir + out(h')`.
* Edge features are `(dx/max(r,1), dy/max(r,1), cos(W h_i, W h_j), q_i, q_j)`
  with `q` from a small shared `C→1` sigmoid conv head.
* E35's semantic candidates are RGB-only (`neighbors.semantic_knn_index`
  gathers exclusively from the RGB tensor), searched in the Chebyshev window
  of radius `R`, and selected by a stable descending sort for determinism.
  Border nodes with fewer than `k` candidates mark the surplus slots invalid;
  they never invent ids.

## Neighbour API (`neighbors.py`)

| Function | Returns | Notes |
|---|---|---|
| `grid8_index(H, W)` | `NeighborTable` | 8-neighbour IR grid, no centre, no self-loops; corners 3 / edges 5 / interior 8 |
| `window_index(H, W, r, include_center=True)` | `NeighborTable` | square `(2r+1)²` window; corner `(r+1)²`, edge midpoint `(r+1)(2r+1)`, interior `(2r+1)²` (edge nodes near a corner are smaller) |
| `semantic_candidates(H, W, R, include_center=True)` | `NeighborTable` | fixed RGB candidate window for E35 |
| `ir_graph_index(H, W, r)` | `IRGraphTable` | E30 combined table; IR ids `<N`, RGB ids `>=N`; per-node grid/window counts |
| `semantic_knn_index(feat_ir, feat_rgb, H, W, radius, k, ...)` | `(index [B,N,k], valid [B,N,k])` | deterministic top-k by cosine in radius; chunked over nodes |
| `node_coords`, `flat_to_yx`, `yx_to_flat` | coordinate helpers | flat = `y*W + x`; round-trip tested |
| `border_class`, `border_counts` | border helpers | corner / edge / interior regression tests |
| `assert_indices_in_bounds`, `assert_no_self_loops` | validators | used by tests |

All fixed tables are cached per `(H, W, r/R)` and returned as CPU tensors;
move them with `.to(device)` and treat them as read-only. Invalid slots
always contain a different in-bounds id (except when `N=1`, where no
alternative exists), so even an unmasked table has no accidental self-loop;
callers must still honour `valid`.

## Tests

```bash
cd experiments/09_graph_fusion/testbed_gfyolo
PYTHONPATH=. python \
    -m pytest test_fuse_modules.py -q
```

Expect **77 passed** on CPU (~3 s). Coverage:

* shape/channel contract for all five rungs, including `C=3` and a 5×7 map;
* E10 concat order is `[IR | RGB]` (hand-set 1x1 weights extract the IR mean);
* E14 gate bias exactly `-4.0`; init identity `< 1e-6` for
  `C=8/32/64/128/256`; zero-RGB and zero-IR at init;
* E21 and E30/E35 attention rows sum to 1 with invalid slots exactly zero;
* grid/window bounds, corner/edge/interior counts, no unintended self-loops,
  combined-table composition;
* semantic candidates inside radius `R`, RGB-only candidates, top-k
  correctness against a brute-force reference, border shortage masked;
* fp16 forward finite; no `nan_to_num` call or forbidden import (AST check);
* one backward gives every parameter a finite, non-zero gradient;
* fixed tables and repeated forwards deterministic;
* parameter counts equal the documented formulas and are printed.

Note: the exact command above lets pytest create its usual
`__pycache__/`/`.pytest_cache/` next to the sources; pass
`-p no:cacheprovider` and `PYTHONDONTWRITEBYTECODE=1` if a clean tree is
required.

## Design caveats / limitations

1. **Gate init identity is not achievable from the bias alone.**
   `sigmoid(-4) = 0.018`, not `< 1e-6`. The bias is kept exactly at `-4.0`
   (as specified) and the final output projection is randomly initialised
   with a tiny scale (`1e-7` for E14, `1e-6` for E21) so the step-0
   correction is below `1e-6` while every parameter still receives a
   non-zero gradient. E30/E35 use `1e-3` (small but non-zero Delta) because
   they only require a negative-bias gate.
2. **Semantic kNN is data-dependent.** The *candidate window* is fixed and
   cached; the top-k selection is rebuilt each forward because the projected
   features change. It is gather-based, stable-sorted and deterministic, but
   not precomputable per `(H,W,r,R,k)` alone.
3. **Dense attention memory.** E21 is linear in `N²` for the attention map
   (chunked, so peak memory is `chunk·N`); `collect_attn=True` materialises
   the full map and should stay off in training.
4. **No PyG parity test.** The plan mentions a unit test matching PyG
   `GATConv` with copied weights; PyG is not installed and E30's edge
   features / gate make the mapping non-trivial, so that check is out of
   scope for this testbed. Edge counts, softmax normalisation and gradient
   flow are tested instead.
5. **CPU-only.** GPU/AMP device behaviour is not exercised here; the fp16
   test runs on CPU. No GPU was touched.

## Files

* `fuse_modules.py` — E10/E14/E21/E30/E35 modules and `count_parameters`.
* `neighbors.py` — grid/window/semantic neighbour tables and validators.
* `test_fuse_modules.py` — the CPU pytest suite.
* `__init__.py` — package exports.
* `README.md` — this file.
