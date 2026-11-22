"""GF-YOLO fusion testbed (experiment 09) -- five pure-PyTorch fusion rungs.

Public API::

    from fuse_modules import (
        FuseConcat1x1,        # E10
        FuseGatedResidual,    # E14
        FuseBiCrossAttention, # E21
        FuseGATLocal,         # E30
        FuseGATSemantic,      # E35
    )

Every module maps ``(f_ir, f_rgb)`` of shape ``[B, C, H, W]`` to ``f_fused``
of the same shape.  Residual rungs also expose ``last_delta`` with
``f_fused == f_ir + last_delta`` (detached).  See ``README.md`` and the
module docstrings for hypotheses, parameter formulas and test instructions.
"""

from .fuse_modules import (
    FuseBiCrossAttention,
    FuseConcat1x1,
    FuseGATLocal,
    FuseGATSemantic,
    FuseGatedResidual,
    count_parameters,
)
from .neighbors import (
    IRGraphTable,
    NeighborTable,
    border_counts,
    grid8_index,
    ir_graph_index,
    semantic_knn_index,
    window_index,
)

__all__ = [
    "FuseConcat1x1",
    "FuseGatedResidual",
    "FuseBiCrossAttention",
    "FuseGATLocal",
    "FuseGATSemantic",
    "count_parameters",
    "IRGraphTable",
    "NeighborTable",
    "border_counts",
    "grid8_index",
    "ir_graph_index",
    "semantic_knn_index",
    "window_index",
]
