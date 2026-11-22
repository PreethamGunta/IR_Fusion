#!/usr/bin/env bash
# Generate the KD arm configs.  Run from the experiment root.
set -euo pipefail
cd "$(dirname "$0")/../src"
PYTHONPATH=. python - <<'PY'
import json
from pathlib import Path
from graphkd import paths

D = paths.CONFIGS / "kd"
D.mkdir(parents=True, exist_ok=True)

BASE = {
    "nc": 4, "k": 40, "conf": 0.01, "temp": 2.0, "warmup_epochs": 5, "kd_every": 4,
    "kd_weight": 1.0, "lambda_node": 1.0, "lambda_edge": 0.5,
    "lambda_struct": 0.1, "lambda_feat": 0.1,
    "cost_alpha": [1.0, 1.0, 0.5, 0.5],
    "sinkhorn_eps": 0.05, "sinkhorn_iters": 20,
    "fgw_eta": 0.5, "fgw_outer": 6,
}
T = {
    "fused": [{"cache": "data/cache/fused", "weight": 1.0, "kind": "fused"}],
    "ir": [{"cache": "data/cache/ir", "weight": 1.0, "kind": "ir"}],
    "multi": [
        {"cache": "data/cache/ir", "weight": 1.0, "kind": "ir"},
        {"cache": "data/cache/rgb", "weight": 1.0, "kind": "rgb"},
    ],
    "three": [
        {"cache": "data/cache/ir", "weight": 1.0, "kind": "ir"},
        {"cache": "data/cache/rgb", "weight": 1.0, "kind": "rgb"},
        {"cache": "data/cache/fused", "weight": 1.0, "kind": "fused"},
    ],
}

CFG = {
    "a1_pred":            {"method": "pred", "teachers": T["fused"]},
    "a2_node":            {"method": "node", "matcher": "hungarian", "teachers": T["fused"]},
    "a2_ir_control":      {"method": "node", "matcher": "hungarian", "teachers": T["ir"]},
    "a3_node_edge":       {"method": "node_edge", "matcher": "hungarian", "teachers": T["fused"]},
    "a4_sinkhorn":        {"method": "sinkhorn", "matcher": "sinkhorn", "teachers": T["fused"],
                            "sinkhorn_iters": 12},
    "a5_fgw":             {"method": "fgw", "matcher": "fgw", "teachers": T["fused"],
                            "fgw_outer": 4, "sinkhorn_iters": 12, "kd_every": 6},
    "a6_partial":         {"method": "partial", "matcher": "partial", "teachers": T["fused"],
                            "sinkhorn_iters": 12, "kd_every": 4},
    "a7_reliability":     {"method": "reliability", "matcher": "hungarian", "teachers": T["fused"],
                            "reliability": True},
    "b_consensus":        {"method": "consensus", "matcher": "hungarian", "teachers": T["multi"],
                            "reliability": True},
    "c_three_teacher":    {"method": "three_teacher", "matcher": "hungarian", "teachers": T["three"],
                            "consistency_weight": 0.05},
    "d_gt_anchor":        {"method": "gt_anchor", "matcher": "hungarian", "teachers": T["fused"],
                            "gt_iou": 0.3},
    "e_condense":         {"method": "condense", "matcher": "hungarian", "teachers": T["fused"],
                            "condensation": {"strategy": "centrality", "k": 30}},
    "f_decomposed":       {"method": "decomposed", "matcher": "hungarian", "teachers": T["fused"],
                            "use_feature_kd": True, "lambda_feat": 0.5},
}
for name, cfg in CFG.items():
    out = {**BASE, **cfg}
    (D / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(name)
PY
