"""Build results/SUMMARY.md from the evaluation JSONs."""
from __future__ import annotations

import json
from pathlib import Path

from graphkd import paths

ARMS = ["a0_gt", "a2_ir_control", "a1_pred_ir", "a3_node_edge_ir", "a5_fgw_ir",
        "a7_reliability_ir", "b_consensus", "a4_sinkhorn_ir", "d_gt_anchor_ir",
        "a2_node", "a1_pred", "a4_sinkhorn", "e_condense", "f_decomposed",
        "c_three_teacher", "a6_partial"]
TEACHERS = ["ir_teacher", "rgb_teacher", "fused_teacher"]


def load(name: str) -> dict | None:
    p = paths.RESULTS / f"{name}_test.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    rows = []
    for name in TEACHERS + ARMS:
        d = load(name)
        if d:
            rows.append((name, d))
    if not rows:
        raise SystemExit("no results yet")
    base = load("a0_gt")
    out = ["# Graph-KD results — aligned FLIR paired protocol", ""]
    out.append("Test split: 609 paired frames. AP is COCO mAP50-95. "
               "Deltas are relative to the GT-only YOLO12n student (`a0_gt`).")
    out.append("")
    out.append("| Arm | AP | AP50 | AP75 | APs | APm | APl | ΔAP | ECE | Brier | Params | ms |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in rows:
        delta = ""
        if base and name != "a0_gt":
            delta = f"{d['AP'] - base['AP']:+.4f}"
        out.append(
            f"| {name} | {d['AP']:.4f} | {d['AP50']:.4f} | {d['AP75']:.4f} | "
            f"{d['APs']:.4f} | {d['APm']:.4f} | {d['APl']:.4f} | {delta} | "
            f"{d.get('ECE', 0):.4f} | {d.get('Brier', 0):.4f} | {d.get('params', 0)/1e6:.2f}M | "
            f"{d.get('latency_ms', 0):.2f} |"
        )
    out.append("")
    best = max((r for r in rows if r[0] not in TEACHERS), key=lambda r: r[1]["AP"])
    out.append(f"## Best student: `{best[0]}`")
    out.append("")
    out.append("| Class | AP |")
    out.append("|---|---:|")
    for c, v in sorted(best[1].get("per_class_AP", {}).items(), key=lambda kv: -kv[1]):
        out.append(f"| {c} | {v:.4f} |")
    out.append("")
    out.append("## Per-arm KD loss terms (last epoch)")
    out.append("")
    out.append("| Arm | last kd terms |")
    out.append("|---|---|")
    for name, _ in rows:
        p = paths.RUNS / name / "kd_terms.jsonl"
        if p.exists():
            lines = p.read_text().strip().splitlines()
            if lines:
                d = json.loads(lines[-1])
                terms = ", ".join(f"{k}={v:.3f}" for k, v in d.items() if k != "epoch")
                out.append(f"| {name} | {terms} |")
    out.append("")
    out.append("## Provenance")
    out.append("")
    for name, d in rows:
        out.append(f"- **{name}**: weights `{d['weights']}` sha256 `{d['weights_sha256'][:16]}…`, "
                   f"split `{d['split_hash']}`, n={d['n_images']}")
    (paths.RESULTS / "SUMMARY.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
