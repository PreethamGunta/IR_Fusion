"""Summarise the GATE-IR-on-FLIR result JSONs into a markdown table."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import RESULTS

ORDER = [
    ("b1_cyclegan", "B1 mixed, no filter"),
    ("b0", "B0 clear-only"),
    ("b2", "B2 fog-only"),
    ("b3", "B3 always-filter"),
    ("g1", "G1 gate + filter"),
    ("g1_gated", "G1 gate + filter (routed)"),
    ("g1_oracle", "G1 oracle gate"),
]


def _load(name: str) -> dict | None:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    rows = []
    for key, label in ORDER:
        clear = _load(f"{key}_clear_vtt") if key not in ("b1_cyclegan", "b2") else (
            _load("b1_cyclegan_clear_vtt") if key == "b1_cyclegan" else None
        )
        fog = _load(f"{key}_fog_vtt") if key not in ("b1_cyclegan",) else _load("b1_cyclegan_fog_vtt")
        if key == "g1":
            fog = _load("g1_fog_filtered")
        if key == "g1_gated":
            fog = _load("g1_gated_fog")
        if key == "g1_oracle":
            fog = _load("g1_oracle_fog")
        rows.append((label, clear, fog))

    print("\n## Detection results (matched VTT test)\n")
    print("| arm | clear mAP50-95 | fog mAP50-95 | clear-fog gap | fog mAP50 | fog P | fog R | n clear/fog |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, clear, fog in rows:
        c = f"{clear['mAP50_95']:.4f}" if clear else "—"
        f = f"{fog['mAP50_95']:.4f}" if fog else "—"
        gap = f"{clear['mAP50_95'] - fog['mAP50_95']:.4f}" if (clear and fog) else "—"
        f50 = f"{fog['mAP50']:.4f}" if fog else "—"
        p = f"{fog['precision']:.4f}" if fog else "—"
        r = f"{fog['recall']:.4f}" if fog else "—"
        n = f"{clear['n_images'] if clear else '—'}/{fog['n_images'] if fog else '—'}"
        print(f"| {label} | {c} | {f} | {gap} | {f50} | {p} | {r} | {n} |")

    gate = _load("gate_cyclegan")
    if gate:
        print("\n## Gate (CycleGAN fog)\n")
        print(f"- val accuracy: {gate['val']['accuracy']:.4f}, F1: {gate['val']['f1']:.4f}")
        for k, v in gate["test"].items():
            print(f"- {k}: acc={v['accuracy']:.4f} F1={v['f1']:.4f} "
                  f"FNR(fog→clear)={v['false_negative_rate_fog_as_clear']:.4f} "
                  f"FPR(clear→fog)={v['false_positive_rate_clear_as_fog']:.4f} n={v['n']}")

    for r in sorted(RESULTS.glob("routing_*.json")):
        d = json.loads(r.read_text())
        print(f"\n## Routing: {d['name']} ({d['gate']})\n")
        print(f"- clear→filter: {d['clear_routed_to_filter']}/{d['n_clear']} "
              f"({d['clear_false_positive_rate']:.4f})")
        print(f"- fog→clear: {d['fog_routed_to_clear']}/{d['n_fog']} "
              f"({d['fog_false_negative_rate']:.4f})")


if __name__ == "__main__":
    main()
