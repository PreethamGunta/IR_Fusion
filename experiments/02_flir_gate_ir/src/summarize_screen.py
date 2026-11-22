"""Summarise the CycleGAN filter screen and pick the winner."""

from __future__ import annotations

import json

from .paths import RESULTS
from .build_screen import DEFAULT_FILTERS


def _load(name: str) -> dict | None:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    rows = []
    for f in DEFAULT_FILTERS:
        clear = _load(f"g1_{f}_clear_vtt")
        fog = _load(f"g1_{f}_fog_vtt")
        gated_fog = _load(f"g1_{f}_gated_fog")
        oracle_fog = _load(f"g1_{f}_oracle_fog")
        rows.append({
            "filter": f,
            "clear_mAP50_95": clear["mAP50_95"] if clear else None,
            "fog_mAP50_95": fog["mAP50_95"] if fog else None,
            "fog_mAP50": fog["mAP50"] if fog else None,
            "gated_fog_mAP50_95": gated_fog["mAP50_95"] if gated_fog else None,
            "oracle_fog_mAP50_95": oracle_fog["mAP50_95"] if oracle_fog else None,
        })

    complete = [r for r in rows if r["fog_mAP50_95"] is not None]
    best = max(complete, key=lambda r: r["fog_mAP50_95"]) if complete else None

    print("\n| filter | clear mAP50-95 | fog mAP50-95 | gated fog | oracle fog |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        def fmt(v):
            return f"{v:.4f}" if v is not None else "—"
        print(f"| {r['filter']} | {fmt(r['clear_mAP50_95'])} | {fmt(r['fog_mAP50_95'])} | "
              f"{fmt(r['gated_fog_mAP50_95'])} | {fmt(r['oracle_fog_mAP50_95'])} |")

    out = {"rows": rows, "best_filter": best["filter"] if best else None,
           "best_fog_mAP50_95": best["fog_mAP50_95"] if best else None}
    (RESULTS / "screen_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nBest filter: {out['best_filter']} ({out['best_fog_mAP50_95']})")
    print(f"-> {RESULTS / 'screen_summary.json'}")


if __name__ == "__main__":
    main()
