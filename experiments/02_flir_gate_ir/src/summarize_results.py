"""Print a compact table of every detection result JSON in results/."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import RESULTS

SKIP_PREFIX = ("gate_", "routing_", "screen_summary", "unet_metrics", "cyclediff_summary")


def main() -> None:
    rows = []
    for p in sorted(RESULTS.glob("*.json")):
        if p.name.startswith(SKIP_PREFIX):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "mAP50_95" not in d:
            continue
        rows.append(d)

    print("\n| name | n | mAP50-95 | mAP50 | P | R |")
    print("|---|---:|---:|---:|---:|---:|")
    for d in sorted(rows, key=lambda x: x["name"]):
        print(f"| {d['name']} | {d.get('n_images', '')} | {d['mAP50_95']:.4f} | "
              f"{d['mAP50']:.4f} | {d['precision']:.4f} | {d['recall']:.4f} |")

    md = ["# All detection results", "", "| name | n | mAP50-95 | mAP50 | P | R |",
          "|---|---:|---:|---:|---:|---:|"]
    for d in sorted(rows, key=lambda x: x["name"]):
        md.append(f"| {d['name']} | {d.get('n_images', '')} | {d['mAP50_95']:.4f} | "
                  f"{d['mAP50']:.4f} | {d['precision']:.4f} | {d['recall']:.4f} |")
    (RESULTS / "all_results.md").write_text("\n".join(md) + "\n")
    print(f"\n-> {RESULTS / 'all_results.md'}")


if __name__ == "__main__":
    main()
