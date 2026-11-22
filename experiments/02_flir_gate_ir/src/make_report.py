"""Build results/REPORT.md from the GATE-IR-on-FLIR result JSONs."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import RESULTS
from .build_screen import DEFAULT_FILTERS


def L(name: str) -> dict | None:
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def row(label: str, clear: str | None, fog: str | None) -> str:
    c = L(clear) if clear else None
    f = L(fog) if fog else None
    cv = f"{c['mAP50_95']:.4f}" if c else "—"
    fv = f"{f['mAP50_95']:.4f}" if f else "—"
    gap = f"{c['mAP50_95'] - f['mAP50_95']:.4f}" if (c and f) else "—"
    n = f"{c['n_images'] if c else '—'}/{f['n_images'] if f else '—'}"
    return f"| {label} | {cv} | {fv} | {gap} | {n} |"


def main() -> None:
    lines = ["# GATE-IR on FLIR — results report", ""]
    screen = L("screen_summary")
    winner = screen["best_filter"] if screen else None

    lines += ["## Phase 1 — CycleGAN 60-epoch filter screen (YOLOv8s)", "",
              "| arm | clear mAP50-95 | fog mAP50-95 | gap | n clr/fog |",
              "|---|---:|---:|---:|---:|",
              row("B1 (no filter)", "b1_yolov8s_60_clear_vtt", "b1_yolov8s_60_fog_vtt")]
    for f in DEFAULT_FILTERS:
        lines.append(row(f"G1 {f}", f"g1_{f}_clear_vtt", f"g1_{f}_fog_vtt"))
    lines.append("")

    if winner:
        lines += [f"## Phase 1 — main comparison, 40-epoch YOLOv12s (winner: {winner})", "",
                  "| arm | clear mAP50-95 | fog mAP50-95 | gap | n clr/fog |",
                  "|---|---:|---:|---:|---:|",
                  row("B1 (no filter)", "b1_yolo12s_40_clear_vtt", "b1_yolo12s_40_fog_vtt"),
                  row(f"G1 {winner}", f"g1_{winner}_40_clear_vtt", f"g1_{winner}_40_fog_vtt"),
                  ""]

        lines += ["## Phase 1 — 100-epoch YOLOv12s confirmation", "",
                  "| arm | clear mAP50-95 | fog mAP50-95 | gap | n clr/fog |",
                  "|---|---:|---:|---:|---:|",
                  row("B1 (no filter)", "b1_yolo12s_100_clear_vtt", "b1_yolo12s_100_fog_vtt"),
                  row("G1 U-Net", "g1_unet_100_clear_vtt", "g1_unet_100_fog_vtt"),
                  ""]

        lines += [f"## Phase 2 — CycleDiff-75 (filter: {winner})", "",
                  "| arm | clear mAP50-95 | fog mAP50-95 | gap | n clr/fog |",
                  "|---|---:|---:|---:|---:|",
                  row("B1 (no filter)", "b1_cyclediff75_yolov8s_40_clear_vtt", "b1_cyclediff75_yolov8s_40_fog_vtt"),
                  row(f"G1 {winner} (CycleGAN-trained)", f"g1_{winner}_cyclediff75_yolov8s_40_clear_vtt",
                      f"g1_{winner}_cyclediff75_yolov8s_40_fog_vtt"),
                  row("G1 U-Net trained on CycleDiff-75", "g1_unet_cd75_40_clear_vtt", "g1_unet_cd75_40_fog_vtt"),
                  ""]

        lines += ["## Phase 2 — 100-epoch YOLOv12s confirmation on CycleDiff-75", "",
                  "| arm | clear mAP50-95 | fog mAP50-95 | gap | n clr/fog |",
                  "|---|---:|---:|---:|---:|",
                  row("B1 (no filter)", "b1_cyclediff75_yolo12s_100_clear_vtt",
                      "b1_cyclediff75_yolo12s_100_fog_vtt"),
                  row("G1 U-Net trained on CycleDiff-75 (stopped at epoch 27, resumable)",
                      "g1_unet_cd75_100_clear_vtt", "g1_unet_cd75_100_fog_vtt"),
                  ""]

    lines += ["## Phase 3 — ablations", "",
              "| arm | clear mAP50-95 | fog mAP50-95 | gap |",
              "|---|---:|---:|---:|",
              row("B0 clear-only (CycleGAN fog)", "b0_40_clear_vtt", "b0_40_fog_vtt"),
              row("B0 clear-only (CycleDiff fog)", "b0_40_clear_vtt", "b0_40_cyclediff75_fog_vtt"),
              row("B2 fog-only (CycleGAN)", None, "b2_40_fog_vtt"),
              row("B2 fog-only (CycleDiff)", None, "b2_cyclediff75_40_fog_vtt")]
    for f in DEFAULT_FILTERS:
        lines.append(row(f"B3 always-on {f}", f"b3_{f}_40_clear_vtt", f"b3_{f}_40_fog_vtt"))
    if winner:
        lines.append(row(f"B3 always-on {winner} (CycleDiff)",
                         f"b3_{winner}_cyclediff75_40_clear_vtt",
                         f"b3_{winner}_cyclediff75_40_fog_vtt"))
    lines.append("")

    gates = []
    for cond in ("cyclegan", "cyclediff75"):
        g = L(f"gate_{cond}")
        if g:
            gates.append(f"### Gate {cond}")
            gates.append("")
            gates.append("| test | acc | F1 | FNR fog→clear | FPR clear→fog |")
            gates.append("|---|---:|---:|---:|---:|")
            for k, v in g["test"].items():
                gates.append(f"| {k} | {v['accuracy']:.4f} | {v['f1']:.4f} | "
                             f"{v['false_negative_rate_fog_as_clear']:.4f} | "
                             f"{v['false_positive_rate_clear_as_fog']:.4f} |")
            gates.append("")
    if gates:
        lines += ["## Gate metrics", ""] + gates

    out = "\n".join(lines) + "\n"
    (RESULTS / "REPORT.md").write_text(out)
    print(out)
    print(f"-> {RESULTS / 'REPORT.md'}")


if __name__ == "__main__":
    main()
