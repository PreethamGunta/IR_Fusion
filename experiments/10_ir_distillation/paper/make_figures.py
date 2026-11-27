#!/usr/bin/env python3
"""Paper figures for experiment 10 (IR-only distillation).

Inputs (all optional — whatever exists is plotted):
  results/<name>_test_metrics.json            student test AP
  results/bootstrap_<name>_vs_e00.json        paired CI vs E00
  results/profiling_students.json             params / latency
  runs/<name>/results.csv                     val curves

The 09 references are read from the sibling experiment:
  09/results/{e00_ir,e30_v2_100ep}_test_metrics.json
  09/results/bootstrap_flagships_vs_e00.json

    python paper/make_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
E09 = REPO / "experiments" / "09_graph_fusion"
RES = EXP / "results"
FIG = EXP / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# name -> (label, colour, metrics_path, runs_path, bootstrap_path)
MODELS = {
    "e00_ir": ("E00 IR-only", "#444444",
               E09 / "results/e00_ir_test_metrics.json", E09 / "runs/e00_ir", None),
    "e30_v2_100ep": ("E30 teacher", "#0072B2",
                     E09 / "results/e30_v2_100ep_test_metrics.json",
                     E09 / "runs/e30_v2_100ep",
                     E09 / "results/bootstrap_flagships_vs_e00.json"),
    "student_12s_decomp": ("12s decomposed", "#009E73",
                           RES / "student_12s_decomp_test_metrics.json",
                           EXP / "runs/student_12s_decomp",
                           RES / "bootstrap_student_12s_decomp_vs_e00.json"),
    "student_12s_direct": ("12s direct", "#E69F00",
                           RES / "student_12s_direct_test_metrics.json",
                           EXP / "runs/student_12s_direct",
                           RES / "bootstrap_student_12s_direct_vs_e00.json"),
    "student_12n_gt": ("12n no-KD", "#CC79A7",
                       RES / "student_12n_gt_test_metrics.json",
                       EXP / "runs/student_12n_gt",
                       RES / "bootstrap_student_12n_gt_vs_e00.json"),
    "student_12n_decomp": ("12n decomposed", "#56B4E9",
                           RES / "student_12n_decomp_test_metrics.json",
                           EXP / "runs/student_12n_decomp",
                           RES / "bootstrap_student_12n_decomp_vs_e00.json"),
}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.4,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=180)
    plt.close(fig)
    print(f"[fig] {FIG / name}.pdf")


def metrics(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d if "AP" in d else None


def delta_ci(path: Path, model: str):
    """Return (mean, lo, hi) of the AP delta vs E00 from a bootstrap JSON."""
    if not path or not path.exists():
        return None
    d = json.loads(path.read_text()).get("deltas", {})
    for key, val in d.items():
        if key.startswith(f"{model}_minus_"):
            ci = val.get("ci95")
            if ci:
                return val["mean"], ci[0], ci[1]
    return None


def curves(run: Path):
    f = run / "results.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(f.open()))
    if not rows:
        return None
    ep = [int(r["epoch"]) for r in rows]
    ap = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    return np.array(ep), np.array(ap)


def fig_test_ap(rows):
    names = [n for n in MODELS if n in rows]
    if len(names) < 2:
        return
    fig, ax = plt.subplots(figsize=(3.4, 0.35 * len(names) + 0.9))
    ys = np.arange(len(names))[::-1]
    for y, n in zip(ys, names):
        label, colour, _, _, bpath = MODELS[n]
        ap = rows[n]["AP"]
        d = delta_ci(bpath, n)
        if d:
            mean, clo, chi = d
            lo, hi = ap + (clo - mean), ap + (chi - mean)
            ax.barh(y, ap, color=colour, height=0.62)
            ax.errorbar(ap, y, xerr=[[ap - lo], [hi - ap]], fmt="none",
                        ecolor="black", elinewidth=0.9, capsize=2)
        else:
            ax.barh(y, ap, color=colour, height=0.62)
        ax.text(ap + 0.004, y, f"{ap:.4f}", va="center", fontsize=6.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([MODELS[n][0] for n in names])
    ax.set_xlabel("locked test AP (mAP$_{50:95}$)")
    ax.set_xlim(0, max(rows[n]["AP"] for n in names) * 1.16)
    ax.set_title("Test AP on the 8,980-image split (95 % CI where available)")
    save(fig, "fig_test_ap")


def fig_kd_ablation(rows):
    groups = {
        "YOLO12s": ["e00_ir", "student_12s_decomp", "student_12s_direct"],
        "YOLO12n": ["e00_ir", "student_12n_gt", "student_12n_decomp"],
    }
    present = {g: [n for n in ns if n in rows] for g, ns in groups.items()}
    if not any(len(v) >= 2 for v in present.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.1))
    for ax, (g, ns) in zip(axes, present.items()):
        if len(ns) < 2:
            ax.set_visible(False)
            continue
        x = np.arange(len(ns))
        ax.bar(x, [rows[n]["AP"] for n in ns],
               color=[MODELS[n][1] for n in ns], width=0.62)
        for xi, n in zip(x, ns):
            ax.text(xi, rows[n]["AP"] + 0.006, f"{rows[n]['AP']:.3f}",
                    ha="center", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([MODELS[n][0] for n in ns], rotation=12)
        ax.set_ylim(0, max(rows[n]["AP"] for n in ns) * 1.2)
        ax.set_ylabel("test AP")
        ax.set_title(g)
    fig.suptitle("KD ablation (E00 is the no-KD baseline of the same student)", y=1.03)
    save(fig, "fig_kd_ablation")


def fig_pareto(rows, prof):
    if not prof:
        return
    pts = []
    for r in prof.get("models", []):
        name = r["model"]
        if name in rows and name in MODELS:
            pts.append((r["latency_fp32_ms"], rows[name]["AP"], r["params_total_m"],
                        MODELS[name][0], MODELS[name][1]))
    if len(pts) < 2:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for ms, ap, params, label, colour in pts:
        ax.scatter(ms, ap, s=18 + params * 2.0, color=colour, edgecolor="black",
                   linewidth=0.4, zorder=3)
        ax.annotate(label, (ms, ap), textcoords="offset points", xytext=(4, 4),
                    fontsize=6.2)
    ax.set_xlabel("A100 latency, batch 1 (ms, FP32)")
    ax.set_ylabel("locked test AP")
    ax.set_title("Accuracy vs cost")
    save(fig, "fig_pareto")


def fig_deltas(rows):
    entries = []
    for n in MODELS:
        if n == "e00_ir" or n not in rows:
            continue
        d = delta_ci(MODELS[n][4], n)
        if d:
            entries.append((MODELS[n][0], d[0] * 100, (d[0] - d[1]) * 100,
                            (d[2] - d[0]) * 100, MODELS[n][1]))
    if not entries:
        return
    fig, ax = plt.subplots(figsize=(3.4, 0.32 * len(entries) + 0.8))
    ys = np.arange(len(entries))[::-1]
    ax.axvline(0, color="black", lw=0.7)
    for y, (label, mean, lo, hi, colour) in zip(ys, entries):
        ax.errorbar(mean, y, xerr=[[lo], [hi]], fmt="o", ms=3.5, color=colour,
                    ecolor=colour, capsize=2)
        ax.annotate(f"{mean:+.2f}", (mean, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6.2)
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in entries])
    ax.set_xlabel(r"$\Delta$AP vs E00 (pp), paired bootstrap 95 % CI")
    ax.set_title("Fusion/distillation gain over the IR-only baseline")
    save(fig, "fig_deltas")


def fig_curves():
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    plotted = False
    for n, (label, colour, _, run, _) in MODELS.items():
        c = curves(run)
        if c is None:
            continue
        ep, ap = c
        ax.plot(ep, ap, lw=1.0, color=colour, label=label)
        k = int(np.argmax(ap))
        ax.plot(ep[k], ap[k], "o", ms=2.6, color=colour)
        plotted = True
    if not plotted:
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("val mAP$_{50:95}$ (1,469)")
    ax.set_title("Validation curves")
    ax.legend(loc="lower right", frameon=False)
    save(fig, "fig_val_curves")


def main() -> None:
    rows = {n: m for n, m in ((n, metrics(MODELS[n][2])) for n in MODELS) if m}
    prof_path = RES / "profiling_students.json"
    prof = json.loads(prof_path.read_text()) if prof_path.exists() else None

    print(f"[fig] models with test metrics: {sorted(rows)}")
    fig_test_ap(rows)
    fig_kd_ablation(rows)
    fig_pareto(rows, prof)
    fig_deltas(rows)
    fig_curves()


if __name__ == "__main__":
    main()
