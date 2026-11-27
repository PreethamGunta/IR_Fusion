#!/usr/bin/env python3
"""Generate every data-driven figure for the WG-Fusion paper.

Inputs (all artifacts of the experiments, nothing hardcoded):
  runs/*/results.csv, runs_g2/*/results.csv  -> val curves
  results/*_test_metrics.json                -> test metrics
  results/bootstrap_*.json                   -> paired delta CIs
  results/oracle_dv_val.json                 -> oracle headroom & strata
  results/gt/dv_v2_ir_val.json               -> class instance counts
Outputs: paper/figures/*.pdf
"""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path(__file__).resolve().parents[1]
FIG = EXP / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=180)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.3, "grid.linewidth": 0.4,
})

ARMS = {  # name: (label, colour)
    "e00_ir": ("IR-only (E00)", "#444444"),
    "e10_v2_100ep": ("Concat (E10)", "#E69F00"),
    "e30_v2_100ep": ("WG-GAT (E30)", "#0072B2"),
    "e35_v2_100ep": ("WG-GAT+sem (E35)", "#009E73"),
    "e41_v2_80ep": ("WG-GAT 2hop (E41)", "#CC79A7"),
    "e30d192_v2_100ep": ("WG-GAT d192", "#56B4E9"),
}
CURVE_RUNS = [("e00_ir", "runs/e00_ir"),
              ("e10_v2_100ep", "runs/e10_v2_100ep"),
              ("e30_v2_100ep", "runs/e30_v2_100ep"),
              ("e35_v2_100ep", "runs/e35_v2_100ep"),
              ("e41_v2_80ep", "runs_g2/e41_v2_80ep"),
              ("e30d192_v2_100ep", "runs_g2/e30d192_v2_100ep")]


def curve(path: Path):
    rows = list(csv.DictReader(open(path)))
    seen = {}
    for r in rows:
        e = int(r["epoch"])
        if e not in seen:
            seen[e] = (float(r["metrics/mAP50-95(B)"]), float(r["metrics/mAP50(B)"]))
    eps = sorted(seen)
    return np.array(eps), np.array([seen[e][0] for e in eps])


def load_metrics(name: str):
    return json.load(open(EXP / "results" / f"{name}_test_metrics.json"))


# ---------------------------------------------------------------- 1. curves
def fig_curves():
    fig, ax = plt.subplots(figsize=(3.35, 2.3))
    for name, path in CURVE_RUNS:
        eps, vals = curve(EXP / path / "results.csv")
        lbl, col = ARMS[name]
        ax.plot(eps, vals, label=lbl, color=col, lw=1.1)
        k = int(np.argmax(vals))
        ax.plot(eps[k], vals[k], "o", ms=2.6, color=col)
    ax.set_xlabel("epoch"); ax.set_ylabel("val mAP$_{50:95}$")
    ax.set_xlim(0, 100); ax.legend(loc="lower right", framealpha=0.9)
    save(fig, "fig_val_curves"); plt.close(fig)


# ------------------------------------------------------------- 2. test AP CI
def fig_test_ap():
    names = ["e00_ir", "e10_v2_100ep", "e30_v2_100ep",
             "e35_v2_100ep", "e41_v2_80ep", "e30d192_v2_100ep"]
    boot = {}
    for f in glob.glob(str(EXP / "results" / "bootstrap_*.json")):
        d = json.load(open(f))
        for k, v in d["deltas"].items():
            if k.endswith("_minus_e00_ir"):
                boot[k.replace("_minus_e00_ir", "")] = v
    ref = load_metrics("e00_ir")["AP"]
    labels, aps, los, his, cols = [], [], [], [], []
    for n in names:
        m = load_metrics(n)
        lbl, col = ARMS[n]
        labels.append(lbl.replace(" (", "\n(")); aps.append(m["AP"])
        if n in boot:
            los.append(ref + boot[n]["ci95"][0]); his.append(ref + boot[n]["ci95"][1])
        else:
            los.append(m["AP"]); his.append(m["AP"])
        cols.append(col)
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    x = np.arange(len(labels))
    ax.bar(x, aps, color=cols, width=0.65,
           yerr=[np.array(aps) - np.array(los), np.array(his) - np.array(aps)],
           capsize=2, ecolor="black", error_kw={"lw": 0.8})
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("test AP"); ax.set_ylim(0.55, 0.615)
    ax.axhline(ref, color="#444444", ls="--", lw=0.8)
    save(fig, "fig_test_ap"); plt.close(fig)


# ----------------------------------------------------------- 3. per class
def fig_per_class():
    names = ["e00_ir", "e10_v2_100ep", "e30_v2_100ep",
             "e35_v2_100ep", "e41_v2_80ep", "e30d192_v2_100ep"]
    classes = ["car", "truck", "bus", "van", "freight_car"]
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    w = 0.13
    x = np.arange(len(classes))
    for i, n in enumerate(names):
        m = load_metrics(n)
        vals = [m["per_class_AP"][c] for c in classes]
        ax.bar(x + (i - 2.5) * w, vals, width=w, label=ARMS[n][0].split(" (")[0],
               color=ARMS[n][1])
    ax.set_xticks(x); ax.set_xticklabels(["car", "truck", "bus", "van", "freight"])
    ax.set_ylabel("test AP"); ax.legend(ncol=2, loc="upper center", fontsize=5.5)
    ax.set_ylim(0.35, 0.85)
    save(fig, "fig_per_class"); plt.close(fig)


# -------------------------------------------------------------- 4. scale
def fig_scale():
    names = ["e00_ir", "e10_v2_100ep", "e30_v2_100ep",
             "e35_v2_100ep", "e41_v2_80ep", "e30d192_v2_100ep"]
    keys = [("APs", "small"), ("APm", "medium"), ("APl", "large")]
    fig, ax = plt.subplots(figsize=(3.35, 2.0))
    x = np.arange(len(keys)); w = 0.13
    for i, n in enumerate(names):
        m = load_metrics(n)
        ax.bar(x + (i - 2.5) * w, [m[k] for k, _ in keys], width=w,
               label=ARMS[n][0].split(" (")[0], color=ARMS[n][1])
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in keys])
    ax.set_ylabel("test AP"); ax.set_ylim(0.15, 0.68)
    ax.legend(ncol=2, loc="upper center", fontsize=5.5)
    save(fig, "fig_scale"); plt.close(fig)


# --------------------------------------------------------------- 5. oracle
def fig_oracle():
    o = json.load(open(EXP / "results" / "oracle_dv_val.json"))
    strata = [("overall", "overall"), ("day", "day"), ("night", "night"),
              ("size_small", "small"), ("size_medium", "medium"),
              ("size_large", "large")]
    ir = [o[f"oracle_gain_{k}"]["recall_ir"] * 100 for k, _ in strata]
    rgb = [o[f"oracle_gain_{k}"]["recall_rgb"] * 100 for k, _ in strata]
    best = [o[f"oracle_gain_{k}"]["recall_best_of_either"] * 100 for k, _ in strata]
    fig, ax = plt.subplots(figsize=(3.35, 2.1))
    x = np.arange(len(strata)); w = 0.26
    ax.bar(x - w, ir, width=w, label="IR-only", color="#444444")
    ax.bar(x, rgb, width=w, label="RGB-only", color="#E69F00")
    ax.bar(x + w, best, width=w, label="oracle", color="#0072B2")
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in strata])
    ax.set_ylabel("recall (%)"); ax.set_ylim(60, 100)
    ax.legend(ncol=3, loc="lower center", fontsize=6)
    save(fig, "fig_oracle"); plt.close(fig)


# -------------------------------------------------------------- 6. deltas
def fig_deltas():
    rows = [
        ("E10 \u2212 E00", "e10_v2_100ep_minus_e00_ir"),
        ("E30 \u2212 E00", "e30_v2_100ep_minus_e00_ir"),
        ("E35 \u2212 E00", "e35_v2_100ep_minus_e00_ir"),
        ("E41 \u2212 E00", "e41_v2_80ep_minus_e00_ir"),
        ("d192 \u2212 E00", "e30d192_v2_100ep_minus_e00_ir"),
        ("E30 \u2212 E10", "e30_v2_100ep_minus_e10_v2_100ep"),
        ("E41 \u2212 E30", "e41_v2_80ep_minus_e30_v2_100ep"),
        ("d192 \u2212 E30", "e30d192_v2_100ep_minus_e30_v2_100ep"),
    ]
    boot = {}
    for f in glob.glob(str(EXP / "results" / "bootstrap_*.json")):
        boot.update(json.load(open(f))["deltas"])
    labels, means, los, his = [], [], [], []
    for lab, key in rows:
        if key not in boot:
            continue
        v = boot[key]
        labels.append(lab)
        means.append(v["mean"] * 100); los.append(v["ci95"][0] * 100); his.append(v["ci95"][1] * 100)
    fig, ax = plt.subplots(figsize=(3.35, 2.3))
    y = np.arange(len(labels))[::-1]
    ax.errorbar(means, y, xerr=[np.array(means) - np.array(los),
                                np.array(his) - np.array(means)],
                fmt="o", ms=3, capsize=2, lw=0.9, color="#0072B2")
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("\u0394AP (pp, paired bootstrap 95% CI)")
    save(fig, "fig_deltas"); plt.close(fig)


# ------------------------------------------------------------ 7. dataset
def fig_dataset():
    gt = json.load(open(EXP / "results" / "gt" / "dv_v2_ir_val.json"))
    names = {c["id"]: c["name"] for c in gt["categories"]}
    counts = {names[cid]: 0 for cid in names}
    for a in gt["annotations"]:
        counts[names[a["category_id"]]] += 1
    o = json.load(open(EXP / "results" / "oracle_dv_val.json"))
    small = sum(o["by_size"]["small"].values())
    med = sum(o["by_size"]["medium"].values())
    large = sum(o["by_size"]["large"].values())
    day = sum(o["by_day_night"]["day"].values())
    night = sum(o["by_day_night"]["night"].values())

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 1.8))
    ax = axes[0]
    ks = list(counts); vs = [counts[k] for k in ks]
    ax.bar(ks, vs, color="#0072B2")
    ax.set_ylabel("instances (val)"); ax.set_title("class distribution")
    for i, v in enumerate(vs):
        ax.text(i, v * 1.01, f"{v}", ha="center", fontsize=5.5)
    ax = axes[1]
    ax.bar(["small", "medium", "large"], [small, med, large],
           color=["#E69F00", "#009E73", "#CC79A7"])
    ax.set_ylabel("instances (val)"); ax.set_title("size / illumination")
    ax.text(2.35, max(small, med, large) * 0.55,
            f"day {day}\nnight {night}", fontsize=6, ha="right", va="center")
    save(fig, "fig_dataset"); plt.close(fig)


if __name__ == "__main__":
    fig_curves(); fig_test_ap(); fig_per_class(); fig_scale()
    fig_oracle(); fig_deltas(); fig_dataset()
    for p in sorted(FIG.glob("*.pdf")):
        print(p.name, p.stat().st_size)
