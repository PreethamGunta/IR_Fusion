"""Build the three figures for the 07 section of the consolidated report.

Run from the experiment directory:

    python -m src.make_report_figures

The script reads the generated test images, the clear test images and
``results/protocol_summary.csv``, and writes PNGs into ``docs/report/figures``.
It does not re-run any experiment.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import labels, paths

CONDITIONS = ["fog", "rain", "snow", "crossover", "sensor", "small_target", "defocus", "occlusion"]
DISPLAY = {
    "clear": "clear (source)",
    "fog": "fog",
    "rain": "rain",
    "snow": "snow",
    "crossover": "thermal crossover",
    "sensor": "sensor / dirty lens",
    "small_target": "small target",
    "defocus": "defocus",
    "occlusion": "static obstruction",
}
OUT = paths.REPO_ROOT / "docs" / "report" / "figures"

# One representative matched frame (first VTT test frame).
STEM = "video-4FRnNpmSmwktFJKjg-frame-000745-L6K5SC6fYjHNC8uff"


def _read(path: Path) -> cv2.Mat:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _clear_img() -> Path:
    return paths.CLEAR_IMG["test"] / f"{STEM}.jpg"


def _clear_lbl() -> Path:
    return paths.CLEAR_LBL["test"] / f"{STEM}.txt"


def _gen_img(cond: str) -> Path:
    return paths.condition_dir(cond, "medium", "test") / "images" / f"{STEM}.jpg"


def _gen_lbl(cond: str) -> Path:
    return paths.condition_dir(cond, "medium", "test") / "labels" / f"{STEM}.txt"


def _draw_boxes(ax, img, rows) -> None:
    h, w = img.shape
    for box in rows:
        x1, y1, x2, y2 = labels.box_pixels(box, w, h)
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor="#33dd55", linewidth=1.1))


def _panel(ax, path: Path, lbl_path: Path, title: str) -> None:
    img = _read(path)
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    _draw_boxes(ax, img, labels.read_yolo(lbl_path))
    ax.set_title(title, fontsize=10, pad=3)
    ax.axis("off")


def figure_conditions() -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 8.7))
    panels = [("clear", _clear_img(), _clear_lbl())] + [
        (c, _gen_img(c), _gen_lbl(c)) for c in CONDITIONS
    ]
    for ax, (key, img_path, lbl_path) in zip(axes.flat, panels):
        _panel(ax, img_path, lbl_path, DISPLAY[key])
    fig.suptitle("FLIR ADAS v2 thermal conditions at medium severity", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "fig_07_conditions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_pairs() -> None:
    selected = ["sensor", "occlusion", "small_target", "defocus"]
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.0))
    for col, cond in enumerate(selected):
        _panel(axes[0, col], _clear_img(), _clear_lbl(), DISPLAY[cond])
        _panel(axes[1, col], _gen_img(cond), _gen_lbl(cond), "")
        axes[0, col].text(-0.04, 0.5, "source", transform=axes[0, col].transAxes,
                          rotation=90, va="center", ha="right", fontsize=10)
        axes[1, col].text(-0.04, 0.5, "generated", transform=axes[1, col].transAxes,
                          rotation=90, va="center", ha="right", fontsize=10)
    fig.suptitle("Source and generated frames for four selected conditions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig_07_pairs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_results() -> None:
    rows = list(csv.DictReader(open(paths.RESULTS / "protocol_summary.csv")))
    order = {c: i for i, c in enumerate(CONDITIONS)}
    rows.sort(key=lambda r: order[r["condition"]])
    names = [DISPLAY[r["condition"]] for r in rows]
    drops = [float(r["drop_rel"]) * 100.0 for r in rows]
    selected = [r["selected"] == "True" for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})
    y = list(range(len(rows)))
    colors = ["#c0392b" if s else "#95a5a6" for s in selected]
    ax1.barh(y, drops, color=colors)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names)
    ax1.invert_yaxis()
    ax1.axvline(5.0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("CM relative mAP@50--95 drop (%)")
    ax1.set_title("Screen: clear-trained detector on A_T")
    ax1.grid(axis="x", alpha=0.25)
    for yi, value in zip(y, drops):
        ax1.text(value + 0.7, yi, f"{value:.1f}", va="center", fontsize=9)

    sel_rows = [r for r in rows if r["selected"] == "True"]
    sel_names = [DISPLAY[r["condition"]] for r in sel_rows]
    cm_gap = [float(r["drop_abs"]) for r in sel_rows]
    mix_gap = [float(r["mix_gap"]) for r in sel_rows]
    yy = list(range(len(sel_rows)))
    height = 0.36
    ax2.barh([v - height / 2 for v in yy], cm_gap, height=height,
             color="#c0392b", label="CM gap")
    ax2.barh([v + height / 2 for v in yy], mix_gap, height=height,
             color="#2e86c1", label="Mix gap")
    ax2.set_yticks(yy)
    ax2.set_yticklabels(sel_names)
    ax2.invert_yaxis()
    ax2.axvline(0.0, color="black", linewidth=1)
    ax2.set_xlabel("clear $-$ adverse mAP@50--95")
    ax2.set_title("Retrained: remaining gap (lower is better)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="x", alpha=0.25)
    for yi, value in zip(yy, mix_gap):
        ax2.text(value + (0.004 if value >= 0 else -0.004), yi + height / 2,
                 f"{value:+.3f}", va="center",
                 ha="left" if value >= 0 else "right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_07_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_conditions()
    figure_pairs()
    figure_results()
    print(f"[figures] wrote three figures to {OUT}")


if __name__ == "__main__":
    main()
