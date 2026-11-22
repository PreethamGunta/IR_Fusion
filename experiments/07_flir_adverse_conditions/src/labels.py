"""YOLO label helpers used by the label-aware conditions.

A row is ``(class_id, cx, cy, w, h)`` with normalised coordinates, exactly as
stored by the FLIR YOLO tree. Rows are copied verbatim unless a condition
actually removes a box (static obstruction); geometry is never altered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

Label = tuple[int, float, float, float, float]


def read_yolo(path: Path) -> list[Label]:
    rows: list[Label] = []
    path = Path(path)
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            rows.append((int(float(parts[0])), *(float(v) for v in parts[1:])))
        except ValueError:
            continue
    return rows


def write_yolo(path: Path, rows: list[Label]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in rows
    )
    path.write_text(text)


def box_pixels(box: Label, width: int, height: int) -> tuple[int, int, int, int]:
    """Clamped pixel rectangle ``(x1, y1, x2, y2)`` with x2 > x1, y2 > y1."""
    _, cx, cy, bw, bh = box
    x1 = int(round((cx - bw / 2.0) * width))
    y1 = int(round((cy - bh / 2.0) * height))
    x2 = int(round((cx + bw / 2.0) * width))
    y2 = int(round((cy + bh / 2.0) * height))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def visible_fraction(mask: np.ndarray, box: Label,
                     width: int, height: int) -> float:
    """Alpha-weighted fraction of a box not covered by the occluder.

    The mask is a soft alpha map, so ``1 - mean(alpha)`` is the fraction of the
    target's thermal signal that survives. A hard threshold would count a
    semi-transparent smudge as fully visible, which is not what the detector
    sees.
    """
    x1, y1, x2, y2 = box_pixels(box, width, height)
    patch = mask[y1:y2, x1:x2]
    if patch.size == 0:
        return 1.0
    return 1.0 - float(patch.mean())


def filter_by_visibility(boxes: list[Label], mask: np.ndarray,
                         width: int, height: int,
                         min_visible: float) -> tuple[list[Label], list[tuple[Label, float]]]:
    """Drop boxes whose visible fraction falls below ``min_visible``.

    This is the only condition that edits ground truth. A box that is fully
    hidden is not a detection failure, so keeping it would score the detector
    against a target that is not in the image.
    """
    kept, dropped = [], []
    for box in boxes:
        vis = visible_fraction(mask, box, width, height)
        if vis >= min_visible:
            kept.append(box)
        else:
            dropped.append((box, vis))
    return kept, dropped
