"""Qualitative contact sheet: source frame vs generated condition.

Draws ground-truth boxes (the generated labels may have fewer boxes for
obstruction) and writes one PNG per condition/severity under results/.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from . import labels, paths


def _annotate(img: np.ndarray, box_rows) -> np.ndarray:
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape
    for box in box_rows:
        x1, y1, x2, y2 = labels.box_pixels(box, w, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return out


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a source/generated contact sheet")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--severity", default="medium")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src_dir = paths.CLEAR_IMG[args.split]
    src_lbl = paths.CLEAR_LBL[args.split]
    gen = paths.condition_dir(args.condition, args.severity, args.split)
    images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in paths.IMG_EXTS)[: args.n]
    if not images:
        raise SystemExit(f"no source images in {src_dir}")

    rows = []
    for src in images:
        original = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        generated = cv2.imread(str(gen / "images" / f"{src.stem}.jpg"), cv2.IMREAD_GRAYSCALE)
        if original is None or generated is None:
            continue
        if generated.shape != original.shape:
            generated = cv2.resize(generated, (original.shape[1], original.shape[0]))
        left = _annotate(original, labels.read_yolo(src_lbl / f"{src.stem}.txt"))
        right = _annotate(generated, labels.read_yolo(gen / "labels" / f"{src.stem}.txt"))
        pair = np.hstack([
            _caption(left, f"source {src.stem[:36]}"),
            _caption(right, f"{args.condition}/{args.severity}"),
        ])
        rows.append(pair)
    if not rows:
        raise SystemExit("no readable image pairs")
    sheet = np.vstack(rows)
    out = paths.RESULTS / "contact_sheets" / f"{args.condition}_{args.severity}_{args.split}.png"
    if args.out:
        out = paths.EXP_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"[sample_sheet] -> {out} ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
