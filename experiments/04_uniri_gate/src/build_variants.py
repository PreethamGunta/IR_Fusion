"""Build UNIRI-TID preprocessing variants for the GATE-IR de-risking baselines.

Variants:
  always : gamma+CLAHE on every frame (always-on restoration)
  oracle : true condition -> identity / gamma / CLAHE
  hard   : predicted condition -> identity / gamma / CLAHE
  soft   : probability-weighted blend of identity / gamma / CLAHE

The gate is the 5-feature MLP (variance, min, max, entropy, Laplacian variance)
trained on the recovered UNIRI weather labels, not on image-statistic clusters.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
UNIRI = ROOT / "data" / "uniri_tid"
SRC = UNIRI / "yolo"
OUT = EXP / "results"
CONDITIONS = ("clear", "fog", "rain")
CLASS_ID = {c: i for i, c in enumerate(CONDITIONS)}
IMG_EXTS = {".jpg", ".jpeg", ".png"}
FEATURE_SIZE = 256


def conditions() -> dict[int, str]:
    with open(UNIRI / "conditions.csv") as f:
        return {int(r["frame_id"]): r["condition"] for r in csv.DictReader(f)}


def _images(split: str) -> list[Path]:
    return sorted(p for p in (SRC / "images" / split).iterdir() if p.suffix.lower() in IMG_EXTS)


def _features(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    x = gray.astype(np.float32) / 255.0
    hist, _ = np.histogram(x, bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64) / (hist.sum() + 1e-12)
    entropy = float(-np.sum(p * np.log2(p + 1e-12)) / 8.0)
    lap = cv2.Laplacian((x * 255).astype(np.float32), cv2.CV_32F)
    return np.array([float(np.var(x)), float(np.min(x)), float(np.max(x)),
                     entropy, float(np.var(lap))], dtype=np.float32)


def build_features(paths: list[Path], workers: int) -> np.ndarray:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return np.vstack(list(ex.map(_features, [str(p) for p in paths], chunksize=32)))


def train_gate(workers: int) -> dict:
    cond = conditions()
    splits = {}
    for split in ("train", "val", "test"):
        imgs = _images(split)
        y = np.array([CLASS_ID[cond[int(p.stem)]] for p in imgs], dtype=int)
        x = build_features(imgs, workers)
        splits[split] = (imgs, x, y)

    scaler = StandardScaler().fit(splits["train"][1])
    clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=300, early_stopping=True,
                        n_iter_no_change=20, random_state=42)
    clf.fit(scaler.transform(splits["train"][1]), splits["train"][2])

    metrics = {"train": int(len(splits["train"][2])), "features":
               ["variance", "min", "max", "entropy", "laplacian_var"], "test": {}}
    preds: dict[str, dict] = {}
    for split, (imgs, x, y) in splits.items():
        probs = clf.predict_proba(scaler.transform(x))
        pred = probs.argmax(1)
        if split == "test":
            metrics["test"] = {
                "accuracy": float(accuracy_score(y, pred)),
                "f1_macro": float(f1_score(y, pred, average="macro")),
                "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            }
        for img, pr, pb in zip(imgs, pred, probs):
            preds[img.stem] = {"pred": int(pr), "probs": [float(v) for v in pb]}

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "gate_uniri.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf}, f)
    (OUT / "gate_uniri_metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "gate_preds.json").write_text(json.dumps(preds))
    print(json.dumps(metrics, indent=2))
    return preds


def gamma(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g = float(np.clip(1.0 + 2.0 * (gray.mean() - 0.5), 0.3, 3.0))
    out = np.power(img.astype(np.float32) / 255.0, g) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _process(job: tuple[str, str, str, str, dict]) -> None:
    src, dst, variant, condition, pred = job
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return
    if variant == "always":
        out = clahe(gamma(img))
    elif variant == "oracle":
        out = img if condition == "clear" else (gamma(img) if condition == "fog" else clahe(img))
    elif variant == "hard":
        p = int(pred["pred"])
        out = img if p == 0 else (gamma(img) if p == 1 else clahe(img))
    elif variant == "soft":
        probs = np.array(pred["probs"], dtype=np.float32)
        probs = probs / (probs.sum() + 1e-8)
        out = np.clip(probs[0] * img + probs[1] * gamma(img) + probs[2] * clahe(img),
                      0, 255).astype(np.uint8)
    else:
        raise ValueError(variant)
    cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 95])


def build_variant(variant: str, preds: dict, workers: int) -> None:
    cond = conditions()
    root = UNIRI / f"yolo_{variant}"
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        jobs = []
        for img in _images(split):
            dst = root / "images" / split / img.name
            lbl_src = SRC / "labels" / split / f"{img.stem}.txt"
            lbl_dst = root / "labels" / split / f"{img.stem}.txt"
            if lbl_src.exists() and not lbl_dst.exists():
                os.symlink(lbl_src.resolve(), lbl_dst)
            jobs.append((str(img), str(dst), variant, cond[int(img.stem)],
                         preds.get(img.stem, {"pred": 0, "probs": [0.34, 0.33, 0.33]})))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_process, jobs, chunksize=16))
        print(f"[variant:{variant}] {split}: {len(jobs)} images")

    yaml = root / f"uniri_{variant}.yaml"
    yaml.write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\n"
        f"test: images/test\nnc: 1\nnames: ['Human']\n"
    )
    print(f"[variant:{variant}] -> {yaml}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="always,soft,hard,oracle")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-gate", action="store_true")
    args = ap.parse_args()

    if args.skip_gate and (OUT / "gate_preds.json").exists():
        preds = json.loads((OUT / "gate_preds.json").read_text())
    else:
        preds = train_gate(args.workers)

    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        build_variant(variant, preds, args.workers)


if __name__ == "__main__":
    main()
