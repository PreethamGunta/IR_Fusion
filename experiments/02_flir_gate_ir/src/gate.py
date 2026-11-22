"""Stage A: binary Clear/Fog gate for FLIR ADAS v2.

Reuses the GATE-IR feature set (variance, min, max, entropy, Laplacian
variance) and a small MLP. Features are computed on 8-bit pseudo-colour
thermal frames, not 14-bit RAW, so the normalization is explicit.

The gate is trained on the FLIR training videos (clear + synthetic fog) and
validated on a video-disjoint subset of those videos. It is tested on the
held-out VTT test videos, and cross-tested on the other fog generator.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .paths import FOG, RESULTS, CLEAR_TEST_IMG, CLEAR_TRAIN_IMG, CLEAR_TRAIN_LBL

VIDEO_RE = re.compile(r"video-([^-]+)-frame-")
FEATURE_NAMES = ["variance", "min", "max", "entropy", "laplacian_var"]
FEATURE_SIZE = 256


def video_id(path: str | Path) -> str:
    m = VIDEO_RE.search(Path(path).name)
    return m.group(1) if m else Path(path).name


def extract_features(path: str | Path) -> np.ndarray:
    """Return the 5 GATE-IR features for one image."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (FEATURE_SIZE, FEATURE_SIZE), interpolation=cv2.INTER_AREA)
    x = gray.astype(np.float32) / 255.0

    variance = float(np.var(x))
    vmin = float(np.min(x))
    vmax = float(np.max(x))

    hist, _ = np.histogram(x, bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64)
    p /= p.sum() + 1e-12
    entropy = float(-np.sum(p * np.log2(p + 1e-12)) / 8.0)  # normalized to [0,1]

    laplacian = cv2.Laplacian((x * 255).astype(np.float32), cv2.CV_32F)
    lap_var = float(np.var(laplacian))

    return np.array([variance, vmin, vmax, entropy, lap_var], dtype=np.float32)


def build_features(paths: list[Path], workers: int = 8) -> np.ndarray:
    paths = [str(p) for p in paths]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        feats = list(ex.map(extract_features, paths, chunksize=64))
    return np.vstack(feats)


def _list_images(d: Path, limit: int | None = None) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
    return files[:limit] if limit else files


def _labelled_pairs(img_dir: Path, lbl_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for img in _list_images(img_dir):
        lbl = lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def _group_split(groups: list[str], val_frac: float, seed: int) -> tuple[set[str], set[str]]:
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    return set(uniq[n_val:]), set(uniq[:n_val])


def _metrics(y_true, y_pred, name: str) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return {
        "name": name,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix_labels_0clear_1fog": cm,
        "false_negative_rate_fog_as_clear": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "false_positive_rate_clear_as_fog": float(fp / (fp + tn)) if (fp + tn) else 0.0,
    }


def train(condition: str = "cyclegan", seed: int = 42, val_frac: float = 0.1,
          workers: int = 8, limit: int | None = None) -> dict:
    clear_pairs = _labelled_pairs(CLEAR_TRAIN_IMG, CLEAR_TRAIN_LBL)
    fog_pairs = _labelled_pairs(FOG[condition]["train_img"], FOG[condition]["train_lbl"])
    if limit:
        clear_pairs = clear_pairs[:limit]
        fog_pairs = fog_pairs[:limit]

    clear_groups = [video_id(p) for p, _ in clear_pairs]
    train_groups, val_groups = _group_split(clear_groups, val_frac, seed)
    # Apply the same video split to both classes so the same scene never appears
    # in train and val (the fog images are derived from the clear ones).
    c_train = [p for p in clear_pairs if video_id(p[0]) in train_groups]
    c_val = [p for p in clear_pairs if video_id(p[0]) in val_groups]
    f_train = [p for p in fog_pairs if video_id(p[0]) in train_groups]
    f_val = [p for p in fog_pairs if video_id(p[0]) in val_groups]

    print(f"[gate] clear train={len(c_train)} val={len(c_val)}; fog train={len(f_train)} val={len(f_val)}")

    x_train = np.vstack([
        build_features([p for p, _ in c_train], workers),
        build_features([p for p, _ in f_train], workers),
    ])
    y_train = np.concatenate([np.zeros(len(c_train), dtype=int), np.ones(len(f_train), dtype=int)])

    x_val = np.vstack([
        build_features([p for p, _ in c_val], workers),
        build_features([p for p, _ in f_val], workers),
    ])
    y_val = np.concatenate([np.zeros(len(c_val), dtype=int), np.ones(len(f_val), dtype=int)])

    scaler = StandardScaler().fit(x_train)
    clf = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=seed,
    )
    clf.fit(scaler.transform(x_train), y_train)

    out = {
        "condition": condition,
        "seed": seed,
        "features": FEATURE_NAMES,
        "train": {"n": int(len(y_train)), "n_videos": len(train_groups)},
        "val": _metrics(y_val, clf.predict(scaler.transform(x_val)), "val"),
        "test": {},
    }

    # Held-out VTT test: clear + the same generator's fog.
    test_sets = {
        "vtt_clear": (CLEAR_TEST_IMG, "clear"),
        "vtt_fog": (FOG[condition]["test_img"], "fog"),
    }
    for other in FOG:
        if other != condition:
            test_sets[f"vtt_fog_{other}"] = (FOG[other]["test_img"], "fog")

    for name, (img_dir, label) in test_sets.items():
        paths = _list_images(img_dir)
        feats = build_features(paths, workers)
        pred = clf.predict(scaler.transform(feats))
        truth = np.full(len(paths), 0 if label == "clear" else 1, dtype=int)
        out["test"][name] = _metrics(truth, pred, name)

    model_path = RESULTS / f"gate_{condition}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf, "features": FEATURE_NAMES}, f)
    out["model_path"] = str(model_path)

    metrics_path = RESULTS / f"gate_{condition}.json"
    metrics_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[gate] saved {model_path} and {metrics_path}")
    return out


def load_gate(condition: str):
    with open(RESULTS / f"gate_{condition}.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["scaler"], bundle["clf"]


def predict_paths(paths: list[Path], condition: str, workers: int = 8) -> np.ndarray:
    scaler, clf = load_gate(condition)
    feats = build_features(paths, workers)
    return clf.predict(scaler.transform(feats))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the binary Clear/Fog gate")
    ap.add_argument("--condition", default="cyclegan", choices=list(FOG))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap images per class")
    args = ap.parse_args()
    train(args.condition, args.seed, args.val_frac, args.workers, args.limit)


if __name__ == "__main__":
    main()
