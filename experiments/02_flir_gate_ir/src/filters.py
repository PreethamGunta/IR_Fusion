"""Stage B: classical fog filters (relative-path, no learned weights).

Implemented filters:
  * gamma        - per-image adaptive gamma (the GATE-IR FogEnhancer idea)
  * clahe        - CLAHE on the LAB lightness channel
  * gamma_clahe  - gamma followed by CLAHE (default "good filter")

All filters operate on 8-bit BGR images and return 8-bit BGR. They are
deliberately cheap: the point of the gate is to skip this cost on clear
frames, so the filter must have a measurable cost.
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# Avoid OpenCV spawning extra threads inside each worker process.
cv2.setNumThreads(1)

FILTERS = ("gamma", "clahe", "gamma_clahe", "dcp")


def _adaptive_gamma(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean = float(gray.mean())
    # Same shape as GATE-IR FogEnhancer: lower mean -> gamma < 1 -> brighten.
    gamma = float(np.clip(1.0 + 2.0 * (mean - 0.5), 0.3, 3.0))
    x = img_bgr.astype(np.float32) / 255.0
    out = np.power(x, gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _clahe(img_bgr: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _box(img: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(
        img.astype(np.float32), ddepth=-1, ksize=(2 * r + 1, 2 * r + 1),
        normalize=True, borderType=cv2.BORDER_REFLECT,
    )


def _guided_filter(guide: np.ndarray, src: np.ndarray, r: int = 40,
                   eps: float = 1e-3) -> np.ndarray:
    """Fast guided filter built from box filters (no cv2.ximgproc needed)."""
    mean_i = _box(guide, r)
    mean_p = _box(src, r)
    var_i = _box(guide * guide, r) - mean_i * mean_i
    cov_ip = _box(guide * src, r) - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return _box(a, r) * guide + _box(b, r)


def _dcp(img_bgr: np.ndarray, patch: int = 15, omega: float = 0.95,
         t0: float = 0.1) -> np.ndarray:
    """Dark Channel Prior dehazing, with a box-filter guided refinement.

    Standard He et al. DCP: dark channel -> atmospheric light -> transmission
    -> recover. Works on 8-bit BGR. No optional OpenCV contrib modules.
    """
    i = img_bgr.astype(np.float32) / 255.0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))

    dark = cv2.erode(np.min(i, axis=2), kernel)
    flat = dark.ravel()
    n = max(1, int(flat.size * 0.001))
    idx = np.argpartition(flat, -n)[-n:]
    a = np.clip(np.max(i.reshape(-1, 3)[idx], axis=0), 1e-3, 1.0)

    norm = i / a
    dark_norm = cv2.erode(np.min(norm, axis=2), kernel)
    t = 1.0 - omega * dark_norm

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = _guided_filter(gray, t, r=40, eps=1e-3)
    t = np.clip(t, t0, 1.0)

    j = (i - a) / t[..., None] + a
    return np.clip(j * 255.0, 0, 255).astype(np.uint8)


def apply_filter(img_bgr: np.ndarray, name: str) -> np.ndarray:
    if name == "gamma":
        return _adaptive_gamma(img_bgr)
    if name == "clahe":
        return _clahe(img_bgr)
    if name == "gamma_clahe":
        return _clahe(_adaptive_gamma(img_bgr))
    if name == "dcp":
        return _dcp(img_bgr)
    raise ValueError(f"unknown filter: {name}")


def _process_one(args: tuple[str, str, str, int]) -> bool:
    src, dst, name, quality = args
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        return False
    out = apply_filter(img, name)
    ok = cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bool(ok)


def filter_dir(src_dir: Path, dst_dir: Path, name: str, workers: int = 8,
               quality: int = 95, limit: int | None = None) -> int:
    """Filter every image in src_dir into dst_dir (same filenames, .jpg)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    srcs = sorted(p for p in Path(src_dir).iterdir() if p.suffix.lower() in exts)
    if limit:
        srcs = srcs[:limit]
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), str(dst_dir / f"{p.stem}.jpg"), name, quality) for p in srcs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_process_one, jobs, chunksize=32))
    n = int(sum(results))
    print(f"[filter:{name}] {src_dir} -> {dst_dir}: {n}/{len(srcs)} images")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply a classical fog filter to a directory")
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--filter", default="gamma_clahe", choices=FILTERS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    filter_dir(args.src, args.dst, args.filter, args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
