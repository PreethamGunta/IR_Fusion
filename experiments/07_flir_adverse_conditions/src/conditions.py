"""The eight adverse conditions for FLIR ADAS v2 thermal frames.

Two families, because they need different treatment:

* ``cyclegan``-suitable (fog, rain, snow, crossover, sensor): a translation
  network could learn the appearance change, and each has a physical
  approximation implemented here so the experiment does not depend on a
  generator that has not been trained for that condition.
* ``physics-only`` (small targets, defocus, obstruction): CycleGAN alters
  geometry or sharp optical properties and would hallucinate structure, so
  these are generated directly on the static images.

All functions take an 8-bit grayscale frame and return an 8-bit grayscale
frame plus (possibly filtered) labels and a small metadata dict. Randomness is
passed in as a seeded ``numpy.random.Generator`` so regenerating a split from
the same seed is bit-reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import labels, severity

cv2.setNumThreads(1)


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    family: str
    method: str
    needs_labels: bool
    modifies_labels: bool
    description: str


REGISTRY: dict[str, ConditionSpec] = {
    "fog": ConditionSpec(
        "fog", "cyclegan", "atmospheric scattering (ASM)",
        False, False,
        "Fog/mist/humidity veil: I = I0*t + A*(1-t); optional depth map, "
        "otherwise uniform transmission.",
    ),
    "rain": ConditionSpec(
        "rain", "cyclegan", "particle rendering",
        False, False,
        "Thermal rain: global contrast attenuation plus motion-blurred cool "
        "droplet streaks.",
    ),
    "snow": ConditionSpec(
        "snow", "cyclegan", "particle simulation + occlusion",
        False, False,
        "Snowfall: cold flake particles, ground accumulation, reduced contrast.",
    ),
    "crossover": ConditionSpec(
        "crossover", "cyclegan", "region temperature blending",
        True, False,
        "Thermal crossover: blend each labelled target toward its local "
        "background temperature by gamma.",
    ),
    "sensor": ConditionSpec(
        "sensor", "cyclegan", "statistical + spatial masking",
        False, False,
        "Sensor degradation: Gaussian noise, fixed-pattern noise, calibration "
        "drift, vignetting and dirty-lens smudges.",
    ),
    "small_target": ConditionSpec(
        "small_target", "physics-only", "downsample/upsample + blur",
        True, False,
        "Long-range simulation: each target crop is reduced to k x k pixels, "
        "upsampled, blurred and blended back.",
    ),
    "defocus": ConditionSpec(
        "defocus", "physics-only", "Gaussian/disk PSF",
        False, False,
        "Out-of-focus optics: Gaussian blur or disk point-spread convolution.",
    ),
    "occlusion": ConditionSpec(
        "occlusion", "physics-only", "alpha-blended occluder mask",
        True, True,
        "Foliage/fence obstruction over the scene; boxes below the visibility "
        "threshold are removed from the labels.",
    ),
}

CONDITION_NAMES = tuple(REGISTRY)


# ── small utilities ─────────────────────────────────────────────────────────

def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _smooth_noise(rng: np.random.Generator, h: int, w: int, scale: int = 16) -> np.ndarray:
    """Zero-mean, unit-std smooth field at frame resolution."""
    lh, lw = max(2, h // scale), max(2, w // scale)
    low = rng.normal(size=(lh, lw)).astype(np.float32)
    low = (low - low.mean()) / (low.std() + 1e-6)
    return cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)


def _sidecar(directory: str | Path | None, stem: str, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy")) -> Path | None:
    """First matching file for ``stem`` in ``directory`` (or None)."""
    if not directory:
        return None
    d = Path(directory)
    for ext in exts:
        p = d / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _load_depth(path: Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None
    if path.suffix == ".npy":
        depth = np.load(path)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    depth = depth.astype(np.float32)
    depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    lo, hi = float(depth.min()), float(depth.max())
    return (depth - lo) / (hi - lo) if hi > lo else np.full(shape, 0.5, np.float32)


def _feather_mask(h: int, w: int, feather_frac: float = 0.15) -> np.ndarray:
    """Ones in the centre, linearly feathered to zero at the box edges."""
    dx = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)[None, :]
    dy = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)[:, None]
    d = np.minimum(dx, dy)
    feather = max(1.0, feather_frac * min(h, w))
    return np.clip(d / feather, 0.0, 1.0)


# ── 1. fog / mist / humidity ────────────────────────────────────────────────

def apply_fog(img, boxes, sev, rng, opts):
    variant = opts.get("variant") or "fog"
    if variant not in severity.FOG_VARIANTS:
        raise SystemExit(f"unknown fog variant {variant!r}")
    t_target = severity.level(severity.FOG_VARIANTS[variant], sev, "fog")
    h, w = img.shape
    atmospheric = opts.get("atmospheric")
    a = float(img.mean() if atmospheric is None else atmospheric)

    depth = _load_depth(_sidecar(opts.get("depth_dir"), opts.get("_stem", "")), (h, w))
    if depth is not None:
        beta = -np.log(max(t_target, 1e-3))
        t = np.exp(-beta * depth)
    else:
        t = np.full((h, w), t_target, np.float32)

    if opts.get("inhomogeneous", True):
        t = t * (1.0 + 0.12 * _smooth_noise(rng, h, w))
    t = np.clip(t, 0.02, 1.0)
    out = img.astype(np.float32) * t + a * (1.0 - t)
    return np.clip(out, 0, 255).astype(np.uint8), boxes, {
        "fog_variant": variant,
        "atmospheric_level": round(a, 2),
        "depth_map": depth is not None,
        "mean_transmission": round(float(t.mean()), 4),
    }


# ── 2. rain ─────────────────────────────────────────────────────────────────

def apply_rain(img, boxes, sev, rng, opts):
    h, w = img.shape
    mean = float(img.mean())
    contrast = severity.level(severity.RAIN_CONTRAST, sev, "rain")
    out = mean + contrast * (img.astype(np.float32) - mean)

    n = int(severity.level(severity.RAIN_DROPS_PER_MP, sev, "rain") * h * w / 1e6)
    amp = severity.level(severity.RAIN_AMPLITUDE, sev, "rain")
    layer = np.zeros((h, w), np.float32)
    lengths = rng.integers(10, 34, n)
    for i in range(n):
        x = int(rng.integers(-20, w + 20))
        y = int(rng.integers(-20, h + 20))
        length = int(lengths[i])
        angle = np.deg2rad(90.0 + rng.normal(0.0, severity.RAIN_ANGLE_JITTER))
        dx, dy = np.cos(angle), -np.sin(angle)
        value = float(rng.normal(-0.6 * amp, 0.5 * amp))
        cv2.line(
            layer, (x, y),
            (int(x + dx * length), int(y + dy * length)),
            value, thickness=int(rng.integers(1, 3)), lineType=cv2.LINE_AA,
        )
    layer = cv2.GaussianBlur(layer, (0, 0), 0.6)
    out = out + layer
    return np.clip(out, 0, 255).astype(np.uint8), boxes, {
        "rain_streaks": n,
        "rain_amplitude": round(amp, 2),
    }


# ── 3. snow ─────────────────────────────────────────────────────────────────

def apply_snow(img, boxes, sev, rng, opts):
    h, w = img.shape
    out = img.astype(np.float32) + severity.level(severity.SNOW_COOLING, sev, "snow")
    mean = float(out.mean())
    out = mean + severity.level(severity.SNOW_CONTRAST, sev, "snow") * (out - mean)

    # Ground accumulation: a soft, low-frequency cold layer weighted to the
    # bottom of the frame. Snow is cold in a white-hot thermal palette.
    frac = severity.level(severity.SNOW_GROUND, sev, "snow")
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    ground = np.clip((yy - (1.0 - 1.6 * frac)) / max(1e-3, 0.8 * frac), 0.0, 1.0)
    ground = ground * np.clip(0.5 + 0.5 * _smooth_noise(rng, h, w), 0.0, 1.0)
    ground = cv2.GaussianBlur(ground, (0, 0), max(1.0, min(h, w) * 0.01))
    out = out * (1.0 - 0.6 * ground) + (mean - 30.0) * (0.6 * ground)

    n = int(severity.level(severity.SNOW_FLAKES_PER_MP, sev, "snow") * h * w / 1e6)
    flake_value = severity.level(severity.SNOW_FLAKE_VALUE, sev, "snow")
    mask = np.zeros((h, w), np.float32)
    layer = np.zeros((h, w), np.float32)
    for _ in range(n):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = int(rng.integers(1, 4))
        v = float(flake_value + rng.normal(0.0, 6.0))
        cv2.circle(mask, (x, y), r, 1.0, -1, lineType=cv2.LINE_AA)
        cv2.circle(layer, (x, y), r, v, -1, lineType=cv2.LINE_AA)
    # Falling flakes are motion-blurred; the same kernel is applied to the
    # value layer and its alpha so the composite stays consistent.
    kernel = np.zeros((5, 3), np.float32)
    kernel[2, 1] = 0.6
    kernel[1, 1] = 0.25
    kernel[3, 1] = 0.25
    kernel[0, 1] = 0.1
    kernel[4, 1] = 0.1
    kernel /= kernel.sum()
    mask = cv2.filter2D(mask, -1, kernel)
    layer = cv2.filter2D(layer, -1, kernel)
    out = out * (1.0 - mask) + layer
    return np.clip(out, 0, 255).astype(np.uint8), boxes, {
        "snowflakes": n,
        "ground_fraction": round(frac, 3),
    }


# ── 4. thermal crossover / high ambient ─────────────────────────────────────

def apply_crossover(img, boxes, sev, rng, opts):
    gamma = opts.get("gamma")
    gamma = severity.level(severity.CROSSOVER_GAMMA, sev, "crossover") if gamma is None else float(gamma)
    h, w = img.shape
    out = img.astype(np.float32)
    applied = 0
    for box in boxes:
        x1, y1, x2, y2 = labels.box_pixels(box, w, h)
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 4:
            continue
        margin = max(4, int(0.35 * max(bw, bh)))
        rx1, ry1 = max(0, x1 - margin), max(0, y1 - margin)
        rx2, ry2 = min(w, x2 + margin), min(h, y2 + margin)
        ring = np.ones((ry2 - ry1, rx2 - rx1), bool)
        ring[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
        samples = out[ry1:ry2, rx1:rx2][ring]
        background = float(np.median(samples)) if samples.size >= 16 else float(np.median(out))
        mask = _feather_mask(bh, bw, 0.15)
        patch = out[y1:y2, x1:x2]
        shifted = (1.0 - gamma) * patch + gamma * background
        out[y1:y2, x1:x2] = patch * (1.0 - mask) + shifted * mask
        applied += 1
    return np.clip(out, 0, 255).astype(np.uint8), boxes, {
        "crossover_gamma": round(gamma, 3),
        "targets_shifted": applied,
    }


# ── 5. sensor noise, dirty lens, calibration drift ──────────────────────────

def _smudge_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), np.float32)
    cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
    for _ in range(int(rng.integers(2, 5))):
        ax = int(rng.integers(max(8, w // 20), max(9, w // 7)))
        ay = int(rng.integers(max(8, h // 20), max(9, h // 7)))
        angle = float(rng.uniform(0, 180))
        dx = int(rng.normal(0, ax // 2))
        dy = int(rng.normal(0, ay // 2))
        cv2.ellipse(mask, (cx + dx, cy + dy), (ax, ay), angle, 0, 360, 1.0, -1,
                    lineType=cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), max(2.0, 0.01 * min(h, w)))
    peak = float(mask.max())
    return mask / peak if peak > 0 else mask


def apply_sensor(img, boxes, sev, rng, opts):
    h, w = img.shape
    out = img.astype(np.float32)

    # Fixed-pattern noise is fixed to the sensor, so it must be identical
    # across frames rather than redrawn per image.
    fpn_rng = np.random.default_rng(int(opts.get("fpn_seed", 20261110)))
    sigma = severity.level(severity.SENSOR_FPN, sev, "sensor")
    fpn = fpn_rng.normal(0.0, sigma, (h, w)).astype(np.float32)
    fpn += fpn_rng.normal(0.0, sigma * 0.8, (1, w)).astype(np.float32)
    n_hot = max(1, int(1e-4 * h * w))
    ys = fpn_rng.integers(0, h, n_hot)
    xs = fpn_rng.integers(0, w, n_hot)
    fpn[ys, xs] += fpn_rng.choice([-1.0, 1.0], n_hot) * severity.level(severity.SENSOR_HOT, sev, "sensor")
    out += fpn

    # Calibration drift: a smooth gain gradient plus vignetting.
    gx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    gy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    drift = severity.level(severity.SENSOR_DRIFT, sev, "sensor")
    out *= 1.0 + drift * (0.6 * gx + 0.4 * gy)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    out *= 1.0 - severity.level(severity.SENSOR_VIGNETTE, sev, "sensor") * np.clip(rr, 0, 1)

    # Dirty lens: a fixed smudge map, blended with a blurred version of the
    # scene plus a small opacity offset (not a uniform dark rectangle).
    smudge_rng = np.random.default_rng(int(opts.get("smudge_seed", 7)))
    smudges = severity.level(severity.SENSOR_SMUDGES, sev, "sensor")
    for _ in range(smudges):
        mask = _smudge_mask(smudge_rng, h, w) * float(rng.uniform(0.35, 0.75))
        blurred = cv2.GaussianBlur(out, (0, 0), max(2.0, 0.012 * min(h, w)))
        offset = float(rng.normal(0.0, 10.0))
        out = out * (1.0 - mask) + (blurred + offset) * mask

    out += rng.normal(0.0, severity.level(severity.SENSOR_NOISE, sev, "sensor"), (h, w))
    return np.clip(out, 0, 255).astype(np.uint8), boxes, {
        "noise_sigma": severity.level(severity.SENSOR_NOISE, sev, "sensor"),
        "fpn_sigma": sigma,
        "smudges": smudges,
    }


# ── 6. small targets (long range) ───────────────────────────────────────────

def apply_small_target(img, boxes, sev, rng, opts):
    pixels = int(opts.get("pixels") or severity.level(severity.SMALL_TARGET_PIXELS, sev, "small_target"))
    blur = severity.level(severity.SMALL_TARGET_BLUR, sev, "small_target")
    h, w = img.shape
    out = img.copy()
    applied, skipped = 0, 0
    for box in boxes:
        x1, y1, x2, y2 = labels.box_pixels(box, w, h)
        bw, bh = x2 - x1, y2 - y1
        if bw < 2 or bh < 2:
            skipped += 1
            continue
        crop = img[y1:y2, x1:x2]
        small = cv2.resize(crop, (pixels, pixels), interpolation=cv2.INTER_AREA)
        up = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
        up = cv2.GaussianBlur(up, (0, 0), blur)
        mask = _feather_mask(bh, bw, 0.18)
        out[y1:y2, x1:x2] = np.clip(
            crop.astype(np.float32) * (1.0 - mask) + up.astype(np.float32) * mask,
            0, 255,
        ).astype(np.uint8)
        applied += 1
    return out, boxes, {
        "target_pixels": pixels,
        "targets_degraded": applied,
        "targets_too_small": skipped,
    }


# ── 7. defocus ──────────────────────────────────────────────────────────────

def apply_defocus(img, boxes, sev, rng, opts):
    psf = opts.get("psf") or "gaussian"
    if psf == "disk":
        radius = int(opts.get("radius") or severity.level(severity.DEFOCUS_DISK_RADIUS, sev, "defocus"))
        k = 2 * radius + 1
        kernel = np.zeros((k, k), np.float32)
        cv2.circle(kernel, (radius, radius), radius, 1.0, -1, lineType=cv2.LINE_AA)
        kernel /= kernel.sum()
        out = cv2.filter2D(img, -1, kernel)
    elif psf == "gaussian":
        sigma = float(opts.get("sigma") or severity.level(severity.DEFOCUS_SIGMA, sev, "defocus"))
        out = cv2.GaussianBlur(img, (0, 0), sigma)
    else:
        raise SystemExit(f"unknown psf {psf!r}; expected gaussian or disk")
    return out, boxes, {"psf": psf}


# ── 8. static obstruction ───────────────────────────────────────────────────

def _fence_mask(h: int, w: int, coverage: float, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((h, w), np.float32)
    period = int(rng.integers(42, 92))
    thickness = max(2, int(round(coverage * period)))
    offset = int(rng.integers(0, period))
    for x in range(offset, w + period, period):
        cv2.rectangle(mask, (x, 0), (x + thickness, h), 1.0, -1)
    for _ in range(int(rng.integers(1, 3))):
        y = int(rng.integers(h // 5, max(h // 5 + 2, 4 * h // 5)))
        rail = max(3, int(thickness * 1.5))
        cv2.rectangle(mask, (0, y), (w, y + rail), 1.0, -1)
    return mask


def _foliage_mask(h: int, w: int, coverage: float, rng: np.random.Generator) -> np.ndarray:
    low = _smooth_noise(rng, h, w, scale=24)
    threshold = float(np.quantile(low, 1.0 - coverage))
    mask = (low >= threshold).astype(np.float32)
    n = max(12, int(coverage * 80))
    for _ in range(n):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        ax = int(rng.integers(6, max(7, w // 10)))
        ay = int(rng.integers(6, max(7, h // 10)))
        cv2.ellipse(mask, (cx, cy), (ax, ay), float(rng.uniform(0, 180)),
                    0, 360, 1.0, -1, lineType=cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (0, 0), 2.0)


def _occluder_mask(h: int, w: int, kind: str, coverage: float, rng, opts) -> tuple[np.ndarray, str]:
    mask_path = _sidecar(opts.get("mask_dir"), opts.get("_stem", ""))
    if mask_path is not None:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, :, 3] if mask.shape[2] == 4 else mask.max(axis=2)
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            hi = float(mask.max())
            if hi > 1.0:
                mask /= hi
            return np.clip(mask, 0, 1), f"file:{mask_path.name}"

    if kind == "fence":
        mask = _fence_mask(h, w, coverage, rng)
    elif kind == "foliage":
        mask = _foliage_mask(h, w, coverage, rng)
    elif kind == "mixed":
        mask = np.maximum(0.9 * _fence_mask(h, w, coverage, rng),
                          _foliage_mask(h, w, coverage, rng))
    else:
        raise SystemExit(f"unknown occluder {kind!r}; expected fence, foliage, mixed or masks")

    # The alpha scale is what makes a condition comparable across severities.
    mean = float(mask.mean())
    if mean > 1e-6:
        mask = np.clip(mask * (coverage / mean), 0.0, 1.0)
    return mask, f"procedural:{kind}"


def apply_occlusion(img, boxes, sev, rng, opts):
    h, w = img.shape
    kind = opts.get("occluder") or "mixed"
    coverage = severity.level(severity.OCCLUSION_COVERAGE, sev, "occlusion")
    mask, source = _occluder_mask(h, w, kind, coverage, rng, opts)

    value = opts.get("occluder_value")
    if value is None:
        value = float(np.percentile(img, 30)) if kind in ("foliage", "mixed") else float(np.median(img))
    value += float(rng.normal(0.0, 5.0))
    out = img.astype(np.float32) * (1.0 - mask) + value * mask

    kept, dropped = labels.filter_by_visibility(
        boxes, mask, w, h,
        float(opts.get("min_visible", severity.OCCLUSION_MIN_VISIBLE)),
    )
    return np.clip(out, 0, 255).astype(np.uint8), kept, {
        "occluder": kind,
        "occluder_source": source,
        "occluder_value": round(value, 2),
        "mask_mean_alpha": round(float(mask.mean()), 4),
        "mask_hard_coverage": round(float((mask >= 0.5).mean()), 4),
        "boxes_in": len(boxes),
        "boxes_dropped": len(dropped),
    }


APPLIERS: dict[str, Callable] = {
    "fog": apply_fog,
    "rain": apply_rain,
    "snow": apply_snow,
    "crossover": apply_crossover,
    "sensor": apply_sensor,
    "small_target": apply_small_target,
    "defocus": apply_defocus,
    "occlusion": apply_occlusion,
}


def apply_condition(name: str, img, boxes, severity_name: str, rng, opts: dict):
    if name not in APPLIERS:
        raise SystemExit(f"unknown condition {name!r}; expected one of {CONDITION_NAMES}")
    if severity_name not in severity.SEVERITIES:
        raise SystemExit(f"unknown severity {severity_name!r}; expected {severity.SEVERITIES}")
    return APPLIERS[name](img, boxes, severity_name, rng, opts)
