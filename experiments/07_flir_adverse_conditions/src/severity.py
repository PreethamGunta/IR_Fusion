"""Severity ladders and degradation measurement shared by every condition.

A severity label is a position on one shared scale, not a physical unit. The
achieved degradation is measured per generated dataset and written to
``generation.json``; that measurement, not the constants here, is the authority
on how strong a condition turned out to be.
"""

from __future__ import annotations

import cv2
import numpy as np

SEVERITIES = ("light", "medium", "heavy")

# --- fog / mist / humidity (atmospheric transmission at the far plane) -------
FOG_VARIANTS = {
    "fog": {"light": 0.75, "medium": 0.55, "heavy": 0.35},
    "mist": {"light": 0.88, "medium": 0.75, "heavy": 0.60},
    "humidity": {"light": 0.93, "medium": 0.85, "heavy": 0.72},
}

# --- rain -------------------------------------------------------------------
RAIN_CONTRAST = {"light": 0.92, "medium": 0.80, "heavy": 0.64}
RAIN_DROPS_PER_MP = {"light": 700, "medium": 1800, "heavy": 4200}
RAIN_AMPLITUDE = {"light": 9.0, "medium": 18.0, "heavy": 30.0}
RAIN_ANGLE_JITTER = 10.0  # degrees either side of vertical

# --- snow -------------------------------------------------------------------
SNOW_FLAKES_PER_MP = {"light": 300, "medium": 900, "heavy": 2200}
SNOW_CONTRAST = {"light": 0.95, "medium": 0.86, "heavy": 0.72}
SNOW_COOLING = {"light": -3.0, "medium": -8.0, "heavy": -16.0}
SNOW_GROUND = {"light": 0.10, "medium": 0.28, "heavy": 0.50}
SNOW_FLAKE_VALUE = {"light": -8.0, "medium": -16.0, "heavy": -28.0}

# --- thermal crossover / high ambient --------------------------------------
CROSSOVER_GAMMA = {"light": 0.15, "medium": 0.30, "heavy": 0.50}

# --- sensor noise, dirty lens, calibration drift ---------------------------
SENSOR_NOISE = {"light": 3.0, "medium": 7.0, "heavy": 14.0}
SENSOR_FPN = {"light": 2.0, "medium": 4.0, "heavy": 8.0}
SENSOR_DRIFT = {"light": 0.02, "medium": 0.05, "heavy": 0.10}
SENSOR_VIGNETTE = {"light": 0.06, "medium": 0.14, "heavy": 0.26}
SENSOR_SMUDGES = {"light": 1, "medium": 2, "heavy": 4}
SENSOR_HOT = {"light": 20.0, "medium": 45.0, "heavy": 90.0}

# --- small targets ----------------------------------------------------------
SMALL_TARGET_PIXELS = {"light": 16, "medium": 8, "heavy": 4}
SMALL_TARGET_BLUR = {"light": 0.6, "medium": 1.1, "heavy": 1.8}

# --- defocus ----------------------------------------------------------------
DEFOCUS_SIGMA = {"light": 1.5, "medium": 3.5, "heavy": 6.0}
DEFOCUS_DISK_RADIUS = {"light": 2, "medium": 4, "heavy": 7}

# --- static obstruction -----------------------------------------------------
OCCLUSION_COVERAGE = {"light": 0.18, "medium": 0.32, "heavy": 0.50}
OCCLUSION_MIN_VISIBLE = 0.25


def level(table: dict, severity: str, name: str = "condition"):
    try:
        return table[severity]
    except KeyError:
        raise SystemExit(f"unknown severity {severity!r} for {name}; "
                         f"expected one of {SEVERITIES}")


def _f32(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) if x.ndim == 3 else x


def measure(before: np.ndarray, after: np.ndarray) -> dict:
    """Cheap, auditable measures of what a condition actually did to a frame.

    contrast_transfer = std(after) / std(before).  A value of 0.55 means the
    frame retained 55 % of its grey-level contrast; it is dimensionless and is
    *not* a physical transmission or visibility.
    """
    b, a = _f32(before), _f32(after)
    sb, sa = float(b.std()), float(a.std())
    hb = float(cv2.Laplacian(b, cv2.CV_32F).var())
    ha = float(cv2.Laplacian(a, cv2.CV_32F).var())
    mse = float(np.mean((b - a) ** 2))
    return {
        "mean_before": round(float(b.mean()), 3),
        "mean_after": round(float(a.mean()), 3),
        "mean_shift": round(float(a.mean() - b.mean()), 3),
        "contrast_transfer": round(sa / sb, 4) if sb > 1e-6 else None,
        "hf_energy_ratio": round(ha / hb, 4) if hb > 1e-6 else None,
        "psnr": round(10.0 * np.log10(255.0 ** 2 / mse), 3) if mse > 1e-9 else None,
    }
