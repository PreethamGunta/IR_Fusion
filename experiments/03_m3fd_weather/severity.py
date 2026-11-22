"""Severity ladders for synthetic weather, shared by Methods 1 and 2.

Why this module exists
----------------------
Fog and rain each carried their own sigma dictionary:

    FOG_SIGMA  = {light: 0.80, medium: 0.60, heavy: 0.40}
    RAIN_SIGMA = {light: 0.85, medium: 0.70, heavy: 0.55}

Measured against the source frames, that made fog remove 20/40/60 % of thermal
contrast where rain removed 15/30/45 % -- a ~1.33x bias toward fog that was
decided before any detector ran. A "fog hurts more than rain" result was the
*predicted* output of the two constants, so reporting it as an empirical finding
was circular.

Severity is now one quantity, measured, and shared by both methods:

    contrast transfer ratio   CTR = std(degraded) / std(source)

There is a single ladder (`CONTRAST_TRANSFER`) and both conditions use it, so at
a given severity both arms lose the same global contrast. What the arms then
differ in is *how* the structure is lost -- Method 1 contracts contrast uniformly
and adds droplet texture, Method 2 varies the contraction with depth -- which is
the comparison the experiment is actually for.

`measure_degradation` records what a generated dataset actually achieved, so a
number quoted in the writeup can be checked against the artefact instead of
against a constant in this file. That measurement, not this ladder, is the
authority on how strong each condition turned out to be.

Scope, stated plainly: CTR is a dimensionless image-contrast measure, not a
physical one. Nothing here maps a severity label onto a rain rate or a
visibility in metres. A severity label is a position on one shared contrast
ladder, and nothing more.
"""

from __future__ import annotations

import cv2
import numpy as np

SEVERITIES = ("light", "medium", "heavy")

# Target CTR per severity: light retains 80 % of source contrast, heavy 40 %.
# These are the *targets*. The achieved value is measured per dataset and
# recorded in its generation.json.
CONTRAST_TRANSFER = {"light": 0.80, "medium": 0.60, "heavy": 0.40}


def contrast_transfer(severity: str) -> float:
    try:
        return CONTRAST_TRANSFER[severity]
    except KeyError:
        raise SystemExit(f"unknown severity {severity!r}; expected one of {SEVERITIES}")


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _hf_rms(img: np.ndarray) -> float:
    return float(np.sqrt(np.mean(cv2.Laplacian(img, cv2.CV_32F) ** 2)))


def measure_degradation(src: np.ndarray, out: np.ndarray) -> dict:
    """Contrast and high-frequency loss actually produced for one frame.

    `hf_retention` is there because CTR alone is not the whole story: a
    spatially uniform blur and a droplet layer can hit the same CTR while
    removing very different amounts of texture, and a detector responds to both.
    """
    s = _gray(src).astype(np.float32)
    o = _gray(out).astype(np.float32)
    s_std, o_std = float(s.std()), float(o.std())
    s_hf, o_hf = _hf_rms(s), _hf_rms(o)
    return {
        "contrast_transfer": (o_std / s_std) if s_std > 0 else None,
        "contrast_drop_pct": (100.0 * (1.0 - o_std / s_std)) if s_std > 0 else None,
        "hf_retention": (o_hf / s_hf) if s_hf > 0 else None,
        "mean_abs_delta": float(np.abs(o - s).mean()),
    }


class DegradationLog:
    """Running mean of `measure_degradation` over a generated dataset.

    Recorded in generation.json under "degradation", so the severity ladder is an
    auditable measurement rather than a claim. Fields that came back None for
    every frame are dropped rather than written as null, matching
    provenance.run_record's convention.
    """

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(self, src: np.ndarray, out: np.ndarray) -> None:
        self._rows.append(measure_degradation(src, out))

    @property
    def count(self) -> int:
        return len(self._rows)

    def summary(self, ndigits: int = 4) -> dict:
        if not self._rows:
            return {}
        keys = self._rows[0].keys()
        out = {}
        for k in keys:
            vals = [r[k] for r in self._rows if r[k] is not None]
            if vals:
                out[k] = round(float(np.mean(vals)), ndigits)
        out["frames"] = len(self._rows)
        return out


def retention_for_target_ctr(base: np.ndarray, retention: np.ndarray,
                             src_std: float, target_ctr: float,
                             iters: int = 30) -> np.ndarray:
    """Scale a (possibly depth-varying) retention field to hit `target_ctr`.

    Severity has to mean the same amount of lost contrast whether the retention is
    uniform (Method 1) or varies per pixel (Method 2), otherwise the two methods
    cannot be compared at equal severity. A constant cannot enforce that, because
    the achieved contrast of an exponential-in-depth retention depends on the
    scene. So it is enforced by measurement instead: one global scalar `g` is
    solved for, per frame, such that

        std(mean + g * retention * (base - mean)) == target_ctr * src_std

    `g` is a single scalar, so the *shape* of `retention` -- which is the whole
    point of Method 2, where it comes from depth -- is untouched; only its scale
    is pinned. Standard deviation is strictly increasing in `g`, so bisection is
    safe and cannot diverge.

    `base` may be colour while `retention` is per-pixel (2-D); the solve happens
    on a broadcast view and the result is returned in `retention`'s own shape, so
    callers keep deciding how to broadcast it.
    """
    if src_std <= 0:
        return retention
    r = retention
    if r.ndim < base.ndim:
        r = r.reshape(r.shape + (1,) * (base.ndim - r.ndim))
    m = float(np.mean(base))
    dev = base.astype(np.float32) - np.float32(m)
    want = target_ctr * src_std
    if want <= 0:
        return np.zeros_like(retention)

    def std_at(g: float) -> float:
        return float(np.std(m + g * r * dev))

    lo, hi = 0.0, 1.0
    while std_at(hi) < want and hi < 64.0:
        hi *= 2.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if std_at(mid) < want:
            lo = mid
        else:
            hi = mid
    return (0.5 * (lo + hi)) * retention