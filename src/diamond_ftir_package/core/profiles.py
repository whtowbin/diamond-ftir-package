"""Map "profiles": one number per pixel straight from the raw spectra, before any analysis.

Like OMNIC's profiles: a band area between two positions above a straight baseline drawn
between two baseline points, a height at one position (optionally above such a baseline),
or the ratio of two profiles. Vectorised over the whole cube, so a 10,000-pixel map
updates while the markers are dragged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid

PROFILE_KINDS = ("area", "height", "ratio")


@dataclass
class Profile:
    """What to show for each pixel. Positions are in spectral-axis units (cm⁻¹)."""

    kind: str = "area"  # area | height | ratio
    lo: float = 1970.0  # band start (height: a single position when lo == hi)
    hi: float = 2040.0  # band end
    baseline: bool = (
        True  # subtract a straight line through baseline_lo and baseline_hi
    )
    baseline_lo: float = 1970.0
    baseline_hi: float = 2040.0
    numerator: Profile | None = None  # ratio only
    denominator: Profile | None = None  # ratio only

    def label(self) -> str:
        if self.kind == "ratio" and self.numerator and self.denominator:
            return f"({self.numerator.label()}) / ({self.denominator.label()})"
        band = f"{min(self.lo, self.hi):g}-{max(self.lo, self.hi):g}"
        return f"peak height {band}" if self.kind == "height" else f"area {band}"


def _nearest(wn: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(wn - value)))


def _band(cube, wn: np.ndarray, p: Profile) -> tuple[np.ndarray, np.ndarray]:
    """The band's values (x increasing, float64), minus the straight baseline if asked.

    Only the band is converted to float64, never the whole cube: a 13,000-pixel map
    would otherwise be copied (hundreds of MB) every time a marker moves."""
    lo, hi = sorted((p.lo, p.hi))
    idx = np.flatnonzero((wn >= lo) & (wn <= hi))
    if idx.size == 0:
        idx = np.array([_nearest(wn, lo)])
    idx = idx[np.argsort(wn[idx])]
    y = np.asarray(cube[..., idx], dtype=np.float64)
    if p.baseline:
        i, j = _nearest(wn, p.baseline_lo), _nearest(wn, p.baseline_hi)
        y0 = np.asarray(cube[..., i : i + 1], dtype=np.float64)
        if i != j:
            y1 = np.asarray(cube[..., j : j + 1], dtype=np.float64)
            y = y - (y0 + (y1 - y0) * (wn[idx] - wn[i]) / (wn[j] - wn[i]))
        else:
            y = y - y0
    return y, wn[idx]


def compute_profile(cube, wn, profile: Profile) -> np.ndarray:
    """One value per spectrum. ``cube`` has shape (..., n_wn); ``wn`` may run either way.

    - area: integral over the band;
    - height: the band's maximum (the peak height), or the value at ``lo`` when
      ``lo == hi``; both above the straight baseline when ``baseline`` is set;
    - ratio: numerator / denominator (NaN where the denominator is 0).
    """
    wn = np.asarray(wn, dtype=np.float64)
    if profile.kind == "ratio":
        if profile.numerator is None or profile.denominator is None:
            raise ValueError("a ratio profile needs a numerator and a denominator")
        top = compute_profile(cube, wn, profile.numerator)
        bottom = compute_profile(cube, wn, profile.denominator)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(bottom != 0, top / bottom, np.nan)
    if profile.kind not in PROFILE_KINDS:
        raise ValueError(
            f"unknown profile kind {profile.kind!r}; use one of {PROFILE_KINDS}"
        )
    y, x = _band(cube, wn, profile)
    if profile.kind == "height":
        return y.max(axis=-1) if y.shape[-1] > 1 else y[..., 0]
    if x.size < 2:
        raise ValueError(
            f"band {profile.lo:g}-{profile.hi:g} contains fewer than 2 points"
        )
    return trapezoid(y, x, axis=-1)
