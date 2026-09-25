"""Diamond thermometry: nitrogen aggregation and platelet degradation temperatures.

Implements, from normalised (1 cm) diamond spectra:

- **Nitrogen aggregation model temperature** T_N: second-order A→B kinetics
  (Taylor et al. 1990) with the revised activation energy and pre-exponential factor
  (Taylor et al. 1996), as used by QUIDDIT and SNAC: Ea/R = 81,160 K, A = 293,608 ppm⁻¹ s⁻¹.
- **Platelet degradation temperature** T_P (Speich et al. 2018, Contrib. Mineral. Petrol.
  173:39). The platelet peak area expected without degradation is P0 = 64 × μB (Eq. 1,
  μB = B-centre absorption = [N_B] / 79.4); degradation is first order, k_P =
  ln(P0/Pt) / t (Eq. 11); and T_P = (Ea/R) / (ln A_P − ln k_P) (Eq. 13).
- **Platelet peak fits following QUIDDIT** (Speich & Kohn 2020, Computers & Geosciences
  144:104558): asymmetric pseudo-Voigt fits of the 3107 cm⁻¹ peak and of the platelet
  region (platelet, 1405 and 1332 cm⁻¹ peaks plus a constant) with QUIDDIT's windows,
  starting values, bounds and constraints, so platelet areas are comparable with
  published QUIDDIT results.
- Platelet "regularity" measures (Speich et al. 2018): fraction of expected platelets
  remaining (diagram one) and the peak-position deviation from Eq. 3 (diagram two).
- Simultaneous nitrogen aggregation and cooling via SNAC (Wincott et al. 2026), vendored.

Temperatures are in °C, durations (mantle residence times) in Ma, areas in cm⁻².
[CITATION NEEDED: Taylor et al. 1990, 1996; Boyd et al. 1994, 1995; Wincott et al. 2026 —
full references in docs/thermometry.md]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d

SECONDS_PER_MA = 1e6 * 365.25 * 24 * 3600
KELVIN = 273.15

# Nitrogen aggregation (Taylor et al. 1990 with constants revised in Taylor et al. 1996)
AGGREGATION_EA_R = 81160.0  # K
AGGREGATION_A = 293608.0  # ppm^-1 s^-1

# Platelet degradation (Speich et al. 2018, Fig. 13). "combined" = natural + experimental
# data (the paper's preferred fit, 88.4 ± 0.9 × 10³ K and 19.7 ± 0.5); the unrounded values
# are those in QUIDDIT's implementation. "natural" = natural diamonds only (73.8 ± 7.2 × 10³ K,
# 9.9 ± 4.8).
PLATELET_CALIBRATIONS: dict[str, tuple[float, float]] = {
    "combined": (88446.0, 19.687),
    "natural": (73.8e3, 9.9),
}
# Eq. 1 (regular reference diamond Mur 235) and Eq. 2 (kernel density of regular samples)
REGULAR_PLATELET_SLOPE = {"reference": 64.0, "kde": 60.9}
B_PPM_PER_CM = 79.4


# ---------------------------------------------------------------------------- temperatures
def nitrogen_temperature(total_n_ppm, b_fraction, duration_ma):
    """Model temperature (°C) from total nitrogen (ppm), fraction of N in B-centres
    (0-1) and mantle residence time (Ma). Array-friendly; NaN where undefined."""
    nt = np.asarray(total_n_ppm, dtype=float)
    frac = np.asarray(b_fraction, dtype=float)
    t = np.asarray(duration_ma, dtype=float) * SECONDS_PER_MA
    with np.errstate(divide="ignore", invalid="ignore"):
        na = nt * (1 - frac)
        arg = ((nt / na) - 1) / (t * nt * AGGREGATION_A)
        temp = -AGGREGATION_EA_R / np.log(arg) - KELVIN
    ok = (nt > 0) & (frac > 0) & (frac < 1) & (t > 0)
    return np.where(ok, temp, np.nan)[()]


def platelet_initial_area(b_ppm, slope: float = REGULAR_PLATELET_SLOPE["reference"]):
    """Expected platelet peak area (cm⁻²) without degradation: slope × μB (Eq. 1)."""
    return slope * np.asarray(b_ppm, dtype=float) / B_PPM_PER_CM


def platelet_temperature(
    platelet_area,
    b_ppm,
    duration_ma,
    calibration: str = "combined",
    slope: float = REGULAR_PLATELET_SLOPE["reference"],
):
    """Platelet degradation temperature (°C), Speich et al. (2018) Eq. 13.

    NaN where no degradation is measurable (area ≥ expected area, i.e. regular or
    subregular behaviour, where the method does not apply) or inputs are invalid.
    """
    ea_r, ln_a = PLATELET_CALIBRATIONS[calibration]
    pt = np.asarray(platelet_area, dtype=float)
    p0 = platelet_initial_area(b_ppm, slope)
    t = np.asarray(duration_ma, dtype=float) * SECONDS_PER_MA
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.log(p0 / pt) / t
        temp = ea_r / (ln_a - np.log(k)) - KELVIN
    ok = (pt > 0) & (p0 > pt) & (t > 0)
    return np.where(ok, temp, np.nan)[()]


def platelet_duration(
    platelet_area,
    b_ppm,
    temperature_c,
    calibration: str = "combined",
    slope: float = REGULAR_PLATELET_SLOPE["reference"],
):
    """Inverse of :func:`platelet_temperature`: residence time (Ma) at a given T (°C)."""
    ea_r, ln_a = PLATELET_CALIBRATIONS[calibration]
    pt = np.asarray(platelet_area, dtype=float)
    p0 = platelet_initial_area(b_ppm, slope)
    k = np.exp(ln_a - ea_r / (np.asarray(temperature_c, dtype=float) + KELVIN))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.log(p0 / pt) / k / SECONDS_PER_MA)[()]


def platelet_regularity(platelet_area, platelet_position, b_ppm) -> dict[str, Any]:
    """Speich et al. (2018) diagrams one and two, as numbers.

    - ``remaining``: Pt / P0 (1 = regular, < 1 = platelets degraded; diagram one, Eq. 1)
    - ``position_deviation``: measured minus expected peak position (cm⁻¹) for the measured
      area, from Eq. 3 (I(B′) = 123.3 x − 167,800). Regular diamonds: 0 ± 2 cm⁻¹;
      subregular (smaller platelets) plot at higher wavenumber.
    """
    p0 = platelet_initial_area(b_ppm)
    pt = np.asarray(platelet_area, dtype=float)
    expected_x = (pt + 167800.0) / 123.3
    with np.errstate(divide="ignore", invalid="ignore"):
        remaining = pt / p0
    return {
        "P0": p0[()] if np.ndim(p0) else float(p0),
        "remaining": remaining[()] if np.ndim(remaining) else float(remaining),
        "position_deviation": (np.asarray(platelet_position, float) - expected_x)[()],
    }


# ---------------------------------------------------------------------------- peak shapes
def _half_lorentz(x, x0, height, hwhm):
    return height * hwhm**2 / ((x - x0) ** 2 + hwhm**2)


def _half_gauss(x, x0, height, width):
    return height * np.exp(-((x - x0) ** 2) / (2 * width**2))


def asym_pseudo_voigt(x, x0, height, hwhm_l, hwhm_r, fraction):
    """QUIDDIT's asymmetric pseudo-Voigt: separate left/right widths, Lorentzian fraction.

    Note: as in QUIDDIT, the Gaussian part uses the width parameters as standard deviations.
    """
    x = np.asarray(x, dtype=float)
    left = x <= x0
    w = np.where(left, hwhm_l, hwhm_r)
    return fraction * _half_lorentz(x, x0, height, w) + (1 - fraction) * _half_gauss(
        x, x0, height, w
    )


def asym_pseudo_voigt_area(height, hwhm_l, hwhm_r, fraction) -> float:
    """Analytic area of :func:`asym_pseudo_voigt` (QUIDDIT's p_area_ana)."""
    return (
        height
        * (hwhm_l + hwhm_r)
        * (fraction * np.pi / 2 + (1 - fraction) * np.sqrt(np.pi / 2))
    )


def _slice(x, y, lo, hi):
    m = (x >= lo) & (x <= hi)
    return x[m], y[m]


def _height_at(x, y, target):
    return float(y[np.argmin(np.abs(x - target))])


def _interp(x, y, grid):
    return interp1d(x, y, kind="linear", bounds_error=False, fill_value=0)(grid)


# ---------------------------------------------------------------------------- QUIDDIT fits
@dataclass
class PeakFit:
    x0: float
    height: float
    hwhm_l: float
    hwhm_r: float
    fraction: float
    area: float
    success: bool

    @property
    def width(self) -> float:
        """QUIDDIT's 'platelet peak width': HWHM_l + HWHM_r."""
        return self.hwhm_l + self.hwhm_r


def fit_3107(x, y) -> PeakFit:
    """Fit the 3107 cm⁻¹ peak as QUIDDIT does: cubic baseline through 3000-3050 and
    3150-3200 cm⁻¹, then an asymmetric pseudo-Voigt on a 0.1 cm⁻¹ grid."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    hx, hy = _slice(x, y, 3000, 3200)
    bg = ((hx >= 3000) & (hx <= 3050)) | ((hx >= 3150) & (hx <= 3200))
    hy = hy - np.polyval(np.polyfit(hx[bg], hy[bg], 3), hx)
    grid = np.arange(hx[0], hx[-1], 0.1)
    data = _interp(hx, hy, grid)

    def sse(p):
        return np.sum((data - asym_pseudo_voigt(grid, *p)) ** 2)

    res = optimize.minimize(
        sse,
        x0=(3107, 0, 1, 1, 0.5),
        method="SLSQP",
        bounds=[(3106, 3108), (0, None), (0.001, 5), (0.001, 5), (0, 1)],
    )
    x0, h, hl, hr, f = res.x
    return PeakFit(
        x0, h, hl, hr, f, asym_pseudo_voigt_area(h, hl, hr, f), bool(res.success)
    )


@dataclass
class PlateletFit(PeakFit):
    centroid: float = np.nan
    h1405: PeakFit | None = None
    b1332: PeakFit | None = None
    offset: float = 0.0

    @property
    def symmetry(self) -> float:
        """QUIDDIT's 'platelet peak symmetry': position minus centroid (cm⁻¹)."""
        return self.x0 - self.centroid


def _three_peaks(grid, p, quiddit_compatible: bool = False):
    b1332 = asym_pseudo_voigt(grid, *p[10:15])
    if quiddit_compatible:
        # QUIDDIT's ultimatepsv uses the 1405 height (H_I) in the Gaussian part of the 1332
        # peak; reproduce it only to match published QUIDDIT numbers exactly.
        x0, h, hl, hr, f = p[10:15]
        w = np.where(grid <= x0, hl, hr)
        b1332 = f * _half_lorentz(grid, x0, h, w) + (1 - f) * _half_gauss(
            grid, x0, p[6], w
        )
    return (
        asym_pseudo_voigt(grid, *p[0:5])
        + asym_pseudo_voigt(grid, *p[5:10])
        + b1332
        + p[15]
    )


def fit_platelet(
    x, y, height_3107: float, quiddit_compatible: bool = False
) -> PlateletFit | None:
    """Fit the platelet region as QUIDDIT does (1327-1420 cm⁻¹; platelet, 1405 and 1332
    peaks plus a constant, fitted together). Returns None when no platelet peak is found.

    Bounds follow QUIDDIT: platelet position within ±1.5 cm⁻¹ of the maximum in
    1350-1380; 1405 height within ±20% of 0.257 × the 3107 height (QUIDDIT's empirical
    relation); 1332 height within ±10% of the measured value; left/right widths of each
    peak within 10 cm⁻¹ of each other. Unlike QUIDDIT, the 1332 peak uses its own height in
    both its Lorentzian and Gaussian parts (QUIDDIT's code uses the 1405 height in the
    Gaussian part); ``quiddit_compatible=True`` reproduces QUIDDIT's behaviour exactly.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    px, py = _slice(x, y, 1327, 1420)
    p2x, p2y = _slice(x, y, 1350, 1380)
    if px.size < 10 or p2x.size < 3:
        return None
    grid = np.arange(px[0], px[-1], 0.1)
    data = _interp(px, py, grid)

    i1405 = 0.257 * height_3107
    i1332 = _height_at(px, py, 1332)
    b_lo, b_hi = (0.0, 0.5) if i1332 <= 0 else (0.9 * i1332, 1.1 * i1332)
    p_max = float(p2x[np.argmax(p2y)])
    cmin = float(np.min(py))
    c_lo, c_hi = (0.0, cmin) if cmin >= 0 else (cmin, 0.0)
    bounds = [
        (p_max - 1.5, p_max + 1.5), (0, None), (0.01, 50), (0.01, 50), (0, 1),
        (1404.5, 1405.5), (0.8 * i1405, 1.2 * i1405), (0.1, 5), (0.1, 5), (0, 1),
        (1331, 1333), (b_lo, b_hi), (0.1, 5), (0.1, 5), (0, 1),
        (c_lo, c_hi),
    ]  # fmt: skip
    cons = [
        {"type": "ineq", "fun": lambda p: 10 - abs(p[7] - p[8])},
        {"type": "ineq", "fun": lambda p: 10 - abs(p[12] - p[13])},
        {"type": "ineq", "fun": lambda p: 10 - abs(p[3] - p[2])},
    ]

    def sse(p):
        return np.sum((data - _three_peaks(grid, p, quiddit_compatible)) ** 2)

    starts = [
        (p_max, 0, 5, 5, 1, 1405, i1405, 5, 5, 1, 1332, i1332, 5, 5, 0, 0),
        (1370, 0, 5, 5, 1, 1405, 0, 5, 5, 1, 1332, 0, 5, 5, 0, 1),  # QUIDDIT's fallback
    ]
    res = None
    for start in starts:
        start = np.clip(start, [b[0] if b[0] is not None else -np.inf for b in bounds],
                        [b[1] if b[1] is not None else np.inf for b in bounds])  # fmt: skip
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = optimize.minimize(
                sse, x0=start, method="SLSQP", bounds=bounds, constraints=cons
            )
        if res.success:
            break
    p = res.x
    if p[1] <= 0:
        return None
    x0, h, hl, hr, f = p[:5]
    lo = x0 - (15 if hl < 1 else 15 * hl)
    hi = x0 + (15 if hr < 1 else 15 * hr)
    sx, _ = _slice(x, y, lo, hi)
    profile = asym_pseudo_voigt(sx, x0, h, hl, hr, f)
    centroid = float(np.average(sx, weights=profile)) if np.any(profile) else np.nan
    return PlateletFit(
        x0, h, hl, hr, f, asym_pseudo_voigt_area(h, hl, hr, f), bool(res.success),
        centroid=centroid,
        h1405=PeakFit(*p[5:10], asym_pseudo_voigt_area(*p[6:10]), True),
        b1332=PeakFit(*p[10:15], asym_pseudo_voigt_area(*p[11:15]), True),
        offset=float(p[15]),
    )  # fmt: skip


# ---------------------------------------------------------------------------- one spectrum
def thermometry_row(
    x,
    y,
    total_n_ppm: float,
    b_ppm: float,
    duration_ma: float,
    calibration: str = "combined",
    quiddit_compatible: bool = False,
) -> dict[str, float]:
    """All thermometry outputs for one normalised spectrum (x, y in cm⁻¹ absorption)."""
    total = float(total_n_ppm)
    b_frac = float(b_ppm) / total if total > 0 else np.nan
    row: dict[str, float] = {
        "T_N (C)": float(nitrogen_temperature(total, b_frac, duration_ma))
    }
    h = fit_3107(x, y)
    row |= {"H3107_height_qd": h.height, "H3107_area_qd": h.area}
    pl = fit_platelet(x, y, h.height, quiddit_compatible)
    nan = float("nan")
    if pl is None:
        row |= dict.fromkeys(
            ("platelet_area_qd", "platelet_x0_qd", "platelet_width_qd", "platelet_symmetry_qd",
             "platelet_P0", "platelet_remaining", "platelet_position_dev", "T_P (C)"), nan)  # fmt: skip
        return row
    reg = platelet_regularity(pl.area, pl.x0, b_ppm)
    row |= {
        "platelet_area_qd": pl.area,
        "platelet_x0_qd": pl.x0,
        "platelet_width_qd": pl.width,
        "platelet_symmetry_qd": pl.symmetry,
        "platelet_P0": float(reg["P0"]),
        "platelet_remaining": float(reg["remaining"]),
        "platelet_position_dev": float(reg["position_deviation"]),
        "T_P (C)": float(
            platelet_temperature(pl.area, b_ppm, duration_ma, calibration)
        ),
    }
    return row


# ---------------------------------------------------------------------------- SNAC
def snac_cooling_model(
    age_core_ma: float,
    age_rim_ma: float,
    age_kimberlite_ma: float,
    core_total_n: float,
    core_b_fraction: float,
    rim_total_n: float,
    rim_b_fraction: float,
    **model_kwargs: Any,
):
    """Fit SNAC's simultaneous aggregation-and-cooling model (Wincott et al. 2026) to core
    and rim nitrogen data. Returns the fitted ``AggregationModel`` (see its ``.run()``
    results, ``plot_T_history()`` and ``save_history()``)."""
    from ._vendor.snac.diamond import Diamond
    from ._vendor.snac.SNACmodel import AggregationModel

    diamond = Diamond(
        age_core=age_core_ma,
        age_rim=age_rim_ma,
        age_kimberlite=age_kimberlite_ma,
        c_NT=core_total_n,
        c_agg=core_b_fraction,
        r_NT=rim_total_n,
        r_agg=rim_b_fraction,
    )
    model = AggregationModel(diamond=diamond, **model_kwargs)
    model.run()
    return model
