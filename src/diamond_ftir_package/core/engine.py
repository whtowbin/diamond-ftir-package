"""Run a Recipe on one spectrum.

``run_recipe`` returns flat output values (for tables, batches and maps) and, per feature,
the arrays needed to draw it (region data, each baseline step, fitted peaks), so the GUI
preview shows exactly what the analysis computed.

Region selection and integration use the generic :class:`Spectrum` methods, so a recipe
reproduces the package's hand-written measurements exactly when given the same settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pybaselines as pybl
from scipy.signal import medfilt

from ..Spectrum_obj import Spectrum, rubberband
from .recipe import BaselineStep, Feature, PeakDef, Recipe

FWHM_TO_SIGMA = 1.0 / 2.354820045


@dataclass
class FeatureResult:
    """Everything computed for one feature. Arrays are on the feature's region."""

    name: str
    x: np.ndarray
    y: np.ndarray
    baselines: list[np.ndarray]
    corrected: np.ndarray
    model: np.ndarray | None
    peak_curves: dict[str, np.ndarray]
    values: dict[str, float] = field(default_factory=dict)
    error: str = ""

    @property
    def baseline(self) -> np.ndarray:
        return (
            np.sum(self.baselines, axis=0) if self.baselines else np.zeros_like(self.y)
        )


@dataclass
class RecipeResult:
    values: dict[str, float]
    features: list[FeatureResult]


# ---------------------------------------------------------------------------- baselines
def baseline_step(x: np.ndarray, y: np.ndarray, step: BaselineStep) -> np.ndarray:
    """Baseline for ``y`` on ``x`` from one step."""
    method = step.method
    if method == "none":
        return np.zeros_like(y)
    fit_y = (
        medfilt(y, step.median_filter)
        if step.median_filter and step.median_filter > 1
        else y
    )
    if method == "asls":
        return pybl.whittaker.asls(fit_y, lam=step.lam, p=step.p)[0]
    if method == "rubberband":
        mid = (x.max() + x.min()) / 2
        bend = step.stretch * (x - mid) ** 2
        return rubberband(x, fit_y + bend) - bend
    if method in ("polynomial", "linear"):
        m = np.zeros_like(x, dtype=bool)
        for lo, hi in step.anchors:
            m |= (x >= lo) & (x <= hi)
        if m.sum() < (step.degree if method == "polynomial" else 1) + 1:
            raise ValueError(
                f"{method} baseline: anchor windows contain too few points"
            )
        degree = step.degree if method == "polynomial" else 1
        coef = np.polyfit(x[m], fit_y[m], degree)
        return np.polyval(coef, x)
    raise ValueError(f"unknown baseline method {method!r}")


# ---------------------------------------------------------------------------- peaks
def _integrate(spec: Spectrum, peak: PeakDef) -> dict[str, float]:
    lo, hi = peak.integrate or [peak.center - peak.fwhm, peak.center + peak.fwhm]
    part = spec.select_range(lo, hi)
    if part.Y.size < 2:
        raise ValueError(f"integration window [{lo}, {hi}] holds fewer than 2 points")
    i = int(np.argmax(part.Y))
    return {
        "area": float(spec.integrate_peak(lo, hi)),
        "height": float(part.Y[i]),
        "center": float(part.X[i]),
        "fwhm": np.nan,
    }


def _lmfit_model(shape: str, prefix: str):
    import lmfit.models as lm

    return {
        "gaussian": lm.GaussianModel,
        "lorentzian": lm.LorentzianModel,
        "voigt": lm.VoigtModel,
        "pseudo_voigt": lm.PseudoVoigtModel,
    }[shape](prefix=prefix)


def _fit_peaks(x: np.ndarray, y: np.ndarray, feature: Feature):
    """Joint lmfit of all peaks in the feature; returns per-peak values and curves."""
    # lmfit parameter names must be valid identifiers, and peak names like "1405" are not:
    # use p0_, p1_, ... internally and translate constraint names written with peak names.
    prefix = {peak.name: f"p{k}_" for k, peak in enumerate(feature.peaks)}
    model = None
    for peak in feature.peaks:
        m = _lmfit_model(feature.model, prefix[peak.name])
        model = m if model is None else model + m
    params = model.make_params()
    step = float(np.median(np.abs(np.diff(x)))) if x.size > 1 else 1.0
    for peak in feature.peaks:
        pre = prefix[peak.name]
        lo, hi = peak.center_range or [peak.center - peak.fwhm, peak.center + peak.fwhm]
        params[pre + "center"].set(value=peak.center, min=lo, max=hi)
        sigma = peak.fwhm * FWHM_TO_SIGMA
        wlo, whi = peak.fwhm_range
        params[pre + "sigma"].set(
            value=sigma,
            min=max(wlo * FWHM_TO_SIGMA, step / 10),
            max=whi * FWHM_TO_SIGMA,
        )
        near = np.abs(x - peak.center) <= max(peak.fwhm, step)
        height = float(np.max(y[near])) if near.any() else float(np.max(y))
        params[pre + "amplitude"].set(
            value=max(height, 1e-12) * peak.fwhm * 1.06, min=0
        )
    for name, expr in feature.constraints.items():
        target = _translate(_param_name(name), prefix)
        if target not in params:
            raise ValueError(f"constraint on unknown parameter '{name}'")
        params[target].set(expr=_translate(_expr(expr), prefix))
    result = model.fit(y, params, x=x)
    comps = result.eval_components(x=x)
    values, curves = {}, {}
    for peak in feature.peaks:
        pre = prefix[peak.name]
        p = result.params
        values[peak.name] = {
            "area": float(p[pre + "amplitude"].value),
            "height": float(p[pre + "height"].value)
            if pre + "height" in p
            else float(np.max(comps[pre])),
            "center": float(p[pre + "center"].value),
            "fwhm": float(p[pre + "fwhm"].value) if pre + "fwhm" in p else np.nan,
        }
        curves[peak.name] = comps[pre]
    ss_res = float(np.sum(result.residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return values, curves, result.best_fit, 1 - ss_res / ss_tot


def _param_name(name: str) -> str:
    """Users write <peak>_fwhm; lmfit's width parameter is sigma (fwhm is derived)."""
    return name[: -len("fwhm")] + "sigma" if name.endswith("_fwhm") else name


def _translate(text: str, prefix: dict[str, str]) -> str:
    """Replace '<peak name>_<param>' with the internal '<p#>_<param>'."""
    import re

    for name in sorted(prefix, key=len, reverse=True):
        text = re.sub(rf"(?<![\w]){re.escape(name)}_(?=[A-Za-z])", prefix[name], text)
    return text


def _expr(expr: str) -> str:
    import re

    return re.sub(r"\b(\w+)_fwhm\b", r"\1_sigma", expr)


# ---------------------------------------------------------------------------- runner
def run_feature(
    x: np.ndarray, y: np.ndarray, feature: Feature, scale: float = 1.0
) -> FeatureResult:
    """Baseline, then integrate or fit the peaks of one feature. ``scale`` multiplies areas
    and heights (e.g. 1 / thickness)."""
    lo, hi = feature.region
    spec = Spectrum(X=np.asarray(x, float), Y=np.asarray(y, float)).select_range(lo, hi)
    rx, ry = spec.X, spec.Y
    baselines, rest = [], ry.copy()
    for step in feature.baseline:
        b = baseline_step(rx, rest, step)
        baselines.append(b)
        rest = rest - b
    corrected = rest
    res = FeatureResult(feature.name, rx, ry, baselines, corrected, None, {})
    try:
        if feature.model == "integrate":
            cspec = Spectrum(X=rx, Y=corrected)
            per_peak = {p.name: _integrate(cspec, p) for p in feature.peaks}
            quality = np.nan
        else:
            per_peak, res.peak_curves, res.model, quality = _fit_peaks(
                rx, corrected, feature
            )
    except Exception as e:  # noqa: BLE001 - report per feature; other features still run
        res.error = f"{type(e).__name__}: {e}"
        per_peak = {
            p.name: dict.fromkeys(feature.outputs, np.nan) for p in feature.peaks
        }
        quality = np.nan
    for pname, vals in per_peak.items():
        for q in feature.outputs:
            v = vals.get(q, np.nan)
            if q in ("area", "height"):
                v = v * scale
            res.values[f"{feature.name}.{pname}.{q}"] = float(v)
    if feature.model != "integrate":
        res.values[f"{feature.name}.fit_r2"] = float(quality)
    return res


def run_recipe(
    x: np.ndarray, y: np.ndarray, recipe: Recipe, thickness: float | None = None
) -> RecipeResult:
    """Run every feature. With ``normalize_by='thickness'`` areas/heights are divided by
    ``thickness`` (required then)."""
    problems = recipe.validate()
    if problems:
        raise ValueError(f"recipe '{recipe.name}': " + "; ".join(problems))
    scale = 1.0
    if recipe.normalize_by == "thickness":
        if not thickness or not np.isfinite(thickness) or thickness <= 0:
            raise ValueError(f"recipe '{recipe.name}' needs a positive thickness")
        scale = 1.0 / thickness
    feats = [run_feature(x, y, f, scale) for f in recipe.features]
    values = {}
    for fr in feats:
        values.update(fr.values)
    return RecipeResult(values, feats)
