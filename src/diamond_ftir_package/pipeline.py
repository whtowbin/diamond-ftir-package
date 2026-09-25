"""Run the full diamond analysis on one spectrum or many files.

This is the single code path shared by the GUI, the command line and Python users, so
they always produce identical numbers for identical :class:`AnalysisParams`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .core import Recipe, resolve_recipe, run_recipe
from .DiamondSpectrum import Diamond_Spectrum
from .params import AnalysisParams

SUPPORTED_SUFFIXES = (".csv", ".spa", ".spc")


def load_spectrum(path: str | Path) -> Diamond_Spectrum:
    """Load a CSV, SPA or SPC file as a :class:`Diamond_Spectrum`."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        from .LoadCSV import CSV_to_IR_Diamond_Spectrum

        spectrum = CSV_to_IR_Diamond_Spectrum(path)
    elif suffix == ".spa":
        from .LoadSPA import Load_SPA

        raw = Load_SPA(str(path))
        if raw is None:
            raise ValueError(f"Could not read {path.name}")
        spectrum = Diamond_Spectrum(
            X=raw.X,
            Y=raw.Y,
            X_Unit="Wavenumber",
            Y_Unit="Absorbance",
            metadata=raw.metadata,
        )
    elif suffix == ".spc":
        from .LoadSPC import SPC_Diamond_FTIR_Spectrum

        spectrum = SPC_Diamond_FTIR_Spectrum(str(path))
    else:
        raise ValueError(
            f"Unsupported file type {suffix!r}; use one of {SUPPORTED_SUFFIXES}"
        )
    if spectrum is None:
        raise ValueError(f"Could not read {path.name}")
    return spectrum


def collect_files(path: str | Path) -> list[Path]:
    """A single file, or every supported file directly inside a folder (sorted)."""
    path = Path(path)
    if path.is_dir():
        return sorted(
            p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES
        )
    return [path]


def run_analysis(
    spectrum: Diamond_Spectrum,
    params: AnalysisParams | None = None,
    baseline_start: tuple[float, float] | None = None,
    thickness_cm: float | None = None,
    thickness_tolerance: float | None = None,
) -> dict[str, Any]:
    """Run the enabled measurements on ``spectrum`` (in place) and return one flat result row.

    ``baseline_start`` optionally seeds the baseline search with a (lam, p) pair; maps pass
    the sample's typical values here. ``thickness_cm`` replaces the thickness fitted from the
    diamond peaks (the fitted value is still reported as ``fitted_typeIIA_ratio``). With
    ``thickness_tolerance`` (decades), the replacement is only used when the fitted value is
    within that many decades of it; otherwise the pixel keeps its own fit and
    ``thickness_from_reference`` is 0. Raises if
    the thickness fit fails (for example when every diamond peak is saturated).
    """
    params = params or AnalysisParams()
    d = params.diamond
    spectrum.fit_baseline(params=d, start=baseline_start)
    lam, p = spectrum.baseline_params
    row: dict[str, Any] = {}
    if thickness_cm is not None:
        fitted = float(spectrum.typeIIA_ratio)
        row["fitted_typeIIA_ratio"] = fitted
        close = thickness_tolerance is None or (
            fitted > 0 and abs(np.log10(fitted / thickness_cm)) <= thickness_tolerance
        )
        row["thickness_from_reference"] = float(close)
        if close:
            spectrum.typeIIA_ratio = float(thickness_cm)
    spectrum.normalize_diamond()
    row |= {
        "typeIIA_ratio": float(spectrum.typeIIA_ratio),
        "baseline_lam": lam,
        "baseline_p": p,
    }
    row |= quality_flags(spectrum, d)

    if params.run_nitrogen:
        spectrum.Nitrogen_fit(params=params.nitrogen)
        row.update({k: float(v) for k, v in spectrum.nitrogen_dict.items()})

    if params.run_hydrogen:
        spectrum.measure_3107_peak(params=params.hydrogen)
        row["Normed_3107_Area"] = getattr(spectrum, "normed_area_3107", np.nan)
        row["Normed_3085_Area"] = getattr(spectrum, "normed_area_3085", np.nan)

    if params.run_platelets:
        spectrum.measure_platelets_and_adjacent(
            params=params.platelet, return_peak_dict=False
        )
        row["Normed_Platelet_Area"] = getattr(spectrum, "normed_area_platelet", np.nan)
        row["Normed_Platelet_Height"] = getattr(
            spectrum, "normed_height_platelet", np.nan
        )
        row["Platelet_Peak_Position"] = getattr(
            spectrum, "platelet_peak_position", np.nan
        )
        row["Normed_1405_Area"] = getattr(spectrum, "normed_area_1405", np.nan)

    th = params.thermometry
    if th.duration_ma > 0 and params.run_nitrogen:
        from .thermometry import thermometry_row

        row |= thermometry_row(
            spectrum.normalized_spectrum.X,
            spectrum.normalized_spectrum.Y,
            row["Total_N ppm"],
            row["B_Nitrogen ppm"],
            th.duration_ma,
            th.calibration,
            th.quiddit_compatible,
        )

    for ref in params.recipes:
        recipe = ref if isinstance(ref, Recipe) else _cached_recipe(str(ref))
        result = run_recipe(
            spectrum.X, spectrum.Y, recipe, thickness=spectrum.typeIIA_ratio
        )
        clash = set(result.values) & set(row)
        if clash:
            raise ValueError(
                f"recipe '{recipe.name}' output names clash with {sorted(clash)}"
            )
        row.update(result.values)

    if params.run_amber:
        spectrum.measure_amber_center(params=params.amber)
        for label, _, _ in params.amber.bands:
            row[f"Amber_{label}_Area"] = getattr(
                spectrum, f"amber_{label}_area_normed", np.nan
            )

    return row


@lru_cache(maxsize=32)
def _cached_recipe_at(ref: str, mtime: float):
    return resolve_recipe(ref)


def _cached_recipe(ref: str):
    """Load a recipe once per process (re-read if the file changed)."""
    path = Path(ref)
    return _cached_recipe_at(ref, path.stat().st_mtime if path.exists() else 0.0)


def quality_flags(spectrum: Diamond_Spectrum, d) -> dict[str, Any]:
    """Per-spectrum quality information; flags only, nothing is excluded.

    - ``diamond_r2``: how well the normalised spectrum matches the type IIa shape over the
      unsaturated phonon windows (1 = perfect). Low values suggest a non-diamond or a failed fit.
    - ``primary_saturated`` / ``nitrogen_saturated``: detector saturation in the 1970-2040 cm-1
      phonon band (thickness then comes from other windows) or in the 1000-1350 cm-1 nitrogen
      band (nitrogen values unreliable).
    - ``QA``: the warnings in words, empty when there are none.
    """
    mask = spectrum.outputdict["mask"]
    ref = spectrum.interpolated_typeIIA_Spectrum.Y[mask]
    norm = spectrum.normalized_spectrum.Y[mask]
    r2 = float(1 - np.sum((norm - ref) ** 2) / np.sum((ref - ref.mean()) ** 2))
    primary = bool(
        spectrum.test_saturation(1970, 2040, d.saturation_cutoff, d.stdev_cut_off)
    )
    n_band = spectrum.Y[(spectrum.X > 1000) & (spectrum.X < 1350)]
    nitrogen = bool(
        spectrum.test_saturation(1000, 1350, 1.8, 0.3) or n_band.max() >= 1.8
    )
    warnings = []
    if r2 < d.min_diamond_r2:
        warnings.append(f"not diamond-like (R² {r2:.2f})")
    if nitrogen:
        warnings.append("nitrogen band saturated")
    if primary:
        warnings.append("primary phonon band saturated")
    return {
        "diamond_r2": r2,
        "primary_saturated": primary,
        "nitrogen_saturated": nitrogen,
        "QA": "; ".join(warnings),
    }


def analyze_file(
    path: str | Path, params: AnalysisParams | None = None
) -> tuple[Diamond_Spectrum | None, dict[str, Any]]:
    """Analyse one file. Never raises: failures come back as ``Status`` in the row."""
    path = Path(path)
    row: dict[str, Any] = {"Filename": path.name}
    spectrum = None
    try:
        spectrum = load_spectrum(path)
        row.update(run_analysis(spectrum, params))
        row["Status"] = "OK"
    except Exception as e:  # noqa: BLE001 - one bad file must not stop a batch
        row["Status"] = f"Error: {e}"
    return spectrum, row


def analyze_paths(
    paths: Iterable[str | Path],
    params: AnalysisParams | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Analyse many files; ``progress(done, total, filename)`` is called after each."""
    paths = list(paths)
    rows = []
    for i, path in enumerate(paths, start=1):
        _, row = analyze_file(path, params)
        rows.append(row)
        if progress:
            progress(i, len(paths), Path(path).name)
    return rows
