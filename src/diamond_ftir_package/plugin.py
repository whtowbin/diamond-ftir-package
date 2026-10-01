"""Diamond FTIR as a workbench plug-in: nitrogen, hydrogen, platelets, amber, thermometry and
thickness-normalised recipes, on single spectra, batches and maps.

Single spectra (and map pixels clicked in the app) go through ``pipeline.run_analysis``;
whole maps through ``maps.process_map``, which runs the same function on every pixel after
its survey and thickness planning. Registered through the ``spectral_workbench.plugins``
entry point, or directly with :func:`register`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from .DiamondSpectrum import Diamond_Spectrum
from .params import CHOICES, AnalysisParams, MapParams
from .workbench.analyses import recipe_curves, resolve
from .workbench.plugins import Analysis, Curve, Registry, SpectrumResult

NAME = "Diamond FTIR"
KEY_COLUMNS = ("QA", "typeIIA_ratio", "Total_N ppm", "B_percent", "Normed_3107_Area")
NITROGEN_COMPONENTS = ("C", "A", "X", "B", "D", "Y")


def diamond_curves(s: Diamond_Spectrum) -> list[Curve]:
    """Baseline and type IIa fit on the raw spectrum; nitrogen fit on the normalised one."""
    curves = []
    base = getattr(s, "baseline", None)
    if base is not None:
        curves.append(Curve("baseline", s.X, base, color="#1b7837", dash=True))
        ref = getattr(getattr(s, "interpolated_typeIIA_Spectrum", None), "Y", None)
        if ref is not None and s.typeIIA_ratio:
            curves.append(
                Curve(
                    "baseline + type IIa × t",
                    s.X,
                    base + s.typeIIA_ratio * ref,
                    color="#2166ac",
                )
            )
    norm = getattr(s, "normalized_spectrum", None)
    if norm is not None:
        curves.append(
            Curve(
                "normalised (1 cm)",
                norm.X,
                norm.Y,
                panel="processed",
                color="#333333",
                width=1.0,
            )
        )
    fit = getattr(s, "nitrogen_plot_fit_params", None)
    if fit is not None:
        comp = fit["fit_component_df"] * fit["fit_params"]
        wn = np.asarray(fit["wn_array"], dtype=float)
        curves.append(
            Curve(
                "nitrogen fit",
                wn,
                comp.sum(axis=1).to_numpy(),
                panel="processed",
                color="#b2182b",
            )
        )
        for k, label in enumerate(NITROGEN_COMPONENTS):
            y = comp.iloc[:, k].to_numpy()
            if np.any(y):
                curves.append(
                    Curve(
                        f"{label} centre",
                        wn,
                        y,
                        panel="processed",
                        width=1.0,
                        dash=True,
                    )
                )
    return curves


def run_spectrum(
    x: np.ndarray, y: np.ndarray, params: AnalysisParams, meta: dict[str, Any]
) -> SpectrumResult:
    from .core import run_recipe
    from .pipeline import run_analysis

    s = Diamond_Spectrum(
        X=x, Y=y, X_Unit="Wavenumber", Y_Unit="Absorbance", metadata=dict(meta)
    )
    out = SpectrumResult()
    try:
        out.values = run_analysis(s, params)
    except Exception as e:  # noqa: BLE001 - show what was fitted up to the failure
        out.error = f"{type(e).__name__}: {e}"
    out.curves = diamond_curves(s)
    for ref in params.recipes if not out.error else ():
        out.curves += recipe_curves(
            run_recipe(s.X, s.Y, resolve(ref), thickness=s.typeIIA_ratio)
        )
    return out


def run_map(
    ds: xr.Dataset, params: AnalysisParams, map_params: MapParams | None, progress=None
) -> xr.Dataset:
    from .maps import process_map

    return process_map(ds, params, map_params or MapParams(), progress=progress)


ANALYSIS = Analysis(
    name=NAME,
    make_params=AnalysisParams,
    run=run_spectrum,
    run_map=run_map,
    make_map_params=MapParams,
    choices=CHOICES,
    skip_fields=("recipes",),
    axis_kinds=("ir",),
    key_columns=KEY_COLUMNS,
    default_layer="Total_N ppm",
    uses_recipes=True,
)


def register(registry: Registry) -> None:
    registry.add_analysis(ANALYSIS)
