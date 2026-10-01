"""Generic analyses shipped with the workbench, and the pixel-by-pixel map runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from ..core import Recipe, resolve_recipe, run_recipe
from .data import pixel, pixel_metadata
from .plugins import Analysis, Curve, Progress, SpectrumResult


@dataclass
class RecipeParams:
    thickness_cm: float = field(
        default=0.0,
        metadata={
            "description": "Sample thickness (cm) for recipes that normalise by thickness; "
            "0 = not known (those recipes then report an error)."
        },
    )
    recipes: tuple = ()  # the ticked recipes, set by the session


def recipe_curves(result) -> list[Curve]:
    """Baseline and fitted model of every recipe feature, drawn on the raw spectrum."""
    curves = []
    for fr in result.features:
        base = fr.baseline
        curves.append(Curve(f"{fr.name} baseline", fr.x, base, dash=True))
        if fr.model is not None:
            curves.append(Curve(f"{fr.name} fit", fr.x, base + fr.model))
    return curves


def resolve(ref) -> Recipe:
    return ref if isinstance(ref, Recipe) else resolve_recipe(str(ref))


def run_recipes(x, y, params: RecipeParams, _meta: dict) -> SpectrumResult:
    out = SpectrumResult()
    errors = []
    for ref in params.recipes:
        recipe = resolve(ref)
        try:
            result = run_recipe(x, y, recipe, thickness=params.thickness_cm or None)
        except ValueError as e:
            errors.append(str(e))
            continue
        out.values.update(result.values)
        out.curves += recipe_curves(result)
        errors += [
            f"{recipe.name}/{f.name}: {f.error}" for f in result.features if f.error
        ]
    out.error = "; ".join(errors)
    return out


RECIPES = Analysis(
    name="Peak recipes",
    make_params=RecipeParams,
    run=run_recipes,
    skip_fields=("recipes",),
    enabled_by_default=False,
    uses_recipes=True,
)


def analyse_map_by_pixel(
    analysis: Analysis,
    ds: xr.Dataset,
    params: Any,
    progress: Progress | None = None,
    mask: np.ndarray | None = None,
) -> xr.Dataset:
    """Run ``analysis.run`` on every (masked) pixel; one result layer per numeric value."""
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    meta = pixel_metadata(ds)
    todo = (
        np.argwhere(mask) if mask is not None else np.argwhere(np.ones((ny, nx), bool))
    )
    layers: dict[str, np.ndarray] = {}
    status = np.full((ny, nx), 1, dtype=np.int8)  # 1 = not analysed
    for k, (i, j) in enumerate(todo, start=1):
        x, y = pixel(ds, int(i), int(j))
        if not np.isfinite(y).any():
            continue
        try:
            res = analysis.run(x, y, params, meta)
        except Exception:  # noqa: BLE001 - one bad pixel must not stop the map
            status[i, j] = 2
            continue
        status[i, j] = 2 if res.error and not res.values else 0
        for name, v in res.values.items():
            if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                layers.setdefault(name, np.full((ny, nx), np.nan))[i, j] = float(v)
        if progress:
            progress(k, len(todo))
    out = xr.Dataset(
        {name: (("y", "x"), arr) for name, arr in layers.items()},
        coords={"y": ds.y.values, "x": ds.x.values},
    )
    out["status"] = (("y", "x"), status)
    return out


__all__ = [
    "RECIPES",
    "RecipeParams",
    "analyse_map_by_pixel",
    "recipe_curves",
    "run_recipes",
]
