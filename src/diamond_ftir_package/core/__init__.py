"""Spectroscopy-agnostic analysis core: recipes of baselines, peaks and peak clusters.

Nothing here depends on diamond, infrared or a GUI; domain packs (``diamond_ftir_package``
itself for diamond FTIR) and the GUIs build on it.
"""

from .engine import FeatureResult, RecipeResult, baseline_step, run_feature, run_recipe
from .recipe import (
    BASELINE_METHODS,
    PEAK_MODELS,
    BaselineStep,
    Feature,
    PeakDef,
    Recipe,
    builtin_recipes,
    resolve_recipe,
)

__all__ = [
    "BASELINE_METHODS",
    "PEAK_MODELS",
    "BaselineStep",
    "Feature",
    "FeatureResult",
    "PeakDef",
    "Recipe",
    "RecipeResult",
    "baseline_step",
    "builtin_recipes",
    "resolve_recipe",
    "run_feature",
    "run_recipe",
]
