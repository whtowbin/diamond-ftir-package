"""The plug-in contract: file readers and analyses.

A domain package (diamond FTIR, a Raman package, ...) registers what it adds:

- a **Loader** turns a file into the workbench data layout (see :mod:`.data`);
- an **Analysis** has a settings dataclass (its form is generated), a function that
  analyses one spectrum and returns values plus curves to draw, and optionally a faster
  whole-map function. Without one, maps are analysed pixel by pixel with the same function,
  so a pixel and a single spectrum with the same settings always agree.

Register in code with ``Registry.add_*``, or from another installed package through the
entry point group ``spectral_workbench.plugins`` (a function taking the registry).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

ENTRY_POINT_GROUP = "spectral_workbench.plugins"
Progress = Callable[[int, int], None]


class Cancelled(Exception):
    """Raised by a progress callback to stop a running analysis."""


@dataclass
class Curve:
    """A line to draw over the spectrum. ``panel`` picks the plot: "raw" (the spectrum as
    loaded) or "processed" (e.g. baseline-corrected / normalised)."""

    name: str
    x: np.ndarray
    y: np.ndarray
    panel: str = "raw"
    color: str | None = None
    width: float = 1.5
    dash: bool = False


@dataclass
class SpectrumResult:
    values: dict[str, Any] = field(default_factory=dict)
    curves: list[Curve] = field(default_factory=list)
    error: str = ""


@dataclass
class Analysis:
    name: str
    make_params: Callable[[], Any]
    run: Callable[[np.ndarray, np.ndarray, Any, dict], SpectrumResult]
    run_map: Callable[[xr.Dataset, Any, Any, Progress | None], xr.Dataset] | None = None
    make_map_params: Callable[[], Any] | None = None
    choices: dict[str, tuple] = field(default_factory=dict)
    skip_fields: tuple[str, ...] = ()  # settings not shown in the form
    axis_kinds: tuple[str, ...] = ("ir", "raman", "other")
    key_columns: tuple[str, ...] = ()  # shown first in result tables
    default_layer: str | None = None  # map layer shown after a run
    enabled_by_default: bool = True
    uses_recipes: bool = False  # receives the ticked recipes as params.recipes

    def applies_to(self, ds: xr.Dataset) -> bool:
        return ds.attrs.get("axis_kind", "ir") in self.axis_kinds


@dataclass
class Loader:
    name: str
    suffixes: tuple[str, ...]
    load: Callable[[Path], xr.Dataset]
    is_map: bool = False


@dataclass
class Registry:
    loaders: list[Loader] = field(default_factory=list)
    analyses: dict[str, Analysis] = field(default_factory=dict)

    def add_loader(self, loader: Loader) -> None:
        self.loaders.insert(0, loader)  # later registrations win

    def add_analysis(self, analysis: Analysis) -> None:
        self.analyses[analysis.name] = analysis

    def loader_for(self, path: Path) -> Loader | None:
        suffix = path.suffix.lower()
        return next((ld for ld in self.loaders if suffix in ld.suffixes), None)

    @property
    def suffixes(self) -> tuple[str, ...]:
        return tuple(sorted({s for ld in self.loaders for s in ld.suffixes}))

    def load_entry_points(self) -> None:
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            ep.load()(self)


def default_registry(with_entry_points: bool = True) -> Registry:
    """Generic readers and the peak-recipe analysis, plus installed plug-ins."""
    from . import analyses, data

    reg = Registry()
    reg.add_loader(Loader("Text / CSV", (".csv", ".txt"), data.load_csv))
    reg.add_loader(Loader("OMNIC SPA", (".spa",), data.load_spa))
    reg.add_loader(Loader("Galactic SPC", (".spc",), data.load_spc))
    reg.add_loader(Loader("OMNIC map", (".map",), data.load_omnic_map, is_map=True))
    reg.add_analysis(analyses.RECIPES)
    if with_entry_points:
        reg.load_entry_points()
    return reg
