"""Qt widgets (qtpy: PySide6 standalone, or napari's Qt) and the workbench main window.

Install with the ``gui`` extra. DataclassForm (a form from any settings dataclass),
SpectrumView (fast plot with navigation and a draggable region), MapView (layers,
histogram, zoom, pixel picking), RecipeEditor (baselines, peaks and clusters with live
preview) and MainWindow (all of them as dockable panels).
"""

from .forms import DataclassForm
from .map_view import MapView
from .recipe_editor import RecipeEditor
from .spectrum_view import SpectrumView
from .window import MainWindow, main

__all__ = [
    "DataclassForm",
    "MainWindow",
    "MapView",
    "RecipeEditor",
    "SpectrumView",
    "main",
]
