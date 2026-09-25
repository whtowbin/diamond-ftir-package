"""Reusable Qt widgets (qtpy: PySide6 standalone, or napari's Qt) and the desktop app.

Install with ``pip install diamond-ftir-package[gui]``. The widgets are spectroscopy-agnostic:
DataclassForm (settings from any dataclass), SpectrumView (fast plot with draggable region),
RecipeEditor (baselines, peaks and clusters with live preview) and MapView (click a pixel to
see its spectrum). The napari plugin uses the same widgets.
"""

from .forms import DataclassForm
from .map_view import MapView
from .recipe_editor import RecipeEditor
from .spectrum_view import SpectrumView

__all__ = ["DataclassForm", "MapView", "RecipeEditor", "SpectrumView"]
