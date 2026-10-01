"""Diamond FTIR desktop app. The reusable widgets now live in ``workbench.qt``; they are
re-exported here so existing imports keep working."""

from ..workbench.qt import DataclassForm, MapView, RecipeEditor, SpectrumView

__all__ = ["DataclassForm", "MapView", "RecipeEditor", "SpectrumView"]
