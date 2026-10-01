"""Spectral workbench: a generic app for single spectra, batches and maps (FTIR, Raman, ...).

Nothing in this subpackage knows about diamonds: domain packages plug in readers and
analyses (see :mod:`.plugins`); :mod:`diamond_ftir_package.plugin` is the first. A test keeps
it that way, so the subpackage can move to its own distribution once a second domain uses it.

- :mod:`.data`: the (y, x, wn) xarray layout and generic readers;
- :mod:`.plugins`: the plug-in contract and registry;
- :mod:`.session`: loading, running, project files (no GUI);
- :mod:`.qt`: the desktop app (optional ``gui`` extra).
"""

from .data import make_cube
from .plugins import Analysis, Curve, Loader, Registry, SpectrumResult, default_registry
from .session import Dataset, Session

__all__ = [
    "Analysis",
    "Curve",
    "Dataset",
    "Loader",
    "Registry",
    "Session",
    "SpectrumResult",
    "default_registry",
    "make_cube",
]
