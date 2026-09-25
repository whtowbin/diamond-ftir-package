"""napari plugin: open OMNIC maps and inspect/analyse them with the package's own widgets.

Install with ``pip install diamond-ftir-package[napari]``; napari then lists "Diamond FTIR"
under Plugins. Suited to large maps: napari displays the full spectral cube lazily and fast.

- Reader: a ``.map`` opens as an image stack (wavenumber × y × x) plus an image of the
  diamond two-phonon band, with the raw dataset kept in the layer metadata.
- Spectrum inspector (dock widget): click a pixel to load its spectrum into the same
  RecipeEditor the desktop app uses; "Run map analysis" runs maps.process_map with the
  editor's recipe and adds every result as a layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

RAW_KEY = "diamond_ftir_raw"


def get_reader(path):
    if isinstance(path, list):
        path = path[0]
    if not str(path).lower().endswith(".map"):
        return None
    return read_map


def read_map(path):
    from .maps import load_map

    path = path[0] if isinstance(path, list) else path
    ds = load_map(path)
    scale = _scale(ds)
    cube = ds.spectra.transpose("wn", "y", "x").values
    band = ds.spectra.sel(wn=slice(1970, 2040)).mean("wn").values
    name = Path(path).stem
    meta = {RAW_KEY: ds}
    return [
        (
            band,
            {"name": f"{name} diamond band", "scale": scale[1:], "metadata": meta},
            "image",
        ),
        (
            cube,
            {
                "name": f"{name} spectra",
                "scale": scale,
                "metadata": meta,
                "visible": False,
            },
            "image",
        ),
    ]


def _scale(ds) -> tuple[float, float, float]:
    dy = float(np.diff(ds.y.values[:2])[0]) if ds.sizes["y"] > 1 else 1.0
    dx = float(np.diff(ds.x.values[:2])[0]) if ds.sizes["x"] > 1 else 1.0
    return (1.0, abs(dy) or 1.0, abs(dx) or 1.0)


class SpectrumInspector(QWidget):
    """Dock widget: pixel spectrum + recipe editor + run the map analysis."""

    def __init__(self, napari_viewer):
        super().__init__()
        from .qtgui import RecipeEditor

        self.viewer = napari_viewer
        self.editor = RecipeEditor()
        self.info = QLabel("Open a .map, then click a pixel.")
        run = QPushButton("Run map analysis (with this recipe)")
        run.clicked.connect(self.run_analysis)
        layout = QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(self.editor, 1)
        layout.addWidget(run)
        self.viewer.mouse_drag_callbacks.append(self._on_click)
        self.result = None

    def _raw_layer(self):
        for layer in reversed(self.viewer.layers):
            if RAW_KEY in getattr(layer, "metadata", {}):
                return layer
        return None

    def _on_click(self, _viewer, event) -> None:
        layer = self._raw_layer()
        if layer is None:
            return
        ds = layer.metadata[RAW_KEY]
        coords = layer.world_to_data(event.position)
        i, j = (round(c) for c in coords[-2:])
        if not (0 <= i < ds.sizes["y"] and 0 <= j < ds.sizes["x"]):
            return
        self.show_pixel(ds, i, j)

    def show_pixel(self, ds, i: int, j: int) -> None:
        t = None
        if self.result is not None and "typeIIA_ratio" in self.result:
            v = float(self.result["typeIIA_ratio"].values[i, j])
            t = v if np.isfinite(v) and v > 0 else None
        self.info.setText(
            f"pixel y={i} x={j}" + (f", thickness {t:.4g} cm" if t else "")
        )
        self.editor.set_spectrum(ds.wn.values, ds.spectra.values[i, j], t)

    def run_analysis(self) -> None:
        from .maps import process_map
        from .params import AnalysisParams, MapParams

        layer = self._raw_layer()
        if layer is None:
            self.info.setText("Open a .map first.")
            return
        ds = layer.metadata[RAW_KEY]
        recipes = (self.editor.recipe,) if self.editor.recipe.features else ()
        params = AnalysisParams(run_platelets=False, recipes=recipes)
        self.info.setText("Running map analysis…")
        self.result = process_map(ds, params, MapParams())
        ok = self.result["status"].values == 0
        for name in self.result.data_vars:
            arr = self.result[name].values
            if arr.ndim != 2 or name == "status":
                continue
            self.viewer.add_image(
                np.where(ok, arr, np.nan).astype(np.float32),
                name=name,
                scale=_scale(ds)[1:],
                colormap="viridis",
                visible=name == "Total_N ppm",
            )
        self.info.setText(f"Done: {int(ok.sum())} pixels analysed.")
