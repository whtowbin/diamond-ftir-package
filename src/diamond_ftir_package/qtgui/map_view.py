"""Map result viewer: image of any result variable; click a pixel to see its spectrum.

Works on the xarray Dataset returned by ``maps.process_map`` plus the raw map Dataset. For
very large maps, use the napari plugin instead (same widgets, napari's viewer).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .spectrum_view import SpectrumView


class MapView(QWidget):
    """``pixel_selected(i, j)`` fires on click (row, column)."""

    pixel_selected = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.result = None
        self.raw = None
        self.variable = QComboBox()
        self.variable.currentTextChanged.connect(self._show_variable)
        self.info = QLabel("")
        self.image = pg.ImageView()
        self.image.ui.roiBtn.hide()
        self.image.ui.menuBtn.hide()
        self.image.getView().scene().sigMouseClicked.connect(self._clicked)
        self.spectrum = SpectrumView()
        top = QHBoxLayout()
        top.addWidget(QLabel("Show:"))
        top.addWidget(self.variable, 1)
        top.addWidget(self.info, 2)
        split = QSplitter()
        split.addWidget(self.image)
        split.addWidget(self.spectrum)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(split)

    def set_data(self, result, raw=None) -> None:
        self.result, self.raw = result, raw
        names = [v for v in result.data_vars if result[v].ndim == 2]
        self.variable.blockSignals(True)
        self.variable.clear()
        self.variable.addItems(names)
        self.variable.blockSignals(False)
        preferred = "Total_N ppm" if "Total_N ppm" in names else names[0]
        self.variable.setCurrentText(preferred)
        self._show_variable(preferred)

    def _show_variable(self, name: str) -> None:
        if self.result is None or not name:
            return
        arr = self.result[name].values.astype(float)
        if "status" in self.result and name != "status":
            arr = np.where(self.result["status"].values == 0, arr, np.nan)
        finite = arr[np.isfinite(arr)]
        levels = (
            (np.percentile(finite, 2), np.percentile(finite, 98))
            if finite.size
            else None
        )
        # pyqtgraph images are (x, y); maps are stored (y, x)
        self.image.setImage(arr.T, autoRange=True, levels=levels)
        self.image.getView().invertY(False)

    def _clicked(self, event) -> None:
        if self.result is None:
            return
        pos = self.image.getImageItem().mapFromScene(event.scenePos())
        j, i = int(pos.x()), int(pos.y())
        ny, nx = self.result.sizes["y"], self.result.sizes["x"]
        if 0 <= i < ny and 0 <= j < nx:
            self.show_pixel(i, j)
            self.pixel_selected.emit(i, j)

    def show_pixel(self, i: int, j: int) -> None:
        name = self.variable.currentText()
        value = float(self.result[name].values[i, j]) if name else np.nan
        self.info.setText(f"pixel y={i} x={j}: {name} = {value:.4g}")
        self.spectrum.clear()
        if self.raw is not None:
            self.spectrum.add_curve(
                self.raw.wn.values,
                self.raw.spectra.values[i, j],
                "spectrum",
                color="#333333",
            )
