"""Fast interactive spectrum plot (pyqtgraph) with an optional draggable region."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget

PALETTE = ["#1b7837", "#2166ac", "#b2182b", "#e08214", "#762a83", "#35978f", "#8c510a"]


class SpectrumView(QWidget):
    """Plot a spectrum plus overlays. ``region_changed(lo, hi)`` fires when the user drags
    the shaded region."""

    region_changed = Signal(float, float)

    def __init__(self, parent=None, x_label="Wavenumber (cm⁻¹)", y_label="Absorbance"):
        super().__init__(parent)
        pg.setConfigOptions(antialias=True, background="w", foreground="k")
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", y_label)
        self.plot.getAxis("left").enableAutoSIPrefix(False)  # no "×0.001" on absorbance
        self.plot.addLegend(offset=(-10, 10))
        self.plot.invertX(x_label.lower().startswith("wavenumber"))
        self.region = pg.LinearRegionItem(brush=(100, 100, 200, 40))
        self.region.setZValue(-10)
        self.region.hide()
        self.region.sigRegionChangeFinished.connect(self._region_moved)
        self.plot.addItem(self.region)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

    def clear(self) -> None:
        self.plot.clear()
        self.plot.addItem(self.region)

    def add_curve(
        self,
        x,
        y,
        name: str,
        color: str | None = None,
        width: float = 1.0,
        dash: bool = False,
    ):
        pen = pg.mkPen(
            color or PALETTE[len(self.plot.listDataItems()) % len(PALETTE)], width=width
        )
        if dash:
            pen.setStyle(pg.QtCore.Qt.PenStyle.DashLine)
        return self.plot.plot(np.asarray(x), np.asarray(y), pen=pen, name=name)

    def show_region(self, lo: float, hi: float) -> None:
        self.region.blockSignals(True)
        self.region.setRegion((lo, hi))
        self.region.blockSignals(False)
        self.region.show()

    def hide_region(self) -> None:
        self.region.hide()

    def _region_moved(self) -> None:
        lo, hi = self.region.getRegion()
        self.region_changed.emit(float(lo), float(hi))
