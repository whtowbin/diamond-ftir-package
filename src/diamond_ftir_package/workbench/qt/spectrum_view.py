"""Fast interactive spectrum plot (pyqtgraph) with navigation tools and a draggable region.

Mouse: wheel zooms, left-drag pans, right-drag stretches an axis, double-click resets.
Toolbar: reset view, fit y to the visible x range, zoom to box. The readout shows the
position under the cursor.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QLabel, QToolBar, QToolButton, QVBoxLayout, QWidget

PALETTE = ["#1b7837", "#2166ac", "#b2182b", "#e08214", "#762a83", "#35978f", "#8c510a"]


def compact_toolbar() -> QToolBar:
    """A toolbar that folds buttons that do not fit into a » menu, so a narrow panel
    never forces the window wider than the screen."""
    bar = QToolBar()
    bar.setMovable(False)
    bar.setFloatable(False)
    return bar


def tool_button(text: str, tip: str, slot, checkable: bool = False) -> QToolButton:
    b = QToolButton()
    b.setText(text)
    b.setToolTip(tip)
    b.setCheckable(checkable)
    (b.toggled if checkable else b.clicked).connect(slot)
    return b


class SpectrumView(QWidget):
    """Plot a spectrum plus overlays. ``region_changed(lo, hi)`` fires when the user drags
    the shaded region; ``line_changed(x)`` when the marker line is dragged."""

    region_changed = Signal(float, float)
    line_changed = Signal(float)
    hovered = Signal(str)  # cursor position text, e.g. for a status bar

    def __init__(
        self,
        parent=None,
        x_label="Wavenumber (cm⁻¹)",
        y_label="Absorbance",
        toolbar=True,
    ):
        super().__init__(parent)
        pg.setConfigOptions(antialias=True, background="w", foreground="k")
        self.plot = pg.PlotWidget()
        self.plot.getAxis("left").enableAutoSIPrefix(False)  # no "×0.001" on absorbance
        self.legend = self.plot.addLegend(
            offset=(-10, 10), brush=(255, 255, 255, 200), labelTextSize="8pt"
        )
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.region = pg.LinearRegionItem(brush=(100, 100, 200, 40))
        self.region.setZValue(-10)
        self.region.hide()
        self.region.sigRegionChangeFinished.connect(self._region_moved)
        self.line = pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen("#762a83", width=1.5)
        )
        self.line.hide()
        self.line.sigPositionChangeFinished.connect(
            lambda: self.line_changed.emit(float(self.line.value()))
        )
        self.readout = QLabel("")
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self._mouse_clicked)
        self.set_axes(x_label, y_label, x_label.lower().startswith("wavenumber"))
        self.clear()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        if toolbar:
            bar = compact_toolbar()
            bar.addWidget(
                tool_button(
                    "Reset",
                    "Show everything (or double-click the plot)",
                    self.reset_view,
                )
            )
            bar.addWidget(
                tool_button(
                    "Fit Y", "Scale y to the data in the visible x range", self.fit_y
                )
            )
            self.box_zoom = tool_button(
                "Box zoom",
                "Drag a rectangle to zoom into (right-click the plot to zoom out)",
                self.set_box_mode,
                True,
            )
            bar.addWidget(self.box_zoom)
            layout.addWidget(bar)
        layout.addWidget(self.plot, 1)
        if toolbar:
            layout.addWidget(self.readout)

    # ------------------------------------------------------------------ content
    def set_axes(self, x_label: str, y_label: str, invert_x: bool) -> None:
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", y_label)
        self.plot.invertX(invert_x)

    def clear(self) -> None:
        self.plot.clear()
        self.plot.addItem(self.region)
        self.plot.addItem(self.line)

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
            pen.setStyle(Qt.PenStyle.DashLine)
        return self.plot.plot(np.asarray(x), np.asarray(y), pen=pen, name=name)

    def show_region(self, lo: float, hi: float) -> None:
        self.region.blockSignals(True)
        self.region.setRegion((lo, hi))
        self.region.blockSignals(False)
        self.region.show()

    def hide_region(self) -> None:
        self.region.hide()

    def show_line(self, x: float) -> None:
        self.line.blockSignals(True)
        self.line.setValue(x)
        self.line.blockSignals(False)
        self.line.show()

    def hide_line(self) -> None:
        self.line.hide()

    # ------------------------------------------------------------------ navigation
    def reset_view(self) -> None:
        self.plot.enableAutoRange()
        self.plot.autoRange()

    def fit_y(self) -> None:
        """Scale y to the curves inside the visible x range (ignores the region)."""
        (x0, x1), _ = self.plot.viewRange()
        lo, hi = np.inf, -np.inf
        for item in self.plot.listDataItems():
            x, y = item.getData()
            if x is None or y is None:
                continue
            m = (x >= min(x0, x1)) & (x <= max(x0, x1)) & np.isfinite(y)
            if m.any():
                lo, hi = min(lo, float(y[m].min())), max(hi, float(y[m].max()))
        if np.isfinite(lo) and hi > lo:
            pad = 0.05 * (hi - lo)
            self.plot.getViewBox().setYRange(lo - pad, hi + pad, padding=0)

    def set_box_mode(self, on: bool) -> None:
        vb = self.plot.getViewBox()
        vb.setMouseMode(vb.RectMode if on else vb.PanMode)

    def set_legend_visible(self, on: bool) -> None:
        self.legend.setVisible(on)

    def _mouse_moved(self, pos) -> None:
        vb = self.plot.getViewBox()
        if self.plot.sceneBoundingRect().contains(pos):
            p = vb.mapSceneToView(pos)
            text = f"x = {p.x():.2f}   y = {p.y():.4g}"
            self.readout.setText(text)
            self.hovered.emit(text)

    def _mouse_clicked(self, event) -> None:
        if event.double():
            self.reset_view()

    def _region_moved(self) -> None:
        lo, hi = self.region.getRegion()
        self.region_changed.emit(float(lo), float(hi))
