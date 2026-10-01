"""Map viewer: any (y, x) layer as an image in stage coordinates, with a real histogram.

- Colour levels from the histogram handles; buttons set them to the data min/max or to the
  2nd-98th percentile. The colour map is chosen from a list.
- Navigation: wheel zooms, drag pans, "Fit" shows the whole map, "Box zoom" zooms to a
  dragged rectangle, "1:1" keeps µm square.
- Click a pixel (or move with the arrow keys once the map has focus) to select it;
  ``pixel_selected(i, j)`` fires with the row and column. A cross marks the pixel.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from .spectrum_view import compact_toolbar, tool_button

HINT = (
    "Click a pixel to see its spectrum · arrow keys: next pixel · wheel: zoom · "
    "drag: pan · double-click: fit"
)
COLORMAPS = ("viridis", "magma", "inferno", "plasma", "cividis", "turbo", "CET-L1")
ARROWS = {
    Qt.Key.Key_Up: (1, 0),
    Qt.Key.Key_Down: (-1, 0),
    Qt.Key.Key_Left: (0, -1),
    Qt.Key.Key_Right: (0, 1),
}


class MapView(QWidget):
    pixel_selected = Signal(int, int)
    layer_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers: dict[str, np.ndarray] = {}
        self.mask: np.ndarray | None = None
        self.xs = self.ys = np.zeros(1)
        self.selected: tuple[int, int] | None = None
        self.layer = QComboBox()
        self.layer.setMinimumContentsLength(14)
        self.layer.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.layer.currentTextChanged.connect(lambda n: self._show_layer(n, fit=False))
        self.layer.setToolTip(
            "Which map to show: the raw-data profile, or any result of the last run"
        )
        self.cmap = QComboBox()
        self.cmap.addItems(COLORMAPS)
        self.cmap.currentTextChanged.connect(self._set_cmap)
        self.readout = QLabel("")

        self.canvas = pg.GraphicsLayoutWidget()
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.view = self.canvas.addViewBox(lockAspect=True, invertY=False)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.view.addItem(self.image)
        self.marker = pg.ScatterPlotItem(
            symbol="+", size=16, pen=pg.mkPen("r", width=2), brush=None
        )
        self.view.addItem(self.marker)
        self.hist = pg.HistogramLUTItem(self.image)
        self.canvas.addItem(self.hist)
        self.canvas.scene().sigMouseClicked.connect(self._clicked)
        self.canvas.scene().sigMouseMoved.connect(self._hover)
        self.canvas.keyPressEvent = self._key  # arrow keys move the selected pixel

        top = compact_toolbar()
        top.addWidget(QLabel("Layer: "))
        top.addWidget(self.layer)
        top.addWidget(QLabel("  Colours: "))
        top.addWidget(self.cmap)
        bar = compact_toolbar()
        bar.addWidget(
            tool_button(
                "Fit",
                "Show the whole map (Ctrl+1, or double-click the map)",
                self.fit_view,
            )
        )
        bar.addWidget(
            tool_button(
                "+", "Zoom in (mouse wheel)", lambda: self.view.scaleBy((0.7, 0.7))
            )
        )
        bar.addWidget(
            tool_button(
                "−",
                "Zoom out (mouse wheel)",
                lambda: self.view.scaleBy((1 / 0.7, 1 / 0.7)),
            )
        )
        self.box_zoom = tool_button(
            "Box zoom",
            "Drag a rectangle on the map to zoom into it",
            self._box_mode,
            True,
        )
        bar.addWidget(self.box_zoom)
        lock = tool_button(
            "1:1", "Keep x and y at the same scale", self.view.setAspectLocked, True
        )
        lock.setChecked(True)
        bar.addWidget(lock)
        bar.addWidget(
            tool_button(
                "Min–max",
                "Colour scale from the data minimum to maximum (drag the histogram handles to set it by hand)",
                self.levels_min_max,
            )
        )
        bar.addWidget(
            tool_button(
                "2–98 %",
                "Colour scale ignoring the 2 % extremes",
                self.levels_percentile,
            )
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(top)
        layout.addWidget(bar)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.readout)
        self.readout.setText(HINT)
        self.readout.setWordWrap(True)
        self._set_cmap(COLORMAPS[0])

    # ------------------------------------------------------------------ data
    def set_coords(self, xs: np.ndarray, ys: np.ndarray) -> None:
        """Stage positions (µm) of the columns and rows."""
        self.xs = np.asarray(xs, dtype=float) if len(xs) else np.zeros(1)
        self.ys = np.asarray(ys, dtype=float) if len(ys) else np.zeros(1)

    def set_layers(
        self, layers: dict[str, np.ndarray], show: str | None = None
    ) -> None:
        """Replace the layer list (name -> (ny, nx) array) and show ``show``."""
        current = show or self.layer.currentText()
        self.layers = {k: np.asarray(v, dtype=float) for k, v in layers.items()}
        self.layer.blockSignals(True)
        self.layer.clear()
        self.layer.addItems(list(self.layers))
        self.layer.blockSignals(False)
        if self.layers:
            self.layer.setCurrentText(
                current if current in self.layers else next(iter(self.layers))
            )
            self._show_layer(self.layer.currentText(), fit=True)

    def update_layer(self, name: str, values: np.ndarray) -> None:
        """Change one layer's values (e.g. a profile while markers move), keeping the view."""
        new = name not in self.layers
        self.layers[name] = np.asarray(values, dtype=float)
        if new:
            self.layer.addItem(name)
        if self.layer.currentText() == name:
            self._show_layer(name, fit=False)

    def show_layer(self, name: str) -> None:
        """Display layer ``name`` (keeps the current zoom)."""
        if name in self.layers and self.layer.currentText() != name:
            self.layer.setCurrentText(name)

    def set_mask(self, mask: np.ndarray | None) -> None:
        self.mask = mask
        self._show_layer(self.layer.currentText(), fit=False, keep_levels=True)

    def current(self) -> np.ndarray | None:
        arr = self.layers.get(self.layer.currentText())
        if arr is None:
            return None
        if self.mask is not None and self.mask.shape == arr.shape:
            arr = np.where(self.mask, arr, np.nan)
        return arr

    def _show_layer(
        self, name: str, fit: bool = True, keep_levels: bool = False
    ) -> None:
        """Draw the current layer. ``fit`` also zooms to the whole map; colour levels are
        reset to the data range unless ``keep_levels`` (e.g. when only the mask changed)."""
        arr = self.current()
        if arr is None:
            self.image.clear()
            return
        levels = (
            self.image.levels if keep_levels and self.image.image is not None else None
        )
        self.image.setImage(arr, autoLevels=False)
        self._place()
        if levels is not None:
            self.image.setLevels(levels)
        else:
            self.levels_min_max()
        if fit:
            self.fit_view()
        self.layer_changed.emit(name)

    def _place(self) -> None:
        """Pixel centres at the stage positions."""
        ny, nx = self.image.image.shape[:2]
        dx = float(np.median(np.diff(self.xs))) if len(self.xs) > 1 else 1.0
        dy = float(np.median(np.diff(self.ys))) if len(self.ys) > 1 else 1.0
        if len(self.xs) != nx or len(self.ys) != ny:
            self.xs, self.ys, dx, dy = (
                np.arange(nx, dtype=float),
                np.arange(ny, dtype=float),
                1.0,
                1.0,
            )
        self.image.setRect(
            pg.QtCore.QRectF(self.xs[0] - dx / 2, self.ys[0] - dy / 2, dx * nx, dy * ny)
        )

    # ------------------------------------------------------------------ levels and colours
    def _finite(self) -> np.ndarray:
        arr = self.current()
        return arr[np.isfinite(arr)] if arr is not None else np.array([])

    def levels_min_max(self) -> None:
        v = self._finite()
        if v.size:
            self.image.setLevels(
                (
                    float(v.min()),
                    float(v.max()) if v.max() > v.min() else float(v.min()) + 1,
                )
            )
            self.hist.setHistogramRange(float(v.min()), float(v.max()))

    def levels_percentile(self) -> None:
        v = self._finite()
        if v.size:
            lo, hi = np.percentile(v, (2, 98))
            self.image.setLevels((float(lo), float(hi) if hi > lo else float(lo) + 1))

    def _set_cmap(self, name: str) -> None:
        self.hist.gradient.setColorMap(pg.colormap.get(name))

    # ------------------------------------------------------------------ navigation and picking
    def fit_view(self) -> None:
        self.view.autoRange(padding=0.02)

    def _box_mode(self, on: bool) -> None:
        self.view.setMouseMode(self.view.RectMode if on else self.view.PanMode)

    def _pixel_at(self, scene_pos) -> tuple[int, int] | None:
        if self.image.image is None:
            return None
        p = self.image.mapFromScene(scene_pos)
        j, i = int(np.floor(p.x())), int(np.floor(p.y()))
        ny, nx = self.image.image.shape[:2]
        return (i, j) if 0 <= i < ny and 0 <= j < nx else None

    def _hover(self, pos) -> None:
        hit = self._pixel_at(pos)
        if hit is None:
            self.readout.setText(HINT)
            return
        i, j = hit
        arr = self.current()
        value = arr[i, j] if arr is not None else np.nan
        self.readout.setText(
            f"row {i}, col {j}   x = {self.xs[j]:.1f} µm, y = {self.ys[i]:.1f} µm   "
            f"{self.layer.currentText()} = {value:.4g}"
        )

    def _clicked(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if event.double():
            self.fit_view()
            return
        hit = self._pixel_at(event.scenePos())
        if hit is not None:
            self.select(*hit)

    def _key(self, event) -> None:
        step = ARROWS.get(event.key())
        if step is None or self.selected is None or self.image.image is None:
            pg.GraphicsLayoutWidget.keyPressEvent(self.canvas, event)
            return
        ny, nx = self.image.image.shape[:2]
        i = min(max(self.selected[0] + step[0], 0), ny - 1)
        j = min(max(self.selected[1] + step[1], 0), nx - 1)
        self.select(i, j)

    def select(self, i: int, j: int, emit: bool = True) -> None:
        self.selected = (i, j)
        if len(self.xs) > j and len(self.ys) > i:
            self.marker.setData([self.xs[j]], [self.ys[i]])
        if emit:
            self.pixel_selected.emit(i, j)
