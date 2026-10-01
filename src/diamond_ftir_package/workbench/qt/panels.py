"""Side panels: profile (what the raw map shows), threshold mask, data browser, results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import QAbstractTableModel, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QTableView,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.profiles import PROFILE_KINDS, Profile


def spin(
    value: float, lo: float = -1e6, hi: float = 1e6, decimals: int = 1
) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setValue(value)
    s.setKeyboardTracking(False)
    return s


# ---------------------------------------------------------------------------- profile
class ProfilePanel(QWidget):
    """Choose the raw-data map: band area, height, or a ratio of two bands.

    The band (numerator for ratios) can also be dragged on the spectrum plot."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.kind = QComboBox()
        self.kind.addItems(PROFILE_KINDS)
        self.lo, self.hi = spin(1970), spin(2040)
        self.baseline = QCheckBox("straight baseline between band ends")
        self.baseline.setChecked(True)
        self.den_lo, self.den_hi = spin(2400), spin(2600)
        form = self.form = QFormLayout(self)
        form.addRow("Show", self.kind)
        form.addRow("Band from", self.lo)
        form.addRow("to", self.hi)
        form.addRow(self.baseline)
        self.den_label = QLabel("Divide by band")
        form.addRow(self.den_label, self.den_lo)
        form.addRow("to", self.den_hi)
        for w in (self.lo, self.hi, self.den_lo, self.den_hi):
            w.valueChanged.connect(self.changed)
        self.kind.currentTextChanged.connect(self._kind_changed)
        self.baseline.toggled.connect(self.changed)
        self._kind_changed(self.kind.currentText())

    def _kind_changed(self, kind: str) -> None:
        for w in (self.den_lo, self.den_hi):
            self.form.setRowVisible(w, kind == "ratio")
        self.changed.emit()

    def set_band(self, lo: float, hi: float) -> None:
        for w, v in ((self.lo, lo), (self.hi, hi)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self.changed.emit()

    def _band(self, lo: float, hi: float, kind: str) -> Profile:
        return Profile(kind, lo, hi, self.baseline.isChecked(), lo, hi)

    def profile(self) -> Profile:
        kind = self.kind.currentText()
        lo, hi = self.lo.value(), self.hi.value()
        if kind == "height":
            # the highest point in the band, above the line through the band ends
            return Profile("height", lo, hi, self.baseline.isChecked(), lo, hi)
        band = self._band(lo, hi, "area")
        if kind == "ratio":
            return Profile(
                "ratio",
                numerator=band,
                denominator=self._band(
                    self.den_lo.value(), self.den_hi.value(), "area"
                ),
            )
        return band


# ---------------------------------------------------------------------------- threshold
class ThresholdPanel(QWidget):
    """Histogram of any layer; pixels outside [low, high] can be hidden on every layer.

    The histogram and the range always start at the layer's own minimum and maximum."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values: np.ndarray | None = None
        self.layer = QComboBox()
        self.layer.currentTextChanged.connect(self._layer_chosen)
        self.enabled = QCheckBox("Hide pixels outside the range (all layers)")
        self.enabled.toggled.connect(self.changed)
        self.lo, self.hi = spin(0, decimals=4), spin(1, decimals=4)
        self.lo.valueChanged.connect(self._spins_moved)
        self.hi.valueChanged.connect(self._spins_moved)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(70)
        self.plot.getAxis("left").setLabel("pixels")
        self.plot.setMouseEnabled(x=True, y=False)
        self.region = pg.LinearRegionItem(brush=(200, 80, 80, 40))
        self.region.sigRegionChangeFinished.connect(self._region_moved)
        self.plot.addItem(self.region)
        self.count = QLabel("")
        self.layers: dict[str, np.ndarray] = {}
        form = QFormLayout()
        form.addRow("Layer", self.layer)
        form.addRow("Keep from", self.lo)
        form.addRow("to", self.hi)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.enabled)
        layout.addWidget(self.count)

    def set_layers(self, layers: dict[str, np.ndarray]) -> None:
        current = self.layer.currentText()
        self.layers = layers
        self.layer.blockSignals(True)
        self.layer.clear()
        self.layer.addItems(list(layers))
        self.layer.blockSignals(False)
        if layers:
            self.layer.setCurrentText(
                current if current in layers else next(iter(layers))
            )
            self._layer_chosen(self.layer.currentText())

    def _layer_chosen(self, name: str) -> None:
        arr = self.layers.get(name)
        self.values = None if arr is None else np.asarray(arr, dtype=float)
        v = self._finite()
        self.plot.clear()
        self.plot.addItem(self.region)
        if v.size:
            lo, hi = float(v.min()), float(v.max())
            counts, edges = np.histogram(
                v,
                bins=min(100, max(10, v.size // 20)),
                range=(lo, hi if hi > lo else lo + 1),
            )
            self.plot.plot(
                edges,
                counts,
                stepMode="center",
                fillLevel=0,
                brush=(60, 60, 140, 120),
                pen="k",
            )
            self._set_range(lo, hi)
        self.changed.emit()

    def _finite(self) -> np.ndarray:
        return (
            self.values[np.isfinite(self.values)]
            if self.values is not None
            else np.array([])
        )

    def _set_range(self, lo: float, hi: float) -> None:
        for w, v in ((self.lo, lo), (self.hi, hi)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self.region.blockSignals(True)
        self.region.setRegion((lo, hi))
        self.region.blockSignals(False)
        self._update_count()

    def _region_moved(self) -> None:
        self._set_range(*self.region.getRegion())
        self.changed.emit()

    def _spins_moved(self) -> None:
        self._set_range(self.lo.value(), self.hi.value())
        self.changed.emit()

    def _update_count(self) -> None:
        m = self.mask()
        v = self._finite()
        kept = (
            int(np.sum(m & np.isfinite(self.values)))
            if m is not None and self.values is not None
            else v.size
        )
        self.count.setText(f"{kept} of {v.size} pixels in range")

    def mask(self) -> np.ndarray | None:
        if self.values is None:
            return None
        with np.errstate(invalid="ignore"):
            return (self.values >= self.lo.value()) & (self.values <= self.hi.value())

    def active_mask(self) -> np.ndarray | None:
        return self.mask() if self.enabled.isChecked() else None


# ---------------------------------------------------------------------------- data browser
class DataBrowser(QTreeWidget):
    """Loaded datasets; batches expand to their files. ``chosen(dataset, file_index)``."""

    chosen = Signal(object, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Data", "Status"])
        self.setColumnWidth(0, 200)
        self.itemSelectionChanged.connect(self._selected)

    def add(self, dataset) -> QTreeWidgetItem:
        top = QTreeWidgetItem([dataset.name, dataset.kind])
        top.setData(0, Qt.ItemDataRole.UserRole, (dataset, 0))
        top.setToolTip(0, "\n".join(str(p) for p in dataset.paths[:20]))
        if dataset.kind == "batch":
            for k, p in enumerate(dataset.paths):
                child = QTreeWidgetItem([Path(p).name, ""])
                child.setData(0, Qt.ItemDataRole.UserRole, (dataset, k))
                top.addChild(child)
        self.addTopLevelItem(top)
        return top

    def select(self, dataset, index: int = 0) -> None:
        for k in range(self.topLevelItemCount()):
            top = self.topLevelItem(k)
            if top.data(0, Qt.ItemDataRole.UserRole)[0] is dataset:
                item = (
                    top.child(index)
                    if dataset.kind == "batch" and top.childCount() > index
                    else top
                )
                self.setCurrentItem(item)
                return

    def set_status(self, dataset, text: str) -> None:
        for k in range(self.topLevelItemCount()):
            top = self.topLevelItem(k)
            if top.data(0, Qt.ItemDataRole.UserRole)[0] is dataset:
                top.setText(1, text)

    def current(self):
        item = self.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item else (None, 0)

    def _selected(self) -> None:
        dataset, index = self.current()
        if dataset is not None:
            self.chosen.emit(dataset, index)


# ---------------------------------------------------------------------------- results
class FrameModel(QAbstractTableModel):
    def __init__(self, frame: pd.DataFrame | None = None):
        super().__init__()
        self.frame = frame if frame is not None else pd.DataFrame()

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self.frame = frame
        self.endResetModel()

    def rowCount(self, parent=None):
        return len(self.frame)

    def columnCount(self, parent=None):
        return len(self.frame.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        v = self.frame.iat[index.row(), index.column()]
        return f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self.frame.columns[section])
        return str(section + 1)


def ordered(frame: pd.DataFrame, first: tuple[str, ...]) -> pd.DataFrame:
    lead = [c for c in ("Filename", "Status", *first) if c in frame.columns]
    return frame[lead + [c for c in frame.columns if c not in lead]]


def map_summary(result) -> pd.DataFrame:
    """Per-layer statistics of a map result (over analysed pixels)."""
    ok = result["status"].values == 0 if "status" in result else None
    rows = []
    for name in result.data_vars:
        if name == "status" or result[name].ndim != 2:
            continue
        v = result[name].values.astype(float)
        v = v[ok & np.isfinite(v)] if ok is not None else v[np.isfinite(v)]
        if v.size:
            p5, med, p95 = np.percentile(v, (5, 50, 95))
            rows.append(
                {
                    "Layer": name,
                    "pixels": v.size,
                    "median": med,
                    "p5": p5,
                    "p95": p95,
                    "min": v.min(),
                    "max": v.max(),
                }
            )
    return pd.DataFrame(rows)


class ResultsPanel(QTabWidget):
    """ "Dataset" tab: one row per file (batches) or per-layer statistics (maps).
    "Selected" tab: every value for the spectrum or pixel being looked at.
    ``row_chosen(k)`` fires when a batch row is clicked."""

    row_chosen = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table_model, self.selected_model = FrameModel(), FrameModel()
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.clicked.connect(lambda idx: self.row_chosen.emit(idx.row()))
        self.selected_table = QTableView()
        self.selected_table.setModel(self.selected_model)
        self.addTab(self.table, "Dataset results")
        self.addTab(self.selected_table, "Selected spectrum")

    def show_frame(self, frame: pd.DataFrame | None) -> None:
        self.table_model.set_frame(frame if frame is not None else pd.DataFrame())
        self.table.resizeColumnsToContents()

    def show_values(self, values: dict) -> None:
        frame = pd.DataFrame({"Value": list(values.values())}, index=list(values))
        frame.insert(0, "Name", frame.index)
        self.selected_model.set_frame(frame.reset_index(drop=True))
        self.selected_table.resizeColumnsToContents()
