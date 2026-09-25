"""Diamond FTIR desktop app (Qt). Start with ``diamond-ftir-app``.

Built from the reusable widgets in this package (DataclassForm, SpectrumView, RecipeEditor,
MapView); the analysis itself is ``pipeline`` / ``maps``, the same code the CLI uses.
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import Recipe, builtin_recipes
from ..params import CHOICES, AnalysisParams, MapParams
from ..pipeline import SUPPORTED_SUFFIXES, analyze_file, collect_files
from .forms import DataclassForm
from .map_view import MapView
from .recipe_editor import RecipeEditor
from .spectrum_view import SpectrumView

SECTIONS = {
    "diamond": "Thickness & baseline",
    "nitrogen": "Nitrogen",
    "hydrogen": "Hydrogen",
    "platelet": "Platelets",
    "thermometry": "Thermometry",
}
TABLE_COLUMNS = (
    "Filename",
    "Status",
    "QA",
    "Total_N ppm",
    "B_percent",
    "Normed_3107_Area",
)


class _Signals(QObject):
    item = Signal(object)
    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)


class Job(QRunnable):
    """Run ``fn(report)`` off the GUI thread; results come back through signals only."""

    def __init__(self, fn):
        super().__init__()
        self.fn, self.signals = fn, _Signals()

    def run(self):
        try:
            self.signals.done.emit(self.fn(self.signals))
        except Exception:  # noqa: BLE001 - shown in the GUI
            self.signals.failed.emit(traceback.format_exc(limit=3))


class SettingsPanel(QWidget):
    """Analysis settings generated from AnalysisParams, plus which recipes to run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.params = AnalysisParams()
        self.extra: list[Recipe] = []  # recipes coming from the editor
        tabs = QTabWidget()
        self.forms = {}
        for section, title in SECTIONS.items():
            form = DataclassForm(getattr(self.params, section), CHOICES)
            self.forms[section] = form
            tabs.addTab(form, title)
        self.toggles = {}
        box = QVBoxLayout()
        for key, label in (
            ("run_nitrogen", "Nitrogen (A, B, C)"),
            ("run_hydrogen", "Hydrogen 3107 / 3085"),
            ("run_platelets", "Platelets and 1405"),
            ("run_amber", "Amber centres"),
        ):
            cb = QCheckBox(label)
            cb.setChecked(getattr(self.params, key))
            cb.toggled.connect(lambda v, k=key: setattr(self.params, k, v))
            self.toggles[key] = cb
            box.addWidget(cb)
        box.addWidget(QLabel("Extra recipes (tick to run):"))
        self.recipe_list = QListWidget()
        for name in sorted(builtin_recipes()):
            self._add_recipe_item(name, name)
        box.addWidget(self.recipe_list)
        row = QHBoxLayout()
        for text, slot in (
            ("Add file…", self._add_file),
            ("Save settings…", self._save),
            ("Load settings…", self._load),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        box.addLayout(row)
        layout = QVBoxLayout(self)
        layout.addLayout(box)
        layout.addWidget(tabs, 1)

    def _add_recipe_item(self, label: str, ref, checked=False) -> None:
        item = QListWidgetItem(label)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )
        item.setData(Qt.ItemDataRole.UserRole, ref)
        self.recipe_list.addItem(item)

    def use_recipe(self, recipe: Recipe) -> None:
        """Add (or replace) an in-memory recipe from the editor, ticked."""
        for k in range(self.recipe_list.count()):
            if self.recipe_list.item(k).text() == f"{recipe.name} (editor)":
                self.recipe_list.takeItem(k)
                break
        self._add_recipe_item(f"{recipe.name} (editor)", recipe, checked=True)

    def _add_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Add recipe", "", "Recipes (*.toml *.json)"
        )
        if path:
            self._add_recipe_item(Path(path).name, path, checked=True)

    def current(self) -> AnalysisParams:
        refs = []
        for k in range(self.recipe_list.count()):
            item = self.recipe_list.item(k)
            if item.checkState() == Qt.CheckState.Checked:
                refs.append(item.data(Qt.ItemDataRole.UserRole))
        return replace(self.params, recipes=tuple(refs))

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save settings", "settings.json", "JSON (*.json)"
        )
        if path:
            p = self.current()
            data = p.to_dict() | {
                "recipes": [r if isinstance(r, str) else r.name for r in p.recipes]
            }
            Path(path).write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load settings", "", "JSON (*.json)"
        )
        if not path:
            return
        data = json.loads(Path(path).read_text())
        self.params = AnalysisParams.from_dict(
            {k: v for k, v in data.items() if k != "recipes"}
        )
        for section, form in self.forms.items():
            form.obj = getattr(self.params, section)
            form.refresh()
        for key, cb in self.toggles.items():
            cb.setChecked(getattr(self.params, key))


class SpectraTab(QWidget):
    """Single files and batches."""

    def __init__(
        self,
        settings: SettingsPanel,
        editor: RecipeEditor,
        pool: QThreadPool,
        parent=None,
    ):
        super().__init__(parent)
        self.settings, self.editor, self.pool = settings, editor, pool
        self.rows, self.spectra = [], {}
        top = QHBoxLayout()
        for text, slot in (
            ("Open files…", self._open_files),
            ("Open folder…", self._open_folder),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            top.addWidget(b)
        self.path_label = QLabel("No input selected")
        top.addWidget(self.path_label, 1)
        self.run_button = QPushButton("Run analysis")
        self.run_button.clicked.connect(self.run)
        top.addWidget(self.run_button)
        export = QPushButton("Export CSV…")
        export.clicked.connect(self._export)
        top.addWidget(export)
        self.progress = QProgressBar()
        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(TABLE_COLUMNS)
        self.table.itemSelectionChanged.connect(self._selected)
        self.view = SpectrumView()
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.table)
        split.addWidget(self.view)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.progress)
        layout.addWidget(split, 1)
        self.files: list[Path] = []

    def _open_files(self) -> None:
        patterns = " ".join(f"*{s}" for s in SUPPORTED_SUFFIXES)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open spectra", "", f"Spectra ({patterns})"
        )
        if paths:
            self.files = [Path(p) for p in paths]
            self.path_label.setText(f"{len(paths)} file(s)")

    def _open_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open folder")
        if path:
            self.files = collect_files(path)
            self.path_label.setText(f"{path} ({len(self.files)} spectra)")

    def run(self) -> None:
        if not self.files:
            QMessageBox.information(
                self, "Nothing to analyse", "Open files or a folder first."
            )
            return
        params = self.settings.current()
        files = list(self.files)
        self.rows, self.spectra = [], {}
        self.table.setRowCount(0)
        self.progress.setMaximum(len(files))
        self.run_button.setEnabled(False)

        def work(signals):
            for k, f in enumerate(files, start=1):
                signals.item.emit(analyze_file(f, params))
                signals.progress.emit(k, len(files))

        job = Job(work)
        job.signals.item.connect(self._add_row)
        job.signals.progress.connect(lambda k, _n: self.progress.setValue(k))
        job.signals.done.connect(lambda _: self.run_button.setEnabled(True))
        job.signals.failed.connect(self._failed)
        self.pool.start(job)

    def _failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        QMessageBox.critical(self, "Analysis failed", message)

    def _add_row(self, result) -> None:
        spectrum, row = result
        self.rows.append(row)
        if spectrum is not None:
            self.spectra[row["Filename"]] = spectrum
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, col in enumerate(TABLE_COLUMNS):
            v = row.get(col, "")
            self.table.setItem(
                r, c, QTableWidgetItem(f"{v:.4g}" if isinstance(v, float) else str(v))
            )

    def _selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        row = self.rows[rows[0].row()]
        s = self.spectra.get(row["Filename"])
        self.view.clear()
        if s is None:
            return
        self.view.add_curve(s.X, s.Y, "spectrum", color="#333333")
        if getattr(s, "baseline", None) is not None:
            self.view.add_curve(s.X, s.baseline, "baseline", color="#1b7837", width=1.5)
        self.editor.set_spectrum(s.X, s.Y, getattr(s, "typeIIA_ratio", None))

    def _export(self) -> None:
        if not self.rows:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results", "results.csv", "CSV (*.csv)"
        )
        if path:
            pd.DataFrame(self.rows).to_csv(path, index=False)


class MapTab(QWidget):
    def __init__(
        self,
        settings: SettingsPanel,
        editor: RecipeEditor,
        pool: QThreadPool,
        parent=None,
    ):
        super().__init__(parent)
        self.settings, self.editor, self.pool = settings, editor, pool
        self.map_params = MapParams()
        self.raw = self.result = None
        top = QHBoxLayout()
        b = QPushButton("Open map…")
        b.clicked.connect(self._open)
        top.addWidget(b)
        self.label = QLabel("No map loaded")
        top.addWidget(self.label, 1)
        self.run_button = QPushButton("Run map analysis")
        self.run_button.clicked.connect(self.run)
        top.addWidget(self.run_button)
        self.progress = QProgressBar()
        self.form = DataclassForm(
            self.map_params, CHOICES, skip=("example_grid", "block_size", "seed")
        )
        self.view = MapView()
        self.view.pixel_selected.connect(self._pixel)
        split = QSplitter()
        split.addWidget(self.form)
        split.addWidget(self.view)
        split.setStretchFactor(1, 4)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.progress)
        layout.addWidget(split, 1)

    def _open(self) -> None:
        from ..maps import load_map

        path, _ = QFileDialog.getOpenFileName(
            self, "Open map", "", "OMNIC maps (*.map *.MAP)"
        )
        if path:
            self.raw = load_map(path)
            self.label.setText(
                f"{Path(path).name}: {self.raw.sizes['y']} × {self.raw.sizes['x']} pixels"
            )

    def run(self) -> None:
        from ..maps import process_map

        if self.raw is None:
            QMessageBox.information(self, "No map", "Open a map first.")
            return
        params, map_params, raw = self.settings.current(), self.map_params, self.raw
        self.run_button.setEnabled(False)

        def work(signals):
            return process_map(
                raw,
                params,
                map_params,
                progress=lambda d, n: signals.progress.emit(d, n),
            )

        job = Job(work)
        job.signals.progress.connect(self._progress)
        job.signals.done.connect(self._done)
        job.signals.failed.connect(
            lambda m: (
                self.run_button.setEnabled(True),
                QMessageBox.critical(self, "Map failed", m),
            )
        )
        self.pool.start(job)

    def _progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(done)

    def _done(self, result) -> None:
        self.run_button.setEnabled(True)
        self.result = result
        self.view.set_data(result, self.raw)

    def _pixel(self, i: int, j: int) -> None:
        if self.raw is None:
            return
        t = (
            float(self.result["typeIIA_ratio"].values[i, j])
            if self.result is not None
            else None
        )
        self.editor.set_spectrum(
            self.raw.wn.values,
            self.raw.spectra.values[i, j],
            t if t and np.isfinite(t) else None,
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Diamond FTIR")
        self.resize(1400, 900)
        self.pool = QThreadPool()
        self.settings = SettingsPanel()
        self.editor = RecipeEditor()
        use = QPushButton("Use this recipe in the analysis")
        use.clicked.connect(lambda: self.settings.use_recipe(self.editor.recipe))
        editor_tab = QWidget()
        el = QVBoxLayout(editor_tab)
        el.addWidget(self.editor, 1)
        el.addWidget(use)
        tabs = QTabWidget()
        tabs.addTab(SpectraTab(self.settings, self.editor, self.pool), "Spectra")
        tabs.addTab(editor_tab, "Recipe editor")
        tabs.addTab(MapTab(self.settings, self.editor, self.pool), "Map")
        split = QSplitter()
        split.addWidget(self.settings)
        split.addWidget(tabs)
        split.setStretchFactor(1, 4)
        self.setCentralWidget(split)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
