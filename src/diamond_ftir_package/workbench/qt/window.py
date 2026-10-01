"""The workbench main window: dockable panels around one session.

Default layout (OMNIC-like): data browser left; map above the spectrum in the middle;
settings, profile and threshold on the right; results along the bottom. Panels can be
moved, tabbed, floated or closed; the layout is remembered, and View > Reset layout
restores the default.

Clicking a file or a map pixel shows its spectrum at once and, with Live fit on, runs the
enabled analyses on it with the current settings in the background (baselines, fits and
values appear when ready). "Run" analyses the whole selected dataset.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
from qtpy.QtCore import QSettings, Qt, QThreadPool, Signal
from qtpy.QtGui import QAction, QKeySequence
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.profiles import compute_profile
from ..data import axis_labels, is_map
from ..plugins import Cancelled
from ..session import PROJECT_SUFFIX, Dataset, Session
from .jobs import Job
from .map_view import MapView
from .panels import (
    DataBrowser,
    ProfilePanel,
    ResultsPanel,
    ThresholdPanel,
    map_summary,
    ordered,
)
from .recipe_editor import RecipeEditor
from .settings_panel import SettingsPanel
from .spectrum_view import SpectrumView, compact_toolbar, tool_button

LAYOUT_VERSION = (
    2  # bump when the default layout changes: old saved layouts are then ignored
)
PROFILE_LAYER = "Profile (raw data)"


class SpectrumPanel(QWidget):
    """The spectrum as loaded (top) and processed curves such as normalised spectra and
    fits (bottom, shown when an analysis provides them). One toolbar drives both plots;
    their x axes are linked."""

    hovered = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.raw = SpectrumView(toolbar=False)
        self.processed = SpectrumView(y_label="Processed", toolbar=False)
        self.processed.plot.setXLink(self.raw.plot)
        self.raw.setMinimumHeight(120)
        for view in (self.raw, self.processed):
            view.hovered.connect(self.hovered)
        bar = compact_toolbar()
        bar.addWidget(
            tool_button(
                "Reset view",
                "Show all of both plots (Ctrl+2, or double-click a plot)",
                self.reset_view,
            )
        )
        bar.addWidget(
            tool_button("Fit Y", "Scale y to the visible x range (Ctrl+3)", self.fit_y)
        )
        bar.addWidget(
            tool_button(
                "Box zoom",
                "Drag a rectangle to zoom into; right-click a plot for more",
                self._box,
                True,
            )
        )
        legend = tool_button(
            "Legend", "Show or hide the curve names", self._legend, True
        )
        legend.setChecked(True)
        bar.addWidget(legend)
        self.show_processed = tool_button(
            "Processed plot",
            "Show the normalised spectrum and fits below",
            self._processed_toggled,
            True,
        )
        self.show_processed.setChecked(True)
        bar.addWidget(self.show_processed)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #d6604d")
        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.addWidget(self.raw)
        self.split.addWidget(self.processed)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 2)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(bar)
        layout.addWidget(self.split, 1)
        layout.addWidget(self.message)
        self.processed.hide()

    def reset_view(self) -> None:
        self.raw.reset_view()
        self.processed.reset_view()

    def fit_y(self) -> None:
        self.raw.fit_y()
        self.processed.fit_y()

    def _box(self, on: bool) -> None:
        self.raw.set_box_mode(on)
        self.processed.set_box_mode(on)

    def _legend(self, on: bool) -> None:
        self.raw.set_legend_visible(on)
        self.processed.set_legend_visible(on)

    def _processed_toggled(self, on: bool) -> None:
        has_curves = bool(self.processed.plot.listDataItems())
        self.processed.setVisible(on and has_curves)
        if on and has_curves:
            total = sum(self.split.sizes()) or self.split.height()
            self.split.setSizes([int(total * 0.6), int(total * 0.4)])

    def show_spectrum(self, x, y, name: str, labels) -> None:
        x_label, y_label, invert = labels
        self.raw.set_axes(x_label, y_label, invert)
        self.processed.set_axes(x_label, "Processed", invert)
        self.raw.clear()
        self.processed.clear()
        self.processed.hide()
        self.message.setText("")
        self.raw.add_curve(x, y, name, color="#333333")
        self.raw.reset_view()

    def add_results(self, results: dict) -> None:
        errors = []
        for name, res in results.items():
            for c in res.curves:
                view = self.processed if c.panel == "processed" else self.raw
                view.add_curve(
                    c.x, c.y, c.name, color=c.color, width=c.width, dash=c.dash
                )
            if res.error:
                errors.append(f"{name}: {res.error}")
        self._processed_toggled(self.show_processed.isChecked())
        self.processed.reset_view()
        self.message.setText("\n".join(errors))


class AnalysisPanel(QWidget):
    """Run controls on top (always visible), then tabs: analysis settings and map display
    (which layer to show, the raw-data profile, and a threshold mask)."""

    RUN_STYLE = (
        "QPushButton { background: %s; color: white; font-weight: bold; border-radius: 5px;"
        " padding: 4px 12px; } QPushButton:disabled { background: #7f7f7f; }"
    )

    def __init__(
        self,
        settings: QWidget,
        layers: QWidget,
        profile: QWidget,
        threshold: QWidget,
        parent=None,
    ):
        super().__init__(parent)
        self.run_button = QPushButton("▶  Run analysis")
        self.run_button.setToolTip(
            "Analyse the whole selected spectrum, batch or map with these settings (Ctrl+R)"
        )
        self.run_button.setMinimumHeight(34)
        self.run_button.setStyleSheet(self.RUN_STYLE % "#1b7837")
        self.run_all_button = QPushButton("Run all")
        self.run_all_button.setToolTip("Analyse every loaded dataset (Ctrl+Shift+R)")
        self.live = QCheckBox("Live fit on click")
        self.live.setToolTip(
            "Fit each spectrum or pixel you click, using these settings (Ctrl+L)"
        )
        self.live.setChecked(True)
        self.progress = QProgressBar()
        self.progress.hide()
        row = QHBoxLayout()
        row.addWidget(self.run_button, 2)
        row.addWidget(self.run_all_button, 1)
        display = QWidget()
        dl = QVBoxLayout(display)
        for title, widget in (
            ("Show on the map (click one)", layers),
            ("What the raw map shows", profile),
            ("Threshold / mask", threshold),
        ):
            box = QGroupBox(title)
            QVBoxLayout(box).addWidget(widget)
            dl.addWidget(box)
        dl.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(display)
        self.tabs = QTabWidget()
        self.tabs.addTab(settings, "Settings")
        self.tabs.addTab(scroll, "Map display")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(row)
        layout.addWidget(self.live)
        layout.addWidget(self.progress)
        layout.addWidget(self.tabs, 1)

    def set_running(self, running: bool) -> None:
        """While running, the Run button becomes a Stop button."""
        self.run_all_button.setEnabled(not running)
        self.run_button.setText("■  Stop" if running else "▶  Run analysis")
        self.run_button.setToolTip(
            "Stop the analysis (results so far are discarded)"
            if running
            else "Analyse the whole selected spectrum, batch or map with these settings (Ctrl+R)"
        )
        self.run_button.setStyleSheet(
            self.RUN_STYLE % ("#b2182b" if running else "#1b7837")
        )
        self.progress.setVisible(running)
        if running:
            self.progress.setValue(0)


HELP_TEXT = """
<h3>Getting started</h3>
<ol><li><b>Open</b> spectra, a folder or a map: File &gt; Open data (Ctrl+O), or drag files onto the window.</li>
<li><b>Explore</b>: click a map pixel or a file to see its spectrum; with <i>Live fit on click</i> the baseline and
fits appear a moment later.</li>
<li><b>Run</b>: press <b>▶ Run analysis</b> (Ctrl+R) to analyse the whole spectrum, batch or map.</li>
<li><b>Save</b> the project (Ctrl+S) or export results (File &gt; Export).</li></ol>
<h3>Map</h3>
<table cellpadding=3>
<tr><td>Click</td><td>show that pixel's spectrum</td></tr>
<tr><td>Arrow keys</td><td>move to the next pixel (click the map first)</td></tr>
<tr><td>Mouse wheel</td><td>zoom</td></tr><tr><td>Drag</td><td>pan</td></tr>
<tr><td>Double-click, Ctrl+1</td><td>fit the whole map</td></tr>
<tr><td>Histogram handles</td><td>set the colour range; <i>Min–max</i> and <i>2–98 %</i> reset it</td></tr>
<tr><td>Map display tab</td><td>band area / height / ratio shown before analysis, and a threshold mask</td></tr></table>
<h3>Spectrum</h3>
<table cellpadding=3>
<tr><td>Mouse wheel</td><td>zoom (over an axis: that axis only)</td></tr>
<tr><td>Drag / right-drag</td><td>pan / stretch</td></tr>
<tr><td>Double-click, Ctrl+2</td><td>reset the view</td></tr><tr><td>Ctrl+3</td><td>fit y to the visible range</td></tr>
<tr><td>Shaded band</td><td>drag it or its edges to choose what the raw map shows</td></tr>
<tr><td>Right-click</td><td>more plot options (export, grid, log axes)</td></tr></table>
<h3>Other shortcuts</h3>
<table cellpadding=3>
<tr><td>Ctrl+Shift+O</td><td>open a folder</td></tr><tr><td>Ctrl+Shift+R</td><td>run every dataset</td></tr>
<tr><td>Ctrl+L</td><td>live fit on/off</td></tr><tr><td>Ctrl+E</td><td>recipe editor</td></tr>
<tr><td>F1</td><td>this help</td></tr></table>
<p>Panels can be dragged, stacked or floated; View &gt; Reset layout puts them back.</p>
"""


class MainWindow(QMainWindow):
    def __init__(self, session: Session | None = None, settings_key: str = "workbench"):
        super().__init__()
        self.session = session or Session()
        self.qsettings = QSettings(
            QSettings.Format.IniFormat,
            QSettings.Scope.UserScope,
            "spectral-workbench",
            settings_key,
        )
        self.pool = QThreadPool.globalInstance()
        self.jobs: set[Job] = set()
        self.dataset: Dataset | None = None
        self.data = None  # xr.Dataset being shown
        self.pixel = (0, 0)
        self.live_token = 0
        self._sized = False
        self.running = False
        self.cancel = threading.Event()
        self._restored = False
        self.setWindowTitle("Spectral workbench")
        self.setAcceptDrops(True)
        self.setDockNestingEnabled(True)
        self._build_panels()
        self._build_docks()
        self._build_actions()
        self.default_state = self.saveState(LAYOUT_VERSION)
        self._restore_layout()
        self.statusBar().showMessage(
            "Open data to begin: File > Open data (Ctrl+O), or drag files here. Help: F1"
        )

    # ------------------------------------------------------------------ construction
    def _build_panels(self) -> None:
        self.browser = DataBrowser()
        self.browser.setMinimumWidth(170)
        self.browser.chosen.connect(self.show_dataset)
        self.map = MapView()
        self.map.pixel_selected.connect(self.show_pixel)
        self.spectrum = SpectrumPanel()
        self.spectrum.hovered.connect(lambda t: self.statusBar().showMessage(t, 4000))
        self.spectrum.raw.region_changed.connect(
            lambda lo, hi: self.profile.set_band(lo, hi)
        )
        self.spectrum.raw.line_changed.connect(
            lambda x: self.profile.set_band(x, self.profile.hi.value())
        )
        self.settings = SettingsPanel(self.session)
        self.settings.changed.connect(self._settings_changed)
        self.profile = ProfilePanel()
        self.profile.changed.connect(self.update_profile)
        self.threshold = ThresholdPanel()
        self.threshold.changed.connect(
            lambda: self.map.set_mask(self.threshold.active_mask())
        )
        self.layer_list = QListWidget()
        self.layer_list.setMaximumHeight(160)
        self.layer_list.setToolTip(
            "Raw-data profile, and every result map of the last run"
        )
        self.layer_list.currentTextChanged.connect(
            lambda n: n and self.map.show_layer(n)
        )
        self.map.layer_changed.connect(self._layer_shown)
        self.analysis = AnalysisPanel(
            self.settings, self.layer_list, self.profile, self.threshold
        )
        self.analysis.run_button.clicked.connect(self._run_clicked)
        self.analysis.run_all_button.clicked.connect(self.run_all)
        self.progress = self.analysis.progress
        self.results = ResultsPanel()
        self.results.row_chosen.connect(
            lambda k: self.dataset and self.browser.select(self.dataset, k)
        )
        self.editor = RecipeEditor()
        use = QPushButton("Use this recipe in the analysis")
        use.clicked.connect(lambda: self.settings.use_recipe(self.editor.recipe))
        self.editor_panel = QWidget()
        el = QVBoxLayout(self.editor_panel)
        el.addWidget(self.editor, 1)
        el.addWidget(use)
        self.setCentralWidget(QWidget())
        self.centralWidget().hide()  # every panel is a dock

    def _dock(self, title: str, widget: QWidget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(title)  # needed to save and restore the layout
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_docks(self) -> None:
        A = Qt.DockWidgetArea
        self.docks = {
            "Data": self._dock("Data", self.browser, A.LeftDockWidgetArea),
            "Map": self._dock("Map", self.map, A.RightDockWidgetArea),
            "Analysis": self._dock("Analysis", self.analysis, A.RightDockWidgetArea),
            "Results": self._dock("Results", self.results, A.BottomDockWidgetArea),
        }
        self.docks["Spectrum"] = self._dock(
            "Spectrum", self.spectrum, A.RightDockWidgetArea
        )
        self.docks["Recipe editor"] = self._dock(
            "Recipe editor", self.editor_panel, A.RightDockWidgetArea
        )
        d = self.docks
        self.splitDockWidget(d["Map"], d["Analysis"], Qt.Orientation.Horizontal)
        self.splitDockWidget(d["Map"], d["Spectrum"], Qt.Orientation.Vertical)
        d["Recipe editor"].setFloating(
            True
        )  # a wide tool: its own window, opened on demand
        d["Recipe editor"].hide()

    def _default_sizes(self) -> None:
        """Panel proportions for the current window size (only meaningful once shown)."""
        d, w, h = self.docks, self.width(), self.height()
        H, V = Qt.Orientation.Horizontal, Qt.Orientation.Vertical
        self.resizeDocks(
            [d["Data"], d["Map"], d["Analysis"]],
            [int(w * 0.15), int(w * 0.57), int(w * 0.28)],
            H,
        )
        self.resizeDocks([d["Map"], d["Spectrum"]], [int(h * 0.42), int(h * 0.38)], V)
        self.resizeDocks([d["Results"]], [int(h * 0.16)], V)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._sized:
            self._sized = True
            if not self._restored:
                self._default_sizes()

    def _action(
        self, menu, text, slot, shortcut=None, checkable=False, tip=""
    ) -> QAction:
        act = QAction(text, self)
        act.setCheckable(checkable)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        keys = act.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
        act.setToolTip(f"{tip or text.rstrip('…')} ({keys})" if keys else (tip or text))
        act.setStatusTip(tip or text)
        (act.toggled if checkable else act.triggered).connect(slot)
        menu.addAction(act)
        return act

    def _build_actions(self) -> None:
        bar = self.menuBar()
        f = bar.addMenu("&File")
        open_data = self._action(
            f,
            "Open data…",
            self.open_files,
            QKeySequence.StandardKey.Open,
            tip="Open spectra or maps",
        )
        open_folder = self._action(
            f,
            "Open folder…",
            self.open_folder,
            "Ctrl+Shift+O",
            tip="Open a folder of spectra as a batch",
        )
        f.addSeparator()
        self._action(f, "Open project…", self.open_project)
        save = self._action(
            f,
            "Save project",
            self.save_project,
            QKeySequence.StandardKey.Save,
            tip="Save settings and results",
        )
        self._action(f, "Save project as…", lambda: self.save_project(ask=True))
        f.addSeparator()
        self._action(f, "Export results…", self.export_results)
        self._action(f, "Export map image…", self.export_map_image)
        f.addSeparator()
        self._action(f, "Quit", self.close, QKeySequence.StandardKey.Quit)
        a = bar.addMenu("&Analysis")
        self.run_action = self._action(
            a,
            "Run analysis",
            self._run_clicked,
            "Ctrl+R",
            tip="Analyse the selected data",
        )
        self._action(a, "Run on all data", self.run_all, "Ctrl+Shift+R")
        self.live_action = self._action(
            a,
            "Live fit on click",
            self.analysis.live.setChecked,
            "Ctrl+L",
            checkable=True,
        )
        self.live_action.setChecked(True)
        self.analysis.live.toggled.connect(self._live_toggled)
        v = bar.addMenu("&View")
        self._action(v, "Fit map", self.map.fit_view, "Ctrl+1")
        self._action(v, "Reset spectrum view", self.spectrum.reset_view, "Ctrl+2")
        self._action(v, "Fit spectrum y", self.spectrum.fit_y, "Ctrl+3")
        v.addSeparator()
        for dock in self.docks.values():
            v.addAction(dock.toggleViewAction())
        v.addSeparator()
        self._action(v, "Reset layout", self.reset_layout)
        t = bar.addMenu("&Tools")
        self._action(
            t,
            "Recipe editor",
            self.show_editor,
            "Ctrl+E",
            tip="Define baselines and peaks to measure",
        )
        h = bar.addMenu("&Help")
        self._action(h, "Controls and shortcuts", self.show_help, "F1")
        tb = self.addToolBar("Main")
        tb.setObjectName("Main toolbar")
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        for act in (open_data, open_folder, save):
            tb.addAction(act)
        tb.addSeparator()
        tb.addAction(self.run_action)
        run = tb.widgetForAction(self.run_action)
        if run is not None:
            run.setStyleSheet(
                "color: white; background: #1b7837; font-weight: bold; border-radius: 4px; padding: 2px 10px;"
            )
        tb.addSeparator()
        tb.addAction(h.actions()[0])

    def show_help(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Controls and shortcuts")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(HELP_TEXT)
        box.exec()

    def show_editor(self) -> None:
        dock = self.docks["Recipe editor"]
        if dock.isFloating() and not dock.isVisible():
            dock.resize(min(1000, self.width()), min(650, self.height()))
        dock.show()
        dock.raise_()

    # ------------------------------------------------------------------ layout
    def _restore_layout(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is not None:  # fit the screen, whatever its size
            g = screen.availableGeometry()
            self.setGeometry(
                g.x() + g.width() // 20,
                g.y() + g.height() // 20,
                int(g.width() * 0.9),
                int(g.height() * 0.9),
            )
        geometry = self.qsettings.value("geometry")
        state = self.qsettings.value("state")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None:  # a layout saved by an older version is ignored
            self._restored = bool(self.restoreState(state, LAYOUT_VERSION))
        if screen is not None:  # never start larger than the screen or off it
            g = screen.availableGeometry()
            if not g.contains(self.frameGeometry()):
                self.setGeometry(
                    g.x() + g.width() // 20,
                    g.y() + g.height() // 20,
                    int(g.width() * 0.9),
                    int(g.height() * 0.9),
                )

    def reset_layout(self) -> None:
        self.restoreState(self.default_state, LAYOUT_VERSION)
        for name, dock in self.docks.items():
            dock.setFloating(name == "Recipe editor")
            dock.setVisible(name != "Recipe editor")
        self._default_sizes()

    # ------------------------------------------------------------------ opening
    def _filters(self) -> str:
        patterns = " ".join(f"*{s}" for s in self.session.registry.suffixes)
        return f"Spectra and maps ({patterns});;All files (*)"

    def open_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open spectra or maps", "", self._filters()
        )
        if paths:
            self.open_paths(paths)

    def open_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open a folder of spectra")
        if path:
            self.open_paths([path])

    def open_paths(self, paths) -> None:
        try:
            added = self.session.open(paths)
        except ValueError as e:
            QMessageBox.warning(self, "Cannot open", str(e))
            return
        for d in added:
            self.browser.add(d)
        if added:
            self.browser.select(added[-1], 0)
        else:
            self.statusBar().showMessage("No supported files found", 5000)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.open_paths([u.toLocalFile() for u in event.mimeData().urls()])

    # ------------------------------------------------------------------ showing data
    def show_dataset(self, dataset: Dataset, index: int = 0) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            data = dataset.load(self.session.registry, index)
        except Exception as e:  # noqa: BLE001 - reader errors shown to the user
            QMessageBox.warning(
                self, "Cannot read", f"{dataset.paths[index].name}: {e}"
            )
            return
        finally:
            QApplication.restoreOverrideCursor()
        changed = dataset is not self.dataset
        self.dataset, self.data = dataset, data
        if is_map(data):
            if changed or self.map.image.image is None:
                self.map.set_coords(data.x.values, data.y.values)
                self.refresh_layers(fit=True)
                self.docks["Map"].raise_()
            ny, nx = data.sizes["y"], data.sizes["x"]
            self.map.select(ny // 2, nx // 2)
        else:
            self.show_pixel(0, 0)
        if changed:
            self._show_dataset_results()

    def _show_dataset_results(self) -> None:
        r = self.dataset.results if self.dataset else None
        if r is None:
            self.results.show_frame(None)
        elif hasattr(r, "data_vars"):
            self.results.show_frame(map_summary(r))
        else:
            keys = tuple(
                c
                for a in self.session.registry.analyses.values()
                for c in a.key_columns
            )
            self.results.show_frame(ordered(r, keys))

    def refresh_layers(self, fit: bool = False) -> None:
        """Map layers: the raw-data profile plus every result of the last run."""
        layers = {PROFILE_LAYER: self._profile_values()}
        r = self.dataset.results if self.dataset else None
        show = None
        if r is not None and hasattr(r, "data_vars"):
            for name in r.data_vars:
                if r[name].dims == ("y", "x"):
                    layers[str(name)] = r[name].values
            defaults = [
                a.default_layer
                for a in self.session.registry.analyses.values()
                if a.default_layer
            ]
            show = next((d for d in defaults if d in layers), None) if fit else None
        self.map.set_layers(layers, show=show)
        self.threshold.set_layers(layers)
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        self.layer_list.addItems(list(layers))
        self.layer_list.blockSignals(False)
        self._layer_shown(self.map.layer.currentText())

    def _layer_shown(self, name: str) -> None:
        items = self.layer_list.findItems(name, Qt.MatchFlag.MatchExactly)
        if items:
            self.layer_list.blockSignals(True)
            self.layer_list.setCurrentItem(items[0])
            self.layer_list.blockSignals(False)

    def _profile_values(self) -> np.ndarray:
        try:
            return compute_profile(
                self.data["spectra"].values, self.data.wn.values, self.profile.profile()
            )
        except ValueError as e:
            self.statusBar().showMessage(f"Profile: {e}", 5000)
            return np.full((self.data.sizes["y"], self.data.sizes["x"]), np.nan)

    def _draw_band(self) -> None:
        p = self.profile.profile()
        band = p.numerator if p.kind == "ratio" and p.numerator else p
        self.spectrum.raw.hide_line()
        self.spectrum.raw.show_region(band.lo, band.hi)

    def update_profile(self) -> None:
        """The profile settings changed: recompute the raw-data map and show it."""
        self._draw_band()
        if self.data is not None and is_map(self.data):
            self.map.update_layer(PROFILE_LAYER, self._profile_values())
            self.map.show_layer(PROFILE_LAYER)
            if self.threshold.layer.currentText() == PROFILE_LAYER:
                self.threshold.set_layers(self.map.layers)

    def show_pixel(self, i: int, j: int) -> None:
        if self.data is None:
            return
        self.pixel = (i, j)
        x, y = self.data.wn.values, self.data["spectra"].values[i, j]
        name = (
            self.dataset.paths[0].name if not is_map(self.data) else f"row {i}, col {j}"
        )
        if self.dataset.kind == "batch":
            name = self.data.attrs.get("source_file", name)
        self.spectrum.show_spectrum(x, y, name, axis_labels(self.data))
        self._draw_band()
        values = self._pixel_values(i, j)
        self.results.show_values(values)
        self.editor.set_spectrum(x, np.asarray(y, float), values.get("typeIIA_ratio"))
        if self.analysis.live.isChecked():
            self.live_fit()

    def _pixel_values(self, i: int, j: int) -> dict:
        r = self.dataset.results if self.dataset else None
        if r is None or not hasattr(r, "data_vars") or not is_map(self.data):
            return {}
        return {
            str(n): float(r[n].values[i, j])
            for n in r.data_vars
            if r[n].dims == ("y", "x")
        }

    # ------------------------------------------------------------------ live fit
    def _live_toggled(self, on: bool) -> None:
        self.live_action.blockSignals(True)
        self.live_action.setChecked(on)
        self.live_action.blockSignals(False)
        if on and self.data is not None:
            self.show_pixel(*self.pixel)

    def _settings_changed(self) -> None:
        if self.analysis.live.isChecked() and self.data is not None:
            self.show_pixel(*self.pixel)

    def live_fit(self) -> None:
        """Analyse the shown spectrum in the background; stale answers are dropped."""
        self.live_token += 1
        token, data, (i, j) = self.live_token, self.data, self.pixel
        self.statusBar().showMessage("Fitting…")
        self._start(
            lambda _s: self.session.analyse_spectrum(data, i, j),
            lambda res: self._live_done(token, res),
        )

    def _live_done(self, token: int, results: dict) -> None:
        if token != self.live_token:
            return
        self.spectrum.add_results(results)
        values = self._pixel_values(*self.pixel)
        for res in results.values():
            values.update({k: v for k, v in res.values.items() if k not in values})
        self.results.show_values(values)
        if "typeIIA_ratio" in values:  # thickness-normalised recipe previews
            self.editor.thickness = values["typeIIA_ratio"]
            self.editor.preview()
        self.statusBar().showMessage("Fit updated", 3000)

    # ------------------------------------------------------------------ full runs
    def _start(self, fn, done, progress=None) -> None:
        job = Job(fn)
        self.jobs.add(job)
        job.signals.done.connect(lambda r: (self.jobs.discard(job), done(r)))
        job.signals.failed.connect(lambda m: (self.jobs.discard(job), self._failed(m)))
        if progress:
            job.signals.progress.connect(progress)
        self.pool.start(job)

    def _failed(self, message: str) -> None:
        if self.running:
            self._set_running(False)
        QMessageBox.critical(self, "Analysis failed", message)

    def _set_running(self, running: bool) -> None:
        self.running = running
        self.analysis.set_running(running)

    def _run_clicked(self) -> None:
        if self.running:
            self.stop()
        else:
            self.run_selected()

    def stop(self) -> None:
        """Ask a running analysis to stop; its worker processes are shut down."""
        if self.running:
            self.cancel.set()
            self.statusBar().showMessage("Stopping…")

    def _progress(self, done: int, total: int) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)

    def run_selected(self) -> None:
        if self.dataset is None:
            QMessageBox.information(
                self, "Nothing selected", "Open data and select it first."
            )
            return
        self.run([self.dataset])

    def run_all(self) -> None:
        self.run(list(self.session.datasets))

    def run(self, datasets: list[Dataset]) -> None:
        if not datasets:
            return
        if self.running:
            return
        self._set_running(True)
        session, cancel = self.session, self.cancel
        cancel.clear()

        def progress(signals, k, n):
            if cancel.is_set():
                raise Cancelled
            signals.progress.emit(k, n)

        def work(signals):
            done = []
            try:
                for d in datasets:
                    session.analyse_dataset(
                        d, progress=lambda k, n: progress(signals, k, n)
                    )
                    done.append(d)
            except Cancelled:
                pass
            return done

        self._start(work, self._run_done, self._progress)

    def _run_done(self, datasets: list[Dataset]) -> None:
        self._set_running(False)
        for d in datasets:
            self.browser.set_status(d, "analysed")
        if self.cancel.is_set():
            self.statusBar().showMessage("Analysis stopped", 8000)
            return
        if self.dataset in datasets:
            if self.data is not None and is_map(self.data):
                self.refresh_layers(fit=True)
                self.analysis.tabs.setCurrentIndex(
                    1
                )  # the result maps are listed there
                self.statusBar().showMessage(
                    "Analysis finished: pick a result under Map display > Show on the map",
                    10000,
                )
            else:
                self.statusBar().showMessage(
                    "Analysis finished: results are in the Results panel", 10000
                )
            self._show_dataset_results()
            self.show_pixel(*self.pixel)

    # ------------------------------------------------------------------ project and export
    def save_project(self, ask: bool = False) -> bool:
        path = self.session.project_path
        if ask or path is None:
            name, _ = QFileDialog.getSaveFileName(
                self,
                "Save project",
                "project" + PROJECT_SUFFIX,
                f"Projects (*{PROJECT_SUFFIX})",
            )
            if not name:
                return False
            path = Path(name)
        saved = self.session.save(path)
        self.statusBar().showMessage(f"Saved {saved.name}", 5000)
        return True

    def open_project(self) -> None:
        if not self._maybe_save():
            return
        name, _ = QFileDialog.getOpenFileName(
            self, "Open project", "", f"Projects (*{PROJECT_SUFFIX})"
        )
        if name:
            self.load_project(name)

    def load_project(self, path) -> None:
        self.session = Session.load(path, registry=self.session.registry)
        self.settings.session = self.session
        self.settings.build()
        self.browser.clear()
        self.dataset = self.data = None
        for d in self.session.datasets:
            self.browser.add(d)
            if d.results is not None:
                self.browser.set_status(d, "analysed")
        missing = [
            str(p)
            for d in self.session.datasets
            for p in d.paths
            if not Path(p).exists()
        ]
        if missing:
            QMessageBox.warning(
                self,
                "Raw data not found",
                "Results are loaded, but these files have moved:\n"
                + "\n".join(missing[:10]),
            )
        if self.session.datasets and not missing:
            self.browser.select(self.session.datasets[0], 0)

    def export_results(self) -> None:
        r = self.dataset.results if self.dataset else None
        if r is None:
            QMessageBox.information(self, "No results", "Run the analysis first.")
            return
        if hasattr(r, "data_vars"):
            from ..session import netcdf_safe

            name, _ = QFileDialog.getSaveFileName(
                self,
                "Export map results",
                f"{self.dataset.name}.nc",
                "NetCDF (*.nc);;CSV, one row per pixel (*.csv)",
            )
            if name.endswith(".csv"):
                r.to_dataframe().reset_index().to_csv(name, index=False)
            elif name:
                netcdf_safe(r).to_netcdf(name)
        else:
            name, _ = QFileDialog.getSaveFileName(
                self, "Export results", f"{self.dataset.name}.csv", "CSV (*.csv)"
            )
            if name:
                r.to_csv(name, index=False)

    def export_map_image(self) -> None:
        arr = self.map.current()
        if arr is None:
            return
        name, _ = QFileDialog.getSaveFileName(
            self,
            "Export map",
            f"{self.map.layer.currentText()}.png",
            "PNG image as shown (*.png);;TIFF, 32-bit values (*.tif)",
        )
        if not name:
            return
        if name.lower().endswith((".tif", ".tiff")):
            from PIL import Image

            Image.fromarray(np.flipud(arr).astype(np.float32), mode="F").save(name)
        else:
            from pyqtgraph import exporters

            exporters.ImageExporter(self.map.view.scene()).export(name)

    # ------------------------------------------------------------------ closing
    def _maybe_save(self) -> bool:
        """True when it is fine to discard the current session."""
        if not self.session.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved work",
            "Save the project (settings and results) before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return answer == QMessageBox.StandardButton.Discard

    def closeEvent(self, event):
        if self.running:
            answer = QMessageBox.question(
                self,
                "Analysis running",
                "An analysis is still running. Stop it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.stop()
            self.pool.waitForDone(30000)  # workers are shut down before we exit
        if not self._maybe_save():
            event.ignore()
            return
        self.qsettings.setValue("geometry", self.saveGeometry())
        self.qsettings.setValue("state", self.saveState(LAYOUT_VERSION))
        event.accept()
        QApplication.quit()  # closing the main window ends the program


def main(
    session_factory=None,
    title: str = "Spectral workbench",
    settings_key: str = "workbench",
) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(session_factory() if session_factory else None, settings_key)
    window.setWindowTitle(title)
    window.show()
    paths = [a for a in sys.argv[1:] if Path(a).exists()]
    if paths:
        window.open_paths(paths)
    return app.exec()
