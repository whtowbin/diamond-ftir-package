"""Qt widgets and napari plugin, run offscreen (no display needed)."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("pyqtgraph")

from qtpy.QtWidgets import QApplication

from diamond_ftir_package.core import (
    BaselineStep,
    resolve_recipe,
    run_recipe,
)

from .synthetic import make_spectrum


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_dataclass_form_edits_object_and_reports_errors(app):
    from diamond_ftir_package.qtgui import DataclassForm

    step = BaselineStep()
    form = DataclassForm(step, {"method": ("asls", "rubberband")})
    form.editors["lam"].setText("2e6")
    form._commit("lam")
    assert step.lam == 2e6
    form.editors["anchors"].setText("900, 920; 1080, 1100")
    form._commit("anchors")
    assert step.anchors == [[900.0, 920.0], [1080.0, 1100.0]]
    form.editors["lam"].setText("abc")
    form._commit("lam")
    assert "lam" in form.error.text() and step.lam == 2e6


def test_recipe_editor_preview_matches_engine(app):
    from diamond_ftir_package.qtgui import RecipeEditor

    s, _ = make_spectrum(a_ppm=200, b_ppm=300, h3107_height=0.5, noise=0.0005)
    s.fit_baseline()
    recipe = resolve_recipe("diamond_hydrogen")
    editor = RecipeEditor(recipe)
    editor.set_spectrum(s.X, s.Y, s.typeIIA_ratio)
    shown = {
        editor.results.item(r, 0).text(): float(editor.results.item(r, 1).text())
        for r in range(editor.results.rowCount())
    }
    expected = run_recipe(s.X, s.Y, recipe, thickness=s.typeIIA_ratio).values
    assert shown["H.3107.area"] == pytest.approx(expected["H.3107.area"], rel=1e-4)


def test_recipe_editor_add_and_remove(app):
    from diamond_ftir_package.qtgui import RecipeEditor

    s, _ = make_spectrum(a_ppm=200, b_ppm=300)
    editor = RecipeEditor()
    editor.set_spectrum(s.X, s.Y, 0.05)
    editor._add_feature(3)
    assert (
        len(editor.recipe.features) == 1 and len(editor.recipe.features[0].peaks) == 3
    )
    editor._add_step()
    assert len(editor.recipe.features[0].baseline) == 2
    editor.current = ("feature", 0, -1)
    editor._remove()
    assert editor.recipe.features == []


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    """Main window with the diamond analysis, its layout kept out of the user's settings."""
    from qtpy.QtCore import QSettings

    from diamond_ftir_package.plugin import NAME
    from diamond_ftir_package.session import diamond_session
    from diamond_ftir_package.workbench.qt import MainWindow

    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path)
    )
    session = diamond_session()
    session.params[NAME].run_hydrogen = session.params[NAME].run_platelets = False
    w = MainWindow(session, settings_key="test")
    yield w
    from qtpy.QtWidgets import QMessageBox

    # a modal dialog would wait forever offscreen: after a failed test, answer "Yes"
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    _wait(w)  # finish runs and deliver their results, so closing never asks
    w.session.settings_dirty = False
    for d in w.session.datasets:
        d.dirty = False
    w.close()


def _wait(w):
    w.pool.waitForDone(60000)
    QApplication.processEvents()


def test_ticked_recipes_reach_the_session(window):
    window.settings.use_recipe(resolve_recipe("diamond_hydrogen"))
    assert any(
        getattr(r, "name", r) == "diamond_hydrogen" for r in window.session.recipes
    )
    assert window.session.settings_dirty


def test_map_click_shows_spectrum_and_live_fit(window):
    from .test_workbench import _map_dataset

    d = _map_dataset(window.session)
    window.browser.add(d)
    window.show_dataset(d)
    assert "Profile (raw data)" in window.map.layers
    profile = window.map.layers["Profile (raw data)"]
    assert profile.shape == (3, 4) and profile[0, 0] < profile[1, 1]  # the hole shows
    window.map.select(1, 2)
    _wait(window)
    names = {i.name() for i in window.spectrum.raw.plot.listDataItems()}
    assert "row 1, col 2" in names and "baseline" in names
    assert window.spectrum.processed.isVisibleTo(window.spectrum)
    assert window.results.selected_model.rowCount() > 5  # live-fit values listed


def test_profile_band_follows_the_spectrum_markers(window):
    from .test_workbench import _map_dataset

    d = _map_dataset(window.session)
    window.show_dataset(d)
    before = window.map.layers["Profile (raw data)"].copy()
    window.spectrum.raw.region_changed.emit(1100.0, 1300.0)
    assert (window.profile.lo.value(), window.profile.hi.value()) == (1100.0, 1300.0)
    assert not np.allclose(
        before, window.map.layers["Profile (raw data)"], equal_nan=True
    )


def test_run_map_adds_result_layers_and_threshold_masks(window):
    from diamond_ftir_package.params import MapParams
    from diamond_ftir_package.plugin import NAME

    from .test_workbench import _map_dataset

    window.session.map_params[NAME] = MapParams(n_jobs=1)
    d = _map_dataset(window.session)
    window.browser.add(d)
    window.show_dataset(d)
    window.run_selected()
    _wait(window)
    assert window.map.layer.currentText() == "Total_N ppm"
    assert window.results.table_model.rowCount() > 3  # per-layer statistics
    th = window.threshold
    th.layer.setCurrentText("Total_N ppm")
    th.lo.setValue(float(np.nanmax(th.values)) + 1)
    th.enabled.setChecked(True)
    assert np.isnan(window.map.current()).all()


def test_run_button_runs_the_selected_dataset(window):
    from diamond_ftir_package.params import MapParams
    from diamond_ftir_package.plugin import NAME

    from .test_workbench import _map_dataset

    window.session.map_params[NAME] = MapParams(n_jobs=1)
    d = _map_dataset(window.session)
    window.browser.add(d)
    window.show_dataset(d)
    window.analysis.run_button.click()
    assert window.running and window.analysis.run_button.text().endswith("Stop")
    _wait(window)
    assert d.results is not None and not window.running
    assert window.analysis.run_button.text().endswith("Run analysis")


def test_layer_list_and_profile_changes_update_the_map(window):
    from diamond_ftir_package.params import MapParams
    from diamond_ftir_package.plugin import NAME

    from .test_workbench import _map_dataset

    window.session.map_params[NAME] = MapParams(n_jobs=1)
    d = _map_dataset(window.session)
    window.browser.add(d)
    window.show_dataset(d)
    window.run_selected()
    _wait(window)
    names = [window.layer_list.item(k).text() for k in range(window.layer_list.count())]
    assert "Profile (raw data)" in names and "B_percent" in names
    window.layer_list.setCurrentRow(names.index("B_percent"))
    assert window.map.layer.currentText() == "B_percent"
    window.profile.kind.setCurrentText("height")  # editing the profile shows it
    assert window.map.layer.currentText() == "Profile (raw data)"
    heights = window.map.layers["Profile (raw data)"]
    assert np.nanmax(heights) > 0  # peak height above the band's baseline, not 0


def test_stop_button_stops_a_run(window):
    from .test_workbench import _map_dataset

    d = _map_dataset(window.session)
    window.browser.add(d)
    window.show_dataset(d)
    window.run([d])  # parallel by default: real worker processes
    window.stop()  # pressed at once; the run stops at its first progress report
    assert window.analysis.run_button.text().endswith("Stop")
    _wait(window)
    assert not window.running and d.results is None
    assert window.analysis.run_button.text().endswith("Run analysis")


def test_help_shortcuts_and_editor_window(window, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    actions = {
        a.text(): a for m in window.menuBar().actions() for a in m.menu().actions()
    }
    assert actions["Controls and shortcuts"].shortcut().toString() == "F1"
    run = actions["Run analysis"]
    assert run.shortcut().toString() == "Ctrl+R"
    native = run.shortcut().toString(run.shortcut().SequenceFormat.NativeText)
    assert native in run.toolTip()  # "⌘R" on macOS, "Ctrl+R" elsewhere
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(self.text()))
    window.show_help()
    assert "Arrow keys" in shown[0]
    editor = window.docks["Recipe editor"]
    assert (
        editor.isFloating() and not editor.isVisible()
    )  # never squeezes the side panel
    window.show_editor()
    assert editor.isVisible()


def test_layout_fits_screen_and_old_layouts_are_ignored(app, tmp_path):
    from qtpy.QtCore import QSettings

    from diamond_ftir_package.session import diamond_session
    from diamond_ftir_package.workbench.qt import MainWindow

    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path)
    )
    old = QSettings(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        "spectral-workbench",
        "fit",
    )
    first = MainWindow(diamond_session(), settings_key="fit")
    old.setValue("state", first.saveState(1))  # a layout saved by version 1
    old.sync()
    w = MainWindow(diamond_session(), settings_key="fit")
    assert not w._restored
    w.show()
    QApplication.processEvents()
    screen = app.primaryScreen().availableGeometry()
    assert screen.contains(w.frameGeometry())
    assert w.docks["Data"].width() >= 170
    for win in (first, w):
        win.session.settings_dirty = False
        win.close()


def test_close_asks_to_save_unsaved_work(window, monkeypatch):
    from qtpy.QtWidgets import QMessageBox

    window.session.settings_dirty = True
    asked = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: asked.append(1) or QMessageBox.StandardButton.Cancel,
    )
    assert not window._maybe_save() and asked
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Discard
    )
    assert window._maybe_save()


def test_map_view_levels_and_picking(app):
    from diamond_ftir_package.workbench.qt import MapView

    view = MapView()
    view.set_coords(np.arange(4) * 25.0, np.arange(3) * 25.0)
    arr = np.arange(12, dtype=float).reshape(3, 4)
    arr[0, 0] = np.nan
    view.set_layers({"a": arr})
    assert tuple(view.image.levels) == (1.0, 11.0)  # data min and max, NaN ignored
    picked = []
    view.pixel_selected.connect(lambda i, j: picked.append((i, j)))
    view.select(2, 3)
    assert picked == [(2, 3)]
    view.set_mask(arr > 5)
    assert np.isnan(view.current()[0, 1]) and view.current()[2, 3] == 11


def test_napari_manifest_is_valid():
    npe2 = pytest.importorskip("npe2")
    from importlib import resources

    manifest = npe2.PluginManifest.from_file(
        resources.files("diamond_ftir_package") / "napari.yaml"
    )
    assert manifest.contributions.readers and manifest.contributions.widgets
    from diamond_ftir_package.napari_plugin import get_reader

    assert get_reader("x.csv") is None and get_reader("x.map") is not None


def test_napari_inspector_widget_with_stand_in_viewer(app):
    """The inspector only uses viewer.layers, viewer.mouse_drag_callbacks and add_image.

    A real headless napari viewer segfaults offscreen with PySide6 (napari's PySide6 support
    is experimental), so a stand-in viewer is used; check the real viewer by hand.
    """
    from types import SimpleNamespace

    from diamond_ftir_package.napari_plugin import RAW_KEY, SpectrumInspector

    from .test_maps import _synthetic_map

    data = _synthetic_map()
    layer = SimpleNamespace(
        metadata={RAW_KEY: data}, world_to_data=lambda pos: (pos[-2], pos[-1])
    )
    added = []
    viewer = SimpleNamespace(
        layers=[layer],
        mouse_drag_callbacks=[],
        add_image=lambda arr, **kw: added.append((kw["name"], arr.shape)),
    )
    widget = SpectrumInspector(viewer)
    assert widget._on_click in viewer.mouse_drag_callbacks
    widget._on_click(viewer, SimpleNamespace(position=(1.0, 2.0)))
    assert "pixel y=1 x=2" in widget.info.text()
    assert widget.editor.x is not None
    widget.run_analysis()
    names = [n for n, _ in added]
    assert "Total_N ppm" in names and all(shape == (3, 4) for _, shape in added)
