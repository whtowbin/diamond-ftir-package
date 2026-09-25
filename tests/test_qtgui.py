"""Qt widgets and napari plugin, run offscreen (no display needed)."""

import os

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


def test_main_window_builds_and_passes_ticked_recipe(app):
    from qtpy.QtCore import Qt

    from diamond_ftir_package.qtgui.app import MainWindow

    w = MainWindow()
    w.settings.use_recipe(resolve_recipe("diamond_hydrogen"))
    params = w.settings.current()
    assert any(getattr(r, "name", r) == "diamond_hydrogen" for r in params.recipes)
    item = w.settings.recipe_list.item(0)
    assert item.flags() & Qt.ItemFlag.ItemIsUserCheckable


def test_map_view_shows_results_and_pixel_spectrum(app):
    from diamond_ftir_package.maps import process_map
    from diamond_ftir_package.params import AnalysisParams, MapParams
    from diamond_ftir_package.qtgui import MapView

    from .test_maps import _synthetic_map

    data = _synthetic_map()
    result = process_map(
        data,
        AnalysisParams(run_hydrogen=False, run_platelets=False),
        MapParams(n_jobs=1),
    )
    view = MapView()
    view.set_data(result, data)
    view.show_pixel(1, 2)
    assert "Total_N ppm" in view.info.text()
    assert len(view.spectrum.plot.listDataItems()) == 1


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
