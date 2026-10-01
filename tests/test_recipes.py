"""Recipe engine: user-defined baselines, single peaks and clusters."""

import numpy as np
import pytest

from diamond_ftir_package.core import (
    BaselineStep,
    Feature,
    PeakDef,
    Recipe,
    builtin_recipes,
    resolve_recipe,
    run_recipe,
)
from diamond_ftir_package.params import AnalysisParams
from diamond_ftir_package.pipeline import run_analysis

from .synthetic import gaussian, make_spectrum


def test_builtin_hydrogen_recipe_reproduces_legacy_measurement_exactly():
    s, _ = make_spectrum(a_ppm=200, b_ppm=300, h3107_height=0.5, noise=0.0005)
    s.fit_baseline()
    s.normalize_diamond()
    s.measure_3107_peak()
    result = run_recipe(
        s.X, s.Y, resolve_recipe("diamond_hydrogen"), thickness=s.typeIIA_ratio
    )
    assert result.values["H.3107.area"] == pytest.approx(s.normed_area_3107, rel=1e-12)
    assert result.values["H.3085.area"] == pytest.approx(s.normed_area_3085, rel=1e-12)


def test_all_builtin_recipes_are_valid():
    for name in builtin_recipes():
        assert resolve_recipe(name).validate() == [], name


def _cluster_recipe(model="gaussian", constraints=None):
    return Recipe(
        name="test",
        normalize_by="none",
        features=[
            Feature(
                name="c",
                region=[900, 1100],
                model=model,
                baseline=[
                    BaselineStep(method="linear", anchors=[[900, 920], [1080, 1100]])
                ],
                peaks=[
                    PeakDef(name="a", center=980, fwhm=15, center_range=[970, 990]),
                    PeakDef(name="b", center=1010, fwhm=15, center_range=[1000, 1020]),
                ],
                constraints=constraints or {},
            )
        ],
    )


def test_cluster_fit_recovers_injected_overlapping_peaks():
    """Injection test: two overlapping Gaussians of known area on a sloped background."""
    x = np.arange(800.0, 1200.0, 1.0)
    rng = np.random.default_rng(0)
    y = 0.2 + 1e-4 * (x - 800) + gaussian(x, 978, 1.0, 14) + gaussian(x, 1012, 0.6, 18)
    y += rng.normal(0, 0.003, x.size)
    true_area = {"a": 1.0 * 14 * 1.0645, "b": 0.6 * 18 * 1.0645}
    v = run_recipe(x, y, _cluster_recipe()).values
    assert v["c.a.center"] == pytest.approx(978, abs=0.5)
    assert v["c.b.center"] == pytest.approx(1012, abs=0.5)
    assert v["c.a.area"] == pytest.approx(true_area["a"], rel=0.03)
    assert v["c.b.area"] == pytest.approx(true_area["b"], rel=0.03)
    assert v["c.fit_r2"] > 0.99


def test_cluster_constraint_ties_widths():
    x = np.arange(800.0, 1200.0, 1.0)
    y = gaussian(x, 978, 1.0, 14) + gaussian(x, 1012, 0.6, 14)
    v = run_recipe(x, y, _cluster_recipe(constraints={"b_fwhm": "a_fwhm"})).values
    assert v["c.a.fwhm"] == pytest.approx(v["c.b.fwhm"], rel=1e-9)


def test_recipe_roundtrip_toml_and_json(tmp_path):
    r = _cluster_recipe(constraints={"b_fwhm": "a_fwhm"})
    for suffix in (".toml", ".json"):
        path = tmp_path / f"r{suffix}"
        r.save(path)
        assert Recipe.load(path) == r


def test_validation_messages_are_readable():
    r = _cluster_recipe()
    r.features[0].region = [1100, 900]
    r.features[0].baseline.append(BaselineStep(method="magic"))
    problems = " ".join(r.validate())
    assert "lo < hi" in problems and "magic" in problems


def test_unknown_setting_in_file_is_an_error(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text('name = "x"\nnormalise_by = "none"\n')
    with pytest.raises(ValueError, match="normalise_by"):
        Recipe.load(p)


def test_recipes_run_through_pipeline_and_maps_use_the_same_path():
    s, _ = make_spectrum(a_ppm=200, b_ppm=300, h3107_height=0.5, noise=0.0005)
    params = AnalysisParams(run_platelets=False, recipes=("diamond_hydrogen",))
    row = run_analysis(s, params)
    assert row["H.3107.area"] == pytest.approx(row["Normed_3107_Area"], rel=1e-12)


def test_rubberband_step_follows_a_steep_tail():
    """A local rubber band handles a sharp falling tail next to a peak (see TODO)."""
    x = np.arange(600.0, 1000.0, 1.0)
    tail = 2.0 * np.exp(-(x - 600) / 40)
    y = tail + gaussian(x, 800, 0.5, 20)
    r = Recipe(
        name="t",
        normalize_by="none",
        features=[
            Feature(
                name="f",
                region=[600, 1000],
                model="gaussian",
                baseline=[BaselineStep(method="rubberband", stretch=0.0)],
                peaks=[PeakDef(name="p", center=800, fwhm=20)],
            )
        ],
    )
    v = run_recipe(x, y, r).values
    assert v["f.p.area"] == pytest.approx(0.5 * 20 * 1.0645, rel=0.1)


def test_numeric_peak_names_and_constraints_work():
    """Peak names like '1405' are not valid lmfit identifiers; the engine maps them."""
    x = np.arange(800.0, 1200.0, 1.0)
    y = gaussian(x, 978, 1.0, 14) + gaussian(x, 1012, 0.6, 14)
    r = _cluster_recipe(constraints={"1012_fwhm": "978_fwhm"})
    r.features[0].peaks[0].name, r.features[0].peaks[1].name = "978", "1012"
    v = run_recipe(x, y, r).values
    assert v["c.978.fwhm"] == pytest.approx(v["c.1012.fwhm"], rel=1e-9)
    assert v["c.978.area"] == pytest.approx(14 * 1.0645, rel=0.02)


def test_builtin_cluster_recipes_fit_a_synthetic_platelet():
    s, _ = make_spectrum(a_ppm=200, b_ppm=300, platelet_height=3.0, noise=0.0005)
    s.fit_baseline()
    res = run_recipe(
        s.X, s.Y, resolve_recipe("diamond_platelet"), thickness=s.typeIIA_ratio
    )
    assert not res.features[0].error
    assert res.values["platelet.B_prime.center"] == pytest.approx(1365, abs=2)
    assert res.values["platelet.B_prime.height"] == pytest.approx(3.0, rel=0.25)
