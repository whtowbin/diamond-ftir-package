"""The generic workbench: data layout, profiles, plug-ins, session and project files."""

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from diamond_ftir_package.core import (
    BaselineStep,
    Feature,
    PeakDef,
    Recipe,
    resolve_recipe,
)
from diamond_ftir_package.core.profiles import Profile, compute_profile
from diamond_ftir_package.maps import process_map
from diamond_ftir_package.params import AnalysisParams, MapParams
from diamond_ftir_package.pipeline import analyze_file
from diamond_ftir_package.plugin import NAME
from diamond_ftir_package.session import diamond_session
from diamond_ftir_package.workbench import Dataset, Session, default_registry, make_cube
from diamond_ftir_package.workbench.analyses import RECIPES

from .synthetic import gaussian, make_spectrum
from .test_maps import _synthetic_map

WORKBENCH = Path(__file__).parents[1] / "src/diamond_ftir_package/workbench"
DOMAIN_MODULES = {
    "DiamondSpectrum",
    "pipeline",
    "maps",
    "params",
    "thermometry",
    "plugin",
    "session",
}


def test_workbench_never_imports_diamond_code():
    """Keeps the workbench movable to its own package: only generic siblings allowed."""
    for path in WORKBENCH.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level >= 2 + (
                path.parent.name == "qt"
            ):
                top = (node.module or "").split(".")[0]
                assert top not in DOMAIN_MODULES, f"{path.name} imports {node.module}"


# ---------------------------------------------------------------------------- profiles
def _triangle_cube():
    wn = np.arange(1000.0, 1101.0)
    peak = np.clip(1 - np.abs(wn - 1050) / 20, 0, None)  # area 20, height 1
    cube = np.stack([peak + 0.5, 2 * peak + 0.1 * (wn - 1000) / 100])[None]  # (1, 2, n)
    return wn, cube


def test_profile_area_height_ratio_and_axis_direction():
    wn, cube = _triangle_cube()
    area = Profile(
        "area", 1020, 1080, baseline=True, baseline_lo=1020, baseline_hi=1080
    )
    np.testing.assert_allclose(compute_profile(cube, wn, area), [[20, 40]], rtol=1e-6)
    np.testing.assert_allclose(
        compute_profile(cube[..., ::-1], wn[::-1], area), [[20, 40]], rtol=1e-6
    )
    peak = Profile(
        "height", 1020, 1080, baseline=True, baseline_lo=1020, baseline_hi=1080
    )
    np.testing.assert_allclose(compute_profile(cube, wn, peak), [[1, 2]], rtol=1e-6)
    np.testing.assert_allclose(
        compute_profile(cube[..., ::-1], wn[::-1], peak), [[1, 2]], rtol=1e-6
    )
    point = Profile(
        "height", 1060, 1060, baseline=True, baseline_lo=1020, baseline_hi=1080
    )
    np.testing.assert_allclose(compute_profile(cube, wn, point), [[0.5, 1]], rtol=1e-6)
    float32 = compute_profile(
        cube.astype(np.float32), wn, area
    )  # no float64 copy needed
    np.testing.assert_allclose(float32, [[20, 40]], rtol=1e-5)
    ratio = Profile(
        "ratio", numerator=area, denominator=Profile("area", 1000, 1010, baseline=False)
    )
    top = compute_profile(cube, wn, area)
    bottom = compute_profile(cube, wn, ratio.denominator)
    np.testing.assert_allclose(compute_profile(cube, wn, ratio), top / bottom)


def test_profile_rejects_empty_band():
    wn, cube = _triangle_cube()
    with pytest.raises(ValueError, match="fewer than 2"):
        compute_profile(cube, wn, Profile("area", 2000, 2100))


# ---------------------------------------------------------------------------- session
def _write_csv(path: Path, **kw) -> Path:
    s, _ = make_spectrum(**kw)
    pd.DataFrame({"wn": s.initial_X, "abs": s.initial_Y}).to_csv(path, index=False)
    return path


def test_open_routes_files_folders_and_maps(tmp_path):
    folder = tmp_path / "batch"
    folder.mkdir()
    for k in range(3):
        _write_csv(folder / f"s{k}.csv", seed=k)
    one = _write_csv(tmp_path / "one.csv")
    (tmp_path / "m.map").write_bytes(b"")  # routed by name, read only when shown
    session = diamond_session()
    added = session.open([folder, one, tmp_path / "m.map"])
    assert [(d.kind, len(d.paths)) for d in added] == [
        ("batch", 3),
        ("map", 1),
        ("spectrum", 1),
    ]
    with pytest.raises(ValueError, match="Unsupported"):
        session.open([tmp_path / "notes.docx"])


def test_batch_rows_match_the_pipeline_exactly(tmp_path):
    """The workbench path (reader -> cube -> plug-in) must equal analyze_file."""
    paths = [
        _write_csv(tmp_path / f"s{k}.csv", a_ppm=100 * (k + 1), b_ppm=200, seed=k)
        for k in range(2)
    ]
    session = diamond_session()
    (d,) = session.open(paths)
    frame = session.analyse_dataset(d)
    for k, path in enumerate(paths):
        _, expected = analyze_file(path, AnalysisParams())
        row = frame.iloc[k].to_dict()
        for key, value in expected.items():
            if isinstance(value, float):
                assert row[key] == pytest.approx(value, rel=1e-12, nan_ok=True), key
            else:
                assert row[key] == value, key


def _map_dataset(session: Session) -> Dataset:
    data = _synthetic_map()
    data.attrs = {"source_file": "syn.map", "axis_kind": "ir"}
    d = Dataset("map", "syn", [Path("syn.map")])
    d._cache[0] = data
    session.datasets.append(d)
    return d


def test_map_run_matches_process_map():
    session = diamond_session()
    session.params[NAME] = AnalysisParams(run_hydrogen=False, run_platelets=False)
    session.map_params[NAME] = MapParams(n_jobs=1)
    d = _map_dataset(session)
    result = session.analyse_dataset(d)
    expected = process_map(d._cache[0], session.params[NAME], MapParams(n_jobs=1))
    np.testing.assert_array_equal(
        result["Total_N ppm"].values, expected["Total_N ppm"].values
    )
    assert d.dirty and session.dirty


def test_live_fit_returns_values_and_curves():
    session = diamond_session()
    session.recipes = ["diamond_hydrogen"]
    d = _map_dataset(session)
    out = session.analyse_spectrum(d._cache[0], 1, 2)[NAME]
    assert not out.error and out.values["Total_N ppm"] > 0
    names = {c.name for c in out.curves}
    assert {"baseline", "nitrogen fit", "normalised (1 cm)"} <= names
    assert "H baseline" in names  # the ticked recipe's baseline is drawn too
    assert "Peak recipes" not in session.analyse_spectrum(
        d._cache[0], 1, 2
    )  # off by default


def test_raman_data_uses_generic_analyses_only():
    """A Raman cube: the diamond (IR) analysis does not apply; recipes do, pixel by pixel."""
    x = np.linspace(100, 1800, 1200)
    cube = np.stack([[gaussian(x, 1332, h, 4) + 0.01 for h in (1.0, 2.0)]])
    ds = xr.Dataset(
        {"spectra": (("y", "x", "wn"), cube)},
        coords={"y": [0.0], "x": [0.0, 1.0], "wn": x},
        attrs={"source_file": "r.map", "axis_kind": "raman"},
    )
    recipe = Recipe(
        name="raman",
        normalize_by="none",
        features=[
            Feature(
                name="D",
                region=[1300, 1360],
                model="integrate",
                baseline=[
                    BaselineStep(method="linear", anchors=[[1300, 1305], [1355, 1360]])
                ],
                peaks=[PeakDef(name="1332", center=1332, integrate=[1310, 1350])],
            )
        ],
    )
    session = diamond_session()
    session.enabled[RECIPES.name] = True
    session.recipes = [recipe]
    assert session.active(ds) == [RECIPES.name]
    d = Dataset("map", "r", [Path("r.map")])
    d._cache[0] = ds
    result = session.analyse_dataset(d)
    areas = [v for k, v in result.data_vars.items() if k.endswith(".area")]
    assert areas and areas[0].values[0, 1] == pytest.approx(
        2 * areas[0].values[0, 0], rel=0.05
    )


def test_project_round_trip(tmp_path):
    session = diamond_session()
    session.params[NAME].nitrogen.d_limit = 0.3
    session.params[NAME] = replace(session.params[NAME], run_amber=True)
    session.enabled[RECIPES.name] = True
    session.recipes = ["diamond_hydrogen", resolve_recipe("diamond_platelet")]
    csv = _write_csv(tmp_path / "a.csv")
    (spec,) = session.open([csv])
    spec.results = pd.DataFrame([{"Filename": "a.csv", "Total_N ppm": 123.4}])
    m = _map_dataset(session)
    m.results = xr.Dataset(
        {"Total_N ppm": (("y", "x"), np.ones((3, 4)))}, attrs={"calibration": {"a": 1}}
    )
    path = session.save(tmp_path / "work")
    assert path.suffix == ".dftir" and not session.dirty

    back = Session.load(path, registry=session.registry)
    assert back.params[NAME].nitrogen.d_limit == 0.3 and back.params[NAME].run_amber
    assert back.params[NAME].amber == session.params[NAME].amber  # tuples restored
    assert back.enabled[RECIPES.name]
    assert (
        back.recipes[0] == "diamond_hydrogen"
        and back.recipes[1].name == "diamond_platelet"
    )
    assert [d.kind for d in back.datasets] == ["spectrum", "map"]
    assert back.datasets[0].results["Total_N ppm"].iloc[0] == 123.4
    np.testing.assert_array_equal(
        back.datasets[1].results["Total_N ppm"].values, np.ones((3, 4))
    )


def test_single_spectrum_cube_and_registry():
    ds = make_cube([1, 2, 3], [4, 5, 6], "x.csv")
    assert ds["spectra"].shape == (1, 1, 3) and ds.attrs["axis_kind"] == "ir"
    reg = default_registry(with_entry_points=False)
    assert reg.loader_for(Path("a.MAP")).is_map and reg.loader_for(Path("a.spa"))
    assert list(reg.analyses) == [RECIPES.name]
