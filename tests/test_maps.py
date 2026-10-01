"""Map processing on a small synthetic map (no real data needed)."""

import numpy as np
import pytest
import xarray as xr

from diamond_ftir_package.maps import (
    STATUS_OFF_SAMPLE,
    STATUS_OK,
    process_map,
    save_map_outputs,
)
from diamond_ftir_package.params import AnalysisParams, MapParams
from diamond_ftir_package.pipeline import run_analysis

from .synthetic import make_spectrum


def _synthetic_map(ny=3, nx=4):
    """A plate with A-rich left half and B-rich right half, plus one off-sample pixel."""
    rows = []
    wn = None
    for i in range(ny):
        row = []
        for j in range(nx):
            a, b = (300, 50) if j < nx // 2 else (50, 300)
            s, _ = make_spectrum(
                a_ppm=a, b_ppm=b, thickness_cm=0.05, noise=0.0005, seed=i * nx + j
            )
            wn = s.initial_X
            row.append(s.initial_Y)
        rows.append(row)
    spectra = np.array(rows, dtype=np.float32)
    spectra[0, 0] = 0.0  # a hole in the sample
    return xr.Dataset(
        {"spectra": (("y", "x", "wn"), spectra)},
        coords={"y": np.arange(ny) * 25.0, "x": np.arange(nx) * 25.0, "wn": wn},
    )


def _params():
    return AnalysisParams(run_hydrogen=False, run_platelets=False)


@pytest.fixture(scope="module")
def data():
    return _synthetic_map()


def test_map_mask_zoning_and_parity_with_single_spectrum(data):
    result = process_map(data, _params(), MapParams(n_jobs=1))
    assert result.status.values[0, 0] == STATUS_OFF_SAMPLE
    assert (result.status.values.ravel()[1:] == STATUS_OK).all()
    b = result["B_percent"].values
    assert np.nanmean(b[:, :2]) < 30  # A-rich half
    assert np.nanmean(b[:, 2:]) > 70  # B-rich half

    # a pixel must give exactly what the single-spectrum path gives
    from diamond_ftir_package import Diamond_Spectrum

    y = data.spectra.values[1, 3].astype(float)
    single = run_analysis(Diamond_Spectrum(X=data.wn.values, Y=y), _params())
    assert result["Total_N ppm"].values[1, 3] == pytest.approx(
        single["Total_N ppm"], rel=1e-6
    )


def test_parallel_matches_serial(data):
    serial = process_map(data, _params(), MapParams(n_jobs=1))
    parallel = process_map(data, _params(), MapParams(n_jobs=2, block_size=3))
    for name in serial.data_vars:
        np.testing.assert_array_equal(serial[name].values, parallel[name].values)


def test_known_thickness_is_applied_and_fitted_value_kept(data):
    result = process_map(
        data,
        _params(),
        MapParams(n_jobs=1, thickness_mode="known", known_thickness_um=500),
    )
    ok = result.status.values == STATUS_OK
    assert np.allclose(result["typeIIA_ratio"].values[ok], 0.05)
    assert "fitted_typeIIA_ratio" in result


def test_calibrated_strategies_seed_every_pixel(data):
    params = _params()
    params.diamond.baseline_method = "single"
    params.diamond.baseline_search = "optimize"
    mp = MapParams(
        n_jobs=1, baseline_strategy="calibrate_fixed", survey_step=1, edge_width_px=0
    )
    result = process_map(data, params, mp)
    ok = result.status.values == STATUS_OK
    lams = result["baseline_lam"].values[ok]
    assert np.allclose(lams, lams[0])  # one calibrated value used everywhere
    assert result.attrs["baseline_start_lam"] == pytest.approx(lams[0])


def test_outputs_are_written(tmp_path, data):
    result = process_map(data, _params(), MapParams(n_jobs=1))
    written = save_map_outputs(result, tmp_path, "demo")
    names = {p.name for p in written}
    assert {
        "demo_results.nc",
        "demo_results.csv",
        "demo_B_percent.png",
        "demo_B_percent.tif",
    } <= names
    reloaded = xr.open_dataset(tmp_path / "demo_results.nc")
    np.testing.assert_array_equal(reloaded["status"].values, result["status"].values)


def _plate_with_rim_and_thick_strip(ny=12, nx=12):
    """A uniform 0.05 cm plate with a partially covered rim and a thick strip on the right."""
    from diamond_ftir_package import (
        Diamond_Spectrum,  # noqa: F401  (ensures package import)
    )

    base, _ = make_spectrum(a_ppm=300, b_ppm=100, thickness_cm=1.0, noise=0.0, seed=0)
    wn, per_cm = base.initial_X, base.initial_Y
    thickness = np.full((ny, nx), 0.05)
    thickness[:, -2:] = 0.5  # thick strip (e.g. mount)
    thickness[0, :] *= 0.3  # partial aperture coverage on the top row
    rng = np.random.default_rng(0)
    spectra = thickness[..., None] * per_cm + rng.normal(0, 0.0005, (ny, nx, wn.size))
    spectra[:, 0] = 0.0  # off-sample column on the left
    ds = xr.Dataset(
        {"spectra": (("y", "x", "wn"), spectra.astype(np.float32))},
        coords={"y": np.arange(ny) * 25.0, "x": np.arange(nx) * 25.0, "wn": wn},
    )
    return ds, thickness


def test_survey_thickness_ignores_rim_and_thick_strip():
    from diamond_ftir_package.maps import (
        THICKNESS_EDGE,
        THICKNESS_OUTLIER,
        THICKNESS_REFERENCE,
    )

    data, _ = _plate_with_rim_and_thick_strip()
    mp = MapParams(
        n_jobs=1,
        thickness_mode="survey",
        thickness_field="constant",
        survey_step=1,  # so the thick strip is surveyed and must be rejected
        edge_width_px=1,
    )
    result = process_map(data, _params(), mp)
    field = json_safe_field(result)
    source = result["thickness_source"].values
    ref = result["thickness_reference"].values

    interior = np.isfinite(ref)
    # the field comes from the plate, not the strip or the rim
    assert np.nanmedian(ref) == pytest.approx(
        np.median(result["fitted_typeIIA_ratio"].values[5:8, 3:8]), rel=0.02
    )
    assert field["points_rejected"] >= 1
    # rim/edge pixels keep their own fit
    assert (source[0, 1:] == THICKNESS_EDGE).all()
    # the thick strip is recognised as an outlier and keeps its own (much larger) thickness
    strip = source[3:-2, -2]
    assert (strip == THICKNESS_OUTLIER).all()
    assert (source[4:8, 3:8] == THICKNESS_REFERENCE).all()
    assert interior.any()


def json_safe_field(result):
    import ast

    return ast.literal_eval(result.attrs["calibration"])["thickness_field"]


def test_stopping_a_parallel_run_leaves_no_worker_processes():
    """A progress callback that raises (the app's Stop / quit) must end the real workers."""
    import multiprocessing

    from diamond_ftir_package.maps import _make_tasks, _run_blocks

    data = _synthetic_map()
    coords = [(i, j) for i in range(3) for j in range(4)]
    tasks = _make_tasks(
        data.wn.values, data.spectra.values, coords, _params(), None, None, 1
    )

    class Stop(Exception):
        pass

    def progress(done, total):
        raise Stop

    with pytest.raises(Stop):
        _run_blocks(tasks, 2, progress, len(coords))
    assert multiprocessing.active_children() == []


def test_unmeasured_pixels_are_masked_on_load():
    from diamond_ftir_package.workbench.data import mask_invalid_pixels

    data = _synthetic_map()
    data.spectra.values[2, 1:] = 4.6e33  # junk left in a map that was stopped early
    mask_invalid_pixels(data)
    assert data.attrs["invalid_pixels"] == 3
    assert (
        np.isnan(data.spectra.values[2, 1:]).all()
        and np.isfinite(data.spectra.values[:2]).all()
    )
