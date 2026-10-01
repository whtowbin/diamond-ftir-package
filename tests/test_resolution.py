"""C-centre factor must use the instrument resolution, never the 1 cm-1 analysis grid."""

import numpy as np
import pytest

from diamond_ftir_package import Diamond_Spectrum
from diamond_ftir_package.DiamondSpectrum import (
    C_center_resolution_correction,
    resolution_from_name,
)
from diamond_ftir_package.params import AnalysisParams, NitrogenParams
from diamond_ftir_package.pipeline import run_analysis

from .synthetic import make_spectrum


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("CBP-0224_50umApt_25umStep_8scans_4wnRes_4-3-25_.map", 4.0),
        ("pre_expt_128sc_2res_200x200ap.map", 2.0),
        ("sample_0.5cm-1res.spa", 0.5),
        ("VP33_map.map", None),
        ("CQ6PS.SPA", None),
    ],
)
def test_resolution_from_file_name(name, expected):
    assert resolution_from_name(name) == expected


def test_liggins_table_values_and_extrapolation():
    assert [C_center_resolution_correction(r) for r in (0.5, 1, 2, 4)] == [
        30,
        37,
        42,
        65,
    ]
    assert C_center_resolution_correction(3) == pytest.approx(53.5)
    assert C_center_resolution_correction(8) == pytest.approx(65 + 4 * 11.5)


def _spectrum(spacing, metadata=None, c_ppm=100.0):
    s, _ = make_spectrum(c_ppm=c_ppm, a_ppm=10, noise=0.0, spacing=spacing)
    return Diamond_Spectrum(X=s.initial_X, Y=s.initial_Y, metadata=metadata)


def test_resolution_sources_in_priority_order():
    s = _spectrum(0.482, {"Filename": "x_4wnRes_.csv"})
    assert s.instrument_resolution(2.0) == (2.0, "setting")
    assert s.instrument_resolution() == (4.0, "file name")
    s = _spectrum(0.482, {"Filename": "CQ6PS.csv"})
    res, source = s.instrument_resolution()
    assert (
        res == pytest.approx(2 * 0.482, rel=1e-3)
        and source == "2 x point spacing (assumed)"
    )


def test_interpolation_does_not_change_the_resolution_used():
    """Coarse (1.93) and fine (0.48) original grids both end on the 1 cm-1 grid; the factor
    must follow the original data, not the analysis grid."""
    for spacing in (1.93, 0.48):
        s = _spectrum(spacing)
        assert np.allclose(np.diff(s.X), 1.0)  # analysis grid
        res, _ = s.instrument_resolution()
        assert res == pytest.approx(
            2 * spacing, rel=1e-3
        )  # Nyquist, from the original grid


def test_c_ppm_scales_with_declared_resolution_and_is_flagged_when_assumed():
    params = AnalysisParams(run_hydrogen=False, run_platelets=False)
    assumed = run_analysis(_spectrum(1.0), params)
    assert "resolution" in assumed["QA"]
    declared = AnalysisParams(run_hydrogen=False, run_platelets=False)
    declared.nitrogen = NitrogenParams(resolution_cm=4.0)
    row = run_analysis(_spectrum(1.0), declared)
    assert row["resolution_source"] == "setting"
    ratio = row["C_Nitrogen ppm"] / assumed["C_Nitrogen ppm"]
    assert ratio == pytest.approx(
        65 / 42, rel=0.02
    )  # assumed: 2 x 1.0 spacing = 2 cm-1
    assert "resolution" not in row["QA"]


def test_map_takes_resolution_from_its_file_name():
    from diamond_ftir_package.maps import process_map
    from diamond_ftir_package.params import MapParams

    from .test_maps import _synthetic_map

    data = _synthetic_map()
    data.attrs["source_file"] = "plate_8scans_4wnRes_.map"
    result = process_map(
        data,
        AnalysisParams(run_hydrogen=False, run_platelets=False),
        MapParams(n_jobs=1),
    )
    ok = result.status.values == 0
    assert np.allclose(result["resolution_cm"].values[ok], 4.0)


def test_coarse_melee_resolution_is_flagged_as_extrapolated():
    params = AnalysisParams(run_hydrogen=False, run_platelets=False)
    params.nitrogen = NitrogenParams(resolution_cm=16.0)
    row = run_analysis(_spectrum(1.0), params)
    assert "extrapolated beyond 4" in row["QA"]


def test_cubic_spline_is_the_default_resampler_and_others_are_available():
    from diamond_ftir_package import Spectrum

    x = np.arange(0.0, 20.0, 0.48)
    s = Spectrum(X=x, Y=np.sin(x))
    grid_default = s.interpolate(1, 18, 1)
    grid_cubic = s.interpolate(1, 18, 1, method="cubic")
    np.testing.assert_array_equal(grid_default.Y, grid_cubic.Y)
    for m, tol in (
        ("akima", 0.02),
        ("makima", 0.02),
        ("pchip", 0.02),
        ("linear", 0.03),
    ):
        got = s.interpolate(1, 18, 1, method=m).Y
        assert np.allclose(got, np.sin(np.arange(1, 18, 1.0)), atol=tol), m
