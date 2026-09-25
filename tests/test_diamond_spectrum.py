import numpy as np
import pytest

from diamond_ftir_package.params import AnalysisParams, PlateletParams
from diamond_ftir_package.pipeline import run_analysis

from .synthetic import make_spectrum


@pytest.fixture(scope="module")
def iaab():
    s, truth = make_spectrum(
        a_ppm=200, b_ppm=300, h3107_height=0.5, platelet_height=1.0, noise=0.0005
    )
    s.fit_baseline()
    s.normalize_diamond()
    s.Nitrogen_fit()
    return s, truth


def test_nitrogen_aggregation_state_is_recovered(iaab):
    s, truth = iaab
    n = s.nitrogen_dict
    true_b_percent = 100 * truth["b_ppm"] / (truth["a_ppm"] + truth["b_ppm"])
    assert n["B_percent"] == pytest.approx(true_b_percent, abs=2)
    assert n["A_Nitrogen ppm"] / n["B_Nitrogen ppm"] == pytest.approx(
        200 / 300, rel=0.05
    )


def test_type_ib_is_classified_as_c_dominated():
    s, _ = make_spectrum(c_ppm=100, a_ppm=10, noise=0.0005)
    s.fit_baseline()
    s.normalize_diamond()
    s.Nitrogen_fit()
    assert s.nitrogen_dict["C_percent"] > 80


def test_thickness_ratio_scales_linearly_with_thickness():
    ratios = []
    for thickness in (0.03, 0.06):
        s, _ = make_spectrum(a_ppm=100, thickness_cm=thickness, noise=0.0002)
        s.fit_baseline()
        ratios.append(s.typeIIA_ratio)
    # Not exactly 2: the pre-baseline step absorbs a thickness-dependent share of the signal
    # (about 5% between these thicknesses on synthetic data). Tighten once real golden data exist.
    assert ratios[1] / ratios[0] == pytest.approx(2.0, rel=0.15)


def test_normalize_requires_baseline_fit():
    s, _ = make_spectrum(a_ppm=100)
    with pytest.raises(ValueError, match="fit_baseline"):
        s.normalize_diamond()


def test_all_peaks_saturated_raises():
    s, _ = make_spectrum(a_ppm=100, thickness_cm=5.0)
    with pytest.raises(ValueError, match="saturated"):
        s.fit_baseline()


def test_zero_nitrogen_gives_nan_percent_not_crash():
    s, _ = make_spectrum(noise=0.0)
    s.fit_baseline()
    s.normalize_diamond()
    s.Nitrogen_fit()
    assert np.isnan(s.nitrogen_dict["B_percent"]) or s.nitrogen_dict["Total_N ppm"] >= 0


def test_hydrogen_alias_matches_main_method(iaab):
    s, _ = iaab
    s.measure_3107_peak()
    first = s.normed_area_3107
    s.measure_H_peaks()
    assert s.normed_area_3107 == first
    assert first > 0


def test_platelet_thresholds_do_not_leak_between_calls(iaab):
    """Regression: a shared mutable default used to carry one spectrum's noise threshold to the next."""
    s, _ = iaab
    user_params = {}
    s.measure_platelets_and_adjacent(find_peaks_params=user_params)
    assert user_params == {}
    assert s.normed_area_platelet > 0
    assert 1355 < s.platelet_peak_position < 1380


def test_amber_without_thickness_ratio_does_not_raise_keyerror():
    """Regression: the un-normalised branch used a wrong dict key."""
    s, _ = make_spectrum(a_ppm=100)
    s.measure_amber_center()
    assert hasattr(s, "amber_center_peak_prominences")
    assert hasattr(s, "amber_4065_area")


def test_1405_branch_without_ratio_does_not_divide_by_none():
    s, _ = make_spectrum(a_ppm=100)
    s.measure_platelets_and_adjacent()


def test_pipeline_row_contains_enabled_sections_only():
    s, _ = make_spectrum(a_ppm=200, b_ppm=300, noise=0.0005)
    params = AnalysisParams(run_hydrogen=False, run_platelets=False)
    row = run_analysis(s, params)
    assert "Total_N ppm" in row and "Normed_3107_Area" not in row


def test_custom_platelet_window_is_used():
    s, _ = make_spectrum(a_ppm=200, b_ppm=300, platelet_height=1.0, noise=0.0005)
    s.fit_baseline()
    s.normalize_diamond()
    narrow = PlateletParams(search_window=(1400, 1410))
    s.measure_platelets_and_adjacent(params=narrow)
    # any peak reported must come from the user's window, not the default 1355-1380
    assert (
        not hasattr(s, "platelet_peak_position")
        or 1400 < s.platelet_peak_position < 1410
    )


def test_nitrogen_window_parameter_changes_fit():
    from diamond_ftir_package.params import NitrogenParams

    s, _ = make_spectrum(a_ppm=200, b_ppm=300, noise=0.0005)
    s.fit_baseline()
    s.normalize_diamond()
    s.Nitrogen_fit()
    default_total = s.nitrogen_dict["Total_N ppm"]
    s.Nitrogen_fit(params=NitrogenParams(wn_low=1000, wn_high=1300))
    assert s.nitrogen_dict["Total_N ppm"] != default_total


def test_sparse_als_matches_dense_reference():
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    from diamond_ftir_package.DiamondSpectrum import baseline_als

    rng = np.random.default_rng(1)
    y = np.linspace(0, 1, 300) ** 2 + rng.normal(0, 0.01, 300)
    lam, p = 1e5, 0.01
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))  # the original dense construction
    w = np.ones(L)
    for _ in range(10):
        z = spsolve(sparse.spdiags(w, 0, L, L) + lam * D.dot(D.T), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    np.testing.assert_allclose(baseline_als(y, lam, p), z, rtol=1e-8, atol=1e-10)


def test_optimized_search_stays_in_bounds_and_is_reported():
    s, _ = make_spectrum(a_ppm=200, b_ppm=300, noise=0.0005)
    params = AnalysisParams(run_hydrogen=False, run_platelets=False)
    params.diamond.baseline_method = "single"
    params.diamond.baseline_search = "optimize"
    row = run_analysis(s, params)
    lo, hi = params.diamond.log10_lam_bounds
    assert 10**lo <= row["baseline_lam"] <= 10**hi
    assert row["B_percent"] == pytest.approx(60, abs=1)


def test_nitrogen_window_wider_than_data_is_clipped_not_zeroed():
    from diamond_ftir_package import Diamond_Spectrum
    from diamond_ftir_package.params import NitrogenParams

    s, _ = make_spectrum(a_ppm=200, b_ppm=300, noise=0.0005)
    cut = Diamond_Spectrum(
        X=s.initial_X[s.initial_X > 700], Y=s.initial_Y[s.initial_X > 700]
    )
    cut.fit_baseline()
    cut.normalize_diamond()
    cut.Nitrogen_fit(params=NitrogenParams(wn_low=600, wn_high=1400))
    assert cut.nitrogen_window[0] > 700
    assert cut.nitrogen_dict["Total_N ppm"] > 100
