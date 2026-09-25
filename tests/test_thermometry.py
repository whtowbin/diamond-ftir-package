"""Thermometry: nitrogen aggregation and platelet degradation (Speich et al. 2018)."""

import numpy as np
import pytest

from diamond_ftir_package import thermometry as th
from diamond_ftir_package.params import AnalysisParams
from diamond_ftir_package.pipeline import run_analysis

from .synthetic import make_spectrum


def test_nitrogen_temperature_matches_vendored_snac():
    from diamond_ftir_package._vendor.snac.aggregation import Temp_N

    for nt, frac, age in [
        (625, 0.863, 3520.0),
        (801, 0.197, 1860.0),
        (2000, 0.49, 3250.0),
    ]:
        # SNAC converts with 273 K, this package with 273.15 K
        assert th.nitrogen_temperature(nt, frac, age) == pytest.approx(
            Temp_N(age, nt, frac) - 0.15, abs=1e-6
        )


def test_nitrogen_temperature_reproduces_paper_example():
    """Speich et al. 2018, Fig. 8: >2000 ppm N, 49% B, Diavik peridotitic (3.25 Ga): ~1090 °C."""
    assert th.nitrogen_temperature(2000, 0.49, 3250) == pytest.approx(1090, abs=5)


def test_platelet_thermometer_reproduces_paper_worked_example():
    """Speich et al. 2018: Pt ≈ 1, P0 ≈ 500 cm-2 at 1640 °C gives ca. 64 ka."""
    b_ppm = 500 / 64 * 79.4
    assert th.platelet_duration(1.0, b_ppm, 1640) * 1e3 == pytest.approx(64, abs=5)


def test_platelet_temperature_and_duration_are_inverse():
    t = th.platelet_temperature(150.0, 620.0, 1500.0)
    assert th.platelet_duration(150.0, 620.0, t) == pytest.approx(1500.0, rel=1e-9)


def test_platelet_temperature_undefined_without_degradation():
    p0 = th.platelet_initial_area(620.0)
    assert np.isnan(th.platelet_temperature(p0 * 1.05, 620.0, 1000))
    assert np.isnan(th.platelet_temperature(0.0, 620.0, 1000))
    assert np.isfinite(th.platelet_temperature(p0 * 0.5, 620.0, 1000))


def test_calibrations_differ_and_are_named():
    a = th.platelet_temperature(100, 620, 1000, "combined")
    b = th.platelet_temperature(100, 620, 1000, "natural")
    assert np.isfinite(a) and np.isfinite(b) and a != b


def test_regularity_measures():
    reg = th.platelet_regularity(
        platelet_area=320.0, platelet_position=(320 + 167800) / 123.3, b_ppm=794.0
    )
    assert reg["P0"] == pytest.approx(640.0)
    assert reg["remaining"] == pytest.approx(0.5)
    assert reg["position_deviation"] == pytest.approx(0.0, abs=1e-9)


def _spectrum_with_platelet(area=120.0, x0=1365.0, hwhm=(5.0, 4.0), fraction=0.6):
    """Normalised synthetic spectrum with an injected asymmetric platelet peak of known area."""
    s, _ = make_spectrum(a_ppm=200, b_ppm=300, h3107_height=0.5, noise=0.0)
    s.fit_baseline()
    s.normalize_diamond()
    x, y = s.normalized_spectrum.X, s.normalized_spectrum.Y.copy()
    height = area / th.asym_pseudo_voigt_area(1.0, *hwhm, fraction)
    y += th.asym_pseudo_voigt(x, x0, height, *hwhm, fraction)
    return x, y, height


@pytest.mark.parametrize("compatible", [True, False])
def test_platelet_fit_recovers_injected_peak(compatible):
    x, y, _ = _spectrum_with_platelet()
    fit = th.fit_platelet(x, y, th.fit_3107(x, y).height, quiddit_compatible=compatible)
    assert fit is not None
    assert fit.x0 == pytest.approx(1365.0, abs=0.5)
    assert fit.area == pytest.approx(120.0, rel=0.08)


def test_asym_pseudo_voigt_area_is_the_integral():
    """Integrate to infinity: Lorentzian tails hold a noticeable share of the area."""
    from scipy.integrate import quad

    def f(x):
        return float(th.asym_pseudo_voigt(np.array([x]), 1365, 2.0, 6.0, 3.0, 0.4)[0])

    total = quad(f, -np.inf, 1365)[0] + quad(f, 1365, np.inf)[0]
    assert total == pytest.approx(
        th.asym_pseudo_voigt_area(2.0, 6.0, 3.0, 0.4), rel=1e-6
    )


def test_pipeline_adds_temperatures_when_duration_given():
    s, _ = make_spectrum(
        a_ppm=200, b_ppm=300, platelet_height=3.0, h3107_height=0.5, noise=0.0005
    )
    params = AnalysisParams(run_platelets=False)
    params.thermometry.duration_ma = 1000.0
    row = run_analysis(s, params)
    assert np.isfinite(row["T_N (C)"])
    assert "platelet_area_qd" in row and "T_P (C)" in row
    off = run_analysis(
        make_spectrum(a_ppm=200, b_ppm=300)[0], AnalysisParams(run_platelets=False)
    )
    assert "T_N (C)" not in off


def test_snac_model_runs_on_example_diamond():
    model = th.snac_cooling_model(3520, 1860, 0, 625, 0.863, 801, 0.197, dt=5)
    assert model is not None
