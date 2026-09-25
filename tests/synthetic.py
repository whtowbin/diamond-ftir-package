"""Build synthetic diamond spectra with known composition for testing.

The spectrum is a thickness-scaled sum of the type IIa reference, the CAXBDY nitrogen
component spectra, and Gaussian peaks for hydrogen (3107 / 3085) and the platelet (1365).
"""

import numpy as np

from diamond_ftir_package import Diamond_Spectrum
from diamond_ftir_package.DiamondSpectrum import (
    CAXBDY,
    C_center_wn_spacing_correction,
    typeIIA_Spectrum,
)

A_PPM_PER_COMP = 16.5
B_PPM_PER_COMP = 79.4
C_PPM_PER_COMP = 0.624332796


def gaussian(x, center, height, fwhm):
    sigma = fwhm / 2.3548
    return height * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def make_spectrum(
    a_ppm=0.0,
    b_ppm=0.0,
    c_ppm=0.0,
    thickness_cm=0.05,
    spacing=1.0,
    h3107_height=0.0,
    platelet_height=0.0,
    noise=0.0,
    seed=0,
):
    """Return (Diamond_Spectrum, truth dict). Heights are in 1 cm-equivalent absorbance."""
    x = np.arange(500.0, 6500.0, spacing)
    ref = np.interp(x, typeIIA_Spectrum.X, typeIIA_Spectrum.Y)

    comps = np.zeros_like(x)
    grid = CAXBDY.index.to_numpy()
    c_corr = C_center_wn_spacing_correction(2 * spacing)  # resolution ~ 2 x spacing
    amounts = {
        "A": a_ppm / A_PPM_PER_COMP,
        "B": b_ppm / B_PPM_PER_COMP,
        "C": c_ppm / (C_PPM_PER_COMP * c_corr),
    }
    for name, amount in amounts.items():
        comps += np.interp(x, grid, CAXBDY[name].to_numpy(), right=0.0) * amount

    per_cm = ref + comps
    per_cm += gaussian(x, 3107, h3107_height, 6.0)
    per_cm += gaussian(x, 1365, platelet_height, 12.0)

    rng = np.random.default_rng(seed)
    y = thickness_cm * per_cm + rng.normal(0, noise, x.size)
    spectrum = Diamond_Spectrum(X=x, Y=y, X_Unit="Wavenumber", Y_Unit="Absorbance")
    truth = {
        "a_ppm": a_ppm,
        "b_ppm": b_ppm,
        "c_ppm": c_ppm,
        "thickness_cm": thickness_cm,
    }
    return spectrum, truth
