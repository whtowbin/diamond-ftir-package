"""Shared plot of one analysed spectrum: data + baseline, and the nitrogen fit.

Used by the GUI and by map example plots so both show the same picture.
"""

from __future__ import annotations

from typing import Any


def plot_spectrum_fit(figure: Any, spectrum: Any, title: str) -> None:
    """Draw onto an empty matplotlib ``figure``; needs ``fit_baseline`` to have run."""
    top = figure.add_subplot(211)
    top.plot(spectrum.X, spectrum.Y, label="Spectrum", lw=1)
    top.plot(spectrum.X, spectrum.baseline, label="Fitted baseline", lw=1)
    top.set_title(title)
    top.set_xlabel("Wavenumber (cm⁻¹)")
    top.set_ylabel("Absorbance")
    top.legend()

    fit = getattr(spectrum, "nitrogen_plot_fit_params", None)
    if fit is None:
        return
    bottom = figure.add_subplot(212)
    wn = fit["wn_array"]
    components = fit["fit_component_df"] * fit["fit_params"]
    bottom.plot(wn, fit["spec_intensity"], label="Normalised spectrum")
    bottom.plot(wn, components.sum(axis=1), label="Fit", linestyle="--")
    for label in ("A", "B", "C", "D"):
        bottom.plot(wn, components[label], label=label, linewidth=1)
    bottom.set_xlabel("Wavenumber (cm⁻¹)")
    bottom.set_ylabel("Absorption coefficient (cm⁻¹)")
    bottom.legend(ncol=5)
