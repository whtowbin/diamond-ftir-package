"""Figures for docs/methods_validation.md, built from synthetic spectra only.

    uv run python scripts/make_method_figures.py      # writes docs/figures/method_*.png

Every spectrum here is synthetic (type IIa reference x thickness + CAXBDY nitrogen
components + chosen background, peaks and noise), so every quantity is known and no real
sample data is published.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diamond_ftir_package import Diamond_Spectrum
from diamond_ftir_package.DiamondSpectrum import CAXBDY, typeIIA_Spectrum
from diamond_ftir_package.params import AnalysisParams, NitrogenParams
from diamond_ftir_package.pipeline import run_analysis
from tests.synthetic import make_spectrum

OUT = ROOT / "docs" / "figures"
METHOD_LABELS = {
    "joint": "joint ASLS + type IIa",
    "multistage": "multistage (old default)",
    "single_opt": "single ASLS, searched",
}
COLORS = {"joint": "#1b7837", "multistage": "#b2182b", "single_opt": "#2166ac"}


def gauss(x, c, fwhm):
    return np.exp(-0.5 * ((x - c) / (fwhm / 2.3548)) ** 2)


def params(kind: str) -> AnalysisParams:
    p = AnalysisParams(run_hydrogen=False, run_platelets=False)
    if kind == "joint":
        p.diamond.baseline_method = "joint"
    elif kind == "multistage":
        p.diamond.baseline_method = "multistage"  # explicit: no longer the default
    elif kind == "single_opt":
        p.diamond.baseline_method, p.diamond.baseline_search = "single", "optimize"
    return p


def nitrogen_absorption(x, a_ppm, b_ppm):
    a = np.interp(x, CAXBDY.index, CAXBDY["A"], left=0, right=0) * a_ppm / 16.5
    b = np.interp(x, CAXBDY.index, CAXBDY["B"], left=0, right=0) * b_ppm / 79.4
    return a + b


def example_stone(thickness=0.1, noise=0.002, seed=3):
    """Synthetic stone: IaAB nitrogen, a platelet peak, a sloped + curved background."""
    s, _ = make_spectrum(
        a_ppm=200,
        b_ppm=300,
        thickness_cm=thickness,
        platelet_height=4.0,
        noise=noise,
        seed=seed,
    )
    x = s.initial_X
    background = 0.25 + 4e-5 * (x - 600) + 3e-9 * (x - 3500) ** 2
    return x, s.initial_Y + background, background


def fig_joint_fit():
    x, y, true_bg = example_stone()
    s = Diamond_Spectrum(X=x, Y=y)
    run_analysis(s, params("joint"))
    t = s.typeIIA_ratio
    ref = s.interpolated_typeIIA_Spectrum.Y
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(s.X, s.Y, color="0.3", lw=0.8, label="measured (synthetic)")
    ax[0].plot(s.X, s.baseline, color="#1b7837", lw=2, label="fitted baseline b")
    ax[0].plot(
        s.X,
        s.baseline + t * ref,
        color="#fdae61",
        lw=1.2,
        ls="--",
        label="b + t × type IIa",
    )
    ax[0].plot(x, true_bg, color="k", lw=1, ls=":", label="true background")
    ax[0].set(
        xlim=(600, 6000),
        xlabel="Wavenumber (cm⁻¹)",
        ylabel="Absorbance",
        title=f"Step 1: joint fit of baseline and thickness\nfitted t = {t:.4f} cm (true 0.1000)",
    )
    ax[0].legend(fontsize=8)
    f = s.nitrogen_plot_fit_params
    comps = f["fit_component_df"] * f["fit_params"]
    wn = f["wn_array"]
    ax[1].plot(
        wn,
        f["spec_intensity"][: len(wn)],
        color="0.3",
        lw=1,
        label="(measured − b) / t",
    )
    ax[1].plot(wn, comps.sum(axis=1), color="k", ls="--", lw=1, label="fit")
    for c, col in (("A", "#2166ac"), ("B", "#b2182b"), ("D", "#999999")):
        ax[1].plot(wn, comps[c], color=col, lw=1, label=c)
    n = s.nitrogen_dict
    ax[1].set(
        xlabel="Wavenumber (cm⁻¹)",
        ylabel="Absorption coefficient (cm⁻¹)",
        title=f"Step 2: nitrogen fit, 950-1350 cm⁻¹\nA {n['A_Nitrogen ppm']:.0f} (true 200), "
        f"B {n['B_Nitrogen ppm']:.0f} (true 300) ppm",
    )
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "method_joint_fit.png", dpi=110)
    plt.close(fig)


def fig_injection():
    x, y, _ = example_stone(noise=0.004)
    dt = 0.03
    slab = dt * (
        np.interp(x, typeIIA_Spectrum.X, typeIIA_Spectrum.Y)
        + nitrogen_absorption(x, 150, 150)
    )
    y_inj = y + slab
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(x, y, color="0.3", lw=0.8, label="spectrum")
    ax[0].plot(x, y_inj, color="#d95f02", lw=0.8, label="spectrum + injected slab")
    ax[0].plot(
        x,
        slab,
        color="k",
        lw=1,
        label=f"injected: {dt} cm × (type IIa + 150 ppm A + 150 ppm B)",
    )
    ax[0].set(
        xlim=(600, 4000),
        xlabel="Wavenumber (cm⁻¹)",
        ylabel="Absorbance",
        title="Injection: add a known slab of diamond",
    )
    ax[0].legend(fontsize=8)
    rec = {}
    for kind, label in METHOD_LABELS.items():
        a, b = Diamond_Spectrum(X=x, Y=y), Diamond_Spectrum(X=x, Y=y_inj)
        run_analysis(a, params(kind))
        run_analysis(b, params(kind))
        rec[kind] = (b.typeIIA_ratio - a.typeIIA_ratio) / dt
        m = (a.X > 1600) & (a.X < 3000)
        ax[1].plot(
            a.X[m],
            a.baseline[m],
            color=COLORS[kind],
            lw=1.5,
            label=label,
        )
        ax[1].plot(b.X[m], b.baseline[m], color=COLORS[kind], lw=1.5, ls="--")
    m = (x > 1600) & (x < 3000)
    ax[1].plot(x[m], y[m], color="0.6", lw=0.6, label="spectrum")
    ax[1].set(
        xlabel="Wavenumber (cm⁻¹)",
        ylabel="Absorbance",
        title="Fitted baselines before (solid) and after (dashed) injection",
    )
    ax[1].legend(fontsize=8)
    kinds = list(rec)
    ax[2].bar(
        [METHOD_LABELS[k] for k in kinds],
        [rec[k] for k in kinds],
        color=[COLORS[k] for k in kinds],
    )
    ax[2].axhline(1, color="k", lw=1, ls=":")
    ax[2].set(
        ylabel="recovered Δt / injected Δt",
        title="Thickness recovery (1.00 = unbiased)",
    )
    ax[2].tick_params(axis="x", labelsize=8, rotation=10)
    for i, k in enumerate(kinds):
        ax[2].text(i, rec[k] + 0.02, f"{rec[k]:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "method_injection.png", dpi=110)
    plt.close(fig)


def fig_leakage():
    x, y, _ = example_stone()
    s = Diamond_Spectrum(X=x, Y=y)
    run_analysis(s, params("joint"))
    base = s.normalized_spectrum.Y.copy()
    X = s.normalized_spectrum.X
    features = {
        "platelet (1365 cm⁻¹)": gauss(X, 1365, 12),
        "residual background (800 cm⁻¹)": gauss(X, 800, 120),
    }
    windows = ["950-1350", "950-1400", "650-1350", "650-1400", "1000-1300"]

    def total_n(yn, w):
        lo, hi = (int(v) for v in w.split("-"))
        s.normalized_spectrum.Y = yn
        s.Nitrogen_fit(params=NitrogenParams(wn_low=lo, wn_high=hi))
        return s.nitrogen_dict["Total_N ppm"]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    m = (X > 600) & (X < 1500)
    ax[0].plot(X[m], base[m], color="0.3", lw=1, label="normalised spectrum")
    for (name, shape), col in zip(
        features.items(), ("#e7298a", "#7570b3"), strict=True
    ):
        ax[0].plot(X[m], base[m] + shape[m], color=col, lw=1, label=f"+ 1 cm⁻¹ {name}")
    for w, col, yr in (
        ("650-1400", "#b2182b", (0.0, 0.06)),
        ("950-1350", "#1b7837", (0.06, 0.12)),
    ):
        lo, hi = (int(v) for v in w.split("-"))
        ax[0].axvspan(
            lo, hi, ymin=yr[0], ymax=yr[1], color=col, alpha=0.6, label=f"window {w}"
        )
    ax[0].set(
        xlabel="Wavenumber (cm⁻¹)",
        ylabel="Absorption coefficient (cm⁻¹)",
        title="Leakage: add one unmodelled feature at a time",
    )
    ax[0].legend(fontsize=8)
    width = 0.38
    for k, ((name, shape), col) in enumerate(
        zip(features.items(), ("#e7298a", "#7570b3"), strict=True)
    ):
        leak = [total_n(base + shape, w) - total_n(base, w) for w in windows]
        ax[1].bar(
            np.arange(len(windows)) + (k - 0.5) * width,
            leak,
            width,
            color=col,
            label=name,
        )
    s.normalized_spectrum.Y = base
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(range(len(windows)), windows)
    ax[1].set(
        xlabel="nitrogen fit window (cm⁻¹)",
        ylabel="change in total N (ppm per 1 cm⁻¹ feature)",
        title="Bias each window picks up (0 = immune)",
    )
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "method_leakage.png", dpi=110)
    plt.close(fig)


def fig_saturation():
    s0, _ = make_spectrum(a_ppm=100, b_ppm=100, thickness_cm=0.35, noise=0.002)
    x = s0.initial_X
    y = np.minimum(s0.initial_Y + 0.2, 3.0) + np.where(
        s0.initial_Y + 0.2 > 3.0, np.random.default_rng(0).normal(0, 0.3, x.size), 0
    )
    s = Diamond_Spectrum(X=x, Y=y)
    mask = s.test_diamond_saturation(2.5, 0.5)
    run_analysis(s, params("joint"))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(s.X, s.Y, color="0.3", lw=0.7, label="measured (synthetic, 0.35 cm)")
    ax.fill_between(
        s.X,
        0,
        3.3,
        where=mask,
        color="#1b7837",
        alpha=0.15,
        label="unsaturated windows used for t",
    )
    sat = (s.X > 1800) & (s.X < 2700) & ~mask
    ax.fill_between(
        s.X, 0, 3.3, where=sat, color="#b2182b", alpha=0.12, label="saturated: weight 0"
    )
    ax.plot(
        s.X,
        s.baseline + s.typeIIA_ratio * s.interpolated_typeIIA_Spectrum.Y,
        color="#fdae61",
        lw=1,
        ls="--",
        label="baseline + t × type IIa",
    )
    ax.set(
        xlim=(1500, 4000),
        ylim=(0, 3.4),
        xlabel="Wavenumber (cm⁻¹)",
        ylabel="Absorbance",
        title=f"Saturated primary band: thickness from the unsaturated windows (t = {s.typeIIA_ratio:.3f}, true 0.350)",
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "method_saturation.png", dpi=110)
    plt.close(fig)


def fig_map_survey():
    from diamond_ftir_package.maps import process_map
    from diamond_ftir_package.params import MapParams
    from tests.test_maps import _plate_with_rim_and_thick_strip

    data, truth = _plate_with_rim_and_thick_strip(ny=16, nx=16)
    mp = MapParams(
        n_jobs=1,
        thickness_mode="survey",
        thickness_field="constant",
        survey_step=1,
        edge_width_px=1,
    )
    res = process_map(data, AnalysisParams(run_hydrogen=False, run_platelets=False), mp)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    im = ax[0].imshow(
        np.where(res.status.values == 0, truth, np.nan), origin="lower", cmap="viridis"
    )
    ax[0].set_title(
        "true thickness (synthetic plate):\nrim at 30% coverage, thick strip at right"
    )
    fig.colorbar(im, ax=ax[0], label="cm")
    own = res["fitted_typeIIA_ratio"].values
    own = np.where(
        np.isfinite(own), own, res["typeIIA_ratio"].values
    )  # edge pixels: own fit
    im = ax[1].imshow(own, origin="lower", cmap="viridis")
    ax[1].set_title("per-pixel fitted thickness")
    fig.colorbar(im, ax=ax[1], label="cm")
    src = res["thickness_source"].values
    cmap = matplotlib.colors.ListedColormap(["#1b7837", "#fdae61", "#b2182b"])
    ax[2].imshow(
        np.where(np.isfinite(src), src, np.nan),
        origin="lower",
        cmap=cmap,
        vmin=0.5,
        vmax=3.5,
    )
    ax[2].set_title(
        "thickness used: green = survey field,\norange = edge (own fit), red = outlier (own fit)"
    )
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "method_map_survey.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (
        fig_joint_fit,
        fig_injection,
        fig_leakage,
        fig_saturation,
        fig_map_survey,
    ):
        f()
        print("done", f.__name__)
