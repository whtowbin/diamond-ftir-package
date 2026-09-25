"""Compare baseline methods on synthetic spectra with known thickness and nitrogen.

Run with: uv run python scripts/baseline_benchmark.py
Prints fitted thickness / true thickness and total N (truth: 1.00 and 500 ppm) for each
baseline variant and background shape. Synthetic backgrounds are simple stand-ins for real
ones, so use this to compare methods, not to certify absolute accuracy.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diamond_ftir_package import Diamond_Spectrum
from diamond_ftir_package.params import AnalysisParams
from diamond_ftir_package.pipeline import run_analysis
from tests.synthetic import make_spectrum

BACKGROUNDS = {
    "none": lambda x: 0 * x,
    "offset+slope": lambda x: 0.2 + 3e-5 * (x - 600),
    "curved": lambda x: 0.1 + 4e-8 * (x - 3000) ** 2,
    "scatter": lambda x: 0.05 + 0.4 * (x / 6000) ** 3,
}
VARIANTS = {
    "multistage": {},
    "multi, no rubber": {"pre_use_rubberband": False},
    "single": {"baseline_method": "single"},
    "single, optimize": {"baseline_method": "single", "baseline_search": "optimize"},
}


def run(thickness: float, background: str, settings: dict) -> dict:
    s, _ = make_spectrum(a_ppm=200, b_ppm=300, thickness_cm=thickness, noise=0.0005)
    x = s.initial_X
    spectrum = Diamond_Spectrum(X=x, Y=s.initial_Y + BACKGROUNDS[background](x))
    params = AnalysisParams(run_hydrogen=False, run_platelets=False)
    for key, value in settings.items():
        setattr(params.diamond, key, value)
    return run_analysis(spectrum, params)


def main() -> None:
    print(
        f"{'background':13s} {'thick':>5s} " + " ".join(f"{v:>18s}" for v in VARIANTS)
    )
    for background in BACKGROUNDS:
        for thickness in (0.03, 0.3):
            cells = []
            for settings in VARIANTS.values():
                try:
                    r = run(thickness, background, settings)
                    cells.append(
                        f"{r['typeIIA_ratio'] / thickness:5.2f} N{r['Total_N ppm']:6.0f}"
                    )
                except Exception as e:  # noqa: BLE001
                    cells.append(f"ERR {type(e).__name__}"[:18])
            print(
                f"{background:13s} {thickness:5} "
                + " ".join(f"{c:>18s}" for c in cells)
            )
    print("truth: thickness ratio 1.00, N 500 ppm")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
