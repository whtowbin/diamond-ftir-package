"""User-tunable analysis parameters.

Every default equals the value that used to be hardcoded in ``DiamondSpectrum.py``,
so calling code that passes no parameters behaves exactly as before. The GUI and CLI
build a :class:`AnalysisParams` from user input and hand it to the pipeline.

Each field carries a short ``description`` in its metadata that the GUI shows as a
tooltip. Literature sources are marked ``[CITATION NEEDED: ...]`` until filled in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

BASELINE_ALGORITHMS = ("Whittaker", "ALS")

# Settings with a fixed set of values; the GUI shows these as drop-down lists.
CHOICES: dict[str, tuple[str, ...]] = {
    "baseline_algorithm": BASELINE_ALGORITHMS,
    "baseline_method": ("joint", "multistage", "single"),
    "baseline_search": ("fixed", "optimize", "grid"),
    "baseline_strategy": ("per_pixel", "calibrate_fixed", "calibrate_local"),
    "thickness_mode": ("per_pixel", "survey", "known"),
    "thickness_field": ("constant", "plane"),
}


def _p(default: Any, description: str) -> Any:
    return field(default=default, metadata={"description": description})


@dataclass
class DiamondFitParams:
    """Thickness normalisation against the type IIa reference spectrum."""

    saturation_cutoff: float = _p(
        2.5, "Absorbance above which a diamond peak counts as saturated."
    )
    stdev_cut_off: float = _p(
        0.5, "Noise (std. dev.) above which a diamond peak counts as saturated."
    )
    baseline_algorithm: str = _p(
        "Whittaker",
        "Baseline algorithm: 'Whittaker' (fast) or 'ALS' (slower, more stable).",
    )
    baseline_method: str = _p(
        "joint",
        "'joint': ASLS baseline and type IIa thickness fitted together (least biased in "
        "injection tests). 'multistage': median filter, mild ASLS and rubber band before the "
        "main baseline. 'single': one ASLS baseline only (the original mapping method).",
    )
    joint_lam: float = _p(
        1e9, "'joint' only: baseline smoothness (lambda) on the 1 cm-1 grid."
    )
    joint_p: float = _p(1e-3, "'joint' only: baseline asymmetry (p).")
    min_diamond_r2: float = _p(
        0.8,
        "Quality flag: spectra whose type IIa shape match (R²) is below this are flagged as "
        "not diamond-like. Flagged, never excluded.",
    )
    baseline_search: str = _p(
        "fixed",
        "'fixed': use lam and p as given. 'optimize': local search for lam and p so the "
        "diamond peaks best match the type IIa shape and 4000-5900 cm-1 is flat. 'grid': coarse "
        "grid over the bounds, then a local search (slower, avoids getting stuck).",
    )
    lam: float = _p(
        1e10, "Main baseline smoothness (lambda); the starting point when optimising."
    )
    p: float = _p(
        5e-6, "Main baseline asymmetry (p); the starting point when optimising."
    )
    log10_lam_bounds: tuple[float, float] = _p(
        (5, 12), "Search range for log10(lambda) when optimising."
    )
    log10_p_bounds: tuple[float, float] = _p(
        (-8, -2), "Search range for log10(p) when optimising."
    )
    search_width: float = _p(
        1.0,
        "When a start point is supplied (maps: the sample's typical lam and p), search only this "
        "many decades either side of it.",
    )
    search_tolerance: float = _p(
        0.02, "Stop the search when log10(lam, p) move less than this."
    )
    search_max_evals: int = _p(
        200, "Maximum baseline evaluations per spectrum when optimising."
    )
    grid_points: int = _p(
        7, "'grid' search: points per axis of the coarse grid (n x n evaluations)."
    )
    flat_range: tuple[float, float] = _p(
        (4000, 5900), "Region (cm-1) that should be flat after baseline removal."
    )
    flat_weight: float = _p(
        0.5,
        "Weight of the flatness term against the type IIa shape term in the search.",
    )
    pre_median_filter: int = _p(21, "Multistage only: median filter width (points).")
    pre_lam: float = _p(
        1e10, "Multistage only: smoothness of the mild first ASLS baseline."
    )
    pre_p: float = _p(
        5e-4, "Multistage only: asymmetry of the mild first ASLS baseline."
    )
    pre_use_rubberband: bool = _p(
        True,
        "Multistage only: subtract the stretched rubber band. On synthetic tests it causes most "
        "of the thickness under-estimate (see docs/baseline_benchmark.md).",
    )
    pre_rubber_stretch: float = _p(
        2e-8, "Multistage only: curvature added for the rubber band."
    )


@dataclass
class NitrogenParams:
    """CAXBDY component fit of the one-phonon region."""

    wn_low: float = _p(950, "Lower wavenumber (cm-1) of the nitrogen fit window.")
    wn_high: float = _p(1350, "Upper wavenumber (cm-1) of the nitrogen fit window.")
    max_C_or_B: float = _p(
        0.01,
        "Max size of the minor component (C in IaAB, B in Ib) relative to the major one.",
    )
    # [CITATION NEEDED: A-centre and B-centre absorption coefficients]
    a_ppm_per_cm: float = _p(16.5, "ppm N per cm-1 of A-component absorption.")
    b_ppm_per_cm: float = _p(79.4, "ppm N per cm-1 of B-component absorption.")
    # [CITATION NEEDED: C-centre absorption coefficient]
    c_ppm_per_cm: float = _p(
        0.624332796,
        "ppm N per cm-1 of C-component absorption (before resolution correction).",
    )
    # [CITATION NEEDED: D-component limit relative to B (Woods correlation)]
    d_limit: float = _p(
        0.435, "Max D-component size relative to the major component in type IaAB fits."
    )


@dataclass
class HydrogenParams:
    """The 3107 and 3085 cm-1 hydrogen peaks."""

    # [CITATION NEEDED: 3107 cm-1 N3VH assignment]
    window: tuple[float, float] = _p(
        (3060, 3180), "Region used for local baseline fitting."
    )
    median_filter: int = _p(21, "Median filter width (points) before baseline fitting.")
    baseline_lam: float = _p(0.1, "ASLS smoothness (lambda) for the local baseline.")
    baseline_p: float = _p(6e-6, "ASLS asymmetry (p) for the local baseline.")
    peak_3107: tuple[float, float] = _p(
        (3103, 3110), "Integration limits for the 3107 peak."
    )
    peak_3085: tuple[float, float] = _p(
        (3082, 3088), "Integration limits for the 3085 peak."
    )


@dataclass
class PlateletParams:
    """Platelet (B') peak near 1365 cm-1 and the 1405 cm-1 peak."""

    # [CITATION NEEDED: platelet peak position and 1405 cm-1 assignment]
    region: tuple[float, float] = _p((1340, 1500), "Region used for baseline fitting.")
    search_window: tuple[float, float] = _p(
        (1355, 1380), "Where to look for the platelet peak."
    )
    noise_window: tuple[float, float] = _p(
        (1380, 1450), "Region used to estimate noise for peak thresholds."
    )
    baseline_lam: float = _p(1000, "ASLS smoothness (lambda) for the first baseline.")
    baseline_p: float = _p(0.001, "ASLS asymmetry (p) for the first baseline.")
    peak_1405: tuple[float, float] = _p(
        (1403, 1407), "Integration limits for the 1405 peak."
    )


@dataclass
class AmberParams:
    """Amber-centre bands, listed as (label, centre, half-width) in cm-1.

    The label names the result attribute (``amber_<label>_area_normed``) and is kept
    for backwards compatibility even where it differs slightly from the centre.
    """

    # [CITATION NEEDED: amber-centre band positions]
    bands: tuple[tuple[int, float, float], ...] = _p(
        (
            (4065, 4060, 10),
            (4165, 4160, 10),
            (4211, 4211, 10),
            (4354, 4354, 10),
            (4495, 4495, 5),
            (4660, 4660, 20),
            (4740, 4740, 20),
            (4850, 4850, 5),
            (4950, 4950, 20),
        ),
        "Amber-centre bands to integrate as (label, centre, half-width) in cm-1.",
    )


@dataclass
class MapParams:
    """How a map is processed (on top of the per-spectrum AnalysisParams)."""

    min_diamond_absorbance: float = _p(
        0.05,
        "Pixels whose mean absorbance in mask_window is below this are treated as off the "
        "sample and skipped.",
    )
    mask_window: tuple[float, float] = _p(
        (1970, 2040),
        "Diamond two-phonon band (cm-1) used to decide whether a pixel is on sample.",
    )
    baseline_strategy: str = _p(
        "calibrate_local",
        "Only used when baseline_search is not 'fixed'. 'per_pixel': full search at every "
        "pixel (slowest). 'calibrate_fixed': search a sample of pixels, then use their median "
        "lam and p everywhere (fastest). 'calibrate_local': start each pixel's search at that "
        "median and search only local_search_width decades around it.",
    )
    local_search_width: float = _p(
        0.5,
        "'calibrate_local': decades either side of the calibrated lam and p to search.",
    )
    survey_step: int = _p(
        4,
        "Coarse survey: analyse every n-th interior pixel first, to find the typical baseline "
        "settings and the thickness field.",
    )
    survey_max_pixels: int = _p(400, "Upper limit on the number of survey pixels.")
    edge_width_px: int = _p(
        3,
        "Pixels this close to the sample edge (or map border) are left out of the survey and "
        "keep their own fitted thickness: the aperture only partly covers the sample there.",
    )
    thickness_mode: str = _p(
        "per_pixel",
        "'per_pixel': thickness from each pixel's own diamond peaks. 'survey': a smooth "
        "thickness field fitted to the coarse survey (suits polished plates); edge pixels and "
        "pixels far from the field keep their own fit. 'known': use known_thickness_um.",
    )
    thickness_field: str = _p(
        "plane", "'survey' mode: 'constant' (flat plate) or 'plane' (allows a wedge)."
    )
    outlier_mad: float = _p(
        3.0,
        "'survey' mode: survey points further than this many robust standard deviations from "
        "the field are ignored, and pixels that far keep their own thickness.",
    )
    known_thickness_um: float = _p(
        0.0, "Measured sample thickness (µm) for thickness_mode='known'."
    )
    n_jobs: int = _p(
        0, "Parallel worker processes: 0 = all cores but one, 1 = no parallelism."
    )
    block_size: int = _p(64, "Pixels sent to a worker at a time.")
    example_grid: tuple[int, int] = _p(
        (3, 3), "Rows x columns of example pixels to plot for checking fits."
    )
    seed: int = _p(
        0, "Random seed for choosing calibration pixels (keeps runs reproducible)."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MapParams:
        return cls(**{k: _restore(cls, k, v) for k, v in data.items()})


@dataclass
class AnalysisParams:
    """All settings for a full analysis run, with a switch per measurement."""

    diamond: DiamondFitParams = field(default_factory=DiamondFitParams)
    nitrogen: NitrogenParams = field(default_factory=NitrogenParams)
    hydrogen: HydrogenParams = field(default_factory=HydrogenParams)
    platelet: PlateletParams = field(default_factory=PlateletParams)
    amber: AmberParams = field(default_factory=AmberParams)
    run_nitrogen: bool = True
    run_hydrogen: bool = True
    run_platelets: bool = True
    run_amber: bool = False
    recipes: tuple[str, ...] = field(
        default=(),
        metadata={
            "description": "Extra analysis recipes to run: built-in names (see "
            "core.builtin_recipes) or paths to .toml/.json recipe files."
        },
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisParams:
        """Build from a (possibly partial) nested dict, e.g. loaded settings JSON."""
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data[f.name]
            if isinstance(value, dict):
                sub_type = type(getattr(cls(), f.name))
                value = sub_type(
                    **{k: _restore(sub_type, k, v) for k, v in value.items()}
                )
            elif isinstance(value, list):
                value = tuple(
                    value
                )  # JSON lists back to the tuple fields (e.g. recipes)
            kwargs[f.name] = value
        return cls(**kwargs)


def _restore(sub_type: type, name: str, value: Any) -> Any:
    """JSON turns tuples into lists; put them back so equality with defaults holds."""
    default = getattr(sub_type(), name)
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(tuple(v) if isinstance(v, list) else v for v in value)
    return value
