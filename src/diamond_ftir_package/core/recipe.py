"""Analysis recipes: user-defined baselines, single peaks and peak clusters, as data.

A recipe is a list of *features*. Each feature has a region of the x axis, a list of baseline
steps applied in order inside that region, and one or more peaks. A feature with one peak is
a single peak; a feature with several peaks is a cluster, fitted jointly with optional
constraints between peak parameters.

Recipes are plain TOML (or JSON) files, so they can be written by hand, edited in the GUI,
shared, and saved with results. Nothing here is specific to diamond or to infrared: x can
be wavenumber, Raman shift or wavelength.

Example (TOML)::

    name = "hydrogen"
    normalize_by = "thickness"

    [[features]]
    name = "H"
    region = [3060, 3180]
    model = "integrate"

    [[features.baseline]]
    method = "asls"
    lam = 0.1
    p = 6e-6
    median_filter = 21

    [[features.peaks]]
    name = "3107"
    center = 3107
    integrate = [3103, 3110]
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

BASELINE_METHODS = ("asls", "rubberband", "polynomial", "linear", "none")
PEAK_MODELS = ("integrate", "gaussian", "lorentzian", "voigt", "pseudo_voigt")
NORMALISATIONS = ("thickness", "none")
OUTPUTS = ("area", "height", "center", "fwhm")


def _f(default: Any, description: str) -> Any:
    kwargs = {"metadata": {"description": description}}
    if isinstance(default, (list, dict)):
        return field(default_factory=lambda: type(default)(default), **kwargs)
    return field(default=default, **kwargs)


@dataclass
class BaselineStep:
    """One baseline step; steps in a feature are applied in order, each to what the
    previous steps left."""

    method: str = _f("asls", "asls | rubberband | polynomial | linear | none")
    lam: float = _f(1e5, "asls: smoothness (larger = stiffer).")
    p: float = _f(1e-3, "asls: asymmetry (smaller = stays further below peaks).")
    median_filter: int = _f(
        0, "Median-filter width (points) applied before fitting the baseline; 0 = off."
    )
    degree: int = _f(1, "polynomial: degree.")
    anchors: list[list[float]] = _f(
        [],
        "polynomial/linear: [[lo, hi], ...] windows known to be peak-free, used for the fit.",
    )
    stretch: float = _f(
        0.0,
        "rubberband: curvature added before the convex hull so it follows concave tails.",
    )


@dataclass
class PeakDef:
    """A peak inside a feature."""

    name: str = _f("peak", "Label used in output column names.")
    center: float = _f(0.0, "Expected position (x units).")
    center_range: list[float] = _f(
        [], "Allowed [lo, hi] for the fitted centre; default centre ± fwhm."
    )
    fwhm: float = _f(10.0, "Starting full width at half maximum (x units).")
    fwhm_range: list[float] = _f([0.5, 500.0], "Allowed [lo, hi] for the fitted width.")
    integrate: list[float] = _f(
        [], "integrate model: [lo, hi] limits; default centre ± fwhm."
    )


@dataclass
class Feature:
    """A region with its own baseline and one peak (single) or several (cluster)."""

    name: str = _f("feature", "Label used in output column names.")
    region: list[float] = _f([0.0, 0.0], "[lo, hi] of the x axis this feature uses.")
    model: str = _f(
        "gaussian", "integrate | gaussian | lorentzian | voigt | pseudo_voigt"
    )
    baseline: list[BaselineStep] = _f([], "Baseline steps, applied in order.")
    peaks: list[PeakDef] = _f([], "One peak (single) or several (cluster).")
    constraints: dict[str, str] = _f(
        {},
        'Cluster constraints as lmfit expressions, e.g. {"b_fwhm": "a_fwhm"} ties peak b\'s '
        "width to peak a's. Parameter names are <peak>_center, _fwhm, _amplitude.",
    )
    outputs: list[str] = _f(list(OUTPUTS), "Quantities reported per peak.")

    @property
    def is_cluster(self) -> bool:
        return len(self.peaks) > 1


@dataclass
class Recipe:
    """A named list of features, with how results are normalised."""

    name: str = _f("recipe", "Recipe name.")
    description: str = _f(
        "", "What the recipe measures and where its settings come from."
    )
    x_unit: str = _f("cm-1", "Unit of the x axis the positions refer to.")
    normalize_by: str = _f(
        "thickness",
        "thickness: divide areas and heights by fitted thickness (1 cm equivalent); none: raw.",
    )
    features: list[Feature] = _f([], "Features to measure.")

    # ------------------------------------------------------------------ validation
    def validate(self) -> list[str]:
        """Human-readable problems; empty when the recipe is usable."""
        problems = []
        if self.normalize_by not in NORMALISATIONS:
            problems.append(f"normalize_by must be one of {NORMALISATIONS}")
        seen = set()
        for feat in self.features:
            where = f"feature '{feat.name}'"
            if feat.name in seen:
                problems.append(f"{where}: duplicate name")
            seen.add(feat.name)
            lo, hi = feat.region
            if not lo < hi:
                problems.append(f"{where}: region must be [lo, hi] with lo < hi")
            if feat.model not in PEAK_MODELS:
                problems.append(f"{where}: model must be one of {PEAK_MODELS}")
            if not feat.peaks:
                problems.append(f"{where}: needs at least one peak")
            for step in feat.baseline:
                if step.method not in BASELINE_METHODS:
                    problems.append(
                        f"{where}: baseline method '{step.method}' not in {BASELINE_METHODS}"
                    )
                if step.method in ("polynomial", "linear") and not step.anchors:
                    problems.append(
                        f"{where}: {step.method} baseline needs anchor windows"
                    )
            names = [p.name for p in feat.peaks]
            if len(set(names)) != len(names):
                problems.append(f"{where}: peak names must be unique")
            for p in feat.peaks:
                if not lo <= p.center <= hi:
                    problems.append(
                        f"{where}: peak '{p.name}' centre {p.center} outside region"
                    )
            for q in feat.outputs:
                if q not in OUTPUTS:
                    problems.append(f"{where}: unknown output '{q}'")
        return problems

    # ------------------------------------------------------------------ (de)serialise
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recipe:
        feats = []
        for fd in data.get("features", []):
            fd = dict(fd)
            fd["baseline"] = [
                BaselineStep(**_known(BaselineStep, s)) for s in fd.get("baseline", [])
            ]
            fd["peaks"] = [PeakDef(**_known(PeakDef, p)) for p in fd.get("peaks", [])]
            feats.append(Feature(**_known(Feature, fd)))
        top = _known(cls, {k: v for k, v in data.items() if k != "features"})
        return cls(**top, features=feats)

    @classmethod
    def load(cls, path: str | Path) -> Recipe:
        path = Path(path)
        if path.suffix.lower() == ".json":
            return cls.from_dict(json.loads(path.read_text()))
        return cls.from_dict(tomllib.loads(path.read_text()))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = _drop_empty(self.to_dict())
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(data, indent=2))
        else:
            import tomli_w

            path.write_text(tomli_w.dumps(data))


def _known(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields the dataclass has; unknown keys raise so typos are not silent."""
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown setting(s) {sorted(unknown)}")
    return dict(data)


def _drop_empty(obj: Any) -> Any:
    """TOML has no null; drop empty lists/dicts so files stay short and readable."""
    if isinstance(obj, dict):
        return {k: _drop_empty(v) for k, v in obj.items() if v not in ([], {}, None)}
    if isinstance(obj, list):
        return [_drop_empty(v) for v in obj]
    return obj


def builtin_recipes() -> dict[str, Path]:
    """Recipes shipped with the package, by name (file stem)."""
    folder = resources.files("diamond_ftir_package") / "recipes"
    return {
        Path(str(p)).stem: Path(str(p))
        for p in folder.iterdir()
        if str(p).endswith(".toml")
    }


def resolve_recipe(ref: str | Path | Recipe) -> Recipe:
    """A Recipe from a Recipe, a built-in name, or a file path."""
    if isinstance(ref, Recipe):
        return ref
    builtin = builtin_recipes()
    if str(ref) in builtin:
        return Recipe.load(builtin[str(ref)])
    return Recipe.load(ref)
