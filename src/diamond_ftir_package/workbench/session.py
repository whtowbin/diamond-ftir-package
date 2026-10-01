"""The working session behind the desktop app, with no GUI code (so it can be tested).

- ``Session.open(paths)`` is the single loader: map files become maps, folders become
  batches, other files single spectra (several at once: a batch).
- Settings are held per analysis and apply to every dataset, spectra and maps alike.
- ``analyse_spectrum`` runs the enabled analyses on one spectrum (the live fit when a pixel
  or file is clicked); ``analyse_dataset`` runs them on a whole batch or map.
- ``save`` / ``load`` write a project file (``.dftir``, a zip): settings, recipes, links to
  the raw data (never copied) and results (CSV for spectra, NetCDF for maps), so a project
  reopens without rerunning.
"""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from ..core import Recipe
from .analyses import analyse_map_by_pixel, resolve
from .data import pixel_metadata
from .plugins import Progress, Registry, SpectrumResult, default_registry

PROJECT_SUFFIX = ".dftir"
PROJECT_VERSION = 1


@dataclass
class Dataset:
    """One loaded thing: a single spectrum, a batch (folder or several files), or a map."""

    kind: str  # "spectrum" | "batch" | "map"
    name: str
    paths: list[Path]
    results: Any = None  # spectra: pd.DataFrame (one row per file); map: xr.Dataset
    dirty: bool = False
    _cache: dict[int, xr.Dataset] = field(default_factory=dict, repr=False)

    def load(self, registry: Registry, index: int = 0) -> xr.Dataset:
        """The data of file ``index`` (the map itself for maps), read once and kept."""
        if index not in self._cache:
            path = self.paths[index]
            loader = registry.loader_for(path)
            if loader is None:
                raise ValueError(f"No reader for {path.name}")
            if self.kind == "batch":
                self._cache.clear()  # keep one file of a batch in memory, not all
            self._cache[index] = loader.load(path)
        return self._cache[index]


def params_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Rebuild a (nested) settings dataclass from JSON; lists go back to tuples."""
    default = cls()
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value, current = data[f.name], getattr(default, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            value = params_from_dict(type(current), value)
        elif isinstance(current, tuple) and isinstance(value, list):
            value = tuple(tuple(v) if isinstance(v, list) else v for v in value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _jsonable(params: Any) -> dict[str, Any]:
    out = asdict(params)
    if "recipes" in out:
        out["recipes"] = []  # held by the session, not per analysis
    return out


@dataclass
class Session:
    registry: Registry = field(default_factory=default_registry)
    params: dict[str, Any] = field(default_factory=dict)  # analysis name -> settings
    map_params: dict[str, Any] = field(default_factory=dict)
    enabled: dict[str, bool] = field(default_factory=dict)
    recipes: list[Recipe | str] = field(default_factory=list)  # ticked recipes to run
    datasets: list[Dataset] = field(default_factory=list)
    project_path: Path | None = None
    settings_dirty: bool = False

    def __post_init__(self) -> None:
        for name, a in self.registry.analyses.items():
            self.params.setdefault(name, a.make_params())
            if a.make_map_params:
                self.map_params.setdefault(name, a.make_map_params())
            self.enabled.setdefault(name, a.enabled_by_default)

    # ------------------------------------------------------------------ loading
    def open(self, paths) -> list[Dataset]:
        """Open files and/or folders; returns the new datasets."""
        added: list[Dataset] = []
        singles: list[Path] = []
        for p in map(Path, paths):
            if p.is_dir():
                loaders = {f: self.registry.loader_for(f) for f in p.iterdir()}
                files = sorted(f for f, ld in loaders.items() if ld and not ld.is_map)
                if files:
                    added.append(Dataset("batch", p.name, files))
                continue
            loader = self.registry.loader_for(p)
            if loader is None:
                raise ValueError(f"Unsupported file: {p.name}")
            if loader.is_map:
                added.append(Dataset("map", p.stem, [p]))
            else:
                singles.append(p)
        if len(singles) == 1:
            added.append(Dataset("spectrum", singles[0].stem, singles))
        elif singles:
            added.append(Dataset("batch", f"{len(singles)} spectra", singles))
        self.datasets.extend(added)
        return added

    def remove(self, dataset: Dataset) -> None:
        self.datasets.remove(dataset)

    @property
    def dirty(self) -> bool:
        return self.settings_dirty or any(d.dirty for d in self.datasets)

    # ------------------------------------------------------------------ running
    def active(self, ds: xr.Dataset) -> list[str]:
        return [
            n
            for n, a in self.registry.analyses.items()
            if self.enabled.get(n) and a.applies_to(ds)
        ]

    def run_params(self, name: str) -> Any:
        params = self.params[name]
        if self.registry.analyses[name].uses_recipes:
            params = replace(params, recipes=tuple(resolve(r) for r in self.recipes))
        return params

    def analyse_spectrum(
        self, ds: xr.Dataset, i: int = 0, j: int = 0
    ) -> dict[str, SpectrumResult]:
        """Every enabled analysis on pixel (i, j) of ``ds``; errors are reported, not raised."""
        x = ds.wn.values
        y = np.asarray(ds["spectra"].values[i, j], dtype=float)
        meta = pixel_metadata(ds)
        out = {}
        for name in self.active(ds):
            try:
                out[name] = self.registry.analyses[name].run(
                    x, y, self.run_params(name), meta
                )
            except Exception as e:  # noqa: BLE001 - shown next to the spectrum
                out[name] = SpectrumResult(error=f"{type(e).__name__}: {e}")
        return out

    def analyse_dataset(
        self, dataset: Dataset, progress: Progress | None = None
    ) -> Any:
        """Full run. Maps give an xr.Dataset of layers, spectra a DataFrame (row per file)."""
        if dataset.kind == "map":
            ds = dataset.load(self.registry)
            layers = []
            for name in self.active(ds):
                a = self.registry.analyses[name]
                if a.run_map is not None:
                    layers.append(
                        a.run_map(
                            ds,
                            self.run_params(name),
                            self.map_params.get(name),
                            progress,
                        )
                    )
                else:
                    layers.append(
                        analyse_map_by_pixel(a, ds, self.run_params(name), progress)
                    )
            result = _merge_layers(layers, ds)
        else:
            rows = []
            for k, path in enumerate(dataset.paths):
                row: dict[str, Any] = {"Filename": path.name}
                try:
                    ds = dataset.load(self.registry, k)
                    errors = []
                    for name, res in self.analyse_spectrum(ds).items():
                        row.update(res.values)
                        if res.error:
                            errors.append(f"{name}: {res.error}")
                    row["Status"] = "; ".join(errors) or "OK"
                except Exception as e:  # noqa: BLE001 - one bad file must not stop a batch
                    row["Status"] = f"Error: {e}"
                rows.append(row)
                if progress:
                    progress(k + 1, len(dataset.paths))
            result = pd.DataFrame(rows)
        dataset.results = result
        dataset.dirty = True
        return result

    # ------------------------------------------------------------------ project files
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != PROJECT_SUFFIX:
            path = path.with_suffix(PROJECT_SUFFIX)
        manifest: dict[str, Any] = {
            "version": PROJECT_VERSION,
            "params": {n: _jsonable(p) for n, p in self.params.items()},
            "map_params": {n: asdict(p) for n, p in self.map_params.items()},
            "enabled": self.enabled,
            "recipes": [
                {"inline": r.to_dict()} if isinstance(r, Recipe) else {"ref": str(r)}
                for r in self.recipes
            ],
            "datasets": [],
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for k, d in enumerate(self.datasets):
                entry = {
                    "kind": d.kind,
                    "name": d.name,
                    "paths": [str(p) for p in d.paths],
                    "results": None,
                }
                if isinstance(d.results, pd.DataFrame):
                    entry["results"] = f"results/{k}.csv"
                    zf.writestr(entry["results"], d.results.to_csv(index=False))
                elif isinstance(d.results, xr.Dataset):
                    entry["results"] = f"results/{k}.nc"
                    with tempfile.TemporaryDirectory() as tmp:
                        nc = Path(tmp) / "r.nc"
                        netcdf_safe(d.results).to_netcdf(nc)
                        zf.write(nc, entry["results"])
                manifest["datasets"].append(entry)
            zf.writestr("project.json", json.dumps(manifest, indent=2, default=str))
        for d in self.datasets:
            d.dirty = False
        self.settings_dirty = False
        self.project_path = path
        return path

    @classmethod
    def load(cls, path: str | Path, registry: Registry | None = None) -> Session:
        path = Path(path)
        session = cls(registry=registry or default_registry(), project_path=path)
        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read("project.json"))
            for name, data in manifest.get("params", {}).items():
                if name in session.params:
                    session.params[name] = params_from_dict(
                        type(session.params[name]), data
                    )
            for name, data in manifest.get("map_params", {}).items():
                if name in session.map_params:
                    session.map_params[name] = params_from_dict(
                        type(session.map_params[name]), data
                    )
            session.enabled.update(
                {
                    k: v
                    for k, v in manifest.get("enabled", {}).items()
                    if k in session.enabled
                }
            )
            session.recipes = [
                Recipe.from_dict(r["inline"]) if "inline" in r else r["ref"]
                for r in manifest.get("recipes", [])
            ]
            for entry in manifest["datasets"]:
                d = Dataset(
                    entry["kind"], entry["name"], [Path(p) for p in entry["paths"]]
                )
                stored = entry.get("results")
                if stored and stored.endswith(".csv"):
                    d.results = pd.read_csv(io.BytesIO(zf.read(stored)))
                elif stored:
                    d.results = xr.load_dataset(io.BytesIO(zf.read(stored)))
                session.datasets.append(d)
        return session


def _merge_layers(layers: list[xr.Dataset], ds: xr.Dataset) -> xr.Dataset:
    """One Dataset of result layers; later analyses do not overwrite earlier names."""
    out = xr.Dataset(coords={"y": ds.y.values, "x": ds.x.values})
    for layer in layers:
        for name in layer.data_vars:
            if layer[name].dims[-2:] != ("y", "x"):
                continue
            key = name if name not in out else f"{name} ({len(out)})"
            out[key] = layer[name]
        out.attrs.update(layer.attrs)
    return out


def netcdf_safe(ds: xr.Dataset) -> xr.Dataset:
    """NetCDF attributes must be simple types."""
    out = ds.copy()
    out.attrs = {
        k: (v if isinstance(v, (str, int, float)) else repr(v))
        for k, v in ds.attrs.items()
    }
    for name in out.data_vars:
        out[name].attrs = {
            k: (v if isinstance(v, (str, int, float)) else repr(v))
            for k, v in out[name].attrs.items()
        }
    return out
