"""The one data layout every part of the workbench uses, and the generic file readers.

Everything loaded is an ``xarray.Dataset`` with a ``spectra`` variable on dims
``(y, x, wn)``: a map is ny × nx spectra, a single spectrum is a 1 × 1 map. ``wn`` is the
spectral axis in cm⁻¹ (wavenumber for IR, Raman shift for Raman). ``attrs`` carry:

- ``source_file``: file name (resolution labels such as "4wnRes" are read from it);
- ``axis_kind``: "ir", "raman" or "other" (sets axis labels and which analyses apply);
- ``metadata``: the reader's own metadata (dict), passed on to analyses.

The same layout is used by xarray-based packages such as hyper-raman, so large Zarr- or
dask-backed maps can be viewed without conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

SPECTRA = "spectra"
AXIS_KINDS = ("ir", "raman", "other")
AXIS_LABELS = {  # x label, y label, draw x decreasing
    "ir": ("Wavenumber (cm⁻¹)", "Absorbance", True),
    "raman": ("Raman shift (cm⁻¹)", "Intensity", False),
    "other": ("x", "y", False),
}


def make_cube(
    x: np.ndarray,
    y: np.ndarray,
    source_file: str = "",
    axis_kind: str = "ir",
    metadata: dict[str, Any] | None = None,
) -> xr.Dataset:
    """One spectrum as a 1 × 1 map."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return xr.Dataset(
        {SPECTRA: (("y", "x", "wn"), y[None, None, :])},
        coords={"y": [0.0], "x": [0.0], "wn": x},
        attrs={
            "source_file": source_file,
            "axis_kind": axis_kind,
            "metadata": dict(metadata or {}),
        },
    )


def pixel(ds: xr.Dataset, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    """Spectral axis and values of pixel (row i, column j)."""
    return ds.wn.values, np.asarray(ds[SPECTRA].values[i, j], dtype=float)


def pixel_metadata(ds: xr.Dataset) -> dict[str, Any]:
    """Metadata an analysis gets with each spectrum from ``ds``."""
    meta = dict(ds.attrs.get("metadata") or {})
    if ds.attrs.get("source_file"):
        meta.setdefault("source_file", ds.attrs["source_file"])
    return meta


def is_map(ds: xr.Dataset) -> bool:
    return ds.sizes["y"] * ds.sizes["x"] > 1


def axis_labels(ds: xr.Dataset | None) -> tuple[str, str, bool]:
    kind = ds.attrs.get("axis_kind", "ir") if ds is not None else "ir"
    return AXIS_LABELS.get(kind, AXIS_LABELS["other"])


# ------------------------------------------------------------------------ generic readers
def read_two_columns(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """x and y from the first two numeric columns of a text file.

    Works with or without a header row (header text becomes non-numeric and is dropped),
    and with comma, semicolon, tab or whitespace separators.
    """
    data = pd.read_csv(path, header=None, sep=None, engine="python")
    data = data.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 10:
        raise ValueError(f"{Path(path).name}: fewer than 10 numeric rows")
    return data.iloc[:, 0].to_numpy(dtype=float), data.iloc[:, 1].to_numpy(dtype=float)


def _guess_kind(*texts: Any) -> str:
    return "raman" if any("raman" in str(t).lower() for t in texts) else "ir"


def load_csv(path: Path) -> xr.Dataset:
    x, y = read_two_columns(path)
    meta = {"Filename": path.name, "Sample": path.name.split("-")[0]}
    return make_cube(x, y, path.name, _guess_kind(path.name), meta)


def load_spa(path: Path) -> xr.Dataset:
    from ..LoadSPA import Load_SPA

    s: Any = Load_SPA(str(path))  # a Spectrum (or None) despite its annotation
    if s is None:
        raise ValueError(f"Could not read {path.name}")
    meta = dict(s.metadata or {})
    meta.setdefault("Filename", path.name)
    kind = _guess_kind(meta.get("xunits", ""), meta.get("units", ""))
    return make_cube(s.X, s.Y, path.name, kind, meta)


def load_spc(path: Path) -> xr.Dataset:
    from ..LoadSPC import Load_SPC

    s: Any = Load_SPC(str(path))  # returns a Spectrum despite its annotation
    meta = dict(s.metadata or {})
    meta.setdefault("Filename", path.name)
    return make_cube(s.X, s.Y, path.name, _guess_kind(path.name), meta)


INVALID_VALUE = 1e6  # no absorbance or count spectrum reaches this; junk bytes do


def mask_invalid_pixels(ds: xr.Dataset) -> xr.Dataset:
    """Set pixels holding non-numbers or absurd values (|value| > 1e6) to NaN, in place.

    OMNIC stores the whole map grid even when a map was stopped early; the pixels never
    measured then hold leftover bytes (values up to ~1e38). Left in, they swamp colour
    scales and pass "is this on the sample?" tests. Their count is kept in
    ``attrs["invalid_pixels"]``.
    """
    values = ds[SPECTRA].values
    with np.errstate(invalid="ignore"):
        bad = ~(np.isfinite(values) & (np.abs(values) <= INVALID_VALUE)).all(axis=-1)
    if bad.any():
        if not values.flags.writeable or not np.issubdtype(values.dtype, np.floating):
            values = values.astype(np.float32 if values.dtype == np.float32 else float)
            ds[SPECTRA] = (ds[SPECTRA].dims, values)
        values[bad] = np.nan
    ds.attrs["invalid_pixels"] = int(bad.sum())
    return ds


def load_omnic_map(path: Path) -> xr.Dataset:
    """Thermo OMNIC ``.map`` (FTIR or Raman)."""
    from ..LoadOmnicMAP import Load_Omnic_Map

    ds = Load_Omnic_Map(str(path))
    info = ds.attrs.get("OmnicInfo", {})
    ds.attrs = {
        "source_file": path.name,
        "axis_kind": _guess_kind(path.name, *info.values())
        if isinstance(info, dict)
        else "ir",
        "metadata": {"source_file": path.name},
    }
    return mask_invalid_pixels(ds)
