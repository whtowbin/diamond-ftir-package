"""Process FTIR maps (many spectra on an x/y grid) with the single-spectrum method.

Every pixel runs through :func:`diamond_ftir_package.pipeline.run_analysis`, the same code
path as single files, the GUI and the CLI, so a pixel and a lone spectrum with the same
settings give the same numbers. What maps add:

- **An on-sample mask**: pixels whose diamond two-phonon absorbance is below
  ``MapParams.min_diamond_absorbance`` (off the stone, holes, edges) are skipped.
- **Calibration**: a random subset of on-sample pixels is analysed first. Its median
  baseline (lam, p) seeds every pixel (``baseline_strategy``), and its median thickness can
  replace the noisy per-pixel estimate (``thickness_mode``).
- **Parallel processing** across CPU cores (``n_jobs``).
- **Outputs**: an xarray Dataset of result maps, saved as NetCDF, CSV, PNG and TIFF.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .DiamondSpectrum import Diamond_Spectrum
from .params import AnalysisParams, MapParams
from .pipeline import run_analysis

STATUS_OK, STATUS_OFF_SAMPLE, STATUS_ERROR = 0, 1, 2
STATUS_LABELS = {
    STATUS_OK: "ok",
    STATUS_OFF_SAMPLE: "off sample",
    STATUS_ERROR: "error",
}


def load_map(path: str | Path) -> xr.Dataset:
    """Load an OMNIC ``.map`` file as a Dataset with ``spectra`` on dims (y, x, wn)."""
    from .LoadOmnicMAP import Load_Omnic_Map

    ds = Load_Omnic_Map(str(path))
    ds.attrs["source_file"] = Path(path).name
    return ds


def on_sample_mask(spectra: xr.DataArray, map_params: MapParams) -> np.ndarray:
    """True where the diamond two-phonon band is strong enough to analyse."""
    low, high = map_params.mask_window
    band = spectra.sel(wn=slice(low, high)).mean("wn").values
    return np.isfinite(band) & (band >= map_params.min_diamond_absorbance)


def _analyse_pixel(
    wn: np.ndarray,
    y: np.ndarray,
    params: AnalysisParams,
    baseline_start: tuple[float, float] | None,
    thickness_cm: float | None,
    tolerance: float | None = None,
) -> dict[str, Any]:
    spectrum = Diamond_Spectrum(
        X=wn, Y=np.asarray(y, dtype=float), X_Unit="Wavenumber", Y_Unit="Absorbance"
    )
    if thickness_cm is not None and not np.isfinite(thickness_cm):
        thickness_cm = None  # edge pixel: keep its own fitted thickness
    return run_analysis(
        spectrum,
        params,
        baseline_start=baseline_start,
        thickness_cm=thickness_cm,
        thickness_tolerance=tolerance,
    )


def _analyse_block(task: tuple) -> list[tuple[int, int, int, dict[str, Any] | str]]:
    """Worker: analyse a block of pixels. Top-level so it can run in another process."""
    wn, coords, block, params, baseline_start, thickness, tolerance = task
    out = []
    for (i, j), y, t in zip(coords, block, thickness, strict=True):
        try:
            row = _analyse_pixel(wn, y, params, baseline_start, t, tolerance)
            out.append((i, j, STATUS_OK, row))
        except Exception as e:  # noqa: BLE001 - one bad pixel must not stop a map
            out.append((i, j, STATUS_ERROR, f"{type(e).__name__}: {e}"))
    return out


def _run_blocks(tasks, jobs, progress=None, total=None):
    """Run pixel blocks inline (jobs == 1) or across worker processes."""
    results = []
    if jobs == 1:
        chunks = map(_analyse_block, tasks)
        for chunk in chunks:
            results.extend(chunk)
            if progress:
                progress(len(results), total)
        return results
    with _single_threaded_workers(), ProcessPoolExecutor(max_workers=jobs) as pool:
        for chunk in pool.map(_analyse_block, tasks):
            results.extend(chunk)
            if progress:
                progress(len(results), total)
    return results


_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextmanager
def _single_threaded_workers():
    """Give each worker process one math-library thread.

    Otherwise every worker's BLAS starts a thread per core and they fight over the CPU.
    Measured on a 10,176-pixel map with 11 workers: 27 s without this, 16 s with it
    (97 s serial). Variables the user already set are left alone. Spawned workers read
    the environment at start-up, so setting it here, in the parent, is enough.
    """
    previous = {k: os.environ.get(k) for k in _THREAD_VARS}
    for k in _THREAD_VARS:
        os.environ.setdefault(k, "1")
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _make_tasks(wn, spectra, coords, params, start, thickness, size, tolerance=None):
    """Blocks of pixels for the workers. ``thickness`` is None (each pixel's own fit),
    a number (same for all), or a (y, x) array (NaN = keep the pixel's own fit)."""

    def t(i, j):
        if thickness is None or np.isscalar(thickness):
            return thickness
        return float(thickness[i, j])

    return [
        (
            wn,
            coords[k : k + size],
            [spectra[i, j] for i, j in coords[k : k + size]],
            params,
            start,
            [t(i, j) for i, j in coords[k : k + size]],
            tolerance,
        )
        for k in range(0, len(coords), size)
    ]


def resolve_jobs(n_jobs: int) -> int:
    """0 means 'all cores but one'; negative counts back from the core count."""
    cores = os.cpu_count() or 1
    if n_jobs == 0:
        return max(1, cores - 1)
    if n_jobs < 0:
        return max(1, cores + n_jobs)
    return n_jobs


def edge_distance(mask: np.ndarray) -> np.ndarray:
    """Distance (in pixels) from each on-sample pixel to the nearest off-sample pixel.

    The map border counts as off sample, so pixels on the outer rows are edge pixels too.
    """
    from scipy import ndimage

    padded = np.pad(mask, 1, constant_values=False)
    return ndimage.distance_transform_edt(padded)[1:-1, 1:-1]


def survey_pixels(mask: np.ndarray, map_params: MapParams) -> list[tuple[int, int]]:
    """Interior pixels on a coarse grid (every ``survey_step`` pixels), edge ring excluded.

    Capped at ``survey_max_pixels`` by taking an even subset of the grid.
    """
    interior = mask & (edge_distance(mask) > map_params.edge_width_px)
    step = max(1, map_params.survey_step)
    grid = np.zeros_like(mask)
    grid[step // 2 :: step, step // 2 :: step] = True
    picks = np.argwhere(interior & grid)
    if len(picks) > map_params.survey_max_pixels:
        keep = np.linspace(0, len(picks) - 1, map_params.survey_max_pixels)
        picks = picks[keep.round().astype(int)]
    return [(int(i), int(j)) for i, j in picks]


def fit_thickness_field(
    ys: np.ndarray, xs: np.ndarray, thickness: np.ndarray, map_params: MapParams
) -> dict[str, Any]:
    """Robust smooth fit of log10(thickness) over the survey points.

    ``thickness_field`` 'constant' fits the median; 'plane' fits a tilted plane (wedge-shaped
    plates). Points more than ``outlier_mad`` robust standard deviations from the fit are
    dropped and the fit repeated, so thick strips, mounts and glitches do not bias it.
    """
    logt = np.log10(thickness)
    keep = np.isfinite(logt)
    constant = map_params.thickness_field == "constant"
    design = (
        np.ones((len(ys), 1))
        if constant
        else np.column_stack([np.ones(len(ys)), ys, xs])
    )
    coef = np.zeros(design.shape[1])
    sigma = np.nan
    for _ in range(10):
        if keep.sum() < design.shape[1] + 2:
            raise ValueError("Too few survey pixels left to fit a thickness field.")
        if constant:
            coef = np.array([np.median(logt[keep])])
        else:
            coef, *_ = np.linalg.lstsq(design[keep], logt[keep], rcond=None)
        resid = logt - design @ coef
        sigma = 1.4826 * np.median(np.abs(resid[keep])) + 1e-6
        new_keep = np.isfinite(logt) & (np.abs(resid) <= map_params.outlier_mad * sigma)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return {
        "coef": coef.tolist(),
        "sigma_decades": float(sigma),
        "tolerance_decades": float(map_params.outlier_mad * sigma),
        "points_used": int(keep.sum()),
        "points_rejected": int(np.isfinite(logt).sum() - keep.sum()),
    }


def evaluate_thickness_field(
    field: dict, shape: tuple[int, int], kind: str
) -> np.ndarray:
    """Thickness (cm) at every pixel from a fitted field."""
    ys, xs = np.mgrid[: shape[0], : shape[1]]
    coef = field["coef"]
    if kind == "constant":
        logt = np.full(shape, coef[0], dtype=float)
    else:
        logt = coef[0] + coef[1] * ys + coef[2] * xs
    return 10.0**logt


def survey(
    wn: np.ndarray,
    spectra: np.ndarray,
    mask: np.ndarray,
    params: AnalysisParams,
    map_params: MapParams,
) -> dict[str, Any]:
    """Coarse pass over interior pixels: typical baseline (lam, p) and a thickness field.

    Uses per-pixel settings exactly as configured in ``params`` (including any baseline
    search), on the grid from :func:`survey_pixels`.
    """
    picks = survey_pixels(mask, map_params)
    if not picks:
        raise ValueError(
            "No interior pixels to survey; lower edge_width_px or min_diamond_absorbance."
        )
    jobs = resolve_jobs(map_params.n_jobs)
    size = max(1, -(-len(picks) // (4 * jobs)))  # small blocks spread over all workers
    tasks = _make_tasks(wn, spectra, picks, params, None, None, size)
    rows = [
        (i, j, row) for i, j, code, row in _run_blocks(tasks, jobs) if code == STATUS_OK
    ]
    if not rows:
        raise ValueError("No survey pixel could be analysed; check the mask threshold.")
    ys = np.array([r[0] for r in rows], dtype=float)
    xs = np.array([r[1] for r in rows], dtype=float)
    lams = np.log10([r[2]["baseline_lam"] for r in rows])
    ps = np.log10([r[2]["baseline_p"] for r in rows])
    thickness = np.array([r[2]["typeIIA_ratio"] for r in rows])
    thickness = np.where(thickness > 0, thickness, np.nan)
    field = fit_thickness_field(ys, xs, thickness, map_params)
    return {
        "baseline_start": (float(10 ** np.median(lams)), float(10 ** np.median(ps))),
        "survey_pixels": len(rows),
        "log10_lam_iqr": float(np.subtract(*np.percentile(lams, [75, 25]))),
        "log10_p_iqr": float(np.subtract(*np.percentile(ps, [75, 25]))),
        "median_thickness_cm": float(np.nanmedian(thickness)),
        "thickness_field": field,
    }


def process_map(
    data: xr.Dataset | str | Path,
    params: AnalysisParams | None = None,
    map_params: MapParams | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> xr.Dataset:
    """Analyse every on-sample pixel of a map and return the results as (y, x) maps.

    ``data`` is a Dataset from :func:`load_map` or a path to a ``.map`` file. The returned
    Dataset has one variable per result (e.g. ``Total_N ppm``, ``B_percent``), plus
    ``status`` (0 ok, 1 off sample, 2 error). Settings, calibration results and per-pixel
    error messages are stored in ``attrs``.
    """
    params = params or AnalysisParams()
    map_params = map_params or MapParams()
    ds = data if isinstance(data, xr.Dataset) else load_map(data)
    wn = ds.wn.values.astype(float)
    spectra = ds.spectra.values
    ny, nx = spectra.shape[:2]
    mask = on_sample_mask(ds.spectra, map_params)

    params = _with_map_resolution(params, ds)
    run_params, start, thickness, tolerance, info = _plan_run(
        wn, spectra, mask, params, map_params
    )

    coords = [tuple(c) for c in np.argwhere(mask)]
    jobs = resolve_jobs(map_params.n_jobs)
    tasks = _make_tasks(
        wn,
        spectra,
        coords,
        run_params,
        start,
        thickness,
        max(1, map_params.block_size),
        tolerance,
    )
    results = _run_blocks(tasks, jobs, progress, len(coords))

    plan = {
        "run_baseline_search": run_params.diamond.baseline_search,
        "run_search_width": run_params.diamond.search_width,
        "baseline_start_lam": start[0] if start else np.nan,
        "baseline_start_p": start[1] if start else np.nan,
        "thickness_tolerance_decades": tolerance if tolerance is not None else np.nan,
    }
    out = _assemble(ds, ny, nx, mask, results, params, map_params, info, jobs, plan)
    _add_thickness_source(out, mask, thickness)
    return out


def _with_map_resolution(params: AnalysisParams, ds: xr.Dataset) -> AnalysisParams:
    """Pixels carry no file name, so take the resolution from the map's name once."""
    from .DiamondSpectrum import resolution_from_name

    if params.nitrogen.resolution_cm:
        return params
    found = resolution_from_name(ds.attrs.get("source_file", ""))
    if not found:
        return params
    return replace(params, nitrogen=replace(params.nitrogen, resolution_cm=found))


def _plan_run(wn, spectra, mask, params, map_params):
    """Decide per-pixel settings: baseline search/start and where thickness comes from.

    Returns ``(run_params, baseline_start, thickness, tolerance, survey_info)`` where
    ``thickness`` is None (own fit everywhere), a number (known thickness) or a (y, x)
    array (survey field; NaN in the edge ring, which keeps its own fit).
    """
    strategy = map_params.baseline_strategy
    if strategy not in ("per_pixel", "calibrate_fixed", "calibrate_local"):
        raise ValueError(f"Unknown baseline_strategy {strategy!r}")
    mode = map_params.thickness_mode
    if mode not in ("per_pixel", "survey", "known"):
        raise ValueError(f"Unknown thickness_mode {mode!r}")
    searching = params.diamond.baseline_search != "fixed"
    need_survey = mode == "survey" or (searching and strategy != "per_pixel")
    info = survey(wn, spectra, mask, params, map_params) if need_survey else {}

    run_params, start = params, None
    if searching and strategy == "calibrate_fixed":
        run_params = replace(
            params, diamond=replace(params.diamond, baseline_search="fixed")
        )
        start = info["baseline_start"]
    elif searching and strategy == "calibrate_local":
        run_params = replace(
            params,
            diamond=replace(params.diamond, search_width=map_params.local_search_width),
        )
        start = info["baseline_start"]

    thickness, tolerance = None, None
    if mode == "known":
        if not map_params.known_thickness_um:
            raise ValueError("thickness_mode='known' needs known_thickness_um")
        thickness = map_params.known_thickness_um * 1e-4
    elif mode == "survey":
        field = info["thickness_field"]
        thickness = evaluate_thickness_field(
            field, mask.shape, map_params.thickness_field
        )
        thickness[edge_distance(mask) <= map_params.edge_width_px] = np.nan
        tolerance = field["tolerance_decades"]
    return run_params, start, thickness, tolerance, info


THICKNESS_OWN, THICKNESS_REFERENCE, THICKNESS_EDGE, THICKNESS_OUTLIER = 0, 1, 2, 3
THICKNESS_SOURCE_LABELS = {
    THICKNESS_OWN: "own fit (thickness_mode per_pixel)",
    THICKNESS_REFERENCE: "survey field or known thickness",
    THICKNESS_EDGE: "own fit: within edge_width_px of the sample edge",
    THICKNESS_OUTLIER: "own fit: too far from the survey field (e.g. mount, thick strip)",
}


def _add_thickness_source(out, mask, thickness) -> None:
    """Per-pixel map saying where each pixel's thickness came from (see labels)."""
    source = np.full(mask.shape, np.nan, dtype=np.float32)
    ok = out["status"].values == STATUS_OK
    if thickness is None:
        source[ok] = THICKNESS_OWN
    else:
        source[ok] = THICKNESS_REFERENCE
        if "thickness_from_reference" in out:
            source[ok & (out["thickness_from_reference"].values == 0)] = (
                THICKNESS_OUTLIER
            )
        reference = np.broadcast_to(
            np.asarray(thickness, dtype=np.float32), mask.shape
        ).copy()
        if not np.isscalar(thickness):
            source[ok & ~np.isfinite(reference)] = THICKNESS_EDGE
        out["thickness_reference"] = (("y", "x"), reference)
    out["thickness_source"] = (("y", "x"), source)
    out.attrs["thickness_source_codes"] = ", ".join(
        f"{k}={v}" for k, v in THICKNESS_SOURCE_LABELS.items()
    )


def _assemble(
    ds, ny, nx, mask, results, params, map_params, calibration, jobs, plan
) -> xr.Dataset:
    status = np.full((ny, nx), STATUS_OFF_SAMPLE, dtype=np.int8)
    keys: list[str] = []
    for _, _, code, row in results:
        if code == STATUS_OK:
            keys.extend(
                k for k, v in row.items() if k not in keys and not isinstance(v, str)
            )
    arrays = {k: np.full((ny, nx), np.nan, dtype=np.float32) for k in keys}
    errors = []
    for i, j, code, row in results:
        status[i, j] = code
        if code == STATUS_OK:
            for k, v in row.items():
                if k in arrays:
                    arrays[k][i, j] = v
        else:
            errors.append(f"y={i} x={j}: {row}")

    out = xr.Dataset(
        {k: (("y", "x"), v) for k, v in arrays.items()}
        | {"status": (("y", "x"), status)},
        coords={"y": ds.y.values, "x": ds.x.values},
    )
    out.attrs = {
        "status_codes": ", ".join(f"{k}={v}" for k, v in STATUS_LABELS.items()),
        "pixels_on_sample": int(mask.sum()),
        "pixels_ok": int((status == STATUS_OK).sum()),
        "pixels_error": int((status == STATUS_ERROR).sum()),
        "n_jobs": jobs,
        "analysis_params": repr(params.to_dict()),
        "map_params": repr(map_params.to_dict()),
        "calibration": repr(calibration),
        "errors": "\n".join(errors[:200]),
        **plan,
    }
    return out


def save_map_outputs(
    result: xr.Dataset,
    out_dir: str | Path,
    stem: str,
    variables: tuple[str, ...] | None = None,
) -> list[Path]:
    """Write NetCDF (everything), a long-format CSV, and a PNG + float32 TIFF per variable."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    nc = out_dir / f"{stem}_results.nc"
    result.to_netcdf(nc)
    written.append(nc)

    table = result.to_dataframe().reset_index()
    csv = out_dir / f"{stem}_results.csv"
    table.to_csv(csv, index=False)
    written.append(csv)

    variables = variables or DEFAULT_MAP_VARIABLES
    for name in variables:
        if name not in result:
            continue
        arr = result[name]
        safe = name.replace(" ", "_").replace("%", "pct").replace("/", "_")
        fig, ax = plt.subplots(figsize=(8, 6))
        arr.where(result.status == STATUS_OK).plot(
            ax=ax, cmap="mako" if _has_mako() else "viridis"
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title(name)
        png = out_dir / f"{stem}_{safe}.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        tif = out_dir / f"{stem}_{safe}.tif"
        Image.fromarray(np.asarray(arr.values, dtype=np.float32)).save(tif)
        written += [png, tif]
    return written


DEFAULT_MAP_VARIABLES = (
    "Total_N ppm",
    "A_Nitrogen ppm",
    "B_Nitrogen ppm",
    "C_Nitrogen ppm",
    "B_percent",
    "typeIIA_ratio",
    "D_comp",
    "Normed_3107_Area",
    "Normed_Platelet_Area",
    "status",
)


def _has_mako() -> bool:
    try:
        import seaborn  # noqa: F401  (registers the 'mako' colormap)
    except ImportError:
        return False
    return True


def save_example_fits(
    data: xr.Dataset,
    out_dir: str | Path,
    stem: str,
    params: AnalysisParams | None = None,
    map_params: MapParams | None = None,
    result: xr.Dataset | None = None,
) -> list[Path]:
    """Plot the baseline and nitrogen fit for a grid of example pixels (quality check).

    Uses the same settings the map run used (read from ``result.attrs`` when given).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotting import plot_spectrum_fit

    params = params or AnalysisParams()
    map_params = map_params or MapParams()
    wn = data.wn.values.astype(float)
    spectra = data.spectra.values
    mask = on_sample_mask(data.spectra, map_params)
    if result is not None:
        a = result.attrs
        run_params = replace(
            params,
            diamond=replace(
                params.diamond,
                baseline_search=a["run_baseline_search"],
                search_width=a["run_search_width"],
            ),
        )
        start = (
            None
            if np.isnan(a["baseline_start_lam"])
            else (a["baseline_start_lam"], a["baseline_start_p"])
        )
        thickness = (
            result["thickness_reference"].values
            if "thickness_reference" in result
            else None
        )
        tol = a["thickness_tolerance_decades"]
        tolerance = None if np.isnan(tol) else tol
    else:
        run_params, start, thickness, tolerance, _ = _plan_run(
            wn, spectra, mask, params, map_params
        )
    ny_ex, nx_ex = map_params.example_grid
    ys = np.linspace(0, spectra.shape[0] - 1, ny_ex + 2)[1:-1].round().astype(int)
    xs = np.linspace(0, spectra.shape[1] - 1, nx_ex + 2)[1:-1].round().astype(int)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in ys:
        for j in xs:
            if not mask[i, j]:
                continue
            fig = plt.figure(figsize=(10, 8))
            title = f"{stem} pixel y={i} x={j}"
            spectrum = Diamond_Spectrum(X=wn, Y=spectra[i, j].astype(float))
            t = (
                thickness
                if thickness is None or np.isscalar(thickness)
                else thickness[i, j]
            )
            if t is not None and not np.isfinite(t):
                t = None
            try:
                run_analysis(
                    spectrum,
                    run_params,
                    baseline_start=start,
                    thickness_cm=t,
                    thickness_tolerance=tolerance,
                )
            except Exception as e:  # noqa: BLE001 - still save a plot that says why
                fig.text(0.5, 0.5, f"{title}\n{e}", ha="center")
            else:
                plot_spectrum_fit(fig, spectrum, title)
            path = out_dir / f"{stem}_fit_y{i}_x{j}.png"
            fig.savefig(path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
    return written
