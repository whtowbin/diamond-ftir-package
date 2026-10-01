"""Injection-recovery bias study of baseline methods and nitrogen windows on real spectra.

    uv run python scripts/bias_study.py local_data/datasets.json [--out local_data/bias_study]

The dataset list is a local, git-ignored JSON file (see docs/design/baseline-and-window.md);
no data or data paths belong in the repository. Each dataset is sampled at random (fixed
seed). Zip members are extracted to a temporary folder outside the repo, deleted afterwards.

For each spectrum the study records:
- outcome: load_error | diamond_saturated | nitrogen_saturated | fit_error | usable;
- quality: point spacing, noise, fitted thickness, background offset, platelet size, and a
  diamond-likeness score (R² of the type IIa shape over the unsaturated phonon windows);
- thickness bias per baseline method, by injection: add dt * (type IIa + known A + B) and
  compare the recovered dt;
- window bias, by leakage: change in total N per 1 cm⁻¹ of an injected platelet peak or
  of residual background below the nitrogen band, and N for each window.

Results go to <out>/results.jsonl (one line per spectrum); summarise them with
scripts/bias_study_report.py.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

METHODS = {
    "multistage": {},
    "multistage_no_rubber": {"pre_use_rubberband": False},
    "single_fixed": {"baseline_method": "single"},
    "single_opt": {"baseline_method": "single", "baseline_search": "optimize"},
    "joint_1e8": {"baseline_method": "joint", "joint_lam": 1e8},
    "joint_1e9": {"baseline_method": "joint", "joint_lam": 1e9},
}
WINDOWS = (
    "950-1350",
    "950-1400",
    "650-1350",
    "650-1400",
    "950-1330",
    "1000-1300",
    "850-1350",
)
A_INJ = B_INJ = 150.0
INJECT_FRACTION = 0.3


# ---------------------------------------------------------------- sampling (parent process)
def sample_items(config: dict, seed: int, tmp: Path) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for ds in config["datasets"]:
        base = {"dataset": ds["label"], "quality": ds.get("quality", "")}
        path = Path(ds["path"])
        n = int(ds.get("n", 50))
        if ds["kind"] == "folder":
            files = sorted(p for p in path.glob(ds.get("pattern", "*")) if p.is_file())
            for f in rng.sample(files, min(n, len(files))):
                items.append(base | {"source": str(f)})
        elif ds["kind"] == "zips":
            zips = sorted(path.glob("*.zip"))
            per_zip = max(1, -(-n // max(1, len(zips))))
            for z in zips:
                with zipfile.ZipFile(z) as zf:
                    members = [
                        m
                        for m in zf.namelist()
                        if m.lower().endswith(ds.get("pattern", ".spc"))
                        and ds.get("include", "") in m
                        and not (ds.get("exclude") and ds["exclude"] in m)
                    ]
                    for k, m in enumerate(
                        rng.sample(members, min(per_zip, len(members)))
                    ):
                        target = tmp / f"{ds['label']}_{z.stem}_{k}{Path(m).suffix}"
                        target.write_bytes(zf.read(m))
                        items.append(base | {"source": str(target)})
        elif ds["kind"] == "map":
            from diamond_ftir_package.maps import (
                edge_distance,
                load_map,
                on_sample_mask,
            )
            from diamond_ftir_package.params import MapParams

            data = load_map(path)
            mask = on_sample_mask(data.spectra, MapParams())
            idx = np.argwhere(mask & (edge_distance(mask) > 3))
            wn = data.wn.values.astype(float)
            for i, j in rng.sample([tuple(p) for p in idx], min(n, len(idx))):
                items.append(
                    base | {"x": wn, "y": data.spectra.values[i, j].astype(float)}
                )
        else:
            raise ValueError(f"unknown kind {ds['kind']!r}")
    for k, it in enumerate(items):
        it["id"] = k
    return items


# ---------------------------------------------------------------- analysis (worker process)
def _params(kind):
    from diamond_ftir_package.params import AnalysisParams

    p = AnalysisParams(run_hydrogen=False, run_platelets=False)
    for k, v in METHODS[kind].items():
        setattr(p.diamond, k, v)
    return p


def _gauss(x, c, fwhm):
    return np.exp(-0.5 * ((x - c) / (fwhm / 2.3548)) ** 2)


def _local_peak(x, y, lo, hi, c_lo, c_hi):
    m = (x >= lo) & (x <= hi)
    xx, yy = x[m], y[m]
    flank = (xx < c_lo) | (xx > c_hi)
    c = np.polyfit(xx[flank], yy[flank], 1)
    return float(max(0.0, (yy - np.polyval(c, xx))[~flank].max()))


def _diff_noise(x, y, lo, hi):
    m = (x >= lo) & (x <= hi)
    d = np.diff(y[m]) / np.sqrt(2)
    return float(1.4826 * np.median(np.abs(d - np.median(d))))


def analyse(item: dict) -> dict:
    from diamond_ftir_package import Diamond_Spectrum
    from diamond_ftir_package.DiamondSpectrum import CAXBDY, typeIIA_Spectrum
    from diamond_ftir_package.params import NitrogenParams
    from diamond_ftir_package.pipeline import load_spectrum, run_analysis

    out = {k: item[k] for k in ("id", "dataset", "quality")}
    quiet = io.StringIO()
    try:
        with contextlib.redirect_stdout(quiet):
            if "source" in item:
                raw = load_spectrum(item["source"])
                x, y = raw.initial_X.astype(float), raw.initial_Y.astype(float)
            else:
                x, y = item["x"], item["y"]
        order = np.argsort(x)
        x, y = x[order], y[order]
        s = Diamond_Spectrum(X=x, Y=y)
    except Exception as e:  # noqa: BLE001
        return out | {"outcome": "load_error", "error": repr(e)[:200]}

    out |= {
        "spacing": float(np.median(np.diff(x))),
        "wn_min": float(x.min()),
        "wn_max": float(x.max()),
        "offset_raw": float(np.median(y[(x >= 4500) & (x <= 5000)]))
        if x.max() > 5000
        else None,
        "noise_raw": _diff_noise(x, y, 4000, 5000) if x.max() > 5000 else None,
        "primary_saturated": bool(s.test_saturation(1970, 2040, 2.5, 0.5)),
    }
    try:
        s.test_diamond_saturation(2.5, 0.5)
    except ValueError:
        return out | {"outcome": "diamond_saturated"}
    n_sat = (
        s.test_saturation(1000, 1350, 1.8, 0.3)
        or s.Y[(s.X > 1000) & (s.X < 1350)].max() >= 1.8
    )

    try:
        ref = Diamond_Spectrum(X=x, Y=y)
        run_analysis(ref, _params("single_opt"))
    except Exception as e:  # noqa: BLE001
        return out | {"outcome": "fit_error", "error": repr(e)[:200]}
    t_ref = float(ref.typeIIA_ratio)
    X, Yn = ref.normalized_spectrum.X, ref.normalized_spectrum.Y
    iia = ref.interpolated_typeIIA_Spectrum.Y
    mask = ref.outputdict["mask"]
    ss_res = np.sum((Yn[mask] - iia[mask]) ** 2)
    ss_tot = np.sum((iia[mask] - iia[mask].mean()) ** 2)
    out |= {
        "t_ref": t_ref,
        "diamond_r2": float(1 - ss_res / ss_tot),
        "noise_norm": _diff_noise(X, Yn, 4000, 5000),
        "platelet": _local_peak(X, Yn, 1340, 1395, 1355, 1380),
        "core_peak": float(Yn[(X >= 1000) & (X <= 1300)].max()),
        "N_ref": float(ref.nitrogen_dict["Total_N ppm"]),
        "B_pct_ref": float(ref.nitrogen_dict["B_percent"]),
    }
    if n_sat:
        return out | {"outcome": "nitrogen_saturated"}
    if t_ref <= 0:
        return out | {"outcome": "fit_error", "error": "non-positive thickness"}

    # thickness bias by injection, per baseline method
    dt = INJECT_FRACTION * t_ref
    comps = A_INJ / 16.5 * np.interp(
        x, CAXBDY.index, CAXBDY["A"], left=0, right=0
    ) + B_INJ / 79.4 * np.interp(x, CAXBDY.index, CAXBDY["B"], left=0, right=0)
    y_inj = y + dt * (np.interp(x, typeIIA_Spectrum.X, typeIIA_Spectrum.Y) + comps)
    rec = {}
    for kind in METHODS:
        try:
            a = Diamond_Spectrum(X=x, Y=y)
            b = Diamond_Spectrum(X=x, Y=y_inj)
            run_analysis(a, _params(kind))
            run_analysis(b, _params(kind))
            same_regions = np.array_equal(a.outputdict["mask"], b.outputdict["mask"])
            rec[kind] = {
                "t": float(a.typeIIA_ratio),
                "N": float(a.nitrogen_dict["Total_N ppm"]),
                "thickness_recovery": float((b.typeIIA_ratio - a.typeIIA_ratio) / dt)
                if same_regions
                else None,
            }
        except Exception as e:  # noqa: BLE001
            rec[kind] = {"error": repr(e)[:120]}
    out["methods"] = rec

    # window bias by leakage (single_opt baseline held fixed)
    base = Yn.copy()

    def fit(yn, window):
        lo, hi = (int(v) for v in window.split("-"))
        ref.normalized_spectrum.Y = yn
        ref.Nitrogen_fit(params=NitrogenParams(wn_low=lo, wn_high=hi))
        return float(ref.nitrogen_dict["Total_N ppm"])

    win = {}
    for w in WINDOWS:
        n0 = fit(base, w)
        win[w] = {
            "N": n0,
            "leak_platelet": fit(base + _gauss(X, 1365, 12), w) - n0,
            "leak_low": fit(base + _gauss(X, 800, 120), w) - n0,
        }
    ref.normalized_spectrum.Y = base
    out["windows"] = win
    return out | {"outcome": "usable"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("local_data/bias_study"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    tmp = Path(tempfile.mkdtemp(prefix="bias_study_"))
    try:
        items = sample_items(config, args.seed, tmp)
        print(
            f"sampled {len(items)} spectra from {len(config['datasets'])} datasets",
            file=sys.stderr,
        )
        args.out.mkdir(parents=True, exist_ok=True)
        done = 0
        with Pool(args.jobs) as pool, open(args.out / "results.jsonl", "w") as fh:
            for row in pool.imap_unordered(analyse, items, chunksize=1):
                fh.write(json.dumps(row) + "\n")
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{len(items)}", file=sys.stderr)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {args.out / 'results.jsonl'}", file=sys.stderr)


if __name__ == "__main__":
    main()
