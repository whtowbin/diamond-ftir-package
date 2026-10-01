# diamond-ftir-package

Calculate nitrogen content and aggregation state, hydrogen (3107 cm⁻¹), platelet (B′) and amber-centre
features of diamond from infrared (FTIR) absorption spectra. Use it from a desktop app, the command
line, or Python.

> **Status:** pre-release (0.0.x). Absolute nitrogen values have not yet been validated against
> reference spectra; see [Known limitations](#known-limitations).

## Install

You need Python 3.10 or newer.

```bash
uv tool install diamond-ftir-package      # or: pipx install diamond-ftir-package
```

*(Standalone downloads for macOS and Windows: [PLACEHOLDER: link once builds exist].)*

## Desktop app

```bash
uv tool install "diamond-ftir-package[gui]"
diamond-ftir-app          # Qt app: spectra, batches, maps and the recipe editor
```

(The earlier tkinter app is still available as `diamond-ftir-gui`.)

### Your own peaks and baselines: recipes

Define extra measurements (local baselines, single peaks, fitted peak clusters) in the
*Recipe editor* tab with a live preview, or as a TOML file, and run them on single spectra,
batches and maps. See [docs/recipes.md](docs/recipes.md).

### Large maps in napari

```bash
pip install "diamond-ftir-package[napari]" "napari[all]"
napari sample.map         # then Plugins → Diamond FTIR → Spectrum inspector
```

1. **Open file…** or **Open folder…** (CSV, SPA or SPC spectra).
2. Tick what to measure, adjust any setting in the tabs on the left (hover a setting for an
   explanation, **Reset to defaults** undoes everything).
3. **Run analysis**. Click a row in the results table to see the fitted baseline and nitrogen fit.
4. **Export results to CSV…**. **Save settings…** stores your settings so a colleague can reuse them.

*[PLACEHOLDER: screenshot of the app]*

## Command line

```bash
diamond-ftir run spectra/ -o results.csv            # a folder, or a single file
diamond-ftir defaults > settings.json               # edit, then:
diamond-ftir run spectra/ --params settings.json -o results.csv
diamond-ftir run spectra/ --recipe my_peaks.toml -o results.csv   # extra measurements
```

The exit code is non-zero if any file failed; failures are listed in the `Status` column.

### Maps (OMNIC `.map`)

```bash
diamond-ftir map sample.map -o Results                         # multistage baseline, all cores
diamond-ftir map sample.map -o Results --thickness-um 850       # polished plate of known thickness
diamond-ftir map sample.map -o Results --baseline-method single \
    --baseline-search optimize --strategy calibrate_fixed       # searched baseline, fast
```

Every pixel runs through the same analysis as a single spectrum. Results are written as
NetCDF, CSV, and a PNG and TIFF per quantity, plus example fit plots for checking. A
10,000-pixel map takes about 16 s on 12 cores with a fixed baseline. See
[docs/baseline_benchmark.md](docs/baseline_benchmark.md) before choosing baseline settings:
the choice changes absolute ppm (B% is robust).

## Python

```python
from diamond_ftir_package.pipeline import load_spectrum, run_analysis
from diamond_ftir_package.params import AnalysisParams

spectrum = load_spectrum("sample.csv")
params = AnalysisParams()
params.nitrogen.max_C_or_B = 0.02
row = run_analysis(spectrum, params)      # one flat dict of results
print(row["Total_N ppm"], row["B_percent"])
```

Input CSVs need two columns: wavenumber (cm⁻¹) and absorbance.

## What it measures

| Result | Meaning |
|---|---|
| `A/B/C_Nitrogen ppm`, `Total_N ppm` | Nitrogen in A, B and C centres, atomic ppm |
| `B_percent`, `C_percent` | Aggregation state: share of total nitrogen in B / C centres |
| `Normed_3107_Area`, `Normed_3085_Area` | Hydrogen-related peak areas, 1 cm equivalent |
| `Normed_Platelet_*`, `Platelet_Peak_Position` | Platelet peak area, height and position |
| `Normed_1405_Area` | 1405 cm⁻¹ peak area |
| `Amber_*_Area` | Amber-centre band areas (optional) |
| `typeIIA_ratio` | Fitted thickness scale against the type IIa reference |
| `T_N (C)`, `T_P (C)`, `platelet_*` | Nitrogen-aggregation and platelet-degradation temperatures (with `--duration-ma`; see [docs/thermometry.md](docs/thermometry.md)) |

Details: [docs/methods.md](docs/methods.md). Every setting: [docs/parameters.md](docs/parameters.md).

## Known limitations

- Very thick or highly absorbing stones saturate the diamond two-phonon peaks; if all three
  intrinsic regions are saturated the file fails with a clear message.
- The default baseline fits the background and the diamond thickness together ("joint"). In
  injection tests on 1,272 real spectra of very different quality it recovered thickness with
  a median error of 0.9% (see [docs/methods_validation.md](docs/methods_validation.md)).
  Absolute ppm still awaits comparison with independently analysed stones.
  *[PLACEHOLDER: validation against reference stones]*
- Results carry quality flags (`QA`, `diamond_r2`, saturation): check them before using values.
- Amber centres and SPC/SPA/map loaders are less tested than the nitrogen workflow.

## Citing

*[PLACEHOLDER: how to cite this software; see `CITATION.cff`]*. The methods rest on published work
that must be cited too; the list is tracked in [docs/CITATIONS_TODO.md](docs/CITATIONS_TODO.md).

## Development

```bash
uv sync
uv run pytest
uv run ruff check && uv run ruff format --check
uv run python scripts/make_parameter_docs.py   # after changing parameters
```
