# Design: reusable GUI, user recipes and distribution

Status: **first increment built** (recipe engine, Qt widgets, desktop app, napari plugin).

## Goals

- Users define their own baselines, single peaks and peak clusters, and run them on single
  spectra, batches and maps without writing code.
- The GUI pieces can be reused for other spectroscopy (other IR samples, Raman, UV-VIS).
- The app is easy to install for non-Python users, and large maps stay responsive.

## Architecture: three layers

| layer | package path | depends on | contents |
|---|---|---|---|
| core | `diamond_ftir_package.core` | numpy, scipy, pybaselines, lmfit | recipe format, recipe engine (no diamond, no GUI) |
| domain pack | `diamond_ftir_package` (params, pipeline, maps, DiamondSpectrum) | core | diamond FTIR: joint type IIa thickness, CAXBDY nitrogen, built-in recipes |
| GUI | `diamond_ftir_package.qtgui`, `napari_plugin` | qtpy, pyqtgraph (+ napari) | reusable widgets, desktop app, napari plugin |

Rules that keep it reusable:

1. **Analysis is data, not code.** Settings are dataclasses (`params.py`, `core.recipe`) with
   a description per field. Recipes are TOML/JSON. Anything the GUI can do can be saved,
   shared and rerun from the command line.
2. **Widgets are generated from the dataclasses** (`DataclassForm`), so new settings appear
   in the GUI without GUI code, and the same widget edits diamond parameters, recipe steps or
   a future Raman pack's settings.
3. **One analysis path.** The GUI, CLI, batches, maps and napari all call
   `pipeline.run_analysis` / `core.run_recipe`. The editor's preview *is* the analysis.
4. **Widgets use qtpy**, so they run on PySide6 (the standalone app) or whatever Qt binding
   napari uses, unchanged.

A new spectroscopy (e.g. Raman) needs a small domain pack: a loader, its normalisation (or
none), and built-in recipes. The recipe engine, editor, spectrum view, map view and napari
plugin carry over as they are.

## Toolkit decision

**Qt (via qtpy) + pyqtgraph, with napari as the large-map viewer.**

- pyqtgraph draws and zooms spectra and images far faster than matplotlib, which matters for
  live previews and clicking through maps.
- napari is Qt-based, so the same widgets become napari dock widgets. That gives lazy,
  GPU-accelerated display of full spectral cubes without writing a map viewer.
- DearPyGui was not chosen: it is fast and small, but its widgets cannot live inside napari,
  which would mean two GUIs.
- The tkinter app (`diamond-ftir-gui`) stays until the Qt app has been used in practice; then
  it can be retired.

## Distribution

| audience | how | contents |
|---|---|---|
| scripts / notebooks | `pip install diamond-ftir-package` | core, diamond pack, CLI |
| desktop users with Python | `uv tool install "diamond-ftir-package[gui]"` → `diamond-ftir-app` | + PySide6, pyqtgraph |
| napari users | `pip install "diamond-ftir-package[napari]"`, or napari's plugin menu once on PyPI | + napari plugin (manifest `napari.yaml`) |
| no Python | standalone app built with PyInstaller in CI (macOS, Windows) | `[gui]` only; napari not bundled |

- **PySide6** (LGPL) is used for the standalone app because it is friendlier for
  redistribution than PyQt (GPL).
- **napari is not bundled** into the standalone app, to keep it small; heavy map users
  install the plugin. Bundle sizes are still to be measured on the first PyInstaller build.
- **napari's Qt backend:** napari's PySide6 support is experimental. In testing here, a
  headless napari viewer crashed with PySide6 offscreen. Recommend napari's default backend
  for napari users (`pip install "napari[all]"`); the plugin works with either through qtpy.
- If another project later needs only the core, split `core` into its own package; the layer
  boundaries above already allow it.

## Testing

- Recipe engine: exact reproduction of the built-in 3107 measurement; injection tests for
  overlapping clusters, constraints, numeric peak names, local rubber band on a steep tail.
- Widgets: run offscreen (`QT_QPA_PLATFORM=offscreen`), so they run in CI: form editing and
  errors, editor preview equals the engine, main window, map view, napari manifest
  validation, inspector widget with a stand-in viewer. The real napari viewer is checked by hand.

## Next steps

- Report fit uncertainties (lmfit standard errors) and flag peaks not significantly above
  noise.
- Validate the platelet and amber example recipes with the injection method; decide whether
  they replace the hand-written measurements.
- Map view: overlay recipe outputs and draw ROIs to average spectra.
- First PyInstaller build; measure bundle size and start-up time.
