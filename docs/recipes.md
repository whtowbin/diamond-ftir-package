# Recipes: your own baselines, peaks and peak clusters

A **recipe** tells the program what to measure beyond the built-in nitrogen analysis:
which region to look at, how to remove the local baseline there, and which peaks to
integrate or fit. Recipes are plain text files (TOML or JSON). You can write them by hand
or build them in the recipe editor, and every recipe runs the same way on a single
spectrum, a batch of files or a map.

Nothing in a recipe is specific to diamond: the same format works for Raman (Raman shift),
UV-VIS (wavelength) or other infrared samples. Set `normalize_by = "none"` when there is no
thickness to divide by.

## A first recipe

```toml
name = "hydrogen"
description = "3107 and 3085 cm-1 peaks"
normalize_by = "thickness"      # divide areas and heights by the fitted thickness

[[features]]
name = "H"
region = [3060, 3180]           # the part of the spectrum this feature uses
model = "integrate"             # integrate | gaussian | lorentzian | voigt | pseudo_voigt

[[features.baseline]]           # baseline steps run in order inside the region
method = "asls"
lam = 0.1
p = 6e-6
median_filter = 21

[[features.peaks]]
name = "3107"
center = 3107
integrate = [3103, 3110]

[[features.peaks]]
name = "3085"
center = 3085
integrate = [3082, 3088]
```

This is the built-in `diamond_hydrogen` recipe. It gives exactly the same numbers as the
package's own 3107 cm⁻¹ measurement, which is checked by a test.

Outputs are named `feature.peak.quantity`, for example `H.3107.area`. Fitted features
also report `feature.fit_r2`.

## Baseline steps

Steps run in order; each one is fitted to what the previous steps left.

| method | settings | use for |
|---|---|---|
| `asls` | `lam` (stiffness), `p` (asymmetry), `median_filter` | smooth backgrounds under peaks |
| `rubberband` | `stretch` (curvature added before the convex hull) | sharp or curved tails beside peaks |
| `polynomial` | `degree`, `anchors` = [[lo, hi], …] peak-free windows | backgrounds you can pin at known clean points |
| `linear` | `anchors` | a straight line between clean windows |
| `none` | – | already flat data |

Keep rubber bands **local** to the region that needs them. A rubber band over the whole
spectrum removed part of the diamond absorption and caused most of the old default's
thickness bias (see [methods_validation.md](methods_validation.md)).

## Single peaks and clusters

A feature with one peak is a single peak; a feature with several peaks is a **cluster**, and
its peaks are fitted together (with lmfit), so overlapping bands share the signal correctly.

Per peak: `center`, `center_range` (allowed [lo, hi]), `fwhm`, `fwhm_range`, and for
`model = "integrate"`, `integrate` limits.

**Constraints** tie cluster parameters together, written with peak names:

```toml
[features.constraints]
b_fwhm = "a_fwhm"                 # peak b has the same width as peak a
b_center = "a_center + 32.5"      # fixed spacing
b_amplitude = "0.5 * a_amplitude" # fixed intensity ratio
```

Parameters are `<peak>_center`, `<peak>_fwhm`, `<peak>_amplitude` (the area).

## Using recipes

- **Desktop app** (`diamond-ftir-app`): build and test a recipe in the *Recipe editor* tab
  on any loaded spectrum or map pixel. Drag the shaded region to set the feature's range;
  every edit re-runs the fit and shows each baseline step, the fit and the peaks. Then press
  *Use this recipe in the analysis*, or save it and tick it in the recipe list.
- **Command line**: `diamond-ftir run spectra/ --recipe my_peaks.toml -o results.csv`
  (repeat `--recipe` for several; `diamond-ftir map file.map --recipe …` for maps).
- **Python**:

  ```python
  from diamond_ftir_package.core import Recipe, run_recipe
  recipe = Recipe.load("my_peaks.toml")
  result = run_recipe(x, y, recipe, thickness=0.12)
  result.values          # {"feature.peak.area": ..., ...}
  ```

- **napari** (large maps): open the `.map` in napari and use the *Spectrum inspector* dock
  widget. It holds the same editor; click a pixel to preview, then run the whole map.

## Built-in recipes

| name | what | status |
|---|---|---|
| `diamond_hydrogen` | 3107 and 3085 cm⁻¹ areas | identical to the package's own measurement |
| `diamond_platelet` | platelet (B′) and 1405 cm⁻¹, fitted jointly | example; validate before relying on it |
| `diamond_amber` | nine amber-centre bands, fitted jointly | example; validate before relying on it |

Validate a new recipe with the injection method in
[methods_validation.md](methods_validation.md): add a known synthetic peak to real spectra
and check how much of it the recipe recovers.

## Known limitations

- Fits report values but not yet uncertainties. A peak that is absent can still get a small
  fitted area from noise (see TODO.md).
