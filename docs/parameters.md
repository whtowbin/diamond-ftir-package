# Parameter reference

Generated from `src/diamond_ftir_package/params.py` by `scripts/make_parameter_docs.py`; edit descriptions there. Every default equals the value that was hardcoded before parameters were exposed. Settings are saved and loaded as JSON (`diamond-ftir defaults` prints them all).

## Thickness and saturation

| Setting | Default | What it does |
|---|---|---|
| `saturation_cutoff` | `2.5` | Absorbance above which a diamond peak counts as saturated. |
| `stdev_cut_off` | `0.5` | Noise (std. dev.) above which a diamond peak counts as saturated. |
| `baseline_algorithm` | `Whittaker` | Baseline algorithm: 'Whittaker' (fast) or 'ALS' (slower, more stable). |
| `baseline_method` | `joint` | 'joint': ASLS baseline and type IIa thickness fitted together (least biased in injection tests). 'multistage': median filter, mild ASLS and rubber band before the main baseline. 'single': one ASLS baseline only (the original mapping method). |
| `joint_lam` | `1000000000.0` | 'joint' only: baseline smoothness (lambda) on the 1 cm-1 grid. |
| `joint_p` | `0.001` | 'joint' only: baseline asymmetry (p). |
| `min_diamond_r2` | `0.8` | Quality flag: spectra whose type IIa shape match (R²) is below this are flagged as not diamond-like. Flagged, never excluded. |
| `baseline_search` | `fixed` | 'fixed': use lam and p as given. 'optimize': local search for lam and p so the diamond peaks best match the type IIa shape and 4000-5900 cm-1 is flat. 'grid': coarse grid over the bounds, then a local search (slower, avoids getting stuck). |
| `lam` | `10000000000.0` | Main baseline smoothness (lambda); the starting point when optimising. |
| `p` | `5e-06` | Main baseline asymmetry (p); the starting point when optimising. |
| `log10_lam_bounds` | `(5, 12)` | Search range for log10(lambda) when optimising. |
| `log10_p_bounds` | `(-8, -2)` | Search range for log10(p) when optimising. |
| `search_width` | `1.0` | When a start point is supplied (maps: the sample's typical lam and p), search only this many decades either side of it. |
| `search_tolerance` | `0.02` | Stop the search when log10(lam, p) move less than this. |
| `search_max_evals` | `200` | Maximum baseline evaluations per spectrum when optimising. |
| `grid_points` | `7` | 'grid' search: points per axis of the coarse grid (n x n evaluations). |
| `flat_range` | `(4000, 5900)` | Region (cm-1) that should be flat after baseline removal. |
| `flat_weight` | `0.5` | Weight of the flatness term against the type IIa shape term in the search. |
| `pre_median_filter` | `21` | Multistage only: median filter width (points). |
| `pre_lam` | `10000000000.0` | Multistage only: smoothness of the mild first ASLS baseline. |
| `pre_p` | `0.0005` | Multistage only: asymmetry of the mild first ASLS baseline. |
| `pre_use_rubberband` | `True` | Multistage only: subtract the stretched rubber band. On synthetic tests it causes most of the thickness under-estimate (see docs/baseline_benchmark.md). |
| `pre_rubber_stretch` | `2e-08` | Multistage only: curvature added for the rubber band. |

## Nitrogen

| Setting | Default | What it does |
|---|---|---|
| `wn_low` | `950` | Lower wavenumber (cm-1) of the nitrogen fit window. |
| `wn_high` | `1350` | Upper wavenumber (cm-1) of the nitrogen fit window. |
| `max_C_or_B` | `0.01` | Max size of the minor component (C in IaAB, B in Ib) relative to the major one. |
| `a_ppm_per_cm` | `16.5` | ppm N per cm-1 of A-component absorption. |
| `b_ppm_per_cm` | `79.4` | ppm N per cm-1 of B-component absorption. |
| `c_ppm_per_cm` | `0.624332796` | ppm N per cm-1 of C-component absorption (before resolution correction). |
| `d_limit` | `0.435` | Max D-component size relative to the major component in type IaAB fits. |

## Hydrogen (3107 / 3085)

| Setting | Default | What it does |
|---|---|---|
| `window` | `(3060, 3180)` | Region used for local baseline fitting. |
| `median_filter` | `21` | Median filter width (points) before baseline fitting. |
| `baseline_lam` | `0.1` | ASLS smoothness (lambda) for the local baseline. |
| `baseline_p` | `6e-06` | ASLS asymmetry (p) for the local baseline. |
| `peak_3107` | `(3103, 3110)` | Integration limits for the 3107 peak. |
| `peak_3085` | `(3082, 3088)` | Integration limits for the 3085 peak. |

## Platelets and the 1405 peak

| Setting | Default | What it does |
|---|---|---|
| `region` | `(1340, 1500)` | Region used for baseline fitting. |
| `search_window` | `(1355, 1380)` | Where to look for the platelet peak. |
| `noise_window` | `(1380, 1450)` | Region used to estimate noise for peak thresholds. |
| `baseline_lam` | `1000` | ASLS smoothness (lambda) for the first baseline. |
| `baseline_p` | `0.001` | ASLS asymmetry (p) for the first baseline. |
| `peak_1405` | `(1403, 1407)` | Integration limits for the 1405 peak. |

## Amber centres

| Setting | Default | What it does |
|---|---|---|
| `bands` | `((4065, 4060, 10), (4165, 4160, 10), (4211, 4211, 10), (4354, 4354, 10), (4495, 4495, 5), (4660, 4660, 20), (4740, 4740, 20), (4850, 4850, 5), (4950, 4950, 20))` | Amber-centre bands to integrate as (label, centre, half-width) in cm-1. |

## Which measurements run

| Setting | Default |
|---|---|
| `run_nitrogen` | `True` |
| `run_hydrogen` | `True` |
| `run_platelets` | `True` |
| `run_amber` | `False` |
