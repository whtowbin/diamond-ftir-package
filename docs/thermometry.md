# Thermometry: nitrogen aggregation and platelet degradation

Two model temperatures can be calculated from a normalised diamond spectrum and a mantle
residence time:

- **T_N**: nitrogen aggregation. How far A-centres have aggregated to B-centres depends
  on temperature and time.
- **T_P**: platelet degradation (Speich et al. 2018). Platelets (the B′ peak near
  1365 cm⁻¹) form as nitrogen aggregates and are destroyed at high temperature. Comparing
  the measured platelet peak area with the area expected for the diamond's B-centres gives a
  second, independent thermometer. It applies to diamonds whose nitrogen aggregation is
  (nearly) complete.

A third tool, **SNAC** (Wincott et al. 2026), models nitrogen aggregation while the diamond
cools, using core, rim and kimberlite ages.

## Using it

```bash
diamond-ftir run spectra/ --duration-ma 2900 -o results.csv
diamond-ftir map sample.map --duration-ma 2900 -o Results
```

In the desktop app: *Thermometry* tab → *duration ma*. In Python:
`params.thermometry.duration_ma = 2900`. A duration of 0 (the default) turns thermometry off.

The mantle residence time is the time between diamond crystallisation (inclusion ages) and
eruption. Speich et al. (2018, Table 1) list examples (for instance Argyle eclogitic 0.40 Ga,
Diavik peridotitic 3.25 Ga, Murowa 2.9 Ga). For long residence times (>1 Ga) an error of a
few hundred Ma changes T_N only slightly; for short ones it matters.

### Output columns

| column | meaning |
|---|---|
| `T_N (C)` | nitrogen aggregation model temperature |
| `H3107_area_qd`, `H3107_height_qd` | 3107 cm⁻¹ peak fitted as in QUIDDIT |
| `platelet_area_qd` | platelet (B′) peak area I(B′), cm⁻², fitted as in QUIDDIT |
| `platelet_x0_qd`, `platelet_width_qd`, `platelet_symmetry_qd` | peak position, width (HWHM_left + HWHM_right), position − centroid |
| `platelet_P0` | expected area without degradation, 64 × μB |
| `platelet_remaining` | I(B′) / P0: 1 = regular, below 1 = platelets degraded |
| `platelet_position_dev` | peak position minus the position expected for the measured area (Eq. 3); regular diamonds 0 ± 2 cm⁻¹ |
| `T_P (C)` | platelet degradation temperature; empty when no degradation is measurable |
| `platelet_diameter_nm`, `platelet_aspect_ratio` | average platelet size from the peak position, D = 221 / (x − 1360) nm and AR = 11.9 / (D + 19.6) + 0.4 (Speich et al. 2017) |

## Equations

**Nitrogen aggregation** (second-order A→B kinetics; Taylor et al. 1990, with the
activation energy and pre-exponential factor revised in Taylor et al. 1996, as used by
QUIDDIT and SNAC):

  T_N = −(Ea/R) / ln[ (N_T/N_A − 1) / (t · N_T · A) ],  Ea/R = 81,160 K, A = 293,608 ppm⁻¹ s⁻¹

where N_T is total nitrogen (ppm), N_A the nitrogen in A-centres and t the residence time (s).

**Platelet degradation** (Speich et al. 2018):

- Eq. 1: expected (undegraded) platelet area P0 = 64 × μB, where μB = [N_B] / 79.4 is the
  absorption due to B-centres (Eq. 2 is an alternative slope of 60.9).
- Eq. 10-11: first-order degradation, P_t = P0 · e^(−k_P t), so k_P = ln(P0 / P_t) / t.
- Eq. 12-13: k_P = A_P · e^(−Ea/RT), so T_P = (Ea/R) / (ln A_P − ln k_P).

| calibration | Ea/R | ln A_P | source |
|---|---|---|---|
| `combined` (default) | 88,446 K | 19.687 | natural + experimental data (paper: 88.4 ± 0.9 × 10³ K, 19.7 ± 0.5, R² 0.996); unrounded values as in QUIDDIT |
| `natural` | 73,800 K | 9.9 | natural diamonds only (73.8 ± 7.2 × 10³ K, 9.9 ± 4.8, R² 0.820) |

The paper reproduces nitrogen-aggregation temperatures within ± 50 °C (mostly ± 30 °C)
with Eq. 13.

## Platelet peak fitting (as in QUIDDIT)

The thermometer was calibrated on platelet areas measured with QUIDDIT, so the platelet
peak is fitted the same way (Speich & Kohn 2020):

1. **3107 cm⁻¹ peak.** Cubic baseline through 3000-3050 and 3150-3200 cm⁻¹, then an
   asymmetric pseudo-Voigt fitted on a 0.1 cm⁻¹ grid.
2. **Platelet region (1327-1420 cm⁻¹).** The platelet, 1405 and 1332 cm⁻¹ peaks (asymmetric
   pseudo-Voigts) plus a constant are fitted together, with QUIDDIT's bounds:
   - platelet position within ± 1.5 cm⁻¹ of the maximum in 1350-1380;
   - 1405 height within ± 20% of 0.257 × the 3107 height (QUIDDIT's empirical relation);
   - 1332 height within ± 10% of the measured value;
   - each peak's left and right widths within 10 cm⁻¹ of each other.
3. **Area** = I · (HWHM_l + HWHM_r) · (σπ/2 + (1 − σ)√(π/2)).

**QUIDDIT compatibility.** QUIDDIT's code uses the 1405 peak's height in the Gaussian part
of the 1332 peak (a slip in `ultimatepsv`).
- The default (`quiddit_compatible = False`) uses the corrected model.
- `True` reproduces QUIDDIT exactly, so areas match published QUIDDIT numbers.

On 19 map spectra the corrected fit gave platelet areas a median 2.9% larger (range
0.9-4.7%; one weak peak 38%). That changes T_P by −0.6 to −3.5 °C, well inside the method's
± 30-50 °C.

### Validation

| check | result |
|---|---|
| T_N vs SNAC's implementation, same inputs | identical except 0.15 °C (273.15 K here, 273 K in SNAC) |
| T_N, paper example (Fig. 8: > 2000 ppm N, 49% B, Diavik peridotitic 3.25 Ga) | 1,093 °C (paper: 1,090 °C) |
| T_P vs QUIDDIT's formula, same inputs | identical except 0.17 °C (Kelvin offset, 365 vs 365.25-day year) |
| Paper's worked example (P_t ≈ 1, P0 ≈ 500 cm⁻², 1640 °C) | 66 ka residence (paper: ca. 64 ka, from rounded inputs) |
| Platelet and 3107 fits vs QUIDDIT's own code on the same 19 normalised map spectra | with `quiddit_compatible=True`: identical (area ratio 1.0000, position and symmetry differences 0.000); 3107 areas identical in both modes |
| Injected asymmetric platelet peak of known area (synthetic) | recovered within 8%, both modes |

## Interpreting results: limits from the paper

- T_P applies to **irregular** (platelet-degraded) diamonds. **Subregular** diamonds (low
  temperature, platelets smaller than expected, position shifted to higher wavenumber) are
  not degraded and the method overestimates their temperature. Check
  `platelet_position_dev` and `platelet_remaining` (diagrams one and two of the paper).
- **Mixing zones**: a beam crossing growth zones mixes their signals; mixing is linear in
  diagram one but not in the others.
- The residence time carries a large systematic uncertainty, especially for rims and short
  histories; the two-stage model (Kohn et al. 2016) or SNAC can help.
- Plastic deformation may also degrade platelets and is not accounted for.

## SNAC: aggregation during cooling

```python
from diamond_ftir_package.thermometry import snac_cooling_model
model = snac_cooling_model(age_core_ma=3520, age_rim_ma=1860, age_kimberlite_ma=0,
                           core_total_n=625, core_b_fraction=0.863,
                           rim_total_n=801, rim_b_fraction=0.197)
model.plot_T_history(); model.save_history("history.csv")
```

SNAC is included in `diamond_ftir_package/_vendor/snac` (MIT licence, pinned commit, see
`VENDORED.md`). It isn't on PyPI, and the PyPI name `snac` belongs to an unrelated package.

## Comparison with DiaMap (Howell et al. 2012)

DiaMap v18 (Howell's Excel workbook, `DiaMap_v18_001_ForNitrogenFitting.xlsx`) was compared
with this package using the workbook's own worked example.

| quantity | DiaMap | this package | note |
|---|---|---|---|
| type IIa scale factor (full pipeline from DiaMap's raw spectrum) | 0.01530 | 0.01697 | joint baseline vs DiaMap's linear baseline |
| A-centre N (full pipeline) | 848.4 ppm | 768.0 ppm | follows the scale factor (−9.5%) |
| A / B from DiaMap's own normalised spectrum (package fit, 1001-1350) | 848.4 / 439.3 ppm | 851.1 / 394.5 ppm | DiaMap fits 1001-1399, including the platelet peak |
| %B | 34.1 | 31.7 | |
| 3107 height | 60.81 | 54.78 | different local baselines |
| platelet area I(B′) | 115.0 cm⁻² | 266.0 cm⁻² | different methods (below) |
| platelet peak position | 1378 | 1377.6 | |

What the comparison showed:

- **The same reference spectra, but not identical.** DiaMap's C, A, B and D component
  spectra match the package's (QUIDDIT's) in scale (median ratios 0.993-1.001), but differ
  locally by up to about 6% (B). DiaMap's own solution has a residual of 227.7 with its
  components and 1,031 with the package's. The small shape differences alone can move B% by a
  few points.
- **Platelet area is method-dependent.** DiaMap sums the positive part of (spectrum − a
  straight baseline between anchor windows, e.g. 1366 and 1414 cm⁻¹, with the 1405 feature
  removed). QUIDDIT fits a pseudo-Voigt whose Lorentzian tails add area. The thermometer's
  P0 = 64 × μB was calibrated on QUIDDIT areas, so DiaMap-type areas must not be used with
  it.
- **Platelet size.** DiaMap labels the Speich et al. (2017) diameter in μm; the paper gives
  nm (D = 221 / (x − 1360) nm). The package uses nm.
- **D-component limit: discrepancy recorded, 0.365 adopted.** DiaMap uses D ≤ 0.435 × B
  (cell I7 of the *Nitrogen Fit* sheet, no source given). Speich et al. (2018) and QUIDDIT use
  0.365 × B, citing Woods (1986, *Proc. R. Soc. Lond. A* 407, 219-238), the published source
  for the D-B relation. The component spectra have the same scaling in both tools, so this is a
  real difference in the limit, not a units effect. **The package now uses 0.365**
  (`NitrogenParams.d_limit`; set 0.435 to reproduce DiaMap). Earlier versions of this package
  used 0.435, copied from DiaMap. In DiaMap's worked example the limit did not bind in the
  package's fit (both values gave the same result). Where it binds, a lower limit lowers D and
  can raise B (and so P0 and T_N slightly).

## Open points

- **D-component limit**: 0.365 adopted (Woods 1986); still to confirm with D. Howell where
  DiaMap's 0.435 came from, in case it reflects a later calibration.
- **Is the type IIa factor the thickness, or thickness ÷ 2.303?** DiaMap computes thickness
  as 2.303 × its type IIa factor (natural-log absorption coefficient). The package and QUIDDIT
  (normalising to 12.3 at 1992 cm⁻¹) treat the reference as absorbance per cm. Nitrogen ppm
  and T_P do not depend on this (all three divide by the same fitted factor), but the reported
  `typeIIA_ratio` and a user-entered known thickness (maps, `thickness_mode="known"`) do. A
  plate of measured thickness settles it: 1.00 mm should give a factor of about 0.10 in one
  convention and 0.043 in the other.
- **QUIDDIT's reference spectra licence.** The QUIDDIT README says "you are free to download
  and use QUIDDIT and all its components", but there is no licence file. Ask the author for
  explicit permission to redistribute the bundled CAXBD / type IIa spectra before a public
  release.

## References

- Speich L., Kohn S.C., Bulanova G.P., Smith C.B. (2018). The behaviour of platelets in
  natural diamonds and the development of a new mantle thermometer. *Contributions to
  Mineralogy and Petrology* 173:39.
- Speich L., Kohn S.C., Wirth R., Bulanova G.P., Smith C.B. (2017). The relationship between
  platelet size and the B′ infrared peak of natural diamonds revisited. *Lithos* 278-281,
  419-426.
- Speich L., Kohn S.C. (2020). QUIDDIT - QUantification of infrared active Defects in Diamond
  and Inferred Temperatures. *Computers & Geosciences* 144, 104558.
  doi:10.1016/j.cageo.2020.104558
- Wincott et al. (2026), SNAC. [CITATION NEEDED: full reference]
- Howell D., O'Neill C.J., Grant K.J., Griffin W.L., Pearson N.J., O'Reilly S.Y. (2012).
  μ-FTIR mapping: distribution of impurities in different types of diamond growth. *Diamond
  and Related Materials* 29, 29-36.
- Howell D., O'Neill C.J., Grant K.J., Griffin W.L., O'Reilly S.Y., Pearson N.J., Stern R.A.,
  Stachel T. (2012). Platelet development in cuboid diamonds: insights from micro-FTIR
  mapping. *Contributions to Mineralogy and Petrology* 164, 1011-1025.
- Woods G.S. (1986). Platelets and the infrared absorption of type Ia diamonds. *Proceedings
  of the Royal Society of London A* 407, 219-238.
- Taylor W.R., Jaques A.L., Ridd M. (1990); Taylor W.R., Canil D., Milledge H.J. (1996).
  [CITATION NEEDED: full references]
- Woods G.S. (1986); Boyd S.R. et al. (1994, 1995); Kohn S.C. et al. (2016); Navon O. et al.
  (2017). [CITATION NEEDED: full references]
