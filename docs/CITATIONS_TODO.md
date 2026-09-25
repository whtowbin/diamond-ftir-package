# Citations still to add

Each row is a place where the code or docs use a published method, constant or assignment.
When you have the reference, replace the `[CITATION NEEDED: ...]` marker there and tick the box.
Search for the marker with `grep -rn "CITATION NEEDED" .`.

| Done | Topic | Where in the code | Notes / hint from existing comments |
|---|---|---|---|
| [ ] | A-centre coefficient 16.5 ppm/cm⁻¹ | `params.py` `a_ppm_per_cm` | |
| [ ] | B-centre coefficient 79.4 ppm/cm⁻¹ | `params.py` `b_ppm_per_cm` | |
| [ ] | C-centre coefficient 0.6243 ppm/cm⁻¹ | `params.py` `c_ppm_per_cm` | |
| [ ] | C-centre spectral-spacing correction `9.7043·s + 25.304` | `DiamondSpectrum.py` `C_center_wn_spacing_correction` | Code comment says Liggins 2010 PhD thesis (Warwick); verify |
| [ ] | D-component limit 0.435 and D–B correlation | `params.py` `d_limit`; `Nitrogen_fit` | Docstring says "Woods" |
| [~] | CAXBDY component spectra / CAXBDY97n | `CAXBDY.py`, `Nitrogen_fit` | Spectra taken from QUIDDIT: L. Speich & S.C. Kohn (2020), Computers & Geosciences 144, 104558, doi:10.1016/j.cageo.2020.104558 (see `Quiddit_Spectra_Files/__init__.py`). Original CAXBD spectra: D. Fisher, De Beers Technologies (CAXBDY97n) — still to cite. |
| [ ] | Diamap program | `Nitrogen_fit` docstring | Howell et al. |
| [ ] | Nitrogen fitting work | `Nitrogen_fit` docstring | Specht et al. |
| [ ] | Type IIa reference spectrum | `typeIIA.py`, class docstring | De Beers Technologies? |
| [ ] | Saturation-based choice of two-phonon regions | `test_diamond_saturation` | |
| [ ] | 3107 cm⁻¹ / N₃VH assignment | `measure_3107_peak` | |
| [ ] | 3085, 3237, 2785 cm⁻¹ hydrogen peaks; NVH⁰ 3123 | TODOs in `DiamondSpectrum.py` | |
| [ ] | Platelet (B′) peak 1355-1380 cm⁻¹ | `measure_platelets_and_adjacent` | |
| [ ] | 1405 cm⁻¹ peak (N₃VH bending mode) | `measure_platelets_and_adjacent` | |
| [ ] | 1450 cm⁻¹ radiation peak | comment in `measure_platelets_and_adjacent` | |
| [ ] | Amber-centre bands (4060…4950 cm⁻¹) | `params.py` `AmberParams` | |
| [ ] | ASLS baseline | `baseline_als`, `als_baseline` | Eilers & Boelens 2005 |
| [ ] | Whittaker smoother | `WhittakerSmoother` | |
| [ ] | pybaselines | dependency | |
| [ ] | Rubber-band correction | `rubberband` | Implementation adapted from a StackExchange answer (R. Kiselev) |
| [ ] | ALS implementation (Stack Overflow) | `baseline_als` | Adapted from user "sparrowcide"; check licence terms |
| [ ] | SPC file reader | `SPC/` | https://github.com/rohanisaac/spc; check licence and add notice |
| [~] | Bundled reference data provenance and licence | `CAXBDY.*`, `typeIIA.*` | From QUIDDIT (above); check QUIDDIT licence terms for redistribution. Y component supplied by Maxwell C. Day (Univ. Padova); Y centre first reported by Hainschwang, Fritsch, Notari & Rondeau (2012), Diamond Relat. Mater. 21, 120-126, doi:10.1016/j.diamond.2011.11.002. |
