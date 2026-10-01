# Citations still to add

Each row is a place where the code or docs use a published method, constant or assignment.
When you have the reference, replace the `[CITATION NEEDED: ...]` marker there and tick the box.
Search for the marker with `grep -rn "CITATION NEEDED" .`.

| Done | Topic | Where in the code | Notes / hint from existing comments |
|---|---|---|---|
| [~] | A-centre coefficient 16.5 ppm/cm⁻¹ | `params.py` `a_ppm_per_cm` | Boyd, Kiflawi & Woods (1994) Phil. Mag. B 69, 1149-1153 (see docs/methods.md#references) |
| [~] | B-centre coefficient 79.4 ppm/cm⁻¹ | `params.py` `b_ppm_per_cm` | Boyd, Kiflawi & Woods (1995) Phil. Mag. B 72, 351-361 |
| [ ] | C-centre coefficient 0.6243 ppm/cm⁻¹ | `params.py` `c_ppm_per_cm` | |
| [~] | C-centre resolution factor (30, 37, 42, 65 at 0.5, 1, 2, 4 cm⁻¹) | `DiamondSpectrum.C_center_resolution_correction` | Table as in DiaMap, "from Liggins 2010 PhD thesis" (Warwick); get the thesis reference and confirm whether it is indexed by resolution or sampling interval. Replaced the earlier linear fit 9.7043·s + 25.304 |
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
| [~] | ASLS baseline | `baseline_als`, `als_baseline`, joint baseline | Eilers & Boelens (2005), Leiden report (see docs/methods.md#references) |
| [ ] | Whittaker smoother | `WhittakerSmoother` | |
| [ ] | pybaselines | dependency | |
| [ ] | Rubber-band correction | `rubberband` | Implementation adapted from a StackExchange answer (R. Kiselev) |
| [ ] | ALS implementation (Stack Overflow) | `baseline_als` | Adapted from user "sparrowcide"; check licence terms |
| [ ] | SPC file reader | `SPC/` | https://github.com/rohanisaac/spc; check licence and add notice |
| [~] | Bundled reference data provenance and licence | `CAXBDY.*`, `typeIIA.*` | From QUIDDIT (above); check QUIDDIT licence terms for redistribution. Y component supplied by Maxwell C. Day (Univ. Padova); Y centre first reported by Hainschwang, Fritsch, Notari & Rondeau (2012), Diamond Relat. Mater. 21, 120-126, doi:10.1016/j.diamond.2011.11.002. |
| [~] | Platelet thermometer (Eqs. 1, 10-13; calibrations) | `thermometry.py` | Speich, Kohn, Bulanova & Smith (2018) Contrib. Mineral. Petrol. 173:39 — cited in docs/thermometry.md |
| [~] | QUIDDIT platelet and 3107 fitting procedure | `thermometry.fit_platelet`, `fit_3107` | Speich & Kohn (2020) Computers & Geosciences 144:104558 |
| [~] | Nitrogen aggregation kinetics constants (81,160 K; 293,608) | `thermometry.nitrogen_temperature` | Taylor, Jaques & Ridd (1990) Am. Mineral. 75, 1290-1310; revised constants Taylor, Canil & Milledge (1996) GCA 60 — verify the 1996 details |
| [ ] | SNAC paper | `_vendor/snac`, `thermometry.snac_cooling_model` | Wincott et al. 2026 — full reference needed |
| [~] | D-component limit: 0.365 adopted (Woods 1986; Speich et al. 2018; QUIDDIT); DiaMap uses 0.435 (no source given) | `params.NitrogenParams.d_limit` | Woods G.S. (1986) Proc. R. Soc. Lond. A 407, 219-238; confirm DiaMap's 0.435 with D. Howell |
| [~] | Resampling to 1 cm⁻¹ with cubic spline | `Spectrum.interpolate`, `resample_values` | Method choice by test (docs/methods.md §1); no external citation needed |
| [~] | Resolution ≈ 2 × sampling interval (Nyquist) | `DiamondSpectrum.instrument_resolution` | Nyquist-Shannon sampling theorem; instrument settings |
| [~] | DiaMap nitrogen fitting workbook | `Nitrogen_fit` docstring ("Diamap Howell et al.") | Howell et al. (2012) Diamond Relat. Mater. 29, 29-36; Howell et al. (2012) Contrib. Mineral. Petrol. 164, 1011-1025 |
| [~] | Platelet size and aspect ratio from peak position | `thermometry.platelet_diameter_nm`, `platelet_aspect_ratio` | Speich et al. (2017) Lithos 278-281, 419-426 |
