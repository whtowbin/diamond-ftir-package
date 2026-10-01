# Methods

How each result is calculated. Every `[CITATION NEEDED: ...]` marks a source still to be added;
they are collected in [CITATIONS_TODO.md](CITATIONS_TODO.md).

## 1. Preparing the spectrum

Spectra are resampled to 1 cm⁻¹ spacing between 601 and 6000 cm⁻¹ (limited by the reference
spectrum range and by the data), using a cubic spline (`Spectrum.interpolate(method=...)`;
Akima, makima, PCHIP and linear are available). This gives every later step the same grid
whatever the instrument's native resolution.

Cubic spline was chosen by test: sharp synthetic peaks were sampled at 0.48, 0.96, 1.93 and
7.7 cm⁻¹ with random phase and resampled to 1 cm⁻¹. Cubic spline had the lowest RMS error and
the smallest loss of peak height at every spacing (mean −4 to −6%, worst about −10%), against
−5 to −8% for Akima (the previous method) and −8 to −11% for linear. Some loss is unavoidable,
because a peak top usually falls between samples. Interpolation does not change the measurement's true
resolution; the C-centre calibration uses that instrument resolution (section 3).

## 2. Thickness normalisation and baseline

The sample is scaled against a type IIa diamond reference spectrum
[CITATION NEEDED: origin of the type IIa reference spectrum (De Beers?)].

1. **Saturation test.** Three intrinsic two-phonon regions are tested in turn: 1970-2040,
   2400-2575, then 3000-3500 cm⁻¹. A region counts as saturated when its mean absorbance exceeds
   `saturation_cutoff` or its noise exceeds `stdev_cut_off`. The first unsaturated region
   (plus 3130-3500 cm⁻¹) decides where the fit is trusted. If none qualifies the file fails.
2. **Baseline and thickness (default: joint fit).** A smooth ASLS baseline and the thickness
   factor t are fitted together: spectrum = baseline + t × type IIa, with ASLS's asymmetric
   weights and saturated points weighted 0 (λ = 1e9, p = 1e-3). Validation on 1,272 real
   spectra: [methods_validation.md](methods_validation.md)
   [CITATION NEEDED: ASLS baseline, Eilers & Boelens 2005].
   The older options remain: `multistage` (median filter, mild ASLS and a stretched rubber band,
   then a main ASLS baseline) and `single` (one ASLS baseline, fixed or searched), with the
   thickness factor as the mean ratio to the reference over the trusted region
   [CITATION NEEDED: pybaselines package] [CITATION NEEDED: rubber-band correction].
3. **Thickness factor.** `typeIIA_ratio` is the fitted scale of the type IIa reference.
   Dividing by it gives the spectrum for a 1 cm path in the reference's units. Whether it
   equals the thickness in cm or the thickness ÷ 2.303 is still open (see
   [thermometry.md](thermometry.md#open-points)); nitrogen ppm does not depend on it.

## 3. Nitrogen (A, B, C centres)

Between `wn_low` and `wn_high` (default 950-1350 cm⁻¹) the normalised spectrum is fitted with a
bounded linear least-squares combination of reference component spectra
(C, A, X, B, D, Y) plus a constant and a linear term
[CITATION NEEDED: CAXBDY component spectra; Fisher, De Beers CAXBDY97n spreadsheet]
[CITATION NEEDED: Diamap, Howell et al.]
[CITATION NEEDED: Specht et al.].

- A first, unconstrained fit decides whether the stone is closer to type Ib (C-dominated) or
  IaAB.
- A second fit applies limits so the minor component cannot grow past `max_C_or_B` times the
  major one; for IaAB stones the D component is also limited (`d_limit`, see below).
- Component heights become atomic ppm using absorption coefficients:
  A = 16.5 and B = 79.4 ppm per cm⁻¹
  [CITATION NEEDED: A- and B-centre calibration coefficients (Boyd et al. 1994, 1995)],
  C = 0.6243 ppm per cm⁻¹ × a factor that depends on the **instrument's spectral
  resolution**. The factor is interpolated through Liggins' values of 30, 37, 42 and 65 at
  0.5, 1, 2 and 4 cm⁻¹, the same table DiaMap uses
  [CITATION NEEDED: Liggins 2010 PhD thesis, University of Warwick].
- **Resolution, not point spacing.** Resolution here means the instrument setting, which is
  about **twice the sampling interval** (Nyquist). A spectrum labelled "4wnRes" or "2res" has
  points roughly every 2 or 1 cm⁻¹; manufacturers round, so it is never exact (4 cm⁻¹ data
  are typically stored every 1.93 cm⁻¹). Some papers and tools call the sampling interval
  the resolution; check which one a source means.
  - The analysis grid is 1 cm⁻¹, so it can represent at best about 2 cm⁻¹ resolution.
  - Resampling onto it does not change a coarser spectrum's resolution (4 cm⁻¹ data stay
    4 cm⁻¹). Finer data are resampled without smoothing.
  - The C factor therefore always uses the instrument resolution of the original measurement.

  The resolution comes from, in order:
  1. the `resolution_cm` setting;
  2. file metadata;
  3. the file name (e.g. `4wnRes`, `2res`; for maps, the map's file name applies to every
     pixel);
  4. **2 × the original point spacing**, flagged in `QA` whenever C is non-zero. Zero-filling
     makes the spacing smaller still, so this can underestimate.

  Other rules:
  - Resolutions above 4 cm⁻¹ (e.g. melee measured at about 16 cm⁻¹) lie beyond Liggins' table.
    The factor is extrapolated and C-centre values are flagged.
  - Set `resolution_cm` when C-centres matter. Results report `resolution_cm` and
    `resolution_source`.
  - **Open:** it is not yet confirmed whether Liggins' table is indexed by true resolution
    (as DiaMap labels it) or by sampling interval. Check the thesis.
- **Reference spectra.** The CAXBD / type IIa component spectra are on a 1 cm⁻¹ grid and are
  believed to be about 2 cm⁻¹ resolution (to be confirmed). Spectra measured at coarser
  resolution have broader sharp features (1332, 1344 cm⁻¹) than the references; broadening
  the references to the sample's resolution is a possible improvement (TODO).
- The D component is limited to 0.365 × B in IaAB fits (Woods 1986, as used by Speich et al.
  2018 and QUIDDIT). DiaMap uses 0.435; see [thermometry.md](thermometry.md#comparison-with-diamap-howell-et-al-2012).
- `B_percent` is B / (A + B + C) × 100. If total nitrogen is zero it is reported as NaN.

## 4. Hydrogen peaks (3107, 3085 cm⁻¹)

A local ASLS baseline is fitted in 3060-3180 cm⁻¹ after a median filter and subtracted; the
peaks are integrated over 3103-3110 and 3082-3088 cm⁻¹ and divided by `typeIIA_ratio`.
The 3107 cm⁻¹ line is assigned to the N₃VH defect
[CITATION NEEDED: 3107 cm⁻¹ assignment and its use as a natural/synthetic indicator].

## 5. Platelet peak and 1405 cm⁻¹ peak

In 1340-1500 cm⁻¹ two baselines are removed (ASLS, then an aggressive rubber-band). Peaks are
found with height and prominence thresholds from the noise in 1380-1450 cm⁻¹. The platelet peak
is the tallest one in 1355-1380 cm⁻¹; its area is integrated over its half-height width
[CITATION NEEDED: platelet (B′) peak position and behaviour]. The 1405 cm⁻¹ peak is reported only
when it rises above twice the local noise, and is thought to be the N₃VH bending mode
[CITATION NEEDED: 1405 cm⁻¹ assignment].

## 6. Amber centres

In 4000-5100 cm⁻¹ a multi-stage baseline is removed and nine bands are integrated over the
windows in `docs/parameters.md`
[CITATION NEEDED: amber-centre band positions and their link to plastic deformation].

## Reference data

`CAXBDY` and `typeIIA` spectra ship with the package as Python dictionaries
[CITATION NEEDED: provenance and licence of the bundled reference spectra].

## References

Full references for sources cited on this page and in [thermometry.md](thermometry.md).
Entries marked *(verify)* still need their details checked against the original.

- Boyd S.R., Kiflawi I., Woods G.S. (1994). The relationship between infrared absorption and
  the A defect concentration in diamond. *Philosophical Magazine B* 69, 1149-1153. (A-centre
  coefficient, 16.5 ppm·cm)
- Boyd S.R., Kiflawi I., Woods G.S. (1995). Infrared absorption by the B nitrogen aggregate in
  diamond. *Philosophical Magazine B* 72, 351-361. (B-centre coefficient, 79.4 ppm·cm)
- Eilers P.H.C., Boelens H.F.M. (2005). Baseline correction with asymmetric least squares
  smoothing. Leiden University Medical Centre report. (ASLS baselines)
- Howell D., O'Neill C.J., Grant K.J., Griffin W.L., Pearson N.J., O'Reilly S.Y. (2012).
  μ-FTIR mapping: distribution of impurities in different types of diamond growth. *Diamond
  and Related Materials* 29, 29-36. (DiaMap)
- Howell D., O'Neill C.J., Grant K.J., Griffin W.L., O'Reilly S.Y., Pearson N.J., Stern R.A.,
  Stachel T. (2012). Platelet development in cuboid diamonds: insights from micro-FTIR
  mapping. *Contributions to Mineralogy and Petrology* 164, 1011-1025. (DiaMap)
- Kohn S.C., Speich L., Smith C.B., Bulanova G.P. (2016). FTIR thermochronometry of natural
  diamonds: a closer look. *Lithos* 265, 148-158. *(verify)* (two-stage annealing model)
- Liggins S. (2010). PhD thesis, University of Warwick. *(verify title)* (C-centre factor
  versus resolution; the table used here is as given in DiaMap)
- Speich L., Kohn S.C., Wirth R., Bulanova G.P., Smith C.B. (2017). The relationship between
  platelet size and the B′ infrared peak of natural diamonds revisited. *Lithos* 278-281,
  419-426. (platelet diameter D = 221 / (x − 1360) nm; aspect ratio)
- Speich L., Kohn S.C., Bulanova G.P., Smith C.B. (2018). The behaviour of platelets in
  natural diamonds and the development of a new mantle thermometer. *Contributions to
  Mineralogy and Petrology* 173:39. (platelet thermometer; D ≤ 0.365 × B citing Woods 1986)
- Speich L., Kohn S.C. (2020). QUIDDIT - QUantification of infrared active Defects in Diamond
  and Inferred Temperatures. *Computers & Geosciences* 144, 104558.
  doi:10.1016/j.cageo.2020.104558 (reference spectra; platelet and 3107 fitting procedure)
- Taylor W.R., Jaques A.L., Ridd M. (1990). Nitrogen-defect aggregation characteristics of
  some Australasian diamonds: time-temperature constraints on the source regions of pipe and
  alluvial diamonds. *American Mineralogist* 75, 1290-1310. (A→B aggregation kinetics)
- Taylor W.R., Canil D., Milledge H.J. (1996). *Geochimica et Cosmochimica Acta* 60,
  4725-4733. *(verify: cited by Speich et al. 2018 for the revised A→B constants)*
- Wincott et al. (2026). SNAC: simultaneous nitrogen aggregation and cooling. *(full
  reference needed)*
- Woods G.S. (1986). Platelets and the infrared absorption of type Ia diamonds. *Proceedings
  of the Royal Society of London A* 407, 219-238. (D component and platelet relations)
- Sampling and resolution: the resolution ≈ 2 × sampling interval relation follows the
  Nyquist-Shannon sampling theorem, as used for FTIR instrument settings.
