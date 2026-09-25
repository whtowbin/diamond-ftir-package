# Methods

How each result is calculated. Every `[CITATION NEEDED: ...]` marks a source still to be added;
they are collected in [CITATIONS_TODO.md](CITATIONS_TODO.md).

## 1. Preparing the spectrum

Spectra are interpolated to 1 cm⁻¹ spacing between 601 and 6000 cm⁻¹ (limited by the reference
spectrum range and by the data). This gives every later step the same grid whatever the
instrument's native resolution. The original spacing is kept because the C-centre calibration
depends on it (section 3).

## 2. Thickness normalisation and baseline

The sample is scaled against a type IIa diamond reference spectrum
[CITATION NEEDED: origin of the type IIa reference spectrum (De Beers?)].

1. **Saturation test.** Three intrinsic two-phonon regions are tested in turn: 1970-2040,
   2400-2575, then 3000-3500 cm⁻¹. A region counts as saturated when its mean absorbance exceeds
   `saturation_cutoff` or its noise exceeds `stdev_cut_off`. The first unsaturated region
   (plus 3130-3500 cm⁻¹) decides where the fit is trusted. If none qualifies the file fails.
2. **Baseline.** A median filter, a mild asymmetric-least-squares (ASLS) baseline and a stretched
   rubber-band baseline are removed, then a final ASLS baseline is optimised so that the
   unsaturated diamond peaks match the reference and the 4000-5900 cm⁻¹ region is flat
   [CITATION NEEDED: ASLS baseline, Eilers & Boelens 2005]
   [CITATION NEEDED: pybaselines package]
   [CITATION NEEDED: rubber-band correction].
3. **Thickness ratio.** `typeIIA_ratio` is the mean ratio of the baseline-subtracted sample to the
   reference over the trusted region. Dividing by it gives a spectrum for a 1 cm path.

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
  major one; for IaAB stones the D component is also limited (`d_limit`)
  [CITATION NEEDED: D-component limit and its correlation with B (Woods)].
- Component heights become atomic ppm using absorption coefficients:
  A = 16.5 and B = 79.4 ppm per cm⁻¹
  [CITATION NEEDED: A- and B-centre calibration coefficients],
  C = 0.6243 ppm per cm⁻¹ × a correction for the instrument's spectral spacing,
  `9.7043 × spacing + 25.304`, from a linear fit to published values
  [CITATION NEEDED: verify — Liggins 2010 PhD thesis, University of Warwick].
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
