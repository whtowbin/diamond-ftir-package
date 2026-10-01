# Baseline methods: what the evidence says

The thickness (type IIa ratio) and therefore every absolute ppm value depend on the
baseline. This page records what has been measured so far, so the choice of default can be
made on evidence. Regenerate the synthetic table with `uv run python scripts/baseline_benchmark.py`.

## The old optimiser never optimised

Up to this change, `fit_diamond_peaks` called `scipy.optimize.minimize` on (lam, p) in
linear units starting at (1e10, 5e-6). Its finite-difference step (~1e-8) is far too small
to change lam ≈ 1e10, so the gradient read as zero and it stopped at iteration 0 on every
spectrum tested. **All earlier results used a fixed ASLS baseline with lam = 1e10,
p = 5e-6.** The default `baseline_search = "fixed"` reproduces exactly that, so existing
numbers are unchanged. The search now works in log10(lam, p) (`"optimize"`, `"grid"`).

## Synthetic spectra (known truth: thickness ratio 1.00, N = 500 ppm, B% = 60)

Type IIa reference × thickness + CAXBDY components + noise + a background:

| background | thickness (cm) | multistage (default) | multistage, no rubber band | single | single, optimize |
|---|---|---|---|---|---|
| none | 0.03 | 0.76 / 654 | 0.95 / 524 | 0.98 / 509 | 1.00 / 500 |
| none | 0.3 | 0.88 / 564 | 0.93 / 534 | 0.97 / 516 | 1.00 / 501 |
| offset + slope | 0.03 | 0.76 / 654 | 0.95 / 524 | 1.56 / 321 | 0.98 / 509 |
| offset + slope | 0.3 | 0.88 / 564 | 0.93 / 534 | 0.97 / 516 | 0.98 / 509 |
| curved | 0.03 | 0.65 / 759 | 0.80 / 625 | 0.82 / 608 | 0.78 / 639 |
| curved | 0.3 | 0.80 / 622 | 0.85 / 586 | 0.87 / 576 | 0.95 / 524 |
| scatter-like | 0.03 | 0.72 / 695 | 0.79 / 629 | 0.83 / 598 | 0.95 / 524 |
| scatter-like | 0.3 | 0.85 / 585 | 0.90 / 553 | 0.93 / 538 | 0.93 / 536 |

Cells are fitted/true thickness and total N (ppm). B% was within 1% of 60 in every cell.

- The multistage pre-baseline under-estimates thickness by 12-35%, mostly because of the
  stretched rubber band (`pre_use_rubberband`).
- `single` with the search on is closest to the truth in 7 of 8 cases. With fixed
  lam = 1e10, `single` cannot follow an offset under a thin sample.
- **Caveat:** these backgrounds are simple shapes. Real spectra (fringes, water, mount
  absorption) may favour a different method. Real spectra with independently known N
  are needed to decide (see TODO.md).

## Real spectra

| file | multistage fixed | single fixed | single optimize |
|---|---|---|---|
| Stone R1, measurement a | 87.5 ppm, B 91.0% | 79.3, 91.0% | 82.8, 91.1% |
| Stone R1, measurement b | 84.4 ppm, B 90.4% | 75.9, 90.4% | 78.9, 90.4% |
| Stone R2 | 83.3 ppm, B 92.2% | 73.4, 92.2% | 79.1, 92.2% |

The two measurements of stone R1 differ by 20% in fitted thickness yet agree within 4-5% in N under every
method, so normalisation is internally consistent. True values are not independently known.

## Map M1: thin polished plate (10,176 on-sample pixels)

- Median total N: multistage fixed 427 ppm; single (calibrated) about 250 ppm. **B% maps are
  identical (median 70.4-70.5%)**. The zoning pattern is the same in all methods.
- The plate is thin: the diamond two-phonon band is about 0.06 absorbance on a
  0.14 offset, and fitted thickness varies by ±30-60% (IQR / median) across pixels with any
  method. For polished plates, `thickness_mode = "known"` (measured thickness) or
  `"survey"` (see below) removes that noise; per-pixel B% does not depend on it.
- A strip at x > 2,800 µm fits about 30× thicker than the rest (edge or mount). It is
  kept in `fitted_typeIIA_ratio` so it can be masked.
- Uniformity of fitted thickness across the plate is *not* a valid way to pick a method:
  single with lam = 1e10 gives the most uniform thickness because it mostly measures the
  constant offset.

## Search cost and the within-sample hypothesis

Per spectrum on this machine: fixed about 4-9 ms; `optimize` about 0.45 s (single) or
80 ms (multistage); `grid` about 0.6-0.75 s. In multistage mode searching barely changes N
(< 0.2%), because the pre-baseline has already removed what the main baseline would fit.

Within the map, optimal log10(lam) had IQR 0.6 and log10(p) IQR 1.0 over 40 pixels. That
is about as wide as between three other stones, so the sample does not narrow the search
much. Seeding each pixel from the sample median still pays off:

| map strategy (single, optimize) | whole map, 11 workers | N vs full per-pixel search |
|---|---|---|
| `per_pixel` | about 7-8 min (estimated from 0.44 s/pixel) | reference |
| `calibrate_local` (±0.5 decade) | 5 min 53 s | median 0.5%, p90 16% |
| `calibrate_fixed` | 15.5 s | median 9%, p90 32% |

B% was identical across all three strategies.

## Against hand-tuned map results (maps M2-M7)

Six maps previously processed by hand with a single ASLS baseline whose `lam`/`p` were tuned
per sample. The accepted result TIFFs served as the reference; re-running that pipeline
reproduced them exactly (median ratio 1.000) on five of the six. Every second pixel was
compared. Only M2, M3 and M5 have enough nitrogen (median 230-950 ppm) to discriminate;
M4, M6 and M7 have median N of 1-17 ppm, where ratios are noise.

| method | N / hand-tuned (M2, M3, M5) | correlation with hand-tuned map |
|---|---|---|
| single ASLS, hand-tuned lam/p (converted to the 1 cm⁻¹ grid) | 1.09-1.16 | 0.86-0.95 |
| single ASLS, survey-calibrated search (automatic) | 1.09-1.15 | 0.85-0.96 |
| multistage (package default) | 1.36-1.47 | 0.40-0.78 |
| thickness × IIa + polynomial background (prototype) | 1.10-1.17 | -0.19-0.41 |

- **The automatic survey-calibrated search reproduces hand-tuning** as closely as the hand-tuned
  values themselves do.
- **The polynomial-background prototype was rejected.** It matched the synthetic tests
  but flattened the real zoning (pixel-to-pixel spread 0.01-0.03 against 0.07-0.10).
- **The remaining 9-17% difference in N comes from the nitrogen fit window, not the baseline.**
  Thickness agrees within 1%. With the hand-tuned pipeline's window (about 650-1400 cm⁻¹
  instead of 950-1350 cm⁻¹) the package agrees within 1%. Which window is correct needs a
  reference [CITATION NEEDED: nitrogen fit window used by CAXBDY / DiaMap / QUIDDIT].
- ASLS `lam` depends on point spacing (roughly spacing⁴). Settings tuned on a map's native
  ~1.93 cm⁻¹ grid must be multiplied by about 14 to give the same smoothness on the
  package's 1 cm⁻¹ grid.

## Map thickness: survey, rim and outliers

`thickness_mode = "survey"` analyses a coarse grid of interior pixels (every `survey_step`
pixels, at most `survey_max_pixels`). It leaves out anything within `edge_width_px` of the
sample edge or map border, and fits a smooth log-thickness field (`constant` or `plane`),
iteratively rejecting points beyond `outlier_mad` robust standard deviations. In the main pass:

- interior pixels within that tolerance of the field use the field (`thickness_source` = 1);
- edge pixels keep their own fit (2). On map M1 the first three pixels from the edge fit
  0.3-1.2 decades thinner, because the aperture only partly covers the sample, and nitrogen
  drops by the same fraction;
- pixels far from the field keep their own fit and are flagged (3), for example a thick strip
  (a mount or a different region) and a one-row acquisition glitch on M1.

On M1 (10,176 pixels): 400 survey pixels, 53 rejected, field scatter 0.024 decades;
7,384 pixels used the field, 1,156 were edge pixels and 1,636 outliers. Total time was 36 s with a baseline search on the survey
pixels and 11 workers.

## Bias study: baseline method × nitrogen fit window (real spectra)

Test set: real spectra whose nitrogen band and at least one diamond phonon region are
unsaturated. That gave 688 spectra (31 single stones plus map pixels from M1, M2, M3 and M5)
for injection and 221 for the window study, plus 120 pixels from two low-N maps (M4, M7).
Real samples have no known answer, so bias was measured by **injection**: known
signals are added to each real spectrum (keeping its real background, noise and extra
peaks) and the change in the result is compared with what was added.

### Thickness (multiplicative bias): add Δt × (type IIa + 150 ppm A + 150 ppm B)

| baseline | recovered Δt / added (median) | p10 | worst group |
|---|---|---|---|
| single ASLS, searched (`single` + `optimize`) | **0.99** | 0.86 | 0.98 (M1) |
| single ASLS, fixed lam = 1e10, p = 5e-6 | 0.95 | 0.48 | 0.48 (M1) |
| multistage (current default) | 0.84 | 0.62 | 0.69 (M1) |

The multistage pre-baseline removes about 16% of the diamond signal on real backgrounds,
so it reports thickness about 16% low and nitrogen about 16% high. The searched single ASLS
is close to unbiased. The added nitrogen was recovered within 1% by every combination: the
nitrogen fit is linear, so multiplicative errors come only from thickness.

### Window (additive bias): inject one unmodelled feature at a time

Change in total N (ppm) per 1 cm⁻¹ of injected feature, median over spectra:

| window | platelet (1365) | residual background below the N band (~800) | noise-driven N spread |
|---|---|---|---|
| 950-1350 (current) | 0.0 | 0.0 | 0.02% of N |
| 950-1400 | -3.4 | -0.1 | 0.02% |
| 650-1350 | 0.0 | -5.6 | 0.02% |
| 650-1400 (map scripts) | -4.2 | -7.4 | 0.02% |

- Platelets in the map pixels measured 6-15 cm⁻¹ (none in the single stones). The
  predicted platelet bias agrees with the observed 1400-vs-1350 difference (correlation
  0.8). Ending at 1400 lowers N by 5-7%, and by 9-21% with a 650 start. This, not the baseline,
  explains the gap between the older map results and the package.
- The lower limit barely matters when the window ends at 1350: starting anywhere from 650
  to 1000 changes N by a median of 0% (p10-p90 ±2-3%).
- Ending below 1350 does matter: 1330 gives +5% and 1300 gives +8% (p90 +16%). The 1330-1350
  range carries the sharp B / N⁺ structure near 1332 cm⁻¹, which the fit needs.
- Noise is not the limiting factor: even on low-N maps (median 13.6 ppm) noise moves N by
  0.3-0.5%, while window choice moves it by -41% to +61%. Widening the window to gain
  precision is not worth the added bias on data like these.

**Conclusion:** 950-1350 cm⁻¹ is the least biased fixed window on this data. Extending
past 1350 is only safe without a platelet peak and then changes nothing measurable, so an
adaptive upper limit adds no benefit here. The large remaining bias is the baseline's
thickness error, which the searched single ASLS removes.
