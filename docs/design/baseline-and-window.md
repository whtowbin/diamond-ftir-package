# Design plan: baselines, thickness and the nitrogen window

Status: **joint baseline adopted as default (lam 1e9, p 1e-3); remaining items open.** Step-by-step method and figures: [../methods_validation.md](../methods_validation.md).

Previous status: draft for review. Evidence is in [../baseline_benchmark.md](../baseline_benchmark.md)
(benchmark and bias-study results). This page gives the goals, what was decided, the plan and
what is still open.

## Goal

Report nitrogen (A, B, C; total N and B%), hydrogen and platelet values that are as faithful
as possible to the diamond, for spectra of very different quality: faceted stones, thin plates,
maps and melee. That means fitting the diamond (thickness) and nitrogen components without
the baseline or the fit window adding bias, and without hand-tuning per sample.

## What bias we mean, and how we measure it

No real spectrum has a known answer, so bias is measured on real spectra by **injection**:
known signals are added to a real spectrum (keeping its real background, noise and extra
peaks), and we check what is recovered.

| bias | caused by | measured by |
|---|---|---|
| multiplicative (thickness) | the baseline removing part of the diamond signal | add Δt × (type IIa + known A + B); recovered Δt / added |
| additive (nitrogen window) | unmodelled features inside the window (platelet, residual background) | add a unit feature; change in total N per 1 cm⁻¹ |
| precision | noise | noise-only realisations; spread of N |

Synthetic spectra are used only for mechanics (every term known). They favoured a
polynomial background model that failed on real maps, so **decisions rest on the real-data
tests**, not on synthetic spectra.

Real data never enters the repository. `scripts/bias_study.py` reads a local, git-ignored
dataset list (`local_data/datasets.json`: label, quality, folder/zip/map path, sample size)
and samples each dataset at random with a fixed seed. `scripts/bias_study_report.py`
summarises the results, with `--anonymise` for tables that go into docs.

## Findings so far (details in baseline_benchmark.md)

1. The old baseline optimiser never moved from its start point; all earlier results used
   a fixed ASLS baseline (lam = 1e10, p = 5e-6). The search now works in log space.
2. Thickness bias, measured on 688 real spectra: searched single ASLS recovers 0.99 of the
   injected thickness, fixed single 0.95 (0.48 on a thin plate), and multistage (current default)
   0.84. So multistage reports N about 16% high.
3. On hand-tuned maps, the automatic searched single ASLS reproduces the hand-tuned results
   (correlation 0.85-0.96) as well as the hand-tuned settings do.
4. A polynomial background (thickness × type IIa + polynomial) was rejected: it matched on
   synthetic spectra but flattened real zoning.
5. Nitrogen window: 950-1350 cm⁻¹ has the least bias. Ending at 1400 is biased low by
   platelets (5-7%; 9-21% with a 650 start). Ending below 1350 drops the 1332 cm⁻¹ structure
   (+5 to +8%). The lower limit barely matters (±2-3%). Noise is negligible next to window
   bias (0.02-0.5% of N).
6. Map thickness: a coarse survey of interior pixels, with a robust thickness field and edge-ring
   and outlier exclusion, gives stable per-pixel thickness on plates.
   Edge pixels (partial aperture coverage) and outlier regions keep their own fits.

## Wider study (quality range)

1,360 spectra were sampled at random from 14 datasets (anonymised D1-D14): high-quality stones,
four maps, four lower-quality stone collections, three melee sets, and two instrument-labelled
"non-diamond" melee sets. The latter are mixed: many are diamonds the instrument
misclassified. 1,272 spectra were usable, meaning their nitrogen band and at least one
diamond phonon region were unsaturated.

| dataset | quality | point spacing (cm⁻¹) | noise (cm⁻¹, normalised) | thickness (cm) | platelet (cm⁻¹) | diamond R² (median) |
|---|---|---|---|---|---|---|
| D1, D2 | high | 0.48 | ≤0.001 | 0.17-0.21 | ~0 | 1.00 |
| D3-D6 | map | 0.96-1.93 | 0.003-0.05 | 0.005-0.044 | 0.1-21 | 1.00 |
| D7-D10 | low | 0.48 | 0.016-0.047 | 0.13-0.15 | 1-1.6 | 0.95-0.99 |
| D11, D12 | low (melee) | 7.7 / 0.48 | 0.003 / 0.13 | 0.014-0.029 | 0.4-2 | 0.93-0.94 |
| D13, D14 | mixed (melee) | 0.48 / 7.7 | 0.17 / 0.005 | 0.013-0.024 | ≤0.4 | 0.88 / 0.55 |

**Thickness recovery** (injection; 1.00 = unbiased; median by group, worst group's p10 in brackets):

| baseline | high | maps | low | melee |
|---|---|---|---|---|
| multistage (current default) | 0.92-0.95 | 0.63-0.91 | 0.95-0.96 | 0.81-0.93 |
| single ASLS, searched | 0.99 | 0.98-0.99 | 0.73-0.96 | 0.69-0.73 |
| **joint ASLS + type IIa, lam 1e8, p 1e-3** | 1.03-1.08 | 1.02 | 1.01 | 1.01 [0.92 overall p10] |
| joint, lam 1e9, p 1e-3 | 1.00-1.01 | 1.01-1.02 | 1.00-1.01 | 1.00 [0.85] |

(The joint rows come from a 30-spectrum-per-dataset subsample; the others use the full sample.)

- No existing baseline is unbiased everywhere. The searched single ASLS is best on maps and
  clean stones but loses 25-30% on noisier stones and melee: its objective compares only the
  *shape* of the diamond band (it divides out the amplitude), so a baseline that eats a fixed
  fraction of the band is not penalised, and the search then drives p to its upper bound
  (43-90% of spectra).
- A *joint* fit removes that mechanism: the spectrum is modelled as a smooth ASLS baseline
  plus t × type IIa, solved together (one extra column in the same penalised least squares,
  with ASLS's asymmetric reweighting and zero weight on saturated points). It needs no search,
  is insensitive to lam and p over two decades, and costs ~22-25 ms per spectrum. (Alternating
  "fit t, then fit ASLS to the rest" diverges and was rejected.)
- **Map check (the test the polynomial model failed): passed.** On interior pixels, the map
  results were split into nitrogen absorbance (N × thickness, which does not depend on
  thickness) and thickness, and compared with the hand-tuned maps:

  | map | method | N × t vs hand-tuned | thickness vs hand-tuned | thickness roughness | N roughness |
  |---|---|---|---|---|---|
  | M3 (little N contrast) | joint | 0.989 | 0.926 | 0.002 | 0.004 |
  | M3 | searched single | 0.791 | 0.798 | 0.002 | 0.004 |
  | M5 (strong zoning) | joint | 0.994 | 0.985 | 0.004 | 0.022 |
  | M5 | searched single | 0.994 | 0.913 | 0.007 | 0.027 |

  The joint fit keeps the nitrogen structure, and gives thickness closer to the hand-tuned maps
  and smoother. The earlier low correlation of its N maps came from thickness detail on maps
  with little N contrast, and from edge pixels. The hand-tuned maps are a reference, not ground
  truth.

**Nitrogen window** (N relative to 950-1350): ending at 1400 lowers N by 1.5-3% on stones
and 5-7% on platelet-rich maps. The 650-1400 window lowers it by 8-21%. Melee are very
sensitive to window choice (-58% to +86%), because their nitrogen signal is weak.
950-1350 remains the least biased choice.

**Diamond-likeness** (R² of the type IIa shape over the unsaturated phonon windows):
90-100% of spectra in the diamond datasets have R² ≥ 0.8. In D14, an instrument-labelled
non-diamond set, 80% fall below 0.8. In D13, also instrument-labelled non-diamond, only 10%
do, consistent with most of D13 being misclassified diamonds. R² is a candidate QA gate.

## Plan

Each step keeps a switch back to the old behaviour and is verified by the bias study before it
becomes a default. Snapshot tests are regenerated only on deliberate changes.

1. ✅ **Done:** joint ASLS + type IIa baseline added and made the default (`baseline_method="joint"`).
   Full-sample rerun (1,109 injections): median thickness error 0.9%, p90 4.9%, 90% of spectra
   within 5%, against 8.6% / 35.5% / 23% for the previous multistage default.
   Original plan text: **Add the joint ASLS + type IIa baseline** (`baseline_method="joint"`; fixed lam ≈ 1e8-1e9 on
   the 1 cm⁻¹ grid, p ≈ 1e-3; no search), with tests and the bias study rerun on the full sample
   (the numbers above use 30 spectra per dataset). Make it the default for single spectra and
   maps once signed off. Multistage and single ASLS stay available. Expected effect on
   today's default: thickness about 5-16% higher, so total N correspondingly lower; B%
   unchanged. *Needs sign-off.*
2. ✅ **Done: lam = 1e9** (lower median error than 1e8 with the same tail; 1e8 over-recovered
   up to 1.07 on clean stones). Original plan text: **Choose lam within 1e8-1e9** from the full-sample rerun: 1e8 over-recovers slightly on
   clean stones (1.03-1.08); 1e9 has a wider low tail on noisy stones (p10 0.85).
3. **Nitrogen window stays 950-1350.** Add a warning when a custom window ends above 1350 and
   a platelet peak is detected (the measured platelet size predicts the bias, correlation 0.8).
4. **Quality flags per spectrum** in every result table: saturated nitrogen band, saturated
   primary phonon band (fit used 2400-2670 or 3130-3500), diamond-likeness R², noise level,
   injection-derived thickness reliability where available.
5. **Diamond-likeness gate** (if the wider study supports it): a diamond R² threshold that
   separates diamond from non-diamond spectra better than instrument labels (the instrument's
   "NONDIAMOND" folders contain many diamonds).
6. **Keep the bias study as a regression tool:** rerun on new datasets, and before changing any
   default.

## Open questions

- Reference method and citation for the nitrogen window and CAXBDY components
  [CITATION NEEDED: nitrogen fit window used by CAXBDY / DiaMap / QUIDDIT].
- Whether melee (thin, low-signal) need their own defaults. The wider study will show whether
  more melee data would help.
- Smoother thickness fields for maps with curved thickness (the plane flags curvature as
  outliers).
