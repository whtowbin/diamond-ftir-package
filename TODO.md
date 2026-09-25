# TODO

## Return to soon
- [ ] **Golden/regression tests with real spectra.** Need 3-5 anonymised real spectra plus trusted
      values (N ppm, B%, H peak areas, platelet height). Store expected values as JSON in
      `tests/golden/`. Until then correctness rests on synthetic-recovery tests only.
- [ ] **Choose the default baseline.** Evidence in `docs/baseline_benchmark.md`: on synthetic
      spectra `single` + `optimize` recovers thickness within 5% in most cases; the default
      multistage under-estimates it by 12-35% (mostly the rubber band). Absolute N on map M1 differs ~1.7x between methods; B% does not. Decide with real spectra of known N,
      then regenerate `tests/data/snapshot.json` deliberately.
- [ ] Map QA flag: mark pixels whose fitted thickness is far from the map median (plate edge,
      mount) when `thickness_mode` is `map_median` or `known`.
- [ ] Fill every `[CITATION NEEDED: ...]` placeholder (see `docs/CITATIONS_TODO.md`).
- [ ] Confirm `archive/` contents can be deleted for good.

- [ ] **Nitrogen fit window: evidence favours keeping 950-1350** (docs/baseline_benchmark.md,
      bias study). Ending at 1400 is biased low by platelets; the lower limit barely matters.
      Optional: warn when a user window ends above 1350 and a platelet peak is detected.
- [x] Default baseline is now joint ASLS + type IIa (lam 1e9); evidence in docs/methods_validation.md.
- [ ] Rerun the window part of the bias study with the joint normalisation, with more melee data.
- [x] Bias-study harness is in scripts/ (bias_study.py, bias_study_report.py, bias_study_figures.py).
- [ ] Survey `thickness_field = "plane"` flags curved-thickness regions as outliers (they fall
      back to their own fit). Consider a smooth 2-D spline field if this matters.

- [ ] **Revisit rubber-band baselines for sharp tails next to important peaks**, especially the
      low-wavenumber range below the nitrogen band (below ~1000 cm-1), where a steep rising or
      falling tail sits right beside the peaks being fitted. A smooth ASLS baseline (including the
      joint fit) can bend into or undershoot such tails. A local rubber band (convex hull) anchored
      on either side may follow them better. Test it with the same injection and leakage methods
      (docs/methods_validation.md): inject a synthetic tail and check nitrogen recovery. Note that
      the old stretched rubber band on the full spectrum caused most of multistage's thickness
      bias, so keep any rubber band local to the region it corrects.

- [ ] **Recipes: report fit uncertainties** (lmfit standard errors) and flag peaks that are
      not significantly above noise; an absent peak can currently get a small area from noise.
- [ ] Validate the example `diamond_platelet` and `diamond_amber` recipes by injection; decide
      whether they replace the hand-written measurements.
- [ ] Check the napari plugin by hand in a real napari window (the automated test uses a
      stand-in viewer; a headless napari viewer crashes offscreen with PySide6).
- [ ] Retire the tkinter GUI once the Qt app has been used in practice.

## Later
- [ ] GUI: edit amber bands in the app (currently via settings JSON only).
- [ ] Profile a 50-spectrum batch before optimising further; only the nitrogen design matrix is
      cached so far.
- [ ] Move `CAXBDY`/`typeIIA` from 7,000-line Python dicts to package data files.
- [ ] Map tab in the GUI (the CLI and Python API support maps now).
- [ ] `LoadSPA` map/directory loaders and `Load_Line_Scan` are untested; OMNIC `.map` works.
- [ ] Row-to-row banding in the multistage N map (map M1): check acquisition drift.
- [ ] Nuitka vs PyInstaller benchmark, only if run or startup time matters.
- [ ] Type hints throughout; add `ty` config and clear its findings.
- [ ] Docs site (mkdocs) once the content settles.
