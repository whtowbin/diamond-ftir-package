# Simple GUI — stepwise build to avoid SEGV

The full GUI lives in `DiamondFTIR_GUI.py`. This folder has a **minimal alternate** so we can add functionality step by step and see where the SEGV appears.

## Step 1: Pure Tk (current)

- **File:** `DiamondFTIR_GUI_simple.py`
- **Launch:** `python -m diamond_ftir_package.gui_launcher_simple`
- **Contains:** One window, label, “Load spectrum (CSV)” (opens file dialog), “Quit”. No matplotlib, no diamond processing imports.

If this **does not** SEGV → Tk is fine; we add package/plotting next.  
If this **does** SEGV → try another Python (e.g. `/usr/bin/python3` or python.org build).

## Step 2: Add loading + processing (no plotting) — DONE

- **File:** `DiamondFTIR_GUI_simple.py` now loads CSV via `LoadCSV.CSV_to_IR_Diamond_Spectrum`, then on “Process” runs `fit_baseline()`, `normalize_diamond()`, `Nitrogen_fit()`, `measure_3107_peak()`, `measure_platelets_and_adjacent()` and shows results in a scrollable text area. No matplotlib.

If Step 2 SEGVs → issue is likely in package import (e.g. DiamondSpectrum at import time).  
If Step 2 is OK → SEGV in the full GUI is likely from matplotlib/TkAgg; we can add plotting in Step 3.

## Step 3: Add plotting (optional)

- Only if Step 2 is stable: import matplotlib in the simple GUI and add one plot (e.g. spectrum X/Y in a single figure). If SEGV appears here, we know it’s Tk + matplotlib.

## Quick test

```bash
cd /Users/henrytowbin/Projects/diamond-ftir-package
uv run python -m diamond_ftir_package.gui_launcher_simple
```

You should get a small window with “Load spectrum (CSV)” and “Quit”. Use that as the base for the next step.
