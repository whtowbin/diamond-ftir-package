# Diamond FTIR GUI Usage Guide

## Overview

The Diamond FTIR GUI provides a user-friendly interface for loading, processing, and visualizing diamond FTIR spectra. It makes it easy to adjust fit parameters and process single spectra or batches of files.

**If you see SEGV with the full GUI**, use the minimal “simple” GUI and add features step by step; see [GUI_SIMPLE_README.md](GUI_SIMPLE_README.md). Launch the simple GUI with:
```bash
python -m diamond_ftir_package.gui_launcher_simple
```

## Installation

The GUI uses standard Python libraries that should already be available:
- `tkinter` (usually included with Python)
- `matplotlib` (for visualization)
- `pandas` (for data handling)
- `numpy` (for numerical operations)

## Launching the GUI

### Option 1: Using the launcher script
```bash
python -m diamond_ftir_package.gui_launcher
```

### Option 2: Direct import
```python
from diamond_ftir_package.DiamondFTIR_GUI import main
main()
```

### Option 3: Run the module directly
```bash
python src/diamond_ftir_package/gui_launcher.py
```

## Features

### File Operations
- **Load Single Spectrum**: Load and process one CSV spectrum file
- **Load Batch Files**: Select multiple CSV files for batch processing
- **Process Current**: Process the currently loaded spectrum with current parameters
- **Process Batch**: Process all loaded batch files

### Parameter Controls

The GUI organizes parameters into tabs:

#### 1. Saturation Tab
- **Saturation Cutoff**: Threshold for detecting saturated peaks (default: 2.5)
- **Stdev Cutoff**: Standard deviation threshold (default: 0.5)
- Lower values = more sensitive saturation detection

#### 2. Baseline Tab
- **Algorithm**: Choose between "Whittaker" (recommended) or "ALS"
- **Lambda**: Smoothness parameter (×10¹⁰, default: 1.0)
  - Higher values = smoother baseline
- **P**: Asymmetry parameter (×10⁻⁶, default: 5.0)
  - Higher values = more asymmetric baseline

#### 3. Nitrogen Tab
- **Max C/B Ratio**: Maximum ratio for minor components (default: 0.01)
- **Wavenumber Low**: Lower bound for nitrogen fitting (default: 950 cm⁻¹)
- **Wavenumber High**: Upper bound for nitrogen fitting (default: 1350 cm⁻¹)
- **Plot Nitrogen Fit**: Checkbox to visualize the nitrogen component fit

#### 4. Platelets Tab
- **Baseline Lambda**: Baseline correction parameter for platelet region (default: 1000)
- **Baseline P**: Baseline asymmetry parameter (default: 0.001)

#### 5. Hydrogen Tab
- **Baseline Lambda**: Baseline correction parameter for hydrogen peaks (default: 0.1)
- **Baseline P**: Baseline asymmetry parameter (×10⁻⁶, default: 6.0)

#### 6. Options Tab
- **Measure Platelet Peaks**: Enable/disable platelet peak measurement
- **Measure Hydrogen Peaks**: Enable/disable hydrogen peak (3107 cm⁻¹) measurement
- **Measure Amber Center**: Enable/disable amber center peak measurement

### Visualization

The spectrum visualization tab provides several plot types:
- **Raw**: Shows the original spectrum with baseline overlay (if fitted)
- **Baseline Corrected**: Shows the spectrum after baseline subtraction
- **Normalized**: Shows the thickness-normalized spectrum (1 cm equivalent)
- **Nitrogen Fit**: Shows the nitrogen component fitting results with individual components

### Results Display

The results tab shows:
- Nitrogen analysis (A, B, C centers, total nitrogen, aggregation state)
- Hydrogen peak measurements (3107 and 3085 cm⁻¹ areas)
- Platelet peak measurements (area, height, position)
- Amber center measurements (if enabled)

### Export

- **Export Results (CSV)**: Save processing results to a CSV file
- **Save Plot**: Save the current visualization as PNG or PDF

## Workflow Example

1. **Load a spectrum**: Click "Load Single Spectrum" and select a CSV file
2. **Adjust parameters**: Navigate through parameter tabs and adjust values as needed
3. **Process**: Click "Process Current" to run the analysis
4. **Review results**: Check the Results tab for quantitative measurements
5. **Visualize**: Use the Spectrum tab to view different processing stages
6. **Export**: Save results and plots as needed

## Batch Processing

For processing multiple files:

1. Click "Load Batch Files" and select multiple CSV files
2. Set your desired parameters
3. Click "Process Batch"
4. A progress dialog will show processing status
5. Results will be displayed in the Results tab as a table
6. Export the batch results to CSV for further analysis

## Tips

- Start with default parameters and adjust based on your spectra
- If saturation warnings appear, you can still process but results may be less reliable
- The nitrogen fit visualization is helpful for understanding component contributions
- Batch processing saves time when analyzing many spectra with the same parameters
- Export results regularly to avoid data loss

## Troubleshooting

**GUI doesn't launch**: Ensure tkinter is installed (usually included with Python)
```bash
python -m tkinter  # Test if tkinter works
```

**Import errors**: Make sure you're running from the project root or have the package installed
```bash
pip install -e .
```

**Processing fails**: Check that your CSV files have the correct format (wavenumber in first column, absorbance in second column)

**Memory issues with batch processing**: Process smaller batches or increase available memory

## Parameter Tuning Guide

### Baseline Fitting
- If baseline is too wavy: Increase Lambda
- If baseline doesn't follow the spectrum: Decrease Lambda or adjust P
- For noisy spectra: Try the ALS algorithm instead of Whittaker

### Nitrogen Fitting
- If C/B ratio seems wrong: Adjust Max C/B Ratio
- For different spectral ranges: Modify wavenumber bounds
- Check the fit visualization to verify component contributions

### Saturation Detection
- If getting false positives: Increase saturation cutoff
- If missing saturated peaks: Decrease saturation cutoff or stdev cutoff
