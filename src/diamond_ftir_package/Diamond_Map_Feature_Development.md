# Diamond FTIR Map Feature Development 

## Goals
- Provide a simple workflow for loading and processing FTIR Maps. 
- Does the most basic map processing functions with minimal manual adjustment.

How do we match data with baselines and interpolated data?
How do we keep track of objects? 
Spectal Data
  - Original Data
  - Interpolated Data
  - Baseline Data
  - Baseline Subtracted Data

Summary Data
  - Concentration
  - Peak Height/Area/Width/Position
  - Peak Ratio
  - Binary Masks
  

## Notes Cut from Code 
Easiest Use Case is a Jupyter Notebook but it would be nice to have some kind of GUI
Most basic functions for processing Spectral Data.
General Functions exposed to users should be able to be chained together.
- Area( wn_low, wn_high)
- Height( wn_low, wn_high) #if only low is given then it is the height of the peak at that point.
- Width( wn_low, wn_high) #if only low is given then it is the width of the peak at that point.
- Basseline( wn_low, wn_high) #if only low is given then it is the width of the peak at that point.
- Median_Filter( window_size)
- Interpolate( wn_low, wn_high, wn_step)
- Plot( wn_low, wn_high)


More advanced funtions that might be exposed for certain use cases
- Conditional Masking - Xarray masking. Should allow for more advanced masking. Based arrays previously generated. 


Functions that opperate on HyperSpectal Maps
Should accomplish easiest actions and be able to apply traditional Xarray methods as well.
One thing I need to decide on is whether to store data as a DataArray or a Dataset.
  - Datasets could hold multile data arrays but it might be overkill since will probably need to explicitly define each subarray.
Accsesor's appear to be the proper tools for this.
https://docs.xarray.dev/en/stable/internals/extending-xarray.html

Functions needed

Goals for baseline algorithms for Diamond Spectra
1. handles noise.
2. well suited for nitrogen bearing diamonds

Platelet Peak height, position, width, area
1405 peak. And maybe note the other peaks in the 1400-1500 range.
3107 peak

Functions needed
    - Interpolation to a common wavenumber grid. 1cm^-1 or 0.5cm^-1
   - baseline fitting functions
   - nitrogen aggregation least squares fitting
   - masking functions for which data to ignore in plots. (e.g. ignore extremely noisy, bad or saturated spectra)
   - peak finding functions
  - peak height functions
  - peak area functions
  - peak width functions
   - Cropping functions. Spatial and spectral
   - Standardize functions for plotting maps and for making TIFFS.
  - Spectra Classification - PCA or alternate.
  - Spectra Classification - Kmeans or alternate.
  - Reset map spatial coordinates to be in mm or um and center the map at 0,0 or make one corner 0,0


Analytical Goals
   - Nitrogen Quantification
   - Nitrogen Aggregation
  - Raman/ PL map analysis  Which peaks are present and how do they vary spatially?
 - Stress and Strain analysis - How do the peaks shift spatially? What can I learn from cross polarized images or step-flow growth.


General Functions
Many of these can be done with Xarray methods, but I want to make them easier to use by assuming which dimension they are opperating on.
  - Interpolation Functions.
  - Plotting functions for maps and spectra
  - TIFF writing functions for maps and spectra
  - Functions for saving and loading maps and spectra
 - Baseline fitting functions
 - Peak fitting functions
 - Spectra Classification - PCA or alternate.
 - Integration
 - Peak height functions
 - Normalization Functions - divide by peak height, area, or sample thickness
 - Cropping functions. Spatial and spectral


Diamond FTIR Specific Functions
  - Nitrogen Quantification / Aggreagation


