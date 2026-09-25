# %%
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import os

from pathlib import Path
import numpy as np


import pybaselines as pybl

from typing import Tuple
import re
import xarray as xr
import imageio
import seaborn as sns

# %%
from DiamondFTIR import LoadSPA as spa
from .LoadSPA import LoadSPA as spa

# %%
dir_path = "/Users/wtowbin/Projects/FTIR Data/CBP-0341_overview_scan_SPA"
map = spa.Load_SPA_Map(dir_path)
# %%

#%% Stacked Map
stacked = map.stack(allpoints=("x", "y"))
array_chunk = stacked.chunk({"allpoints": 1})


def baseline_ASLS(
    spectrum, lam=1e6, p=0.002
):
    baseline = pybl.whittaker.asls(spectrum.values, lam=lam, p=p)[0]
    # baseline_array = xr.DataArray({"wn": spectrum.wn, "values": baseline})
    baseline_array = baseline

    return baseline_array  # returning a numpy array appears to work best.


def interpolate_to_common_grid(da):
    # Interpolate to a common grid 1cm^-1 # Consider 0.5cm^-1
    wn_low = np.round(da.wn[0], decimals=0)
    wn_high = np.round(da.wn[-1], decimals=0)
    wn_new = np.arange(wn_low, wn_high, 1)
    da_interp = da.interp(wn=wn_new, method="cubic")
    return da_interp

interpolated = interpolate_to_common_grid(stacked).unstack("allpoints")


baselines = stacked.spectra.groupby("allpoints").map(baseline_test).unstack("allpoints")
baselines_interp = interpolate_to_common_grid(baselines)

Basline_subtracted = map.spectra - baselines

#%%





@xr.register_dataset_accessor("map")
@Dataclass
class MapAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._baselines = None

    def __post_init__(self):


    def baseline_ASLS(self, lam=1e6, p=0.002):
        baseline = pybl.whittaker.asls(self._obj.values, lam=lam, p=p)[0]
        # baseline_array = xr.DataArray({"wn": spectrum.wn, "values": baseline})
        baseline_array = baseline

        return baseline_array # P
    
    def Baseline
    
    def interpolate_to_common_grid(self):
        # Interpolate to a common grid 1cm^-1 # Consider 0.5cm^-1
        wn_low = np.round(self._obj.wn[0], decimals=0)
        wn_high = np.round(self._obj.wn[-1], decimals=0)
        wn_new = np.arange(wn_low, wn_high, 1)
        da_interp = self._obj.interp(wn=wn_new, method="cubic")
        return da_interp
    
    
    interpolated = interpolate_to_common_grid(stacked).unstack("allpoints") 
    
    def plot(self):
        self._obj.plot()
        plt.show()

    def plot_baseline(self):
        baseline = self.baseline_ASLS()
        plt.plot(self._obj.wn, baseline)
        plt.show()  

# %%