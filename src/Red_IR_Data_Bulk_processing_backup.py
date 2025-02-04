#%%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import xarray as xr
from diamond_ftir_package import LoadCSV
from tenacity import retry, stop_after_attempt

from typing import Dict

#%%
font = {  'family': 'Avenir Next',
    "weight": "normal",
    "size": "16",
}
plt.rc("font", **font)

output_folder = "Results"
Path(output_folder).mkdir(parents=True, exist_ok=True)
#%%

IR_DIR = Path("data/IR and UV-VIS Data/IR Data")

def list_files(dir, filetype = "*.csv"):
    p = Path(dir)
    return list(sorted(p.glob(filetype)))
# %%

FTIR_files = list_files(IR_DIR)

Nitrogen_concentrations = []

@retry(stop=stop_after_attempt(2))
def fit_N_Diamond(filepath):
    Spectrum = LoadCSV.CSV_to_IR_Diamond_Spectrum(filepath)
    Nitrogen_Saturation = Spectrum.test_saturation(900,1400, 2,0.5)
    if Nitrogen_Saturation is False:
        Spectrum.fit_baseline()
        Spectrum.normalize_diamond()
        Spectrum.Nitrogen_fit(plot_fit = False)
        dict = Spectrum.nitrogen_dict
        dict["filename"] = filepath.name
        dict["name"] = filepath.name.split("-")[0]

        # Measure H Peaks
        Spectrum.measure_3107_peak()
        dict["Normed_3107_Area"] = Spectrum.normed_area_3107
        dict["Normed_3085_Area"] = Spectrum.normed_area_3085

        Spectrum.measure_platelets_and_adjacent()
        dict["Normed_Platelet_Area"] = Spectrum.normed_area_platelet


        return dict
    else:
        return None 

Count = 0 
for filepath in FTIR_files:
    print(f"Count:{Count}")
    try:
        Count = Count + 1
        calculated_N = fit_N_Diamond(filepath)
        if type(calculated_N) is type({}):
            Nitrogen_concentrations.append(calculated_N)
    except:
        print("An expection occured with file: {filepath.name}")


df = pd.DataFrame(Nitrogen_concentrations)

df.to_csv("Red_Diamond_Nitrogen.csv")

# %%

fig, ax = plt.subplots()
df.plot("B_percent",'Total_N',  kind = "scatter", alpha = 0.3, ax = ax)
ax.set_xlabel("Percent B Aggregates")
ax.set_ylabel("Total Nitrogen ppm")

# %%
