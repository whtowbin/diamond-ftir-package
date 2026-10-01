# %%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import xarray as xr
from diamond_ftir_package import LoadCSV
from tenacity import retry, stop_after_attempt

from typing import Dict

# %%
font = {
    "family": "Avenir Next",
    "weight": "normal",
    "size": "16",
}
plt.rc("font", **font)

output_folder = "Results"
Path(output_folder).mkdir(parents=True, exist_ok=True)
# %%

IR_DIR = Path("data/IR and UV-VIS Data/IR Data")


def list_files(dir, filetype="*.csv"):
    p = Path(dir)
    return list(sorted(p.glob(filetype)))


# %%

FTIR_files = list_files(IR_DIR)

Nitrogen_concentrations = []
