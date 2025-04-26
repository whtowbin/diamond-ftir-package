# %%
import pathlib
import numpy as np
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import re
import xarray as xr

try:
    from .Spectrum_obj import Spectrum
    from .DiamondSpectrum import Diamond_Spectrum
    from .SPC import spc

except:
    from Spectrum_obj import Spectrum
    from DiamondSpectrum import Diamond_Spectrum
    from SPC import spc
# %%
# %%


def Load_SPC(filepath: str) -> tuple[np.ndarray, np.ndarray, dict]:
    path = pathlib.Path(filepath)
    SPC_File = spc.File(path)
    metadata = {
        "year": SPC_File.year,
        "month": SPC_File.month,
        "day": SPC_File.day,
        "filename": path.name,
    }
    spec = Spectrum(SPC_File.x, SPC_File.sub[0].y, metadata=metadata)
    return spec


def SPC_Diamond_FTIR_Spectrum(filepath: str):
    spec = Load_SPC(filepath)
    return Diamond_Spectrum(spec.X, spec.Y)


# %%
