#%%
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from .Spectrum_obj import Spectrum
from .DiamondSpectrum import Diamond_Spectrum
#%%


def CSV_to_IR_Diamond_Spectrum(filepath):
    filepath = Path(filepath) # ensure path is a pathlib object
    data = pd.read_csv(filepath)
    wavenumber = data.iloc[:,0].to_numpy()
    intensities = data.iloc[:,1].to_numpy()

    sample_name = filepath.name.split("-")[0]

    try:
        Metadata = {
            "Filename": filepath.name,
            "Sample": sample_name,
            "cheese":"yes please"
        }
        x_data = wavenumber
        # Data = {"X": x_data, "Y": intensities}
        # return Data, Metadata

        SPA_Spectrum = Diamond_Spectrum(
            X=wavenumber,
            Y=intensities,
            X_Unit="Wavenumber",
            Y_Unit="Absorbance",
            metadata=Metadata,
        )
        return SPA_Spectrum
    except Exception as e:
        print(f"An exception occured: {e}")


def CSV_to_IR_Spectrum(filepath):
    filepath = Path(filepath) # ensure path is a pathlib object
    data = pd.read_csv(filepath)
    wavenumber = data.iloc[:,0].to_numpy()
    intensities = data.iloc[:,1].to_numpy()

    sample_name = filepath.name.split("-")[0]

    try:
        Metadata = {
            "Filename": filepath.name,
            "Sample": sample_name,
            "cheese":"yes please"
        }
        x_data = wavenumber
        # Data = {"X": x_data, "Y": intensities}
        # return Data, Metadata

        SPA_Spectrum = Spectrum(
            X=wavenumber,
            Y=intensities,
            X_Unit="Wavenumber",
            Y_Unit="Absorbance",
            metadata=Metadata,
        )
        return SPA_Spectrum
    except Exception as e:
        print(f"An exception occured: {e}")


def CSV_to_Xarray(filepath):
    # Loads a Single CSV into Xarray for processing

    filepath = Path(filepath) # ensure path is a pathlib object
    data = pd.read_csv(filepath)
    wavenumber = data.iloc[:,0].to_numpy()
    intensities = data.iloc[:,1].to_numpy()

    sample_name = filepath.name.split("-")[0]

    unit_names = {"x": "um", "y": "um", "wn": "cm^-1", "data": "absorbance"}
    unit_long_names = {"x": "microns", "y": "microns", "wn": "wavenumbers", "data": "absorbance"}
    metadata = {"sample_name" : sample_name, "unit_names": unit_names, "unit_long_names": unit_long_names}


    DataArray = xr.DataArray(
        [[intensities]],
        dims=("y", "x", "wn"),
        coords={"y": np.arange(1), "x": np.arange(1), "wn": wavenumber},
        attrs= metadata
    )
    return DataArray

    # data = {"spectra": (["y", "x", "wn"], intensities)}

    # coords = {
    #     "x": np.arange(1),
    #     "y": np.arange(1),
    #     "wn": wavenumber,
    # }

    # dataset = xr.Dataset(
    #     data_vars= data,
    #     coords= coords,
    #     attrs= metadata
    # )

    # return dataset
# %%

