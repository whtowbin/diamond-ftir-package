#%%
from pathlib import Path
import numpy as np
import scipy.signal
import pandas as pd
#%%
from .Spectrum_obj import Spectrum

#%%
# Import Diamond Type IIa Spectra for normalizing thickness 
typeIIAPath = Path("typeIIa.csv")
typeIIA = pd.read_csv(
    typeIIAPath,
    names=["wn", "absorbance"],
)
typeIIA = typeIIA.set_index("wn")
#%%
typeIIA_Spectrum = Spectrum(
            X=typeIIA['wn'],
            Y=typeIIA['absorbance'],
            X_Unit="Wavenumber",
            Y_Unit="Absorbance",
            metadata=Metadata,
        )

#%%
class Diamond_Spectrum(Spectrum):
    def diamonds(self):
        print("Diamonds are Forever")