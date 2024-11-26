#%%
from pathlib import Path
import numpy as np
import scipy.signal
import scipy.optimize as optimize
import pybaselines as pybl
from copy import deepcopy
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Union
import scipy.sparse as sparse

from .Spectrum_obj import Spectrum

from .typeIIA import typeIIA_json
from .CAXBD import CAXBD_json

#%%

# Type Spectra are only imported once outside of the class so that they dont fill up the memory in long loops, by creating multiple identical objects
typeIIA = pd.DataFrame(typeIIA_json)
#typeIIA = typeIIA.set_index(keys=["wn"]) 

typeIIA_Spectrum = Spectrum(
X=typeIIA['wn'],
Y=typeIIA['absorbance'],
X_Unit="Wavenumber",
Y_Unit="Absorbance",
)
        

CAXBD = pd.DataFrame(CAXBD_json )
#CAXBD = CAXBD.set_index(keys=["wn"])



#TODO This should actually be turned into a matrix for fitting
# CAXBD_Spectrum = Spectrum(
#         X=CAXBD['wn'],
#         Y=CAXBD['A'],
#         X_Unit="Wavenumber",
#         Y_Unit="Absorbance",
#     )

@dataclass()
class Diamond_Spectrum(Spectrum):


    def diamonds(self):
        print("Diamonds are Forever")

    def interpolate_to_diamond(self):
        spec_min = np.round(self.X.min())+1
        spec_max = np.round(self.X.max()) -1

        typeIIA_min = np.round(typeIIA_Spectrum.X.min())
        typeIIA_max = np.round(typeIIA_Spectrum.X.max())

        #set minimum wavenumber to 600 since most data is useless below that with our currents systems
        wn_min = max(spec_min,typeIIA_min, 601)
        wn_max = min(spec_max, typeIIA_max, 6000)

        
        self.interpolate(wn_min, wn_max, 1, inplace= True)

        self.interpolated_typeIIA_Spectrum = typeIIA_Spectrum.interpolate(wn_min, wn_max, 1)
    
    def test_diamond_saturation(self) :
        """Tests if intrinsic diamond FTIR peaks are saturated in a spectrum and returns a dictionary the best peak range to fit over. 
        Only to be used on non-thickness normalized spectra that have not been baseline corrected. 
        Assumes that saturation is sequential i.e. if primary is unsaturated then all others are unsaturated too. 
        """
        main_diamond_sat = self.test_saturation(X_low=1970,X_high=2040, saturation_cutoff=2.5, stdev_cut_off=0.5)
        secondary_diamond_sat = self.test_saturation(X_low=2400,X_high=2575, saturation_cutoff=2.5, stdev_cut_off=0.5)
        third_diamond_sat = self.test_saturation(X_low=2400,X_high=2575, saturation_cutoff=2.5, stdev_cut_off=0.5)
         
        if main_diamond_sat == False:
            fit_mask_idx  =  (((self.X > 1800) & (self.X < 2313)) | (self.X > 2390) & (self.X < 2670))
        
        elif (main_diamond_sat == True) & (secondary_diamond_sat == False):
            fit_mask_idx  = (self.X > 2390) & (self.X < 2670)
        
        elif (main_diamond_sat == True) & (secondary_diamond_sat == True) & (third_diamond_sat == False):
            fit_mask_idx  = (self.X > 3130) & (self.X < 3500)

        fit_mask_idx = fit_mask_idx & ((self.X > 3130) & (self.X < 3500)) & ((self.X > 1400) & (self.X < 1800)) & ((self.X > 680) & (self.X < 900))
        return fit_mask_idx

    # def baseline_error_diamond_fit(self,ideal_diamond = typeIIA_Spectrum, data_mask = fit_mask_idx):
    #         self.baseline_ASLS(lam = 1000000, p = 0.0005)

    def fit_diamond_peaks_ALS(self):
        ideal_diamond = self.interpolated_typeIIA_Spectrum
        """ Fits a diamond spectrum to an ideal spectrum accounting for saturated peaks. Spectrum needs to be interpolated to the same spacing as the typeIIA diamond spectrum.  

        Args:
            ideal_diamond (_type_, optional): _description_. Defaults to typeIIA_Spectrum.
        """
        fit_mask_idx = self.test_diamond_saturation()

        def baseline_diamond_fit_R_squared(baseline_input_tuple, spectrum_wavenumber = self.X ,spectrum_intensity = self.median_filter(11).Y, typeIIA_intensity=ideal_diamond.Y, mask_idx_list=fit_mask_idx):
            lam, p = baseline_input_tuple
            print(f"lam = {lam}, p = {p}")
            baseline = baseline_als(spectrum_intensity, lam=lam, p=p)
            baseline_subtracted = spectrum_intensity - baseline 
            baseline_subtracted_masked = baseline_subtracted[mask_idx_list]
            typeIIA_masked = typeIIA_intensity[mask_idx_list]
            fit_ratio =  baseline_subtracted_masked/ typeIIA_masked
            
            # Force Baseline to fit flat part of spectrum
            flat_range_idx = (spectrum_wavenumber > 4000) & (spectrum_wavenumber < 5000)
            weight_factor = 0.0001 # Sets balance of residulas between typeIIA and flat baseline section
            flat_baseline_residuals_squared = ((baseline_subtracted[flat_range_idx])**2).sum() * weight_factor 

            typeIIa_residuals_squared = (( (baseline_subtracted_masked/fit_ratio) - typeIIA_masked)**2).sum() 

            Total_residuals_squares = flat_baseline_residuals_squared + typeIIa_residuals_squared
            print(f" total Residuals squared {Total_residuals_squares}")
            #return np.log(Total_residuals_squares)
            return Total_residuals_squares
        
        #p_opt = optimize.minimize(baseline_diamond_fit_R_squared, (100000000,0.0005), args = (self.Y, ideal_diamond.Y, fit_mask_idx),bounds=((1e2, 1e8), (1e-9,1)),method='Nelder-Mead')
        #p_opt = optimize.minimize(baseline_diamond_fit_R_squared, x0=(6,-4), args = (self.Y, ideal_diamond.Y, fit_mask_idx),bounds=((4, 9), (-9,-4)),method='Powell')
        p_opt = optimize.differential_evolution(baseline_diamond_fit_R_squared, bounds=((1e5, 1e10), (1e-7,0.001)), x0=(10000000,0.0005), tol = 1000000000, atol = 100000)
        #p_opt = optimize.differential_evolution(baseline_diamond_fit_R_squared, bounds=((4, 9), (-9,-4)), x0=(6,-4))
        #p_opt = optimize.basinhopping(baseline_diamond_fit_R_squared, (1000000,0.0005), minimizer_kwargs = {"args": (self.Y, ideal_diamond.Y, fit_mask_idx)})
        #p_opt = optimize.basinhopping(baseline_diamond_fit_R_squared, x0=(6,-4))
        #p_opt = optimize.dual_annealing(baseline_diamond_fit_R_squared, bounds=((1e2, 1e10), (1e-9,1e-2)), args = (self.Y, ideal_diamond.Y, fit_mask_idx))
        #p_opt = optimize.dual_annealing(baseline_diamond_fit_R_squared, bounds=((4, 9), (-8,-3)), x0=(6,-4) )
        #p_opt = optimize.least_squares(baseline_diamond_fit_R_squared, (1000000,0.00005), args = (self.Y, ideal_diamond.Y, fit_mask_idx),bounds=((1e2, 1e10), (1e-9, 1e-2)) )
        #p_opt = optimize.least_squares(baseline_diamond_fit_R_squared, x0=(6,-4), args = (self.Y, ideal_diamond.Y, fit_mask_idx),bounds=((4, 9), (-8,-3)) )
        return p_opt
        # Apply mask to baselined data. ( Maybe even optimize baseline to best fit using scipy optimzie)


    def fit_diamond_peaks(self):
            ideal_diamond = self.interpolated_typeIIA_Spectrum
            """ Fits a diamond spectrum to an ideal spectrum accounting for saturated peaks. Spectrum needs to be interpolated to the same spacing as the typeIIA diamond spectrum.  

            Args:
                ideal_diamond (_type_, optional): _description_. Defaults to typeIIA_Spectrum.
            """
            fit_mask_idx = self.test_diamond_saturation()

            def baseline_diamond_fit_R_squared(baseline_input_tuple, spectrum_wavenumber = self.X ,spectrum_intensity = self.median_filter(11).Y, typeIIA_intensity=ideal_diamond.Y, mask_idx_list=fit_mask_idx):
                lam, p = baseline_input_tuple
                print(f"lam = {lam}, p = {p}")
                baseline = pybl.whittaker.asls(spectrum_intensity, lam=lam, p=p)[0]
                baseline_subtracted = spectrum_intensity - baseline 
                baseline_subtracted_masked = baseline_subtracted[mask_idx_list]
                typeIIA_masked = typeIIA_intensity[mask_idx_list]
                fit_ratio =  baseline_subtracted_masked/ typeIIA_masked
                
                # Force Baseline to fit flat part of spectrum
                flat_range_idx = (spectrum_wavenumber > 4000) & (spectrum_wavenumber < 5000)
                weight_factor = 0.0001 # Sets balance of residulas between typeIIA and flat baseline section
                flat_baseline_residuals_squared = ((baseline_subtracted[flat_range_idx])**2).sum() * weight_factor 

                typeIIa_residuals_squared = (( (baseline_subtracted_masked/fit_ratio) - typeIIA_masked)**2).sum() 

                Total_residuals_squares = flat_baseline_residuals_squared + typeIIa_residuals_squared
                print(f" total Residuals squared {Total_residuals_squares}")
                #return np.log(Total_residuals_squares)
                return Total_residuals_squares
            
            #p_opt = optimize.minimize(baseline_diamond_fit_R_squared, (100000000,0.0005), args = (self.Y, ideal_diamond.Y, fit_mask_idx),bounds=((1e2, 1e8), (1e-9,1)),method='Nelder-Mead')
            #p_opt = optimize.minimize(baseline_diamond_fit_R_squared, x0=(6,-4), args = (self.Y, ideal_diamond.Y, fit_mask_idx),bounds=((4, 9), (-9,-4)),method='Powell')
            p_opt = optimize.differential_evolution(baseline_diamond_fit_R_squared, bounds=((1e5, 1e10), (1e-7,0.001)), x0=(10000000,0.0005), tol = 1000000000, atol = 100000)
            #p_opt = optimize.differential_evolution(baseline_diamond_fit_R_squared, bounds=((4, 9), (-9,-4)), x0=(6,-4))
            #p_opt = optimize.basinhopping(baseline_diamond_fit_R_squared, (1000000,0.0005), minimizer_kwargs = {"args": (self.Y, ideal_diamond.Y, fit_mask_idx)})
            #p_opt = optimize.basinhopping(baseline_diamond_fit_R_squared, x0=(6,-4))
            #p_opt = optimize.dual_annealing(baseline_diamond_fit_R_squared, bounds=((1e2, 1e10), (1e-9,1e-2)), args = (self.Y, ideal_diamond.Y, fit_mask_idx))
            #p_opt = optimize.dual_annealing(baseline_diamond_fit_R_squared, bounds=((4, 9), (-8,-3)), x0=(6,-4) )
            #p_opt = optimize.least_squares(baseline_diamond_fit_R_squared, (1000000,0.00005), args = (self.Y, ideal_diamond.Y, fit_mask_idx),bounds=((1e2, 1e10), (1e-9, 1e-2)) )
            #p_opt = optimize.least_squares(baseline_diamond_fit_R_squared, x0=(6,-4), args = (self.Y, ideal_diamond.Y, fit_mask_idx),bounds=((4, 9), (-8,-3)) )
            return p_opt

    def __post_init__(self):
            super().__post_init__() # Call the __post_init__ method for the Spectrum_object super class then add additional features. 
            # add or subtract 1 to keep rounded data in range
            self.interpolate_to_diamond()
            
            return self

# %%

# test saturation:
# if raw average is greater than 2.5 and (spectrum - median) has a large stdev dont use main peak.  
# %%

def baseline_als(y, lam, p, niter=10):
    """
    Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005 implemented on stackoverflow by user: sparrowcide
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    """
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        #Z = W + lam * D.dot(D.transpose())
        Z = W + lam * np.dot(D,D.T)
        z = sparse.linalg.spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z
