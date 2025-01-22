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
from tenacity import retry # Function to retry failed fitting algorithms for a set number of times
from scipy.spatial import ConvexHull

from .Spectrum_obj import Spectrum

from .typeIIA import typeIIA_json
from .CAXBD import CAXBD_json

import matplotlib.pyplot as plt

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
            print("Primary Diamond Peaks Are Saturated")
            
        elif (main_diamond_sat == True) & (secondary_diamond_sat == True) & (third_diamond_sat == False):
            fit_mask_idx  = (self.X > 3130) & (self.X < 3500)
            print("Secondary Diamond Peaks Are Saturated")

        fit_mask_idx = fit_mask_idx & ((self.X > 3130) & (self.X < 3500)) & ((self.X > 1400) & (self.X < 1800)) & ((self.X > 680) & (self.X < 900))
        return fit_mask_idx

    # def baseline_error_diamond_fit(self,ideal_diamond = typeIIA_Spectrum, data_mask = fit_mask_idx):
    #         self.baseline_ASLS(lam = 1000000, p = 0.0005)

    def fit_diamond_peaks(self, baseline_algorithm = "Whittaker"):
        """ Fits a diamond spectrum to an ideal spectrum accounting for saturated peaks. Spectrum needs to be interpolated to the same spacing as the typeIIA diamond spectrum.  

        Args:
            ideal_diamond (_type_,
                optional): _description_. Defaults to typeIIA_Spectrum.
        """
    
        fit_mask_idx = self.test_diamond_saturation()
        baseline_func = select_baseline_func(baseline_algorithm)
            
        ideal_diamond_Y = self.interpolated_typeIIA_Spectrum.Y
        fit_mask_idx = self.test_diamond_saturation()
        X = self.X
       
        Y_rubber = self.median_filter(11).baseline_aggressive_rubberband(Y_stretch=0.00000001)
        Y_sub = self.median_filter(11).Y  - Y_rubber
        def baseline_diamond_fit_R_squared(baseline_input_tuple, spectrum_wavenumber = X ,spectrum_intensity = Y_sub, 
                                            typeIIA_intensity=ideal_diamond_Y, mask_idx_list=fit_mask_idx):
            lam, p = baseline_input_tuple
            print(f"lam = {lam}, p = {p}")
            
            baseline = baseline_func(spectrum_intensity, lam=lam, p=p)


            baseline_subtracted = spectrum_intensity - baseline 
            baseline_subtracted_masked = baseline_subtracted[mask_idx_list]
            typeIIA_masked = typeIIA_intensity[mask_idx_list]
            fit_ratio =  baseline_subtracted_masked/ typeIIA_masked
            
            # Force Baseline to fit flat part of spectrum
            flat_range_idx = (spectrum_wavenumber > 4000) & (spectrum_wavenumber < 5000)
            weight_factor = 0.0001 # Sets balance of residuals between typeIIA and flat baseline section
            flat_baseline_residuals_squared = ((baseline_subtracted[flat_range_idx])**2).sum() * weight_factor 

            typeIIa_residuals_squared = (( (baseline_subtracted_masked/fit_ratio) - typeIIA_masked)**2).sum() 

            Total_residuals_squares = flat_baseline_residuals_squared + typeIIa_residuals_squared
            print(f" total Residuals squared {Total_residuals_squares}")
            #return np.log(Total_residuals_squares)
            return Total_residuals_squares
        
        p_opt = optimize.differential_evolution(baseline_diamond_fit_R_squared, bounds=((1e7, 1e10), (1e-7,0.001)), x0=(10000000,0.0005), tol = 1000)

        baseline_opt  = baseline_func(Y_sub , lam=p_opt.x[0], p=p_opt.x[1])
        baseline_out =   baseline_opt + Y_rubber
        
        return baseline_out
        

        # return {"Baseline":baseline_out, "fit_ratio": 
        # - Baseline how to best output the fit ratio and update the spectrum object. 

    def fit_baseline(self):
        try:

            baseline = self.fit_diamond_peaks(baseline_algorithm = "Whittaker")
            
        except (np.linalg.LinAlgError, RuntimeError) as e:
            baseline = self.fit_diamond_peaks(baseline_algorithm = "ALS")
            if e is np.linalg.LinAlgError:
                print(e)
                print("error caught. Fitting Baseline with alternate baseline function")

        except Exception as e:
            print(e)

        return baseline

    def baseline_rubberband(self):
        baseline = rubberband(self.X, self.Y)
        return baseline


    def baseline_aggressive_rubberband(self, Y_stretch=0.0001, plot_intermediate = False):
        midpoint_X = round((max(self.X) - min(self.X))/2)
        nonlinear_offset = Y_stretch * (self.X - midpoint_X)**2
        Y_alt = self.Y + nonlinear_offset
        baseline = rubberband(self.X, Y_alt)

        if plot_intermediate ==True:
            plt.plot(self.X, Y_alt)
            plt.plot(self.X, baseline)
        
        return (baseline - nonlinear_offset)


    def normalize_diamond(self, TypeIIA_ratio: float):
        """returns spectrum normalized to 1 cm thickness based on unsaturated Diamond peak heights"""
        normalized_absorbance = (self.Y - self.baseline) / TypeIIA_ratio # Is the fit baseline already thickness corrected?
        pass

    def Nitrogen_fit(self, CAXBD = CAXBD):
        # Fits Nitrogen Aggregation peaks using the C,A,X,B,D spectra developed by XYZ at Maidenhead
        pass

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



def als_baseline(intensities, asymmetry_param=0.05, smoothness_param=5e5,
                 max_iters=10, conv_thresh=1e-5, verbose=False):
  '''Computes the asymmetric least squares baseline.
  * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
  smoothness_param: Relative importance of smoothness of the predicted response.
  asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                       Setting p=1 is effectively a hinge loss.
  '''
  smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
  # Rename p for concision.
  p = asymmetry_param
  # Initialize weights.
  w = np.ones(intensities.shape[0])
  for i in range(max_iters):
    z = smoother.smooth(w)
    mask = intensities > z
    new_w = p*mask + (1-p)*(~mask)
    conv = np.linalg.norm(new_w - w)
    if verbose:
      print(i+1, conv)
    if conv < conv_thresh:
      break
    w = new_w
  else:
    print('ALS did not converge in %d iterations' % max_iters)
  return z


class WhittakerSmoother(object):
  def __init__(self, signal, smoothness_param, deriv_order=1):
    self.y = signal
    assert deriv_order > 0, 'deriv_order must be an int > 0'
    # Compute the fixed derivative of identity (D).
    d = np.zeros(deriv_order*2 + 1, dtype=int)
    d[deriv_order] = 1
    d = np.diff(d, n=deriv_order)
    n = self.y.shape[0]
    k = len(d)
    s = float(smoothness_param)

    # Here be dragons: essentially we're faking a big banded matrix D,
    # doing s * D.T.dot(D) with it, then taking the upper triangular bands.
    diag_sums = np.vstack([
        np.pad(s*np.cumsum(d[-i:]*d[:i]), ((k-i,0),), 'constant')
        for i in range(1, k+1)])
    upper_bands = np.tile(diag_sums[:,-1:], n)
    upper_bands[:,:k] = diag_sums
    for i,ds in enumerate(diag_sums):
      upper_bands[i,-i-1:] = ds[::-1][:i+1]
    self.upper_bands = upper_bands

  def smooth(self, w):
    foo = self.upper_bands.copy()
    foo[-1] += w  # last row is the diagonal
    return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)



def rubberband(x, y):
    """
    Rubber band baseline from
    # Find the convex hull R Kiselev on stack overflow
    https://dsp.stackexchange.com/questions/2725/how-to-perform-a-rubberband-correction-on-spectroscopic-data
    """
    v = ConvexHull(np.array(list(zip(x, y)))).vertices
    # Rotate convex hull vertices until they start from the lowest one
    v = np.roll(v, -v.argmin())
    # Leave only the ascending part
    v = v[: v.argmax()]

    # Create baseline using linear interpolation between vertices
    return np.interp(x, x[v], y[v])




def select_baseline_func(baseline_algorithm = "Whittaker"):
    """ Fits a diamond spectrum to an ideal spectrum accounting for saturated peaks. Spectrum needs to be interpolated to the same spacing as the typeIIA diamond spectrum.  

    Args:
        ideal_diamond (_type_,
            optional): _description_. Defaults to typeIIA_Spectrum.
    """
    def baseline_Whittaker_internal(spectrum_intensity, lam, p):
        return pybl.whittaker.asls(spectrum_intensity, lam, p)[0]

    match baseline_algorithm:
        case "Whittaker":
            baseline_func = baseline_Whittaker_internal
        case "ALS":

            baseline_func = baseline_als
        case _:
            print("Incorrect Baseline Option Selected")

    return baseline_func
