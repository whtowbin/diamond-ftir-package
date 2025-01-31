#%%
from pathlib import Path
from copy import deepcopy
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

import scipy.linalg
from scipy.optimize import nnls, lsq_linear

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
CAXBD = CAXBD.set_index(keys=["wn"])

# CAXBD_Spectra = CAXBD_Spectra.set_index("wn")

# CAXBD_Spectra = CAXBD_Spectra[wn_low:wn_high]
# CAXBD_Spectra_np = CAXBD_Spectra.to_numpy()  # Numpy array for matrix fitting



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

        # Adds a bunch of other non saturated regions to the baseline that are useful for fitting baselines to diamonds
        fit_mask_idx = (fit_mask_idx | (((self.X > 3130) & (self.X < 3500)) ))
        #| ((self.X > 1400) & (self.X < 1800)) | ((self.X > 680) & (self.X < 900))))
        return fit_mask_idx

    # def baseline_error_diamond_fit(self,ideal_diamond = typeIIA_Spectrum, data_mask = fit_mask_idx):
    #         self.baseline_ASLS(lam = 1000000, p = 0.0005)

    def fit_diamond_peaks(self, baseline_algorithm: str = "Whittaker", inplace: bool = False):
        """ Fits a diamond spectrum to an ideal spectrum accounting for saturated peaks. Spectrum needs to be interpolated to the same spacing as the typeIIA diamond spectrum.  

        Args:
            ideal_diamond (_type_,
                optional): _description_. Defaults to typeIIA_Spectrum.
        """
    
        fit_mask_idx = self.test_diamond_saturation()
        baseline_func = select_baseline_func(baseline_algorithm)
            
        ideal_diamond_Y = self.interpolated_typeIIA_Spectrum.Y

        X = self.X
        Y_filter = self.median_filter(21).Y
        # Subtract a mild ASLS Baseline
        Y_ASLS = baseline_func(Y_filter,lam = 1e10, p = 0.0005)
        Y_subtracted = Y_filter - Y_ASLS
        # Subtract a Semi-agressive rubberband baseline
        #Y_rubber = self.median_filter(21).baseline_aggressive_rubberband(Y_stretch=0.00000001).Y
        Y_rubber = baseline_aggressive_rubberband(X, Y_subtracted,Y_stretch=0.00000001)
        Y_subtracted  = Y_subtracted - Y_rubber

        # Fit a more aggressive ASLS Baseline to the baseline subtracted values
        def baseline_diamond_fit_R_squared(baseline_input_tuple:tuple[float,float], spectrum_wavenumber = X ,spectrum_intensity = Y_subtracted, 
                                            typeIIA_intensity =ideal_diamond_Y, mask_idx_list =fit_mask_idx, ):
            """Function to fit a baseline and a thickness normalized "Ideal" TypeIIA to a given diamond FTIR spectrum and calculate the residuals using an optimization function 
                Written to be semi-optimized for the optimiziaiton loop
            Args:
                baseline_input_tuple tuple[float,float]: Tuple of inputs for Asymmetric Least squares baseline fitting function. Lam and P
                spectrum_wavenumber (NDARRAY, optional): Array of Spectrum X intercepts typically wavenumber. Defaults to X.
                spectrum_intensity (NDARRAY, optional): Diamond FTIR Spectrum Intensity Measurements typically absorbance. Defaults to Y_subtracted.
                typeIIA_intensity (NDARRAY optional): Ideal TypeIIA Diamond FTIR Spectrum Intensity Measurements typically absorbance. Defaults to ideal_diamond_Y.
                mask_idx_list (NDARRAY[int], optional): List or array of integers for index of there to evaluate functions. Defaults to fit_mask_idx.

            Returns:
                _type_: _description_
            """
            lam, p = baseline_input_tuple
            print(f"lam = {lam}, p = {p}")
            
            baseline = baseline_func(spectrum_intensity, lam=lam, p=p)


            baseline_subtracted = spectrum_intensity - baseline 
            baseline_subtracted_masked = baseline_subtracted[mask_idx_list]
            typeIIA_masked = typeIIA_intensity[mask_idx_list]
            fit_ratio =  np.mean(baseline_subtracted_masked/ typeIIA_masked)  
            
            # Force Baseline to fit flat part of spectrum
            flat_range_idx = (spectrum_wavenumber > 4000) & (spectrum_wavenumber < 5900) 

            #This Weight Factor should probably be something that can be fine tuned 
            weight_factor = 0.5#0.1 # Sets balance of residuals between typeIIA and flatness of the baseline section
            flat_baseline_residuals_squared = ((baseline_subtracted[flat_range_idx])**2).sum() * weight_factor 

            # Attempts to weight the residuals under the unsaturated daimond peaks more heavily
            typeIIa_residuals_squared = (( (baseline_subtracted_masked/fit_ratio) - typeIIA_masked)**2).sum() 

            Total_residuals_squares = flat_baseline_residuals_squared + typeIIa_residuals_squared
            #print(f" total Residuals squared {Total_residuals_squares}")
            #return np.log(Total_residuals_squares)
            return Total_residuals_squares
        
        p_opt = optimize.differential_evolution(baseline_diamond_fit_R_squared, bounds=((1e7, 1e12), (1e-9,0.001)), x0=(1000000000,0.0005), maxiter = 1000, atol = 1000000000, tol = 1000000000000000000)
        #p_opt = optimize.dual_annealing(baseline_diamond_fit_R_squared, bounds=((1e6, 1e12), (1e-9,0.001)), x0=(10000000,0.0005), )#tol = 1000000000000000, maxiter = 200, atol = 100)

        baseline_opt  = baseline_func(Y_subtracted , lam=p_opt.x[0], p=p_opt.x[1])
        baseline_out =   baseline_opt + Y_rubber + Y_ASLS

        baseline_subtracted = self.Y - baseline_out 
        baseline_subtracted_masked = baseline_subtracted[fit_mask_idx]
        typeIIA_masked =  ideal_diamond_Y[fit_mask_idx]
        fit_ratio =  np.mean(baseline_subtracted_masked/ typeIIA_masked)

        if inplace == False:

            # it might be better to return a full copy of the object not just the baseline as a spectrum
            Spectrum_out = Spectrum(X,baseline_out)
            Spectrum_out.typeIIA_ratio = fit_ratio
            return Spectrum_out
        
        else:
            self.typeIIA_ratio = deepcopy(fit_ratio)
            self.baseline = deepcopy(baseline_out)
            # output intermediate calcs for diagnosics 
            self.outputdict  = {"mask": fit_mask_idx, "baseline_subtracted":baseline_subtracted, "TypeIIA_Y": ideal_diamond_Y, "Fit_ratio": fit_ratio}



    def fit_baseline(self):
        try:

            self.fit_diamond_peaks(baseline_algorithm = "Whittaker", inplace=True)
            
        except (np.linalg.LinAlgError, RuntimeError) as e:
            self.fit_diamond_peaks(baseline_algorithm = "ALS", inplace=True)
            if e is np.linalg.LinAlgError:
                print(e)
                print("error caught. Fitting Baseline with alternate baseline function")

        except Exception as e:
            print(e)


    def normalize_diamond(self, inplace = True):
        """returns spectrum normalized to 1 cm thickness based on unsaturated Diamond peak heights"""
        try:
            normalized_absorbance = (self.Y - self.baseline) / self.typeIIA_ratio # Is the fit baseline already thickness corrected?
        
            normalized_spectrum = Spectrum(
                    X=self.X,
                    Y=normalized_absorbance,
                    X_Unit="Wavenumber",
                    Y_Unit="Absorbance",
                    )
            if inplace ==False:
                return normalized_spectrum
            else:
                self.normalized_spectrum = normalized_spectrum


        except Exception as e:
            print(e)
            "Diamond Spectrum object must have a baseline and typeIIA_ratio fit prior to using this method, try using the fit_baseline() method before calling this"


    def Nitrogen_fit(self, CAXBD = CAXBD, plot_fit = False):
        # Fits Nitrogen Aggregation peaks using the C,A,X,B,D spectra developed by Fischer at Maidenhead

        # Select Same Range as CAXBD in Diamond Spectrum
        # Do a non-negative Least squares fit of the CAXBD components 
        # Output the components in terms of both fit parameters and concentration of nitrogen.
            # params = np.linalg.lstsq(CAXBD_matrix, spectrum.values, rcond=None)[0]
        wn_low = 900
        wn_high = 1400
        
        CAXBD_select = CAXBD.loc[wn_low:wn_high]
        CAXBD_matrix = CAXBD_select.to_numpy()
        wn_array = CAXBD_select.index.to_numpy()
        
        
        offset_pos = np.ones_like(wn_array)
        offset_neg = offset_pos * -1

        linear_pos = deepcopy(wn_array) #np.arange(len(wn_array))
        linear_neg = linear_pos * -1

        linear_array = np.vstack((offset_pos, offset_neg, linear_pos, linear_neg))
        CAXBD_matrix = np.hstack((CAXBD_matrix, linear_array.T))

        labels = ["C", "A", "X", "B","D", "offset+", "offset-", "linear+", "linear-"]
        fit_component_df = pd.DataFrame(CAXBD_matrix, columns= labels, index= wn_array)

        spec = self.normalized_spectrum
        spec_intensity = spec.select_range(wn_low, wn_high+1).Y

        try:
            params = nnls(
                CAXBD_matrix,
                spec_intensity,  # .values
            )[0]
            if plot_fit == True:
                fig, ax = plt.subplots()
                ax.plot(wn_array, spec_intensity, label = "Spectrum")
                fit_comp = (fit_component_df * params) #(CAXBD_select * params)
                model_spectrum = fit_comp.sum(axis=1, numeric_only=True)
                model_spectrum.plot(label = "Fit Spectrum")
                fit_comp.plot(ax = ax)
                ax.legend()

            # Assumes all params are positive. I think this is correct but That depends on the purpose of the X and D components
        except ValueError as e:
            print("Value Error")
            print(e)
            params = np.zeros(5)

        C_comp = params[0] 
        A_comp = params[1]
        X_comp = params[2]
        B_comp = params[3] 
        D_comp = params[4] 
        A_Nitrogen = params[1] * 16.5
        B_Nitrogen = params[3] * 79.4 
        wn_spacing = self.initial_X[1]- self.initial_X[0]
        C_correction = C_center_wn_spacing_correction(wn_spacing) # Correction based on spectral resolution in Liggins 2010 PhD thesis
        C_Nitrogen = params[0] * 0.624332796 * C_correction 

        Total_N = A_Nitrogen + B_Nitrogen + C_Nitrogen
        B_percent = B_Nitrogen / Total_N * 100
        C_percent = C_Nitrogen/ Total_N * 100
# C = fit_param[0]  This needs to be multiplied by a molar absorptivity and as well as a correction for spectral resolution Liggins 2010 Thesis Warwick University
        nitrogen_dict = {
            "C_comp" :C_comp, 
            "A_comp" : A_comp,
            "X_comp" : X_comp,
            "B_comp" : B_comp,
            "D_comp" : D_comp,
            "A_Nitrogen" : A_Nitrogen,
            "B_Nitrogen" : B_Nitrogen,
            "C_Nitrogen" : C_Nitrogen,
            "Total_N" : Total_N,
            "B_percent" : B_percent,
            "C_percent" : C_percent,
            }
        self.nitrogen_dict = nitrogen_dict


    def Nitrogen_fit_leastSQ(self, CAXBD = CAXBD, plot_fit = False):
        # Fits Nitrogen Aggregation peaks using the C,A,X,B,D spectra developed by Fischer at Maidenhead

        # Select Same Range as CAXBD in Diamond Spectrum
        # Do a non-negative Least squares fit of the CAXBD components 
        # Output the components in terms of both fit parameters and concentration of nitrogen.
            # params = np.linalg.lstsq(CAXBD_matrix, spectrum.values, rcond=None)[0]
        wn_low = 900
        wn_high = 1400
        
        CAXBD_select = CAXBD.loc[wn_low:wn_high]
        CAXBD_matrix = CAXBD_select.to_numpy()
        wn_array = CAXBD_select.index.to_numpy()
        
        
        offset= np.ones_like(wn_array)
        linear= deepcopy(wn_array) #np.arange(len(wn_array))
        

        linear_array = np.vstack((offset, linear))
        CAXBD_matrix = np.hstack((CAXBD_matrix, linear_array.T))

        labels = ["C", "A", "X", "B","D", "offset", "linear",]
        fit_component_df = pd.DataFrame(CAXBD_matrix, columns= labels, index= wn_array)

        spec = self.normalized_spectrum
        spec_intensity = spec.select_range(wn_low, wn_high+1).Y
        bounds = np.array([(0,np.inf),(0,np.inf),(0,np.inf),(0,np.inf),(0,np.inf),(-np.inf,np.inf),(-np.inf,np.inf)]).T
        try:
            params = lsq_linear(
                CAXBD_matrix,
                spec_intensity,  bounds=bounds
            )["x"]
            if plot_fit == True:
                fig, ax = plt.subplots()
                ax.plot(wn_array, spec_intensity, label = "Spectrum")
                fit_comp = (fit_component_df * params) #(CAXBD_select * params)
                model_spectrum = fit_comp.sum(axis=1, numeric_only=True)
                model_spectrum.plot(label = "Fit Spectrum")
                fit_comp.plot(ax = ax)
                ax.legend()

            # Assumes all params are positive. I think this is correct but That depends on the purpose of the X and D components
        except ValueError as e:
            print("Value Error")
            print(e)
            params = np.zeros(5)

        C_comp = params[0] 
        A_comp = params[1]
        X_comp = params[2]
        B_comp = params[3] 
        D_comp = params[4] 
        A_Nitrogen = params[1] * 16.5
        B_Nitrogen = params[3] * 79.4 
        wn_spacing = self.initial_X[1]- self.initial_X[0]
        C_correction = C_center_wn_spacing_correction(wn_spacing) # Correction based on spectral resolution in Liggins 2010 PhD thesis
        C_Nitrogen = params[0] * 0.624332796 * C_correction 

        Total_N = A_Nitrogen + B_Nitrogen + C_Nitrogen
        B_percent = B_Nitrogen / Total_N * 100
        C_percent = C_Nitrogen/ Total_N * 100
# C = fit_param[0]  This needs to be multiplied by a molar absorptivity and as well as a correction for spectral resolution Liggins 2010 Thesis Warwick University
        nitrogen_dict = {
            "C_comp" :C_comp, 
            "A_comp" : A_comp,
            "X_comp" : X_comp,
            "B_comp" : B_comp,
            "D_comp" : D_comp,
            "A_Nitrogen" : A_Nitrogen,
            "B_Nitrogen" : B_Nitrogen,
            "C_Nitrogen" : C_Nitrogen,
            "Total_N" : Total_N,
            "B_percent" : B_percent,
            "C_percent" : C_percent,
            }
        self.nitrogen_dict = nitrogen_dict
    

    def measure_3107_peak(self):
        pass

# Alt Peaks 3085, 

    def measure_platelets():
        pass

    def __post_init__(self):
            super().__post_init__() # Call the __post_init__ method for the Spectrum_object super class then add additional features. 
            # add or subtract 1 to keep rounded data in range
            self.interpolate_to_diamond()
            self.typeIIA_ratio = None
            
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


def baseline_aggressive_rubberband(x, y, Y_stretch: float =0.0001, plot_intermediate:bool = False):
    midpoint_X = round((max(x) - min(x))/2)
    nonlinear_offset = Y_stretch * (x - midpoint_X)**2 
    y_alt = y + nonlinear_offset
    baseline = rubberband(x, y_alt)

    
    return (baseline - nonlinear_offset)



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



def C_center_wn_spacing_correction(wn_spacing:float) -> float:
    return (9.7043*wn_spacing + 25.304) # Function derived from Linear fit to values determined in Liggins et al. 2010 Phd Thesis
# %%
