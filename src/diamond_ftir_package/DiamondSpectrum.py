# %%
from pathlib import Path
from copy import deepcopy
import numpy as np

# import scipy.signal
import scipy.optimize as optimize
import pybaselines as pybl
from copy import deepcopy
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Union
import scipy.sparse as sparse
from tenacity import retry  # Function to retry failed fitting algorithms for a set number of times
from scipy.spatial import ConvexHull


# import scipy.linalg
from scipy.optimize import nnls, lsq_linear

import matplotlib.pyplot as plt

try:
    from .Spectrum_obj import Spectrum
    from .typeIIA import typeIIA_json
    from .CAXBDY import CAXBDY_json

except:
    from Spectrum_obj import Spectrum
    from typeIIA import typeIIA_json
    from CAXBDY import CAXBDY_json

# from warnings import deprecated

# %%

# Type Spectra are only imported once outside of the class so that they dont fill up the memory in long loops, by creating multiple identical objects
typeIIA = pd.DataFrame(typeIIA_json)
# typeIIA = typeIIA.set_index(keys=["wn"])

typeIIA_Spectrum = Spectrum(
    X=typeIIA["wn"],
    Y=typeIIA["absorbance"],
    X_Unit="Wavenumber",
    Y_Unit="Absorbance",
)


CAXBDY = pd.DataFrame(CAXBDY_json)
CAXBDY = CAXBDY.set_index(keys=["wn"])


@dataclass()
class Diamond_Spectrum(Spectrum):
    """
    Specialized spectrum class for diamond FTIR analysis with methods for nitrogen content and defect classification.

    The Diamond_Spectrum class extends the base Spectrum class with diamond-specific functionality,
    including thickness normalization using type IIa diamond reference, nitrogen content estimation,
    platelet defect analysis, and identification of spectral features associated with various diamond defects.

    The class automatically interpolates input spectra to 1 cm⁻¹ spacing within the range of 601-6000 cm⁻¹
    to ensure consistent processing across different instruments and measurement conditions.

    Features:
        - Automatic detection of saturated diamond peaks
        - Thickness normalization using type IIa diamond reference spectra
        - Nitrogen content quantification using CAXBDY fitting method
        - Measurement of platelet peaks, amber centers, and hydrogen-related defects
        - Specialized baseline correction optimized for diamond spectra

    Attributes:
        All attributes from the parent Spectrum class, plus:
        interpolated_typeIIA_Spectrum (Spectrum): Reference type IIa spectrum interpolated to match sample
        typeIIA_ratio (float): Thickness normalization factor based on diamond intrinsic peaks
        normalized_spectrum (Spectrum): Thickness-normalized spectrum (1 cm equivalent)
        nitrogen_dict (Dict): Results of nitrogen content analysis

    Example:
        ```python
        # Load a diamond spectrum from a file
        diamond = Diamond_Spectrum.from_file("sample123.csv", X_Unit="cm⁻¹", Y_Unit="Absorbance")

        # Process the spectrum
        diamond.fit_baseline()
        diamond.normalize_diamond()
        diamond.Nitrogen_fit()
        diamond.measure_platelets_and_adjacent()
        diamond.measure_3107_peak()

        # Access results
        print(f"Total nitrogen: {diamond.nitrogen_dict['Total_N ppm']} ppm")
        print(f"Aggregation state: {diamond.nitrogen_dict['B_percent']}% B")
        print(f"Platelet peak: {diamond.normed_area_platelet}")
        print(f"3107 cm⁻¹ peak area: {diamond.normed_area_3107}")
        ```

    Notes:
        - Automatically handles spectra with saturated diamond intrinsic peaks by selecting alternative peaks
        - Based on reference spectra and methods from diamond research literature
        - Calculation of nitrogen content follows methodologies established by De Beers Technologies
    """

    def diamonds(self):
        print("Diamonds are Forever")

    def interpolate_to_diamond(self):
        """
        Interpolates the spectrum to a standard spectral range and resolution for diamond analysis.

        This method prepares both the current spectrum and the reference type IIa spectrum by:
        1. Determining the overlapping spectral range between the current spectrum and the
           reference type IIa spectrum
        2. Limiting the range to 601-6000 cm⁻¹ (the most useful range for diamond analysis)
        3. Interpolating both spectra to 1 cm⁻¹ spacing for consistent analysis

        The interpolation is performed in-place on the current spectrum, and the interpolated
        type IIa reference spectrum is stored in the `interpolated_typeIIA_Spectrum` attribute.
        This standardization is essential for thickness normalization and reliable peak analysis.

        Notes:
            This method is automatically called during object initialization and ensures
            consistent spectral resolution across all diamond processing steps.
            Most diamond-specific analysis methods require this interpolation to have been
            performed first.

        No parameters are required as the method uses the pre-loaded reference spectrum.
        """
        spec_min = np.round(self.X.min()) + 1
        spec_max = np.round(self.X.max()) - 1

        typeIIA_min = np.round(typeIIA_Spectrum.X.min())
        typeIIA_max = np.round(typeIIA_Spectrum.X.max())

        # set minimum wavenumber to 600 since most data is useless below that with our currents systems
        wn_min = max(spec_min, typeIIA_min, 601)
        wn_max = min(spec_max, typeIIA_max, 6000)

        self.interpolate(wn_min, wn_max, 1, inplace=True)

        self.interpolated_typeIIA_Spectrum = typeIIA_Spectrum.interpolate(wn_min, wn_max, 1)

    def test_diamond_saturation(self, saturation_cutoff=2.5, stdev_cut_off=0.5):
        """
        Detects saturation in diamond intrinsic absorption peaks and selects appropriate regions for thickness normalization.

        This method systematically tests three key regions in diamond FTIR spectra for detector saturation:
        1. Primary two-phonon diamond peaks (1970-2040 cm⁻¹)
        2. Secondary two-phonon diamond peaks (2400-2575 cm⁻¹)
        3. Three-phonon diamond peaks (3000-3500 cm⁻¹)

        The method assumes sequential saturation: if primary peaks are unsaturated, secondary and tertiary
        peaks are also unsaturated. This provides robustness for fitting diamond spectra with various
        thicknesses and collection conditions where saturation is common.

        Returns:
            numpy.ndarray: Boolean mask array indicating regions of the spectrum to use for baseline
            and thickness normalization fitting. True values in the mask correspond to spectral regions
            that should be included in fitting procedures.

        Notes:
            - This method should only be used on raw (not baseline-corrected) and non-thickness-normalized spectra
            - Saturation detection uses both absolute intensity thresholds and standard deviation metrics
            - The returned mask is used by the diamond peak fitting algorithm to avoid saturated regions
            - For very thick diamonds where all intrinsic peaks are saturated, an exception is raised
            - Additional non-saturated regions (e.g., 3130-3500 cm⁻¹) are always included in the mask

        Raises:
            Exception: If all three diamond peak regions are saturated, making thickness normalization impossible

        See Also:
            fit_diamond_peaks: Uses the mask from this method to perform baseline and thickness fitting
            test_saturation: Lower-level method that tests individual regions for saturation
        """
        main_diamond_sat = self.test_saturation(
            X_low=1970,
            X_high=2040,
            saturation_cutoff=saturation_cutoff,
            stdev_cut_off=stdev_cut_off,
        )
        secondary_diamond_sat = self.test_saturation(
            X_low=2400,
            X_high=2575,
            saturation_cutoff=saturation_cutoff,
            stdev_cut_off=stdev_cut_off,
        )
        third_diamond_sat = self.test_saturation(
            X_low=3000,
            X_high=3500,
            saturation_cutoff=saturation_cutoff,
            stdev_cut_off=stdev_cut_off,
        )

        if main_diamond_sat == False:
            fit_mask_idx = ((self.X > 1800) & (self.X < 2313)) | (self.X > 2390) & (self.X < 2670)

        elif (main_diamond_sat == True) & (secondary_diamond_sat == False):
            fit_mask_idx = (self.X > 2390) & (self.X < 2670)
            print("Primary Diamond Peaks Are Saturated")

        elif (
            (main_diamond_sat == True)
            & (secondary_diamond_sat == True)
            & (third_diamond_sat == False)
        ):
            fit_mask_idx = (self.X > 3130) & (self.X < 3500)
            print("Secondary Diamond Peaks Are Saturated")

        elif (
            (main_diamond_sat == True)
            & (secondary_diamond_sat == True)
            & (third_diamond_sat == True)
        ):
            raise Exception(
                "All diamond peaks  are saturated and thickness correction cannot be determined"
            )

        # Adds a bunch of other non saturated regions to the baseline that are useful for fitting baselines to diamonds
        fit_mask_idx = (
            fit_mask_idx | ((self.X > 3130) & (self.X < 3500))  #
            # | ((self.X > 1450) & (self.X < 1750))
            # | ((self.X > 680) & (self.X < 900))
        )
        return fit_mask_idx

    # def baseline_error_diamond_fit(self,ideal_diamond = typeIIA_Spectrum, data_mask = fit_mask_idx):
    #         self.baseline_ASLS(lam = 1000000, p = 0.0005)

    def fit_diamond_peaks(
        self,
        baseline_algorithm: str = "Whittaker",
        inplace: bool = False,
        saturation_cutoff=2.5,
        stdev_cut_off=0.5,
    ):
        """
        Fits a sophisticated baseline to diamond spectra and calculates thickness normalization factor.

        This method performs a multi-stage baseline correction optimized specifically for diamond FTIR spectra:
        1. Detects which diamond intrinsic peaks are unsaturated using test_diamond_saturation()
        2. Applies a coarse median filter and initial baseline removal
        3. Applies a rubberband correction to remove broad curvature
        4. Optimizes a final baseline using the reference type IIa spectrum as a guide
        5. Calculates the thickness normalization factor by comparing unsaturated diamond peaks
           to the type IIa reference

        The method handles saturated regions automatically by using the mask from test_diamond_saturation()
        to only fit against unsaturated diamond intrinsic peaks.

        Args:
            baseline_algorithm (str, optional): Algorithm to use for asymmetric least squares baseline
                correction. Options are "Whittaker" (recommended, uses PyBaselines implementation) or
                "ALS" (custom implementation, slower but more stable for some spectra). Defaults to "Whittaker".
            inplace (bool, optional): If True, stores the baseline and typeIIA_ratio in the current object.
                If False, returns a new Spectrum object with the baseline. Defaults to False.

        Returns:
            Spectrum or self: If inplace=False, returns a new Spectrum object with the calculated
            baseline as Y values and typeIIA_ratio attribute. If inplace=True, modifies the current
            object by setting its baseline attribute and typeIIA_ratio, and returns self.

        Notes:
            - The method requires the spectrum to have been interpolated with interpolate_to_diamond() first
            - The optimization balances fitting the diamond intrinsic peaks while maintaining flat regions
            - The calculated typeIIA_ratio is used for thickness normalization in normalize_diamond()
            - For very noisy spectra, try using median_filter() before applying this method

        Raises:
            Exception: If diamond peak saturation testing fails or optimization cannot converge

        See Also:
            test_diamond_saturation: Detects which diamond peaks are saturated
            normalize_diamond: Uses the typeIIA_ratio to create a thickness-normalized spectrum
        """
        try:
            fit_mask_idx = self.test_diamond_saturation(saturation_cutoff, stdev_cut_off)

        except Exception as e:
            print(e)

        baseline_func = select_baseline_func(baseline_algorithm)

        ideal_diamond_Y = self.interpolated_typeIIA_Spectrum.Y

        X = self.X
        Y_filter = self.median_filter(21).Y
        # Subtract a mild ASLS Baseline
        Y_ASLS = baseline_func(Y_filter, lam=1e10, p=0.0005)
        Y_subtracted = Y_filter - Y_ASLS
        # Subtract a Semi-agressive rubberband baseline
        # Y_rubber = self.median_filter(21).baseline_aggressive_rubberband(Y_stretch=0.00000001).Y
        Y_rubber = baseline_aggressive_rubberband(X, Y_subtracted, Y_stretch=0.00000002)
        Y_subtracted = Y_subtracted - Y_rubber

        # Fit a more aggressive ASLS Baseline to the baseline subtracted values
        def baseline_diamond_fit_R_squared(
            baseline_input_tuple: tuple[float, float],
            spectrum_wavenumber=X,
            spectrum_intensity=Y_subtracted,
            typeIIA_intensity=ideal_diamond_Y,
            mask_idx_list=fit_mask_idx,
        ):
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
            # print(f"lam = {lam}, p = {p}")

            baseline = baseline_func(spectrum_intensity, lam=lam, p=p)

            baseline_subtracted = spectrum_intensity - baseline
            baseline_subtracted_masked = baseline_subtracted[mask_idx_list]
            typeIIA_masked = typeIIA_intensity[mask_idx_list]
            fit_ratio = np.mean(baseline_subtracted_masked / typeIIA_masked)

            # Force Baseline to fit flat part of spectrum
            flat_range_idx = (spectrum_wavenumber > 4000) & (spectrum_wavenumber < 5900)

            # This Weight Factor should probably be something that can be fine tuned
            weight_factor = 0.5  # 0.1 # Sets balance of residuals between typeIIA and flatness of the baseline section
            flat_baseline_residuals_squared = (
                (baseline_subtracted[flat_range_idx]) ** 2
            ).sum() * weight_factor

            # Attempts to weight the residuals under the unsaturated daimond peaks more heavily
            typeIIa_residuals_squared = (
                ((baseline_subtracted_masked / fit_ratio) - typeIIA_masked) ** 2
            ).sum()

            Total_residuals_squares = flat_baseline_residuals_squared + typeIIa_residuals_squared
            # print(f" total Residuals squared {Total_residuals_squares}")
            # return np.log(Total_residuals_squares)
            return Total_residuals_squares

        # p_opt = optimize.differential_evolution(baseline_diamond_fit_R_squared, bounds=((1e8, 1e14), (1e-9,0.00001)), x0=(1e10,0.000005), maxiter = 500, atol = 1000000000, tol = 1000000000000000000)
        # p_opt = optimize.dual_annealing(baseline_diamond_fit_R_squared, bounds=((1e8, 1e14), (1e-9,0.00001)), x0=(1e10,0.000005),  maxiter = 200)
        p_opt = optimize.minimize(
            baseline_diamond_fit_R_squared,
            bounds=((1e8, 1e14), (1e-9, 0.0001)),
            x0=(1e10, 0.000005),
        )

        baseline_opt = baseline_func(Y_subtracted, lam=p_opt.x[0], p=p_opt.x[1])
        baseline_out = baseline_opt + Y_rubber + Y_ASLS

        baseline_subtracted = self.Y - baseline_out
        baseline_subtracted_masked = baseline_subtracted[fit_mask_idx]
        typeIIA_masked = ideal_diamond_Y[fit_mask_idx]
        fit_ratio = np.mean(baseline_subtracted_masked / typeIIA_masked)

        if inplace == False:
            # it might be better to return a full copy of the object not just the baseline as a spectrum
            Spectrum_out = Spectrum(X, baseline_out)
            Spectrum_out.typeIIA_ratio = fit_ratio
            return Spectrum_out

        else:
            self.typeIIA_ratio = deepcopy(fit_ratio)
            self.baseline = deepcopy(baseline_out)
            # output intermediate calcs for diagnosics
            self.outputdict = {
                "mask": fit_mask_idx,
                "baseline_subtracted": baseline_subtracted,
                "TypeIIA_Y": ideal_diamond_Y,
                "Fit_ratio": fit_ratio,
            }

    def fit_baseline(self, saturation_cutoff=2.5, stdev_cut_off=0.5):
        try:
            self.fit_diamond_peaks(
                baseline_algorithm="Whittaker",
                inplace=True,
            )

        except (np.linalg.LinAlgError, RuntimeError) as e:
            self.fit_diamond_peaks(
                baseline_algorithm="ALS",
                inplace=True,
                saturation_cutoff=saturation_cutoff,
                stdev_cut_off=stdev_cut_off,
            )
            if e is np.linalg.LinAlgError:
                print(e)
                print("error caught. Fitting Baseline with alternate baseline function")

        except Exception as e:
            print(e)

    def normalize_diamond(self, inplace=True):
        """returns spectrum normalized to 1 cm thickness based on unsaturated Diamond peak heights"""
        try:
            normalized_absorbance = (
                self.Y - self.baseline
            ) / self.typeIIA_ratio  # Is the fit baseline already thickness corrected?

            normalized_spectrum = Spectrum(
                X=self.X,
                Y=normalized_absorbance,
                X_Unit="Wavenumber",
                Y_Unit="Absorbance",
            )
            if inplace == False:
                return normalized_spectrum
            else:
                self.normalized_spectrum = normalized_spectrum

        except Exception as e:
            print(e)
            "Diamond Spectrum object must have a baseline and typeIIA_ratio fit prior to using this method, try using the fit_baseline() method before calling this"

    def Nitrogen_fit(self, CAXBDY=CAXBDY, plot_fit=False, max_C_or_B=0.01):
        """
        Quantifies nitrogen content and aggregation state using the CAXBDY component fitting method.

        This method analyzes the 950-1350 cm⁻¹ region to determine nitrogen content by fitting
        reference spectra for different nitrogen defects in diamond:
        - C centers: isolated substitutional nitrogen (Type Ib)
        - A centers: nitrogen pairs (Type IaA)
        - B centers: four nitrogen atoms surrounding a vacancy (Type IaB)
        - X: N+ component
        - D: Platelet related Peak
        - Y: Component found in Type Ib diamonds

        The method applies appropriate constraints to ensure physically meaningful results:
        - For Type IaAB diamonds, C and X components are limited to 10% of the highest peak
        - D component is limited based on Woods' linear correlation with B component
        - Spectral resolution correction is applied to C center calculations

        The results are stored in the `nitrogen_dict` attribute with comprehensive information
        about nitrogen concentrations and aggregation states.

        Args:
            CAXBDY (pandas.DataFrame, optional): Reference nitrogen component spectra.
                Defaults to the pre-loaded CAXBDY dataset.
            plot_fit (bool, optional): Whether to plot the component fitting results.
                Useful for visually assessing fit quality. Defaults to False.
            max_C_or_B (float, optional): Maximum ratio for minor components (C in Type IaAB or
                B in Type Ib). Controls the balance between component types. Defaults to 0.1.

        Notes:
            - This method requires a normalized spectrum (use normalize_diamond() first)
            - Nitrogen concentrations are calculated in atomic ppm using standard calibration factors:
             - * A centers: 16.5 ppm per cm⁻¹ of absorption
             - * B centers: 79.4 ppm per cm⁻¹ of absorption
             - * C centers: Variable based on spectral resolution (Liggins 2010)
            - Aggregation state is expressed as B/(A+B+C) percentage
            - For accurate results, spectra should be thickness-normalized to 1 cm
            - The component fitting includes linear offset correction

        Example:
            ```python
            diamond = Diamond_Spectrum.from_file("sample.csv")
            diamond.fit_baseline()
            diamond.normalize_diamond()
            diamond.Nitrogen_fit(plot_fit=True)
            print(f"Total nitrogen: {diamond.nitrogen_dict['Total_N ppm']} ppm")
            print(f"Aggregation state: {diamond.nitrogen_dict['B_percent']}% B")
            ```

        References:
            Based on methodology developed by D. Fisher (De Beers Technologies, Maidenhead)
            for the CAXBDY97n Excel spreadsheet and further refined by Liggins (2010).
            Inspired By Diamap Program by Howell et al.
            and work By Specht et al.
        """
        wn_low = 950
        wn_high = 1350  # 1400

        CAXBDY_select = CAXBDY.loc[wn_low:wn_high]
        CAXBDY_matrix = CAXBDY_select.to_numpy()
        wn_array = CAXBDY_select.index.to_numpy()

        offset = np.ones_like(wn_array)
        linear = np.arange(len(offset)) - (wn_high - wn_low)

        linear_array = np.vstack((offset, linear))
        CAXBDY_matrix = np.hstack((CAXBDY_matrix, linear_array.T))

        labels = [
            "C",
            "A",
            "X",
            "B",
            "D",
            "Y",
            "offset",
            "linear",
        ]
        fit_component_df = pd.DataFrame(CAXBDY_matrix, columns=labels, index=wn_array)

        spec = self.normalized_spectrum
        spec_intensity = spec.select_range(wn_low, wn_high + 1).Y

        wn_spacing = self.initial_X[1] - self.initial_X[0]
        C_correction = C_center_wn_spacing_correction(wn_spacing)

        # I should make the bounds limit the height of C or B depending on if its a typa 1aAB or 1b diamond
        bounds = np.array(
            [
                (0, np.inf),
                (0, np.inf),
                (0, np.inf),
                (0, np.inf),
                (0, np.inf),
                (0, np.inf),
                (-np.inf, np.inf),
                (-np.inf, np.inf),
            ]
        ).T

        try:
            params = lsq_linear(CAXBDY_matrix, spec_intensity, bounds=bounds)["x"]

            A_Nitrogen = params[1] * 16.5
            B_Nitrogen = params[3] * 79.4
            C_Nitrogen = params[0] * 0.624332796 * C_correction

            type1b_factor = max([params[0], params[1]])
            type1a_factor = max([params[1], params[3]])

            # Restrict Final N-Fit  if B centers are greater than C centers and vice versa
            # Type 1b fits
            if B_Nitrogen / (A_Nitrogen + B_Nitrogen) < C_Nitrogen / (C_Nitrogen + A_Nitrogen):
                bounds2 = np.array(
                    [
                        (0, np.inf),
                        (0, np.inf),
                        (0, np.inf),
                        (0, type1b_factor * max_C_or_B),
                        (0, type1b_factor * max_C_or_B / 10),
                        (0, type1b_factor * max_C_or_B / 10),
                        (-np.inf, np.inf),
                        (-np.inf, np.inf),
                    ]
                ).T

            else:  # type 1a fits
                # C centers limited to less than 10% of B centers
                bounds2 = np.array(
                    [
                        (0, max_C_or_B * type1a_factor),
                        (0, np.inf),
                        (0, max_C_or_B * type1a_factor),
                        (0, np.inf),
                        (0, 0.435 * type1a_factor),
                        (0, np.inf),
                        (-np.inf, np.inf),
                        (-np.inf, np.inf),
                    ]
                ).T

            params = lsq_linear(CAXBDY_matrix, spec_intensity, bounds=bounds2)["x"]

            if plot_fit == True:
                Plot_Nitrogen(params, fit_component_df, wn_array, spec_intensity)

            # Assumes all params are positive. I think this is correct but That depends on the purpose of the X and D components
        except ValueError as e:
            print("Value Error")
            print(e)
            params = np.zeros(5)

        params = np.round(params, 8)
        C_comp = params[0]
        A_comp = params[1]
        X_comp = params[2]
        B_comp = params[3]
        D_comp = params[4]
        Y_comp = params[5]
        A_Nitrogen = np.round(params[1] * 16.5, 1)
        B_Nitrogen = np.round(params[3] * 79.4, 1)
        C_Nitrogen = np.round(params[0] * 0.624332796 * C_correction, 1)

        Total_N = np.round(A_Nitrogen + B_Nitrogen + C_Nitrogen, 1)
        AB_Nitrogen = np.round(A_Nitrogen + B_Nitrogen, 1)
        AC_Nitrogen = np.round(A_Nitrogen + C_Nitrogen, 1)
        B_percent = np.round(B_Nitrogen / Total_N * 100, 1)
        C_percent = np.round(C_Nitrogen / Total_N * 100, 1)

        # C = fit_param[0]  This needs to be multiplied by a molar absorptivity and as well as a correction for spectral resolution Liggins 2010 Thesis Warwick University
        nitrogen_dict = {
            "C_comp": C_comp,
            "A_comp": A_comp,
            "X_comp": X_comp,
            "B_comp": B_comp,
            "D_comp": D_comp,
            "Y_comp": Y_comp,
            "A_Nitrogen ppm": A_Nitrogen,
            "B_Nitrogen ppm": B_Nitrogen,
            "C_Nitrogen ppm": C_Nitrogen,
            "A+B Nitrogen ppm": AB_Nitrogen,
            "A+C Nitrogen ppm": AC_Nitrogen,
            "Total_N ppm": Total_N,
            "B_percent": B_percent,
            "C_percent": C_percent,
        }
        self.nitrogen_dict = nitrogen_dict

        self.nitrogen_plot_fit_params = {
            "fit_params": params,
            "fit_component_df": fit_component_df,
            "wn_array": wn_array,
            "spec_intensity": spec_intensity,
        }

    # @deprecated(
    #     "This method will be removed and replaced with a more general function for quantifying diamond hydrogen defects: Measure_H_defects()"
    # )
    def measure_3107_peak(self):
        """
        Measures and quantifies the hydrogen-related 3107 cm⁻¹ peak and adjacent 3085 cm⁻¹ peak.

        This method analyzes the 3060-3180 cm⁻¹ region to identify and measure hydrogen-related
        defect peaks in the diamond spectrum. The 3107 cm⁻¹ peak is the most common hydrogen-related
        feature in natural diamonds and is associated with the N3VH defect (nitrogen-vacancy-hydrogen
        complex). The method:

        1. Applies specialized baseline correction optimized for this spectral region
        2. Integrates the peak areas at 3107 cm⁻¹ and 3085 cm⁻¹
        3. Normalizes the areas by the diamond thickness factor if available

        The results are stored as attributes in the Diamond_Spectrum object, allowing for
        subsequent analysis of hydrogen content and defect correlations.

        No parameters are required as the method uses the spectral data already stored in the object.

        Attributes Set:
            If thickness normalization has been performed (typeIIA_ratio exists):
                normed_area_3107 (float): Thickness-normalized area of the 3107 cm⁻¹ peak
                normed_area_3085 (float): Thickness-normalized area of the 3085 cm⁻¹ peak
            Otherwise:
                area_3107 (float): Raw area of the 3107 cm⁻¹ peak
                area_3085 (float): Raw area of the 3085 cm⁻¹ peak

        Notes:
            - The method automatically applies appropriate baseline correction parameters
              optimized for the 3107 cm⁻¹ region
            - For accurate quantification, the spectrum should be thickness-normalized using
              normalize_diamond() before calling this method
            - The 3107 cm⁻¹ peak is often used as an indicator of natural versus synthetic
              origin in certain diamond types
            - Additional hydrogen-related peaks (e.g., 3237 cm⁻¹, 2785 cm⁻¹) are mentioned
              in comments but not currently measured by this method

        See Also:
            normalize_diamond: For thickness normalization
            measure_platelets_and_adjacent: For measuring platelet-related peaks
            measure_amber_center: For measuring amber center features
        """
        spectrum = self.select_range(3060, 3180)
        baseline = self.select_range(3060, 3180).median_filter(21).baseline_ASLS(lam=0.1, p=6e-6)
        subtracted = spectrum - baseline
        area_3107 = subtracted.integrate_peak(3103, 3110)  # (3100, 3115)
        area_3085 = subtracted.integrate_peak(3082, 3088)

        # Maybe add NVH0 (3123 cm-1), and then list all peaks above a certain prominence
        # 3237, 3107,and 2785

        if self.typeIIA_ratio != None:
            self.normed_area_3107 = area_3107 / self.typeIIA_ratio

            self.normed_area_3085 = area_3085 / self.typeIIA_ratio

        else:
            self.area_3107 = area_3107
            self.area_3085 = area_3085

    # Alt Peaks 3085,

    def measure_H_peaks(self, plot=False):
        """
        Measures and quantifies the hydrogen-related 3107 cm⁻¹ peak and adjacent 3085 cm⁻¹ peak.

        This method analyzes the 3060-3180 cm⁻¹ region to identify and measure hydrogen-related
        defect peaks in the diamond spectrum. The 3107 cm⁻¹ peak is the most common hydrogen-related
        feature in natural diamonds and is associated with the N3VH defect (nitrogen-vacancy-hydrogen
        complex). The method:

        1. Applies specialized baseline correction optimized for this spectral region
        2. Integrates the peak areas at 3107 cm⁻¹ and 3085 cm⁻¹
        3. Normalizes the areas by the diamond thickness factor if available

        The results are stored as attributes in the Diamond_Spectrum object, allowing for
        subsequent analysis of hydrogen content and defect correlations.

        No parameters are required as the method uses the spectral data already stored in the object.

        Attributes Set:
            If thickness normalization has been performed (typeIIA_ratio exists):
                normed_area_3107 (float): Thickness-normalized area of the 3107 cm⁻¹ peak
                normed_area_3085 (float): Thickness-normalized area of the 3085 cm⁻¹ peak
            Otherwise:
                area_3107 (float): Raw area of the 3107 cm⁻¹ peak
                area_3085 (float): Raw area of the 3085 cm⁻¹ peak

        Notes:
            - The method automatically applies appropriate baseline correction parameters
              optimized for the 3107 cm⁻¹ region
            - For accurate quantification, the spectrum should be thickness-normalized using
              normalize_diamond() before calling this method
            - The 3107 cm⁻¹ peak is often used as an indicator of natural versus synthetic
              origin in certain diamond types
            - Additional hydrogen-related peaks (e.g., 3237 cm⁻¹, 2785 cm⁻¹) are mentioned
              in comments but not currently measured by this method

        See Also:
            normalize_diamond: For thickness normalization
            measure_platelets_and_adjacent: For measuring platelet-related peaks
            measure_amber_center: For measuring amber center features
        """
        spectrum = self.select_range(3060, 3180)
        baseline = self.select_range(3060, 3180).median_filter(21).baseline_ASLS(lam=0.1, p=6e-6)
        subtracted = spectrum - baseline
        area_3107 = subtracted.integrate_peak(3103, 3110)  # (3100, 3115)
        area_3085 = subtracted.integrate_peak(3082, 3088)

        # Maybe add NVH0 (3123 cm-1), and then list all peaks above a certain prominence
        # 3237, 3107,and 2785

        if plot == True:
            spectrum.plot()
            baseline.plot()

        if self.typeIIA_ratio != None:
            self.normed_area_3107 = area_3107 / self.typeIIA_ratio

            self.normed_area_3085 = area_3085 / self.typeIIA_ratio

        else:
            self.area_3107 = area_3107
            self.area_3085 = area_3085

    def measure_platelets_and_adjacent(
        self,
        baseline1_param={"lam": 1000, "p": 0.001},
        find_peaks_params={},
        plot=False,
        return_peak_dict=True,
    ):
        """
        Analyzes the platelet peak and adjacent features in the 1340-1500 cm⁻¹ region of diamond spectra.

        This method performs multi-stage baseline correction and peak detection to identify and measure:
        1. The B' platelet peak (typically 1360-1375 cm⁻¹), which correlates with aggregated nitrogen
           and provides information about diamond formation and treatment history
        2. The 1405 cm⁻¹ peak, is thought to be the bending mode of N3VH the same defect with a stretching mode at 3107

        The method applies specialized baseline corrections that are optimized for isolating these
        features from the complex spectral background in this region.

        Args:
            baseline1_param (dict, optional): Parameters for the initial ASLS baseline correction.
                Defaults to {"lam": 1000, "p": 0.001}.
            find_peaks_params (dict, optional): Parameters for the peak finding algorithm.
                If empty, the method automatically sets height and prominence thresholds based on
                local noise levels. Defaults to {}.
            plot (bool, optional): If True, plots the baseline-corrected spectra and intermediate
                processing steps. Useful for method verification. Defaults to False.
            return_peak_dict (bool, optional): If True, returns the complete peak information
                dictionary from scipy.signal.find_peaks. Defaults to True.

        Returns:
            dict or None: If return_peak_dict is True, returns a dictionary of peak properties
            including positions, heights, prominences, and widths. Otherwise returns None.

        Attributes Set:
            If thickness normalization has been performed (typeIIA_ratio exists):
                normed_area_platelet (float): Thickness-normalized area of the platelet peak
                normed_height_platelet (float): Thickness-normalized height of the platelet peak
                normed_area_1405 (float): Thickness-normalized area of the 1405 cm⁻¹ peak
                normed_height_1405 (float): Thickness-normalized height of the 1405 cm⁻¹ peak
            Otherwise:
                area_platelet (float): Raw area of the platelet peak
                height_platelet (float): Raw height of the platelet peak
                area_1405 (float): Raw area of the 1405 cm⁻¹ peak
                height_1405 (float): Raw height of the 1405 cm⁻¹ peak

            platelet_peak_position (float): Wavenumber position of the platelet peak

        Notes:
            - The platelet peak is often used in diamond research to calculate a regularity factor
              when combined with B-center nitrogen content
            - For accurate results, the spectrum should be thickness-normalized using normalize_diamond()
              before calling this method
            - If no platelet peak is found, corresponding attributes will not be set
            - The 1405 cm⁻¹ peak is only measured if its height exceeds twice the local noise level

        See Also:
            Nitrogen_fit: For determining nitrogen content
            normalize_diamond: For thickness normalization
        """
        # get platelet peak parameters and identify additional peaks in the range from 1340 to 1500
        spec = self.select_range(1340, 1500)
        baseline1 = spec.median_filter(5).baseline_ASLS(**baseline1_param)
        baseline_subtracted1 = spec - baseline1
        baseline2 = baseline_subtracted1.median_filter(5).baseline_aggressive_rubberband(0.00000001)
        baseline_subtracted2 = baseline_subtracted1 - baseline2

        if not find_peaks_params.__contains__("height"):
            stdev = baseline_subtracted2.select_range(1380, 1450).Y.std()
            find_peaks_params["height"] = stdev * 2
            find_peaks_params["prominence"] = stdev

        peaks = baseline_subtracted2.find_peaks(
            **find_peaks_params, **{"width": (None, None), "rel_height": 0.5, "distance": 5}
        )  # sets relative peak height for the width to 0.5 for full width half max and distance for 5 data points between peaks

        # Define platelet peak range to search
        platelet_peak_condition = np.where((peaks["peaks_wn"] > 1355) & (peaks["peaks_wn"] < 1380))

        platelet_peak_position = peaks["peaks_wn"][platelet_peak_condition]
        platelet_peak_prominence = peaks["prominences"][platelet_peak_condition]
        platelet_peak_height = peaks["peak_heights"][platelet_peak_condition]
        platelet_peak_width = peaks["widths_wn"][platelet_peak_condition]

        if len(platelet_peak_position) != 0:
            max_platelet_range_idx = np.argmax(platelet_peak_height)
            platelet_peak_position = platelet_peak_position[max_platelet_range_idx]
            platelet_peak_height = platelet_peak_height[max_platelet_range_idx]
            platelet_peak_width = platelet_peak_width[max_platelet_range_idx]
            platelet_peak_prominence = platelet_peak_prominence[max_platelet_range_idx]

            try:
                platelet_peak_area = baseline_subtracted2.integrate_peak(
                    X_low=platelet_peak_position - platelet_peak_width / 2,
                    X_high=platelet_peak_position + platelet_peak_width / 2,
                )

                if self.typeIIA_ratio != None:
                    self.normed_area_platelet = platelet_peak_area / self.typeIIA_ratio
                    self.normed_height_platelet = platelet_peak_height / self.typeIIA_ratio

                else:
                    self.area_platelet = platelet_peak_area
                    self.height_platelet = platelet_peak_height

                self.platelet_peak_position = platelet_peak_position

            except Exception as e:
                print(e)
                print("Could not find platelet peak automatically")

        smoothed_1405_range = (
            baseline_subtracted2.select_range(1380, 1480).median_filter(3).gaussian_filter(1)
        )

        baseline_1405 = smoothed_1405_range.baseline_ASLS(lam=15, p=0.001)
        baseline_subtracted3_1405 = baseline_subtracted2.select_range(1380, 1480) - baseline_1405

        noise_1405 = baseline_subtracted3_1405.select_range(1385, 1420).Y.std()
        height_1405 = baseline_subtracted3_1405.select_range(1403, 1407).Y.max()
        area_1405 = baseline_subtracted3_1405.integrate_peak(1403, 1407)

        if height_1405 > noise_1405 * 2:
            if self.typeIIA_ratio != None:
                self.normed_area_1405 = area_1405 / self.typeIIA_ratio
                self.normed_height_1405 = height_1405 / self.typeIIA_ratio

            else:
                self.area_1405 = area_1405 / self.typeIIA_ratio
                self.height_1405 = height_1405 / self.typeIIA_ratio
        else:
            self.normed_area_1405 = np.nan
            self.normed_height_1405 = np.nan
            self.area_1405 = np.nan
            self.height_1405 = np.nan

        # Peaks to find
        # 1344, 1405
        # 1450 cm–1 radiation peak
        # Platelet between 1355 and 1375
        if plot == True:
            baseline_subtracted2.plot()
            smoothed_1405_range.plot()
            baseline_1405.plot()

        if return_peak_dict == True:
            return peaks

    def measure_amber_center(self, plot_initial=False, plot_subtracted=False):
        """
        Analyzes the amber center features in the 4000-5100 cm⁻¹ region of diamond FTIR spectra.

        This method performs sophisticated baseline correction and peak detection to identify and
        measure amber center features, which are optical defects frequently observed in natural
        brown diamonds. The method utilizes a multi-stage baseline correction approach to isolate
        the characteristic absorption peaks in this spectral region.

        The method automatically measures areas of known amber center peaks at:
        - 4065 cm⁻¹:
        - 4165 cm⁻¹:
        - 4211 cm⁻¹:
        - 4354 cm⁻¹:
        - 4495 cm⁻¹:
        - 4660 cm⁻¹:
        - 4740 cm⁻¹:
        - 4850 cm⁻¹:
        - 4950 cm⁻¹:
        Args:
            plot_initial (bool optional): Whether to plot the original spectrum and initial
                baseline. Useful for debugging. Defaults to False.
            plot_subtracted (bool, optional): Whether to plot the baseline-subtracted spectrum
                with detected peaks. Defaults to False.

        Returns:
            dict: Peak properties dictionary with positions, heights, prominences and widths
                of all detected peaks in the amber center region.

        Attributes Set:
            amber_center_peak_positions (array): Wavenumber positions of all detected peaks

            If thickness normalization has been performed (typeIIA_ratio exists):
                amber_center_peak_heights_normed (array): Thickness-normalized heights of all detected peaks
                amber_center_peak_prominences_normed (array): Thickness-normalized prominences of all peaks
                amber_XXXX_area_normed (float): Thickness-normalized area of each specific peak,
                    where XXXX represents the approximate peak position (e.g., amber_4065_area_normed)
            Else:
                amber_center_peak_heights (array): Raw heights of all detected peaks
                amber_center_peak_prominences (array): Raw prominences of all detected peaks

        Notes:
            - Amber centers are associated with plastic deformation in natural brown diamonds
            - These features can provide insights into diamond formation conditions and treatment history
            - For quantitative analysis, the spectrum should be thickness-normalized using
              normalize_diamond() prior to calling this method
            - The fine-tuned baseline correction parameters are optimized for typical amber center features

        See Also:
            normalize_diamond: For thickness normalization
            find_complex_peaks: For the underlying peak detection algorithm
        """

        peaks_output = self.find_complex_peaks(
            (3990, 6000),
            peak_range=(4000, 5100),
            noise_range=(4000, 5000),
            plot_initial=plot_initial,
            plot_subtracted=plot_subtracted,
            fine_gaussian_filter=True,
            fine_median_filter=True,
            baseline2_stretch_param=0.0000000015,
            find_peaks_params={"width": 2, "rel_height": 0.5, "distance": 5},
            fine_median_filter_len=3,
            fine_gaussian_filter_len=5,
            baseline1_param={"lam": 100000, "p": 0.0005},
            return_baseline_subtracted=True,
        )
        peaks = peaks_output["peak_dict"]
        baseline_subtracted = peaks_output["baseline_subtracted"]
        baseline_subtracted_smoothed = peaks_output["baseline_subtracted_smoothed"]

        self.amber_center_peak_positions = peaks["peaks_wn"]
        # [[4060,10],[4160,20], [4211,10], [4354, 10], [4495, 5], [4660,20 ], [4850, 15], [4950,40]]

        if self.typeIIA_ratio != None:
            self.amber_center_peak_heights_normed = peaks["peak_heights"] / self.typeIIA_ratio
            self.amber_center_peak_prominences_normed = peaks["prominences"] / self.typeIIA_ratio

            self.amber_4065_area_normed = (
                baseline_subtracted.integrate_peak(4060 - 10, 4060 + 10) / self.typeIIA_ratio
            )
            self.amber_4165_area_normed = (
                baseline_subtracted.integrate_peak(4160 - 10, 4160 + 10) / self.typeIIA_ratio
            )
            self.amber_4211_area_normed = (
                baseline_subtracted.integrate_peak(4211 - 10, 4211 + 10) / self.typeIIA_ratio
            )
            self.amber_4354_area_normed = (
                baseline_subtracted.integrate_peak(4354 - 10, 4354 + 10) / self.typeIIA_ratio
            )
            self.amber_4495_area_normed = (
                baseline_subtracted.integrate_peak(4495 - 5, 4495 + 5) / self.typeIIA_ratio
            )
            self.amber_4660_area_normed = (
                baseline_subtracted.integrate_peak(4660 - 20, 4660 + 20) / self.typeIIA_ratio
            )

            self.amber_4740_area_normed = (
                baseline_subtracted.integrate_peak(4740 - 20, 4740 + 20) / self.typeIIA_ratio
            )

            self.amber_4850_area_normed = (
                baseline_subtracted.integrate_peak(4850 - 5, 4850 + 5) / self.typeIIA_ratio
            )
            self.amber_4950_area_normed = (
                baseline_subtracted.integrate_peak(4950 - 20, 4950 + 20) / self.typeIIA_ratio
            )
        else:
            self.amber_center_peak_heights = peaks["peak_heights"]
            self.amber_center_peak_prominences = peaks["peak_prominences"]

        return peaks

    def __post_init__(self):
        super().__post_init__()  # Call the __post_init__ method for the Spectrum_object super class then add additional features.
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
        # Z = W + lam * D.dot(D.transpose())
        Z = W + lam * np.dot(D, D.T)
        z = sparse.linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def als_baseline(
    intensities,
    asymmetry_param=0.05,
    smoothness_param=5e5,
    max_iters=10,
    conv_thresh=1e-5,
    verbose=False,
):
    """Computes the asymmetric least squares baseline.
    * http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
    smoothness_param: Relative importance of smoothness of the predicted response.
    asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
                         Setting p=1 is effectively a hinge loss.
    """
    smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
    # Rename p for concision.
    p = asymmetry_param
    # Initialize weights.
    w = np.ones(intensities.shape[0])
    for i in range(max_iters):
        z = smoother.smooth(w)
        mask = intensities > z
        new_w = p * mask + (1 - p) * (~mask)
        conv = np.linalg.norm(new_w - w)
        if verbose:
            print(i + 1, conv)
        if conv < conv_thresh:
            break
        w = new_w
    else:
        print("ALS did not converge in %d iterations" % max_iters)
    return z


class WhittakerSmoother(object):
    def __init__(self, signal, smoothness_param, deriv_order=1):
        self.y = signal
        assert deriv_order > 0, "deriv_order must be an int > 0"
        # Compute the fixed derivative of identity (D).
        d = np.zeros(deriv_order * 2 + 1, dtype=int)
        d[deriv_order] = 1
        d = np.diff(d, n=deriv_order)
        n = self.y.shape[0]
        k = len(d)
        s = float(smoothness_param)

        # Here be dragons: essentially we're faking a big banded matrix D,
        # doing s * D.T.dot(D) with it, then taking the upper triangular bands.
        diag_sums = np.vstack(
            [
                np.pad(s * np.cumsum(d[-i:] * d[:i]), ((k - i, 0),), "constant")
                for i in range(1, k + 1)
            ]
        )
        upper_bands = np.tile(diag_sums[:, -1:], n)
        upper_bands[:, :k] = diag_sums
        for i, ds in enumerate(diag_sums):
            upper_bands[i, -i - 1 :] = ds[::-1][: i + 1]
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


def baseline_aggressive_rubberband(
    x, y, Y_stretch: float = 0.0001, plot_intermediate: bool = False
):
    midpoint_X = round((max(x) - min(x)) / 2)
    nonlinear_offset = Y_stretch * (x - midpoint_X) ** 2
    y_alt = y + nonlinear_offset
    baseline = rubberband(x, y_alt)

    return baseline - nonlinear_offset


def select_baseline_func(baseline_algorithm="Whittaker"):
    """Fits a diamond spectrum to an ideal spectrum accounting for saturated peaks. Spectrum needs to be interpolated to the same spacing as the typeIIA diamond spectrum.

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


def C_center_wn_spacing_correction(wn_spacing: float) -> float:
    return (
        9.7043 * wn_spacing + 25.304
    )  # Function derived from Linear fit to values determined in Liggins et al. 2010 Phd Thesis


# %%


def Plot_Nitrogen(params, fit_component_df, wn_array, spec_intensity):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(wn_array, spec_intensity, label="Spectrum")
    fit_comp = fit_component_df * params  # (CAXBDY_select * params)
    model_spectrum = fit_comp.sum(axis=1, numeric_only=True)
    model_spectrum.plot(label="Fit Spectrum")
    fit_comp.iloc[:, 0:6].plot(ax=ax)
    fit_comp.iloc[:, 6:].sum(axis=1, numeric_only=True).plot(label="Linear_Offset")
    ax.legend()
    ax.set_xlabel("Wavenumber (1/cm)")
    ax.set_ylabel("Absorptivity (1/cm)")


def edit_plot(
    spectrum_name,
    output_path=None,
    subfolder: Union[None, str] = None,
    set_title=True,
    save_file=True,
    dpi=400,
    dimensions=(12, 8),
):
    ax = plt.gca()
    fig = plt.gcf()
    fig.set_dpi(dpi)
    fig.set_size_inches(*dimensions)

    name = spectrum_name.split(".")[0]
    if output_path == None:
        if subfolder != None:
            output_path = Path(f"Results/Figures/{subfolder}").mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(f"Results/Figures").mkdir(parents=True, exist_ok=True)

    spectrum_name.split["."][0]
    if set_title:
        ax.set_title(spectrum_name)

    if save_file:
        plt.savefig(f"{name}.png")
