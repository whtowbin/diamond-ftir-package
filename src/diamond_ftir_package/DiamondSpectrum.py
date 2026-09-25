# %%
import logging
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybaselines as pybl
import scipy.sparse.linalg  # noqa: F401  (makes sparse.linalg available)
from scipy import optimize, sparse
from scipy.linalg import solveh_banded
from scipy.optimize import lsq_linear
from scipy.signal import medfilt

from .CAXBDY import CAXBDY_json
from .params import (
    AmberParams,
    DiamondFitParams,
    HydrogenParams,
    NitrogenParams,
    PlateletParams,
)
from .Spectrum_obj import Spectrum, rubberband
from .typeIIA import typeIIA_json

logger = logging.getLogger(__name__)

# from warnings import deprecated

# %%
# TODO Add Kwargs to methods to reduce hardcoding of parameters
# TODO Add additional Hydrogen defect peaks such as 3237 cm-1, 2785 cm-1
# TODO List stats for initial spectral resolution and other paramters
# TODO Add function to mask tops of nitrogen peaks to avoid saturation
# TODO add output flags for potentially bad fits in diamond and Nitrogen fitting
# TODO add plot of diamond fit, platlets, and Hyrdrogen Peaks and as an optional flag, 
# TODO Improve plot layout and add customization such as title, axis labels, etc.

# TODO write entry point scripts to run single or batches of spectra with various output options. dicts, csv, json, etc.
# TODO write import script to load various spectra formats and convert to the Spectrum object

#%%
# Type Spectra are only imported once outside of the class so that they dont fill up the memory in long loops, by creating multiple identical objects
typeIIA = pd.DataFrame(typeIIA_json)
# typeIIA = typeIIA.set_index(keys=["wn"])

typeIIA_Spectrum = Spectrum(
    X=typeIIA["wn"].to_numpy(),
    Y=typeIIA["absorbance"].to_numpy(),
    X_Unit="Wavenumber",
    Y_Unit="Absorbance",
)


CAXBDY = pd.DataFrame(CAXBDY_json)
CAXBDY = CAXBDY.set_index(keys=["wn"])


_DEFAULT_CAXBDY = CAXBDY
_NITROGEN_LABELS = ["C", "A", "X", "B", "D", "Y", "offset", "linear"]


def _build_nitrogen_design(CAXBDY, wn_low, wn_high):
    """Design matrix (components + offset + linear) for the nitrogen fit window."""
    CAXBDY_select = CAXBDY.loc[wn_low:wn_high]
    wn_array = CAXBDY_select.index.to_numpy()
    offset = np.ones_like(wn_array)
    linear = np.arange(len(offset)) - (wn_high - wn_low)
    matrix = np.hstack((CAXBDY_select.to_numpy(), np.vstack((offset, linear)).T))
    fit_component_df = pd.DataFrame(matrix, columns=_NITROGEN_LABELS, index=wn_array)
    return matrix, wn_array, fit_component_df


@lru_cache(maxsize=8)
def _nitrogen_design(wn_low, wn_high):
    """Cached design matrix for the bundled CAXBDY reference (rebuilt only if the window changes)."""
    return _build_nitrogen_design(_DEFAULT_CAXBDY, wn_low, wn_high)


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
            logger.info("Primary diamond peaks are saturated; using 2400-2670 cm-1")

        elif (
            (main_diamond_sat == True)
            & (secondary_diamond_sat == True)
            & (third_diamond_sat == False)
        ):
            fit_mask_idx = (self.X > 3130) & (self.X < 3500)
            logger.info("Secondary diamond peaks are saturated; using 3130-3500 cm-1")

        elif (
            (main_diamond_sat == True)
            & (secondary_diamond_sat == True)
            & (third_diamond_sat == True)
        ):
            raise ValueError(
                "All diamond peaks  are saturated and thickness correction cannot be determined"
            )

        # Adds a bunch of other non saturated regions to the baseline that are useful for fitting baselines to diamonds
        fit_mask_idx = (
            fit_mask_idx | ((self.X > 3130) & (self.X < 3500))
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
        params: DiamondFitParams | None = None,
        start: tuple[float, float] | None = None,
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
            params (DiamondFitParams, optional): Full baseline settings (method, search, lam/p,
                bounds). When given it overrides the three keyword settings above.
            start (tuple, optional): (lam, p) to start the search from instead of
                ``params.lam, params.p``. Maps use this to start each pixel near the
                sample's typical values.

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
        params = params or DiamondFitParams(
            saturation_cutoff=saturation_cutoff,
            stdev_cut_off=stdev_cut_off,
            baseline_algorithm=baseline_algorithm,
        )
        fit_mask_idx = self.test_diamond_saturation(params.saturation_cutoff, params.stdev_cut_off)
        baseline_func = select_baseline_func(params.baseline_algorithm)
        ideal_diamond_Y = self.interpolated_typeIIA_Spectrum.Y
        X = self.X

        if params.baseline_method == "joint":
            lam, p = start if start is not None else (params.joint_lam, params.joint_p)
            weights = joint_fit_weights(X, fit_mask_idx)
            baseline_out, fit_ratio = joint_asls_diamond(
                self.Y, ideal_diamond_Y, lam=lam, p=p, weights=weights
            )
            self.baseline_params = (float(lam), float(p))
            baseline_subtracted = self.Y - baseline_out
        else:
            pre_baseline, Y_subtracted = diamond_pre_baseline(X, self.Y, baseline_func, params)
            objective = DiamondBaselineObjective(
                X, Y_subtracted, ideal_diamond_Y, fit_mask_idx, baseline_func, params
            )
            lam, p = search_baseline_params(objective, params, start=start)
            self.baseline_params = (lam, p)

            baseline_opt = baseline_func(Y_subtracted, lam=lam, p=p)
            baseline_out = baseline_opt + pre_baseline

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

    # TODO Rename this funtion to be a clear that this both fits the baseline and the typeIIA ratio
    def fit_baseline(
        self,
        saturation_cutoff=2.5,
        stdev_cut_off=0.5,
        baseline_algorithm="Whittaker",
        params: DiamondFitParams | None = None,
        start: tuple[float, float] | None = None,
    ):
        """Fit the diamond baseline and the type IIa thickness ratio in place.

        Falls back to the slower ALS algorithm if the sparse solve fails. Any other error
        (for example all diamond peaks saturated) is raised to the caller.
        """
        params = params or DiamondFitParams(
            saturation_cutoff=saturation_cutoff,
            stdev_cut_off=stdev_cut_off,
            baseline_algorithm=baseline_algorithm,
        )
        try:
            self.fit_diamond_peaks(inplace=True, params=params, start=start)

        except (np.linalg.LinAlgError, RuntimeError) as e:
            if params.baseline_algorithm == "ALS":
                raise
            logger.warning("%r: fitting baseline with the alternate ALS function", e)
            self.fit_diamond_peaks(
                inplace=True, params=replace(params, baseline_algorithm="ALS"), start=start
            )

    def normalize_diamond(self, inplace=True):
        """returns spectrum normalized to 1 cm thickness based on unsaturated Diamond peak heights"""
        if getattr(self, "baseline", None) is None or self.typeIIA_ratio is None:
            raise ValueError(
                "Diamond Spectrum object must have a baseline and typeIIA_ratio fit prior to "
                "using this method, try using the fit_baseline() method before calling this"
            )
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
        self.normalized_spectrum = normalized_spectrum

    def instrument_resolution(self, setting: float = 0.0) -> tuple[float, str]:
        """Instrument resolution (cm-1) and where it came from.

        Order: an explicit setting; a 'resolution' entry in the file metadata; the file
        name (e.g. '4wnRes'); otherwise twice the ORIGINAL point spacing (before
        interpolation; Nyquist), flagged as assumed. Resolution is the instrument setting,
        about twice the sampling interval; OMNIC may also zero-fill, making the spacing
        smaller still. Set it explicitly when it matters (C-centres).
        """
        if setting and setting > 0:
            return float(setting), "setting"
        meta = self.metadata or {}
        for key in ("resolution", "Resolution"):
            if meta.get(key):
                return float(meta[key]), "file metadata"
        for key in ("Filename", "filename", "Title", "source_file"):
            found = resolution_from_name(meta.get(key, "")) if meta.get(key) else None
            if found:
                return found, "file name"
        # Nyquist: resolution is about twice the sampling interval. Zero-filled data have a
        # smaller spacing still, so this can underestimate; it is flagged.
        spacing = float(np.median(np.abs(np.diff(self.initial_X))))
        return 2.0 * spacing, "2 x point spacing (assumed)"

    def _store_measurement(self, name: str, value) -> None:
        """Set ``normed_<name>`` if thickness-normalised, otherwise ``<name>``."""
        if self.typeIIA_ratio is not None:
            setattr(self, f"normed_{name}", value / self.typeIIA_ratio)
        else:
            setattr(self, name, value)

    def Nitrogen_fit(self, CAXBDY=CAXBDY, plot_fit=False, max_C_or_B=None, params=None):
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
                B in Type Ib). Overrides ``params.max_C_or_B`` when given. Defaults to 0.01.
            params (NitrogenParams, optional): Fit window, conversion factors and limits.
                Defaults to ``NitrogenParams()``, the historical hardcoded values.

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
        params = params or NitrogenParams()
        spec = self.normalized_spectrum
        # Clip the window to where both the data and the reference components exist, so a
        # window wider than the data narrows instead of failing (it used to return zeros).
        wn_low = int(max(params.wn_low, np.ceil(spec.X.min()), CAXBDY.index.min()))
        wn_high = int(min(params.wn_high, np.floor(spec.X.max()) - 1, CAXBDY.index.max()))
        if (wn_low, wn_high) != (params.wn_low, params.wn_high):
            logger.info("Nitrogen window clipped to %s-%s cm-1 by the data", wn_low, wn_high)
        self.nitrogen_window = (wn_low, wn_high)
        if max_C_or_B is None:
            max_C_or_B = params.max_C_or_B
        if CAXBDY is _DEFAULT_CAXBDY:
            CAXBDY_matrix, wn_array, fit_component_df = _nitrogen_design(wn_low, wn_high)
        else:
            CAXBDY_matrix, wn_array, fit_component_df = _build_nitrogen_design(CAXBDY, wn_low, wn_high)

        spec_intensity = spec.select_range(wn_low, wn_high + 1).Y

        resolution, source = self.instrument_resolution(params.resolution_cm)
        self.nitrogen_resolution = (resolution, source)
        C_correction = C_center_resolution_correction(resolution)

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
            fit = lsq_linear(CAXBDY_matrix, spec_intensity, bounds=bounds)["x"]

            A_Nitrogen = fit[1] * params.a_ppm_per_cm
            B_Nitrogen = fit[3] * params.b_ppm_per_cm
            C_Nitrogen = fit[0] * params.c_ppm_per_cm * C_correction

            type1b_factor = max([fit[0], fit[1]])
            type1a_factor = max([fit[1], fit[3]])

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
                        (0, params.d_limit * type1a_factor),
                        (0, np.inf),
                        (-np.inf, np.inf),
                        (-np.inf, np.inf),
                    ]
                ).T

            fit = lsq_linear(CAXBDY_matrix, spec_intensity, bounds=bounds2)["x"]

            if plot_fit == True:
                Plot_Nitrogen(fit, fit_component_df, wn_array, spec_intensity)

            # Assumes all params are positive. I think this is correct but That depends on the purpose of the X and D components
        except ValueError as e:
            logger.warning("Nitrogen fit failed (%s); reporting zeros", e)
            fit = np.zeros(8)

        fit = np.round(fit, 8)
        C_comp, A_comp, X_comp, B_comp, D_comp, Y_comp = fit[:6]
        A_Nitrogen = np.round(fit[1] * params.a_ppm_per_cm, 1)
        B_Nitrogen = np.round(fit[3] * params.b_ppm_per_cm, 1)
        C_Nitrogen = np.round(fit[0] * params.c_ppm_per_cm * C_correction, 1)

        Total_N = np.round(A_Nitrogen + B_Nitrogen + C_Nitrogen, 1)
        AB_Nitrogen = np.round(A_Nitrogen + B_Nitrogen, 1)
        AC_Nitrogen = np.round(A_Nitrogen + C_Nitrogen, 1)
        if Total_N > 0:
            B_percent = np.round(B_Nitrogen / Total_N * 100, 1)
            C_percent = np.round(C_Nitrogen / Total_N * 100, 1)
        else:
            B_percent = C_percent = np.nan

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
            "fit_params": fit,
            "fit_component_df": fit_component_df,
            "wn_array": wn_array,
            "spec_intensity": spec_intensity,
        }

    # @deprecated(
    #     "This method will be removed and replaced with a more general function for quantifying diamond hydrogen defects: Measure_H_defects()"
    # )
    # add other H peaks such as 3237 cm-1, 2785 cm-1
    def measure_3107_peak(self, params=None, plot=False):
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

        Args:
            params (HydrogenParams, optional): Windows, baseline settings and integration limits.
            plot (bool, optional): Plot the region and its baseline.

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
        params = params or HydrogenParams()
        lo, hi = params.window
        spectrum = self.select_range(lo, hi)
        baseline = spectrum.median_filter(params.median_filter).baseline_ASLS(
            lam=params.baseline_lam, p=params.baseline_p
        )
        subtracted = spectrum - baseline

        # Maybe add NVH0 (3123 cm-1), and then list all peaks above a certain prominence
        # 3237, 3107,and 2785
        self._store_measurement("area_3107", subtracted.integrate_peak(*params.peak_3107))
        self._store_measurement("area_3085", subtracted.integrate_peak(*params.peak_3085))

        if plot:
            spectrum.plot()
            baseline.plot()

    def measure_H_peaks(self, plot=False, params=None):
        """Alias of :meth:`measure_3107_peak` kept for backwards compatibility."""
        return self.measure_3107_peak(params=params, plot=plot)

    def measure_platelets_and_adjacent(
        self,
        baseline1_param=None,
        find_peaks_params=None,
        plot=False,
        return_peak_dict=True,
        params=None,
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
                Defaults to ``{"lam": params.baseline_lam, "p": params.baseline_p}``.
            find_peaks_params (dict, optional): Parameters for the peak finding algorithm.
                If empty, the method automatically sets height and prominence thresholds based on
                local noise levels. Defaults to None. The dict is copied, never mutated.
            params (PlateletParams, optional): Regions and windows. Defaults to ``PlateletParams()``.
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
        params = params or PlateletParams()
        if baseline1_param is None:
            baseline1_param = {"lam": params.baseline_lam, "p": params.baseline_p}
        find_peaks_params = dict(find_peaks_params or {})  # copy: never leak thresholds between spectra
        spec = self.select_range(*params.region)
        baseline1 = spec.median_filter(5).baseline_ASLS(**baseline1_param)
        baseline_subtracted1 = spec - baseline1
        baseline2 = baseline_subtracted1.median_filter(5).baseline_aggressive_rubberband(0.00000001)
        baseline_subtracted2 = baseline_subtracted1 - baseline2

        if "height" not in find_peaks_params:
            stdev = baseline_subtracted2.select_range(*params.noise_window).Y.std()
            find_peaks_params["height"] = stdev * 2
            find_peaks_params["prominence"] = stdev

        peaks = baseline_subtracted2.find_peaks(
            **find_peaks_params, width=(None, None), rel_height=0.5, distance=5
        )  # sets relative peak height for the width to 0.5 for full width half max and distance for 5 data points between peaks

        # Define platelet peak range to search
        search_low, search_high = params.search_window
        platelet_peak_condition = np.where(
            (peaks["peaks_wn"] > search_low) & (peaks["peaks_wn"] < search_high)
        )

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

                self._store_measurement("area_platelet", platelet_peak_area)
                self._store_measurement("height_platelet", platelet_peak_height)

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
        height_1405 = baseline_subtracted3_1405.select_range(*params.peak_1405).Y.max()
        area_1405 = baseline_subtracted3_1405.integrate_peak(*params.peak_1405)

        if height_1405 > noise_1405 * 2:
            self._store_measurement("area_1405", area_1405)
            self._store_measurement("height_1405", height_1405)
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

    def measure_amber_center(self, plot_initial=False, plot_subtracted=False, params=None):
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
            params (AmberParams, optional): Bands to integrate. Defaults to ``AmberParams()``.

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

        params = params or AmberParams()
        ratio = self.typeIIA_ratio
        suffix = "_normed" if ratio is not None else ""
        scale = ratio if ratio is not None else 1.0

        setattr(self, f"amber_center_peak_heights{suffix}", peaks["peak_heights"] / scale)
        setattr(self, f"amber_center_peak_prominences{suffix}", peaks["prominences"] / scale)
        for label, centre, half_width in params.bands:
            area = baseline_subtracted.integrate_peak(centre - half_width, centre + half_width)
            setattr(self, f"amber_{label}_area{suffix}", area / scale)

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


# [CITATION NEEDED: Eilers & Boelens 2005 ASLS; check licence of the StackExchange implementation]
def baseline_als(y, lam, p, niter=10):
    """
    Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005 implemented on stackoverflow by user: sparrowcide
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    """
    L = len(y)
    # Second-difference operator, built sparse (equal to np.diff(np.eye(L), 2) without the
    # dense L x L matrix, which took seconds and ~250 MB for a 5,000-point spectrum).
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2), format="csc")
    penalty = lam * (D @ D.T)
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L))
        Z = (W + penalty).tocsc()
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


class WhittakerSmoother:
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


def baseline_aggressive_rubberband(
    x, y, Y_stretch: float = 0.0001, plot_intermediate: bool = False
):
    midpoint_X = round((max(x) - min(x)) / 2)
    nonlinear_offset = Y_stretch * (x - midpoint_X) ** 2
    y_alt = y + nonlinear_offset
    baseline = rubberband(x, y_alt)

    return baseline - nonlinear_offset


def joint_fit_weights(x, fit_mask):
    """Weight 0 for saturated parts of the two-phonon band (1800-2700 cm-1 outside the
    unsaturated fit mask), 1 elsewhere. Saturated points carry no thickness information."""
    untrusted = (x > 1800) & (x < 2700) & ~fit_mask
    return (~untrusted).astype(float)


def joint_asls_diamond(y, reference, lam, p, weights, max_iter=50):
    """Fit y = smooth baseline + t * reference in one penalised least-squares problem.

    Minimises  sum_i w_i (y_i - b_i - t r_i)^2 + lam * |D2 b|^2  over the baseline b (any
    smooth curve, as in ASLS) and the scalar thickness t, with ASLS's asymmetric reweighting
    (w = p above the model, 1 - p below) on top of the given weights.

    Unlike fitting ASLS first and scaling the reference afterwards, the baseline cannot
    absorb part of the diamond band, because the band is part of the model. On real spectra
    (injection tests, docs/methods_validation.md) this removed the 5-30% thickness
    under-estimate of the other baselines. Returns (baseline, t).
    """
    from scipy.sparse.linalg import splu

    y = np.asarray(y, dtype=float)
    ref = np.asarray(reference, dtype=float)
    n = y.size
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    penalty = lam * (D.T @ D)
    w0 = np.asarray(weights, dtype=float)
    w = w0.copy()
    t, b = 0.0, np.zeros(n)
    for _ in range(max_iter):
        lu = splu((sparse.diags(w) + penalty).tocsc())
        b_y = lu.solve(w * y)  # (W + P)^-1 W y
        b_r = lu.solve(w * ref)  # (W + P)^-1 W r
        t = float(np.sum(w * ref * (y - b_y)) / np.sum(w * ref * (ref - b_r)))
        b = b_y - t * b_r
        resid = y - b - t * ref
        w_new = w0 * np.where(resid > 0, p, 1 - p)
        if np.array_equal(w_new > 0.5, w > 0.5):
            break
        w = w_new
    return b, t


def diamond_pre_baseline(x, y, baseline_func, params):
    """Remove the broad background before the main baseline fit.

    Returns ``(pre_baseline, y_subtracted)``. For ``baseline_method="multistage"`` this is a
    median filter, a mild ASLS and a stretched rubber band (the original method). For
    ``"single"`` nothing is removed, so the main ASLS fit is the only baseline (the method
    of the original mapping script).
    """
    if params.baseline_method == "single":
        return np.zeros_like(y), np.asarray(y, dtype=float)
    y_filter = medfilt(y, params.pre_median_filter)
    y_asls = baseline_func(y_filter, lam=params.pre_lam, p=params.pre_p)
    y_subtracted = y_filter - y_asls
    if not params.pre_use_rubberband:
        return y_asls, y_subtracted
    y_rubber = baseline_aggressive_rubberband(x, y_subtracted, Y_stretch=params.pre_rubber_stretch)
    return y_asls + y_rubber, y_subtracted - y_rubber


class DiamondBaselineObjective:
    """Misfit of a candidate (lam, p) baseline: diamond peaks vs the type IIa shape, plus flatness.

    Kept as a class (not a closure) so maps can reuse it and so it pickles for parallel work.
    """

    def __init__(self, x, y_subtracted, reference, mask, baseline_func, params):
        self.y = y_subtracted
        self.baseline_func = baseline_func
        self.mask = mask
        self.reference_masked = reference[mask]
        low, high = params.flat_range
        self.flat_idx = (x > low) & (x < high)
        self.flat_weight = params.flat_weight
        self.evaluations = 0

    def __call__(self, lam: float, p: float) -> float:
        self.evaluations += 1
        try:
            baseline = self.baseline_func(self.y, lam=lam, p=p)
        except (np.linalg.LinAlgError, ValueError):
            return np.inf  # ill-conditioned (lam, p): steer the search away instead of failing
        subtracted = self.y - baseline
        masked = subtracted[self.mask]
        ratio = np.mean(masked / self.reference_masked)
        flat = (subtracted[self.flat_idx] ** 2).sum() * self.flat_weight
        shape = ((masked / ratio - self.reference_masked) ** 2).sum()
        return flat + shape

    def log10(self, log_params) -> float:
        """Objective in log10(lam), log10(p): the scale the search actually works in."""
        return self(10.0 ** log_params[0], 10.0 ** log_params[1])


def search_baseline_params(objective, params, start=None) -> tuple[float, float]:
    """Choose (lam, p) for the main baseline according to ``params.baseline_search``.

    - ``"fixed"``: use ``start`` or ``(params.lam, params.p)`` as they are. This is what the
      original code effectively did (its optimiser never moved; see TODO.md).
    - ``"optimize"``: bounded Nelder-Mead in log10 space, starting from ``start`` or
      ``(params.lam, params.p)``. Bounds are ``params.log10_lam_bounds``/``log10_p_bounds``,
      narrowed to +/- ``params.search_width`` decades around ``start`` when one is given.
    - ``"grid"``: a ``grid_points`` x ``grid_points`` log grid over the same bounds, then
      Nelder-Mead within one grid step of the best point. Slower but robust to plateaus.
    """
    lam0, p0 = start if start is not None else (params.lam, params.p)
    if params.baseline_search == "fixed":
        return float(lam0), float(p0)
    if params.baseline_search not in ("optimize", "grid"):
        raise ValueError(f"Unknown baseline_search {params.baseline_search!r}")

    x0 = np.log10([lam0, p0])
    bounds = [params.log10_lam_bounds, params.log10_p_bounds]
    if start is not None and params.search_width is not None:
        w = params.search_width
        bounds = [
            (max(bounds[0][0], x0[0] - w), min(bounds[0][1], x0[0] + w)),
            (max(bounds[1][0], x0[1] - w), min(bounds[1][1], x0[1] + w)),
        ]

    if params.baseline_search == "grid":
        # The objective is stepwise in (lam, p) because ASLS weights are 0/1-like, so a local
        # optimiser can stall on a plateau. Scan a coarse log grid first, then refine locally
        # within one grid step of the best point.
        n = params.grid_points
        lam_axis = np.linspace(*bounds[0], n)
        p_axis = np.linspace(*bounds[1], n)
        best = min(
            ((objective.log10((a, b)), a, b) for a in lam_axis for b in p_axis),
            key=lambda t: t[0],
        )
        x0 = np.array(best[1:])
        step = [(bounds[0][1] - bounds[0][0]) / (n - 1), (bounds[1][1] - bounds[1][0]) / (n - 1)]
        bounds = [
            (max(bounds[0][0], x0[0] - step[0]), min(bounds[0][1], x0[0] + step[0])),
            (max(bounds[1][0], x0[1] - step[1]), min(bounds[1][1], x0[1] + step[1])),
        ]

    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    result = optimize.minimize(
        objective.log10,
        x0=x0,
        method="Nelder-Mead",
        bounds=bounds,
        options={
            "xatol": params.search_tolerance,
            "fatol": 0.0,
            "maxfev": params.search_max_evals,
            "initial_simplex": _initial_simplex(x0, bounds),
        },
    )
    lam, p = 10.0 ** result.x
    return float(lam), float(p)


def _initial_simplex(x0, bounds, fraction=0.25):
    """A simplex spanning a quarter of each bound, so the first steps cross ASLS plateaus.

    scipy's default simplex moves each coordinate by only 5%, which on a stepwise objective
    often lands on the same plateau and ends the search at once.
    """
    x0 = np.asarray(x0, dtype=float)
    simplex = [x0.copy()]
    for k, (low, high) in enumerate(bounds):
        vertex = x0.copy()
        delta = fraction * (high - low)
        vertex[k] = x0[k] + delta if x0[k] + delta <= high else x0[k] - delta
        simplex.append(vertex)
    return np.array(simplex)


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


# C-centre calibration factor versus instrument resolution (cm-1), as tabulated in DiaMap
# (Howell et al. 2012) "from Liggins 2010 PhD thesis". [CITATION NEEDED: Liggins 2010 thesis]
C_CENTRE_RESOLUTION_TABLE = ((0.5, 30.0), (1.0, 37.0), (2.0, 42.0), (4.0, 65.0))


def C_center_resolution_correction(resolution: float) -> float:
    """C-centre factor for the instrument's spectral resolution (cm-1), NOT the point spacing
    and NOT the 1 cm-1 analysis grid: interpolation does not change the true resolution.

    Piecewise-linear through Liggins' tabulated values; outside 0.5-4 cm-1 the end segments
    are extended (flagged in docs as an extrapolation).
    """
    res = np.array([r for r, _ in C_CENTRE_RESOLUTION_TABLE])
    fac = np.array([f for _, f in C_CENTRE_RESOLUTION_TABLE])
    r = float(resolution)
    if r < res[0]:
        return float(fac[0] + (r - res[0]) * (fac[1] - fac[0]) / (res[1] - res[0]))
    if r > res[-1]:
        return float(fac[-1] + (r - res[-1]) * (fac[-1] - fac[-2]) / (res[-1] - res[-2]))
    return float(np.interp(r, res, fac))


# Earlier name; the argument must be the instrument resolution.
C_center_wn_spacing_correction = C_center_resolution_correction


def resolution_from_name(name: str) -> float | None:
    """Resolution written in an OMNIC-style file name, e.g. '..._4wnRes_...' or '..._2res_...'."""
    import re

    m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:wn|cm-?1)?\s*res(?:olution)?(?![a-z])", str(name), re.I)
    return float(m.group(1)) if m else None


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
    subfolder: None | str = None,
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
    if output_path is None:
        output_path = Path("Results/Figures") / (subfolder or "")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if set_title:
        ax.set_title(spectrum_name)

    if save_file:
        plt.savefig(output_path / f"{name}.png")
