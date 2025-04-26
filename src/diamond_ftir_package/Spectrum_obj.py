# %%
import numpy as np

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Union
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
from copy import deepcopy
from scipy.signal import medfilt, find_peaks
from scipy.integrate import simpson as simpson_integrate
from scipy.spatial import ConvexHull

import pybaselines as pybl
import scipy.sparse as sparse

from scipy.ndimage import gaussian_filter1d


@dataclass(order=True)
class Spectrum:
    """
    A comprehensive data class for spectroscopic data processing and analysis.

    The Spectrum class provides methods for common spectroscopic data manipulation, including
    baseline correction, peak detection, interpolation, integration, filtering, and visualization.
    Most methods support method chaining and can operate either in-place or by returning new
    Spectrum objects.

    Attributes:
        X (np.ndarray): Independent variable data array (typically wavenumber or wavelength).
        Y (np.ndarray): Dependent variable data array (typically absorbance or intensity).
        X_Unit (str): Unit for X values (e.g., "cm⁻¹", "nm").
        Y_Unit (str): Unit for Y values (e.g., "Absorbance", "Intensity").
        metadata (Dict): Optional dictionary for storing spectrum metadata.
        kwargs (Dict): Optional dictionary for additional parameters.
        baseline (np.ndarray): Stores baseline data when baseline correction is applied.
        initial_X (np.ndarray): Preserves original X data through transformations.
        initial_Y (np.ndarray): Preserves original Y data through transformations.
        initial_spectrum (Spectrum): Preserves the complete initial state.

    Notes:
        - The class automatically ensures X data is in ascending order during initialization
        - Most processing methods accept an 'inplace' parameter which determines whether
          to modify the current object or return a new one
        - Multiple baseline correction algorithms are available for different spectral types
        - Peak finding methods range from simple detection to complex multi-stage algorithms
        - Arithmetic operations are supported for combining and manipulating spectra

    Example:
        ```python
        # Create a spectrum object
        spec = Spectrum(X=wavenumbers, Y=absorbance, X_Unit="cm⁻¹", Y_Unit="Absorbance")

        # Process data with method chaining
        processed = (spec
                    .interpolate(400, 4000, step=2)
                    .median_filter(window=5)
                    .baseline_ALS(lam=1e7, p=0.001, inplace=True))

        # Find and analyze peaks
        peaks = processed.find_complex_peaks(
            noise_range=(3800, 4000),
            plot_subtracted=True
        )

        # Calculate area of a specific peak
        area = processed.select_range(1600, 1700).integrate_peak()
        ```

    Raises:
        ValueError: If invalid parameters are provided to processing methods.
    """

    X: np.ndarray = None  # Sequential array; typically wavenumber or wavelength
    Y: np.ndarray = None  # Typically intensity of absorbance
    X_Unit: str = None
    Y_Unit: str = None
    metadata: Dict = None
    kwargs: Dict = None  # defines kwargs, should probably have a metadata dict
    baseline = None

    def __post_init__(self):
        """method automatially called after initialization"""
        if self.X[0] > self.X[-1]:  # Ensures Spectrum X axis is in ascending order.
            self.X = np.flip(self.X)
            self.Y = np.flip(self.Y)
        # Preserves the initial unmodified spectrum
        self.initial_X = deepcopy(self.X)
        self.initial_Y = deepcopy(self.Y)
        self.initial_spectrum = deepcopy(self)

    def select_range(
        self, X_low: Union[int, float], X_high: Union[int, float], inplace: bool = False
    ):
        """Selects a subset of the data based on the selected range for x axis points.
          Finds closest datapoints to given ranges.

        Args:
            X_low (int or float): low
            X_high (int or float): _description_
        """
        index_low = np.argmin(np.abs(self.X - X_low))
        index_high = np.argmin(np.abs(self.X - X_high))
        X_slice = slice(index_low, index_high)
        if (
            self.X[0] > self.X[-1]
        ):  # Check if array is asscending or descending # TODO decide if this is redundant with above check
            X_slice = slice(index_high, index_low)
        if inplace is False:
            other = deepcopy(self)
            other.X = deepcopy(other.X[X_slice])
            other.Y = deepcopy(other.Y[X_slice])
            return other
        else:
            self.X = deepcopy(self.X[X_slice])
            self.Y = deepcopy(self.Y[X_slice])
            return self

    def integrate_peak(
        self,
        X_low: Union[int, float, None] = None,
        X_high: Union[int, float, None] = None,
    ):
        """
        Calculates the integrated area under the curve within the specified x-axis range.

        Uses Simpson's rule to numerically integrate the spectrum between the specified x-axis bounds.
        This method is useful for quantitative analysis, calculating peak areas, and comparing
        relative intensities of spectral features.

        Args:
            X_low (Union[int, float, None], optional): The lower bound of the x-axis range to integrate.
                If None, uses the minimum x-value in the spectrum. Defaults to None.
            X_high (Union[int, float, None], optional): The upper bound of the x-axis range to integrate.
                If None, uses the maximum x-value in the spectrum. Defaults to None.

        Returns:
            float: The integrated area under the curve in the specified range. Units are
            the product of X and Y units (e.g., if X is in cm⁻¹ and Y is absorbance,
            the result will be in absorbance·cm⁻¹).

        Example:
            ```python
            # Calculate the area of a specific peak
            area = spectrum.integrate_peak(1600, 1700)
            print(f"The peak area is {area:.2f}")

            # Calculate the total area of the spectrum
            total_area = spectrum.integrate_peak()
            ```

        Notes:
            The integration uses scipy's implementation of Simpson's rule for numerical
            integration, which provides good accuracy for most spectroscopic data.
        """
        # I should figure out how to include baselines in data.
        if X_low == None:  # Assume Whole Spectrum is to be intergated
            X_low = min(self.X)
        if X_high == None:  # Assume Whole Spectrum is to be intergated
            X_high = max(self.X)

        spec = self.select_range(X_low, X_high)
        # integrated = quad_integrate(spec)   Doesnt work like the old version
        integrated = simpson_integrate(spec.Y, x=spec.X)
        return integrated

    def peak_height(
        self,
        X_low: Union[int, float, None] = None,
        X_high: Union[int, float, None] = None,
    ):
        """
        Calculates the average height (intensity) of the spectrum within the specified x-axis range.

        This method is useful for quickly assessing peak heights or average signal levels
        in a region of interest without performing baseline correction or peak finding.

        Args:
            X_low (Union[int, float, None], optional): The lower bound of the x-axis range.
                If None, uses the minimum x-value in the spectrum. Defaults to None.
            X_high (Union[int, float, None], optional): The upper bound of the x-axis range.
                If None, uses the maximum x-value in the spectrum. Defaults to None.

        Returns:
            float: The average y-value (height/intensity) in the specified range.
                Units match the Y_Unit of the spectrum.

        Example:
            ```python
            # Calculate the average height of a specific peak
            height = spectrum.peak_height(1600, 1700)
            print(f"The average peak height is {height:.3f}")

            # Calculate the average intensity of the entire spectrum
            avg_intensity = spectrum.peak_height()
            ```

        Notes:
            This is a simple averaging method and does not perform baseline correction.
            For more accurate peak height measurements, consider subtracting a baseline first.
        """
        # I should figure out how to include baselines in data.
        if X_low == None:  # Assume Whole Spectrum is to be intergated
            X_low = min(self.X)
        if X_high == None:  # Assume Whole Spectrum is to be intergated
            X_high = max(self.X)

        spec = self.select_range(X_low, X_high)
        # integrated = quad_integrate(spec)   Doesnt work like the old version
        height = np.mean(spec.Y)
        return height

    def plot(self, ax=None, plot_baseline=False, plot_initial=False, *args, **kwargs):
        """plots a spectrum as on a matplotlib plot.
        Args:
            ax (_type_, optional): matplotlib axis object
            *args: passed to matplotlib.pyplot.plot() func
            **kwargs: passed to matplotlib.pyplot.plot() func
        Returns:
            ax: matplotlib axis object
        """
        if ax is None:
            ax = plt.gca()

        if plot_initial == True:
            Xmin = self.X.min()
            Xmax = self.X.max()

            spec = self.initial_spectrum.select_range(Xmin, Xmax)

            ax.plot(spec.X, spec.Y, *args, **kwargs)
        else:
            ax.plot(self.X, self.Y, *args, **kwargs)

        ax.set_xlabel(self.X_Unit)
        ax.set_ylabel(self.Y_Unit)

        if plot_baseline == True:
            ax.plot(self.X, self.baseline)

        return ax

    def interpolate(
        self,
        X_low: Union[int, float],
        X_high: Union[int, float],
        step: Union[int, float] = 1,
        inplace: bool = False,
    ):
        """Interpolates Spectrum using SciPy's Akima 1D Interpolator

        Args:
            X_low (Union[int, float]): _description_
            X_high (Union[int, float]): _description_
            step (Union[int, float], optional): _description_. Defaults to 1.
            inplace (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if X_low < self.X.min():
            print(
                f"{X_low} is outside of spectral range of input (min:{self.X.min()}, max:{self.X.max()})"
            )

        if X_high > self.X.max():
            print(
                f"{X_high} is outside of spectral range of input (min:{self.X.min()}, max:{self.X.max()})"
            )

        X_interp = np.arange(start=X_low, stop=X_high, step=step)
        interpolater = Akima1DInterpolator(self.X, self.Y)
        Y_interpolated = interpolater(X_interp)

        if inplace == False:
            other = deepcopy(self)
            other.X = deepcopy(X_interp)
            other.Y = deepcopy(Y_interpolated)
            return other
        else:
            self.X = deepcopy(X_interp)
            self.Y = deepcopy(Y_interpolated)
            return self

    def median_filter(self, window: int = 5, inplace: bool = False):
        """Apply a median filter to the spectrum. Generally removes narrow spikes in data with minimal peak distortion

        Args:
            window (int, optional): width of window for kernel filter. Must be an odd number Defaults to 5.
            inplace (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        if (window % 2) != 1:
            raise ValueError("window should be an odd integer.")

        filtered_Y = medfilt(self.Y, window)

        if inplace == False:
            other = deepcopy(self)
            other.Y = deepcopy(filtered_Y)
            return other
        else:
            self.Y = deepcopy(filtered_Y)
            return self

    def gaussian_filter(self, sigma: int = 5, inplace: bool = False, outlier_reject: bool = False):
        """Apply a 1D Gaussian Kernel filter to the spectrum. Smooths spectrum

        Args:
            sigma (int, optional): standard deviation for Gaussian kernel
            inplace (bool, optional): _description_. Defaults to False.
            outlier_reject:(bool, optional): Determine where to apply a 3x3 median filter prior to smoothing

        Returns:
            _type_: _description_
        """
        if outlier_reject == True:
            spec = self.median_filter(3)
        else:
            spec = self

        filtered_Y = gaussian_filter1d(spec.Y, sigma)

        if inplace == False:
            other = deepcopy(self)
            other.Y = deepcopy(filtered_Y)
            return other
        else:
            self.Y = deepcopy(filtered_Y)
            return self

    def baseline_ASLS(self, lam: float = 1e6, p: float = 0.0005, inplace: bool = False):
        """
        Applies the Asymmetric Least Squares Smoothing baseline correction using PyBaselines.

        This method uses the Whittaker smoother implementation from the PyBaselines package
        to calculate a flexible baseline that preferentially fits the lower points of the spectrum.
        This algorithm is particularly effective for spectra with peaks that primarily extend
        in the positive direction from the baseline.

        Args:
            lam (float, optional): Smoothing parameter. Controls the flexibility of the baseline.
                Higher values create smoother, less flexible baselines. Defaults to 1e6.
            p (float, optional): Asymmetry parameter. Controls the balance between fitting
                points above vs. below the baseline. Smaller values create baselines that
                follow the lower points of the data. Defaults to 0.0005.
            inplace (bool, optional): If True, stores the baseline in the current object's
                baseline attribute. If False, returns a new Spectrum object containing
                the baseline. Defaults to False.

        Returns:
            Spectrum or self: If inplace=False, returns a new Spectrum object with the
            calculated baseline as Y values. If inplace=True, modifies the current object
            by setting its baseline attribute and returns self.

        Notes:
            - For most IR/Raman spectra, the default parameters work well
            - This implementation uses the optimized version from PyBaselines which is
              faster than the custom ALS implementation
            - For spectra with complex baselines, compare results with baseline_ALS or
              baseline_aggressive_rubberband

        See Also:
            baseline_ALS: Alternative custom implementation of asymmetric least squares

        References:
            P. H. C. Eilers, H. F. M. Boelens, "Baseline Correction with Asymmetric Least Squares
            Smoothing," Leiden University Medical Centre Report (2005).
        """
        baseline = pybl.whittaker.asls(self.Y, lam=lam, p=p)[0]

        if inplace == False:
            return Spectrum(self.X, baseline)
        else:
            self.baseline = baseline

    def baseline_ALS(self, lam: float = 1e6, p: float = 0.0005, niter=10, inplace: bool = False):
        """
        Applies Asymmetric Least Squares Smoothing baseline correction algorithm.

        Implements the method by P. Eilers and H. Boelens (2005). This is an alternative
        to the Whitaker Smoother in PyBaselines ASLS. It is slower but often more
        numerically stable for complex spectra.

        Args:
            lam (float, optional): Smoothing parameter. Controls the flexibility of the
                baseline - higher values create smoother baselines. Defaults to 1e6.
            p (float, optional): Asymmetry parameter. Controls the balance between fitting
                points above vs. below the curve - smaller values create baselines that
                follow the lower points of the data. Defaults to 0.0005.
            niter (int, optional): Number of iterations. More iterations may provide better
                fits but increase computation time. Defaults to 10.
            inplace (bool, optional): If True, stores the baseline in the current object's
                baseline attribute. If False, returns a new Spectrum object containing
                the baseline. Defaults to False.

        Returns:
            Spectrum or self: If inplace=False, returns a new Spectrum object with the
            calculated baseline as Y values. If inplace=True, modifies the current object
            by setting its baseline attribute and returns self.

        Notes:
            - For spectra with broad peaks, larger `lam` values are recommended
            - For spectra with sharp peaks or narrow features, consider smaller `lam` values
            - `p` should typically be in the range of 0.001-0.1 for most spectra

        References:
            Eilers, P. H. C., & Boelens, H. F. M. (2005). Baseline correction with
            asymmetric least squares smoothing. Leiden University Medical Centre Report.

            Implemented on stackoverflow by user: sparrowcide
            https://stackoverflow.com/questions/29156532/python-baseline-correction-library
        """
        y = self.Y
        L = len(y)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            # Z = W + lam * D.dot(D.transpose())
            Z = W + lam * np.dot(D, D.T)
            z = sparse.linalg.spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        baseline = z
        if inplace == False:
            return Spectrum(self.X, baseline)
        else:
            self.baseline = baseline

    def baseline_rubberband(self, inplace=False):
        baseline = rubberband(self.X, self.Y)
        if inplace == False:
            return Spectrum(self.X, baseline)
        else:
            self.baseline = baseline

    def baseline_aggressive_rubberband(
        self, Y_stretch=0.0001, inplace=False, plot_intermediate=False
    ):
        """
        Applies an enhanced rubberband baseline correction with nonlinear distortion.

        This method improves upon the standard rubberband correction by adding a parabolic
        distortion to the Y values before finding the convex hull, allowing the baseline
        to better adapt to complex curved backgrounds. The distortion is then subtracted
        from the resulting baseline.

        Args:
            Y_stretch (float, optional): Controls the amount of parabolic distortion applied.
                Higher values create more aggressive baseline curvature. Defaults to 0.0001.
            inplace (bool, optional): If True, stores the baseline in the current object's
                baseline attribute. If False, returns a new Spectrum object containing
                the baseline. Defaults to False.
            plot_intermediate (bool, optional): If True, plots the distorted spectrum and
                resulting baseline for visual inspection. Useful for parameter tuning.
                Defaults to False.

        Returns:
            Spectrum or self: If inplace==False, returns a new Spectrum object with the
            calculated baseline as Y values. If inplace==True, modifies the current object
            by setting its baseline attribute and returns self.

        Notes:
            - The method calculates a parabolic distortion centered at the midpoint of the X range
            - The last point of the baseline is adjusted to match the second-to-last point to
              avoid edge artifacts
            - Increasing Y_stretch makes the baseline more curved and aggressive
            - For subtle baselines in relatively flat spectra, use smaller Y_stretch values

        Example:
            ```python
            # Calculate baseline and store in the spectrum object
            spec.baseline_aggressive_rubberband(Y_stretch=0.0002, inplace=True)

            # Subtract baseline from spectrum
            baseline_corrected = spec - spec.baseline

            # Or calculate and return baseline as a new spectrum
            baseline = spec.baseline_aggressive_rubberband(Y_stretch=0.0001)
            ```
        """
        midpoint_X = round((max(self.X) - min(self.X)) / 2)
        nonlinear_offset = Y_stretch * (self.X - midpoint_X) ** 2
        Y_alt = self.Y + nonlinear_offset
        baseline = rubberband(self.X, Y_alt) - nonlinear_offset
        baseline[-1] = baseline[-2]

        if plot_intermediate == True:
            plt.plot(self.X, Y_alt)
            plt.plot(self.X, baseline)

        if inplace == False:
            return Spectrum(self.X, baseline)
        else:
            self.baseline = baseline

    def find_peaks(self, *args, **kwargs):
        """
        Identifies peaks in the spectrum data using scipy.signal.find_peaks and returns positions in x-axis units.

        This method wraps scipy.signal.find_peaks functionality while adding spectrum-specific features:
        - Maps peak indices to corresponding x-axis values
        - Calculates peak widths in x-axis units when applicable
        - Preserves all original peak properties with additional metadata

        Args:
            *args: Positional arguments passed to scipy.signal.find_peaks
            **kwargs: Keyword arguments passed to scipy.signal.find_peaks
                Common parameters include:
                - height (float or tuple): Required height of peaks
                - threshold (float or tuple): Required threshold of peaks
                - distance (float): Required minimal horizontal distance between peaks
                - prominence (float or tuple): Required prominence of peaks
                - width (float or tuple): Required width of peaks
                - rel_height (float): Relative height at which peak width is measured (0-1.0)

        Returns:
            dict: Peak properties dictionary containing:
                - peaks_idx (array): Indices of peaks in the original spectrum
                - peaks_wn (array): X-axis values of peaks
                - widths_wn (array): Peak widths in x-axis units (if widths are calculated)
                - All other properties returned by scipy.signal.find_peaks (heights, prominences, etc.)

        Notes:
            For detailed information on peak detection parameters, refer to the scipy.signal.find_peaks documentation:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

        Example:
            ```python
            spec = Spectrum(X, Y)
            # Find peaks with minimum height 0.1 and minimum distance of 5 data points
            peaks = spec.find_peaks(height=0.1, distance=5)
            print(f"Found {len(peaks['peaks_wn'])} peaks at wavenumbers: {peaks['peaks_wn']}")
            ```
        """
        found_peaks = find_peaks(self.Y, *args, **kwargs)
        peaks_idx = found_peaks[0]
        peaks_wn = self.X[peaks_idx]
        peak_properties = found_peaks[1]
        peak_properties["peaks_wn"] = peaks_wn
        peak_properties["peaks_idx"] = peaks_idx
        if peak_properties.__contains__("widths"):
            wn_spacing = self.X[1] - self.X[0]
            peak_properties["widths_wn"] = peak_properties["widths"] * wn_spacing
        return peak_properties

        # Maybe set minimum prominence based on standard deviation of a baseline subtracted region known to to have no peaks.

    def find_complex_peaks(
        self,
        baseline_range: Tuple = (None, None),
        noise_range: Tuple = (None, None),
        peak_range: Tuple = (None, None),
        baseline1_param: Dict = {"lam": 1e7, "p": 0.005},
        baseline2_stretch_param: float = 0.0000000001,
        rough_median_filter_len: int = 21,
        fine_median_filter_len: int = 3,
        fine_gaussian_filter_len: int = 3,
        fine_gaussian_filter: bool = False,
        fine_median_filter: bool = False,
        peak_noise_mult: int = 2,
        find_peaks_params={"width": (None, None), "rel_height": 0.5, "distance": 5},
        plot_initial: bool = False,
        plot_subtracted: bool = False,
        plot_peak_locations: bool = True,
        return_baseline_subtracted=False,
    ):
        """
        Identifies peaks in a complex spectrum using a two-stage baseline correction and adaptive peak detection.

        This method uses a sophisticated approach to peak detection in complex spectra:
        1. Applies initial median filtering to remove noise spikes
        2. Performs first baseline correction using ASLS
        3. Performs second baseline correction using aggressive rubberband
        4. Detects peaks with adaptive threshold based on noise estimation
        5. Optionally applies additional filtering and visualization

        Args:
            baseline_range (Tuple, optional): The (low, high) x-range for baseline correction.
                If (None, None), uses full spectrum. Defaults to (None, None).
            noise_range (Tuple, optional): The (low, high) x-range used to estimate noise for peak detection.
                Should be a region with minimal peaks. Defaults to (None, None).
            peak_range (Tuple, optional): The (low, high) x-range to search for peaks.
                If (None, None), uses full spectrum. Defaults to (None, None).
            baseline1_param (Dict, optional): Parameters for ASLS baseline correction.
                Higher 'lam' gives smoother baseline, higher 'p' increases asymmetry.
                Defaults to {"lam": 1e7, "p": 0.005}.
            baseline2_stretch_param (float, optional): Stretch parameter for aggressive rubberband.
                Higher values increase baseline curvature. Defaults to 0.0000000001.
            rough_median_filter_len (int, optional): Window size for initial noise filtering.
                Must be odd integer. Defaults to 21.
            fine_median_filter_len (int, optional): Window size for final median filtering.
                Must be odd integer. Defaults to 3.
            fine_gaussian_filter_len (int, optional): Sigma for final Gaussian filtering.
                Defaults to 3.
            fine_gaussian_filter (bool, optional): Whether to apply Gaussian filter after baseline correction.
                Defaults to False.
            fine_median_filter (bool, optional): Whether to apply median filter after baseline correction.
                Defaults to False.
            peak_noise_mult (int, optional): Multiplier for noise standard deviation to set minimum peak height.
                Higher values require more prominent peaks. Defaults to 2.
            find_peaks_params (Dict, optional): Parameters passed to scipy.signal.find_peaks.
                See scipy documentation for details. Default sets FWHM peak width detection
                and minimum distance between peaks.
            plot_initial (bool, optional): Whether to plot original spectrum and baseline.
                Defaults to False.
            plot_subtracted (bool, optional): Whether to plot baseline-subtracted spectrum.
                Defaults to False.
            plot_peak_locations (bool, optional): Whether to mark detected peaks on the plot.
                Defaults to True.
            return_baseline_subtracted (bool, optional): Whether to return baseline-subtracted spectra
                along with peak information. Defaults to False.

        Returns:
            dict or peak_properties:
                If return_baseline_subtracted is False:
                    Returns peak properties dictionary with peaks_wn (peak positions),
                    peaks_idx (indices), prominences, widths, etc.
                If return_baseline_subtracted is True:
                    Returns dict with:
                    - "peak_dict": Peak properties dictionary
                    - "baseline_subtracted": Baseline-subtracted spectrum
                    - "baseline_subtracted_smoothed": Filtered baseline-subtracted spectrum

        Notes:
            - Adjusting baseline parameters greatly affects peak detection performance
            - For noisy spectra, enabling fine filtering is recommended
            - For accurate noise estimation, select a noise_range with minimal peaks
        """

        if baseline_range[0] != None:
            spec = self.select_range(baseline_range[0], baseline_range[1])
        else:
            spec = self

        if noise_range[0] != None:
            noise_X_low, noise_X_high = noise_range

        baseline1 = spec.median_filter(rough_median_filter_len).baseline_ASLS(**baseline1_param)
        baseline_subtracted1 = spec - baseline1
        baseline2 = baseline_subtracted1.median_filter(
            rough_median_filter_len
        ).baseline_aggressive_rubberband(baseline2_stretch_param)
        baseline_subtracted2 = baseline_subtracted1 - baseline2

        if not find_peaks_params.__contains__("height"):
            stdev_range = baseline_subtracted2.select_range(noise_X_low, noise_X_high)
            stdev = (stdev_range - stdev_range.median_filter(rough_median_filter_len)).Y.std()
            find_peaks_params["height"] = stdev * peak_noise_mult
            find_peaks_params["prominence"] = stdev

        baseline_subtracted2_filtered = baseline_subtracted2

        if fine_median_filter == True:
            baseline_subtracted2_filtered = baseline_subtracted2_filtered.median_filter(
                fine_median_filter_len
            )

        if fine_gaussian_filter == True:
            baseline_subtracted2_filtered = baseline_subtracted2_filtered.gaussian_filter(
                fine_gaussian_filter_len
            )

        if peak_range[0] != None:
            baseline_subtracted2_filtered = baseline_subtracted2_filtered.select_range(*peak_range)

        peaks = baseline_subtracted2_filtered.find_peaks(
            **find_peaks_params
        )  # sets relative peak height for the width to 0.5 for full width half max and distance for 5 data points between peaks
        print(find_peaks_params.keys())
        if plot_initial == True:
            spec.plot(label="Spectrum")
            (baseline1 + baseline2).plot(label="Baseline")
            plt.legend()

        if plot_initial == True & plot_subtracted == True:
            fig, ax = plt.subplots()

        if plot_subtracted == True:
            if peak_range[0] != None:
                baseline_subtracted2.select_range(*peak_range).plot(label="Spectrum")
            else:
                baseline_subtracted2.plot(label="Spectrum")

            baseline_subtracted2_filtered.plot(label="Smoothed Spectrum ")
            if plot_peak_locations == True:
                plt.vlines(
                    peaks["peaks_wn"],
                    ymin=baseline_subtracted2_filtered.Y[peaks["peaks_idx"]] - peaks["prominences"],
                    ymax=baseline_subtracted2_filtered.Y[peaks["peaks_idx"]],
                    color="k",
                )
            plt.legend()
        if return_baseline_subtracted == True:
            return {
                "peak_dict": peaks,
                "baseline_subtracted": baseline_subtracted2,
                "baseline_subtracted_smoothed": baseline_subtracted2_filtered,
            }

        else:
            return peaks

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Spectrum(self.X, self.Y * other)

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Spectrum(self.X, self.Y / other)

    def __add__(self, other):
        if not isinstance(other, (Spectrum, int, float)):
            return NotImplemented

        if isinstance(other, Spectrum):
            return Spectrum(self.X, self.Y + other.Y)

        if isinstance(other, (int, float)):
            return Spectrum(self.X, self.Y + Y)

    def __sub__(self, other):
        if not isinstance(other, (Spectrum, int, float)):
            return NotImplemented

        if isinstance(other, Spectrum):
            return Spectrum(self.X, self.Y - other.Y)

        if isinstance(other, (int, float)):
            return Spectrum(self.X, self.Y - other)

    def test_saturation(
        self,
        X_low: Union[int, float],
        X_high: Union[int, float],
        saturation_cutoff: Union[int, float],
        stdev_cut_off: Union[int, float],
        smoothing_window: int = 21,
    ) -> bool:
        """Tests if the spectrum is saturated in a region either by total intensity of by standard deviation. Best Suited for IR spectra where saturation point is known to be around 2 absorbance units and signal gets noisier the more saturated it is. An alternate method for Raman would be to check for plateaus in a peak.

        Args:
            X_low (Union[int, float]): _description_
            X_high (Union[int, float]): _description_
            saturation_cutoff (Union[int, float]): _description_
            stdev_cut_off (Union[int, float]): _description_
            smoothing_window (int, optional): Number of points to use in the median filter used to compare if the noise is too high in a region. Defaults to 21.

        Returns:
            bool: boolean true if the selected region meets the saturation criteria. False if not
        """
        data = self.select_range(X_low=X_low, X_high=X_high)
        smoothed = data.median_filter(11)
        mean = data.Y.mean()
        stdev = np.std(data.Y - smoothed.Y)
        if mean > saturation_cutoff or stdev > stdev_cut_off:
            return True
        else:
            return False


# %%
# To Do
# Data Class should include Original Data and Interpolated Data
# X = np.arange(400.4, 6000.6, step=20)
# sigma = np.random.rand(*X.shape) * 1000000
# Y = 0.00015 * X**3 - X**2 + 8000 + sigma
# # %%
# example = Spectrum(X=X, Y=Y, X_Unit="Wavenumber", Y_Unit="Absorbance")
# example.plot(linestyle="dashed")

# example.interpolate_spectrum(400, 6000).selcted_range(800, 5000).plot(c="r")
# # %%
# example.interpolate_spectrum()

# # %%


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


# %%
