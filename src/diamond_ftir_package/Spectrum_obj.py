# %%
import numpy as np

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Union
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator
from copy import deepcopy
from scipy.signal import medfilt
import pybaselines as pybl

@dataclass(order=True)
class Spectrum:
    """Spectrum Object with method for common data processing.
    Most methods support chained calls and returning values inplace.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
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

    def plot(self, ax=None, plot_baseline= False, *args, **kwargs):
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
            print(f"{X_low} is outside of spectral range of input (min:{self.X.min()}, max:{self.X.max()})")
        
        if X_high > self.X.max():
            print(f"{X_high} is outside of spectral range of input (min:{self.X.min()}, max:{self.X.max()})")

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
    
    def baseline_ASLS(self, lam: float = 1e6, p: float = 0.0005, inplace: bool = False):
        baseline = pybl.whittaker.asls(self.Y, lam=lam, p=p)[0]
        
        if inplace == False:
            return Spectrum(self.X, baseline)
        else: self.baseline = baseline
    
    def __mul__(self,other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Spectrum(self.X, self.Y * other)
        
    def __div__(self,other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Spectrum(self.X, self.Y / other)
    
    def __add__(self,other):
        if not isinstance(other, (Spectrum, int, float)):
            return NotImplemented
        
        if isinstance(other,Spectrum):
            return Spectrum(self.X, self.Y + other.Y)
               
        if isinstance(other,(int,float)):
            return Spectrum(self.X, self.Y + Y)
    
    def __sub__(self,other):
        if not isinstance(other, (Spectrum, int, float)):
            return NotImplemented
        
        if isinstance(other,Spectrum):
            return Spectrum(self.X, self.Y - other.Y)
        
        if isinstance(other,(int,float)):
            return Spectrum(self.X, self.Y - other)


    def test_saturation(self, X_low: Union[int, float], X_high: Union[int, float], saturation_cutoff: Union[int, float], stdev_cut_off: Union[int, float], smoothing_window:int = 21 ) -> bool:
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
