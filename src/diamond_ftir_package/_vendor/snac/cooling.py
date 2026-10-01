"""Module providing cooling functions.
"""
import numpy as np


def linear_cool(T_start, time, rate):
    """
    Calculate the temperature after a given time of linear cooling.

    PARAMS:
    --------
    T_start: starting temperature (Celsius)
    time: elapsed time (Myr)
    rate: exponential cooling rate (1/Myr)

    RETURNS:
    --------
    T: temperature (Celsius) after 'time' Myr
    """
    return T_start - time * rate


def exponential_cool(T_start, time, rate):
    """
    Calculate the temperature after a given time of exponential cooling.

    PARAMS:
    --------
        T_start: starting temperature (Celsius)
        time: elapsed time (Myr)
        rate: exponential cooling rate (1/Myr)

    RETURNS:
        T: temperature (Celsius) after 'time' Myr
    """
    return T_start * np.exp(-rate * time)
