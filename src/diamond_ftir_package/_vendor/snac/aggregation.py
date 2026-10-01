"""Module for nitrogen aggregation calculations and cooling history scenarios.
"""

import numpy as np

from .cooling import linear_cool

_EAR = 81160
_PREEXP = 293608

_SCENARIOS = [
    'continuous', 'hot_spike', 'rapid_ascent'
    ]


def aggregate(NA, T, t):
    """input: NT, NA, T and t
    returns NA_final after aggregation for time t at temperature T
    accepts t in Myr, T in deg. C
    """
    rate_const = _PREEXP * np.exp(
        -_EAR/(T+273)
        )
    NA_final = NA/(1 + rate_const * t * NA)

    return NA_final


def Temp_N(t, NT, IaB):
    """
    calculate T in degrees Celsius based on nitrogen aggregation.

    INPUT:
    ------
    t | float/int: elapased time (in Myr)
    NT | float/int: total Nitrogen concentration (ppm)
    IaB | float/int: proportion of B-centres (0 to 1)

    RETURNS:
    T | float: temperature (degrees Celsius)
    """
    t *= 1e6 * 365.25 * 24 * 60 * 60  # convert Myr to seconds
    NA = NT * (1-IaB)
    T = (-81160/(np.log(((NT/NA)-1)/(t*NT*293608))))

    return T-273


def aggregate_and_cool(params, *args, **kwargs):
    """"
    Function used in the optimization of the cooling rate and starting
    temperature of a sample to match the observed nitrogen aggregation data.

    Note: by default, this function returns the sum of the squared differences
    between the modelled and observed nitrogen aggregation values. However, if
    the keyword argument return_history=True is provided, the full history of
    NA (core and rim) and Temperature values will be returned instead.

    REQUIRED INPUT:
    --------
    params: tuple of parameters to be optimized
        T_start | int: starting temperature (Celsius)
        cooling_rate | float: cooling rate (K/Myr)
    args: tuple of arguments for the aggregation function
        durations: list of durations (Myr)
        age_core: age of the core (Ma)
        age_rim: age of the rim (Ma)
        c_NT: total nitrogen concentration in the core (ppm)
        r_NT: total nitrogen concentration in the rim (ppm)
        c_agg: observed nitrogen aggregation in the core (proportion of N
            in B-centres, 0-1)
        r_agg: observed nitrogen aggregation in the rim (proportion of N
            in B-centres, 0-1)

    OPTIONAL INPUT (BY KEYWORD):
    --------
    cooling_function | function: function to calculate temperature at each
        time step (default: linear_cool)
    T_scenario | str: the cooling scenario, one of the following:
        'continuous' assumes continouous cooling.
            No additional parameters are required.
        'hot_spike' assumes a sharp spike in temperature at a specified time
            followed by rapid cooling until reaching continous trajectory.
            Requires scenario_params = (T_pulse, t_pulse_start, pulse_duration)
            Note: T_pulse is the relative increase in temperature
        'rapid_ascent' assumes instantaneous ascent to shallower depth at a
            specified time (i.e. drop in temperature).
            Cooling continues after ascent at the same rate as before.
            Requires scenario_params = (T_drop, t_ascent)
            Note: T_drop is the relative drop in temperature
    scenario_params: additional parameters for specific cooling scenarios
        (default: None). See T_scenario options for details.
    return_history: boolean indicating whether to return the full history of
        NA and T values (default: False)

    RETURNS:
    --------
    error | float (default): sum of the squared differences between the
        modelled and observed nitrogen aggregation values.
    OR (if return_history=True): tuple consisting of
        NA_core: list of NA values in the core at each time step
        NA_rim: list of NA values in the rim at each time step
        T_all: list of temperatures at each time step
    """
    # unpack parameters and arguments
    T_start, cooling_rate = params
    durations, age_core, age_rim, c_NT, r_NT, c_agg, r_agg = args

    # initialize variables
    c_NA0 = c_NT
    c_NA1 = 0
    r_NA0 = r_NT
    r_NA1 = 0

    # unpack variadic keyword arguments
    cooling_function = kwargs.get('cooling_function', linear_cool)
    T_scenario = kwargs.get('T_scenario', 'continuous')
    scenario_params = kwargs.get('scenario_params', None)
    return_history = kwargs.get('return_history', False)

    NA_core = []
    NA_rim = []
    T_all = []

    # iterate over the cooling steps
    i = 0
    for duration in durations:
        # calculate T for each duration depending on the cooling scenario
        if T_scenario == 'continuous':
            # simple continuous cooling
            T = cooling_function(T_start, duration, cooling_rate)

        elif T_scenario == 'hot_spike':
            # sharp spike in temperature at specified time followed by rapid
            # cooling until reaching continous trajectory
            T_pulse, t_pulse_start, pulse_duration = scenario_params

            # cooling before pulse
            if duration < t_pulse_start:
                T = cooling_function(T_start, duration, cooling_rate)

            # temperature spike
            # (linear interpolation between start and end of pulse)
            elif (duration >= t_pulse_start) and (duration <= (
                    t_pulse_start + pulse_duration)):
                # interpolate linearly between T_start_pulse and T_after_pulse
                T_start_pulse = cooling_function(
                    T_start, t_pulse_start, cooling_rate) + T_pulse
                T_after_pulse = cooling_function(
                    T_start, t_pulse_start + pulse_duration, cooling_rate)
                T = T_start_pulse + (T_after_pulse - T_start_pulse) * (
                    (duration - t_pulse_start) / pulse_duration)

            # cooling continues after pulse
            elif duration > (t_pulse_start + pulse_duration):

                T = cooling_function(T_start, duration, cooling_rate)

        elif T_scenario == 'rapid_ascent':
            # instantaneous ascent to shallower depth at specified time
            # (i.e. drop in temperature)
            T_drop, t_ascent = scenario_params

            # before ascent starts (continuous cooling):
            if duration < t_ascent:
                T = cooling_function(T_start, duration, cooling_rate)

            # after rapid ascent and drop in T
            # (return to original cooling rate):
            elif duration >= t_ascent:
                T = cooling_function(
                    cooling_function(T_start, t_ascent, cooling_rate) - T_drop,
                    duration - t_ascent,
                    cooling_rate
                    )

        else:
            raise ValueError(
                f"Invalid T_scenario. Must be one of {', '.join(_SCENARIOS)}"
                )

        # work out duration increment from duration list
        if i == 0:
            d_t = duration
        else:
            d_t = durations[i]-durations[i-1]

        # before rim grows, only core aggregates:
        if (age_core - duration) > age_rim:
            c_NA1 = aggregate(c_NA0, T, d_t * 1e6 * 365.25 * 24 * 60 * 60)
            c_NA0 = c_NA1

        # after rim has grown, core and rim now both aggregate
        elif (age_core - duration) < age_rim:
            c_NA1 = aggregate(c_NA0, T, d_t * 1e6 * 365.25 * 24 * 60 * 60)
            c_NA0 = c_NA1

            r_NA1 = aggregate(r_NA0, T, d_t * 1e6 * 365.25 * 24 * 60 * 60)
            r_NA0 = r_NA1

        # increment i and append values to lists
        # it may not be necessary to store all values,
        # but might be useful for debugging/plotting
        i += 1
        NA_core.append(c_NA0)
        NA_rim.append(r_NA0)
        T_all.append(T)

    # retrieve final aggregation values
    c_agg_model = 1-(NA_core[-1]/c_NT)
    r_agg_model = 1-(NA_rim[-1]/r_NT)

    # calculate error
    # multiplying by 1000 for scaling purposes.
    error = (
        (r_agg - r_agg_model)**2 + (c_agg - c_agg_model)**2
        ) * 1e3

    # convert histories to numpy arrays for efficient numeric operations
    NA_core = np.array(NA_core)
    NA_rim = np.array(NA_rim)
    T_all = np.array(T_all)

    # optionally return the full history of NA and T values
    if return_history:
        return NA_core, NA_rim, T_all

    return error
