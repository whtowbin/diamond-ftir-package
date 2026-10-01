from functools import partial
import os

import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt
import csv
import json

from collections.abc import Callable

from .diamond import Diamond
from .cooling import linear_cool
from .aggregation import aggregate_and_cool, _SCENARIOS


class AggregationModel:
    """Class to model nitrogen aggregation and cooling history of a diamond.
    """

    def __init__(
            self,
            diamond: Diamond,
            cooling_rate0: float = 0.01, T_start0: int = 1200,
            rate_bounds=(0.001, 0.12), T_bounds=(1000, 1450), dt: int = 1,
            T_scenario: str = 'continuous', scenario_params=None,
            cooling_function: Callable = linear_cool
            ):
        """Initialise AggregationModel object.

        PARAMS:
        --------
        diamond | diamond.Diamond : Diamond object with measured parameters
        cooling_rate0 | float : initial guess for cooling rate (deg.C/Myr)
        T_start0 | int : initial guess for starting temperature (deg.C)
        rate_bounds | tuple of float : bounds for cooling rate during fitting
            of the form (lower, upper)
        T_bounds | tuple of int : bounds for starting temperature during
            fitting of the form (lower, upper)
        dt | int : time step for model calculations (Myr)
        scenario | str : cooling scenario to use, one of 'continuous',
            'hot_spike', 'rapid_ascent'
        scenario_params | dict (optional): parameters for the cooling scenario.

        For more information for scenarios and scenario_params, see
        documentation in aggregation.py
        """

        if T_scenario not in _SCENARIOS:
            raise ValueError(
                "Unrecognised scenario. "
                f"Must be one of {', '.join(_SCENARIOS)}."
                )

        self.diamond = diamond
        self.cooling_rate0 = cooling_rate0
        self.T_start0 = T_start0

        self.rate_bounds = rate_bounds
        self.T_bounds = T_bounds

        self.dt = dt

        self.T_scenario = T_scenario
        self.scenario_params = scenario_params
        self.cooling_function = cooling_function

        self.fitted = False

    def aggregate_and_cool_partial(self, return_history: bool = False):
        """Create a partial function of aggregate_and_cool with fixed cooling
        parameters. This can be used in optimization routines.
        """
        return partial(
            aggregate_and_cool,
            cooling_function=self.cooling_function,
            T_scenario=self.T_scenario,
            scenario_params=self.scenario_params,
            return_history=return_history
            )

    def get_durations(self):
        """
        Determine incremental durations for model calulations.

        RETURNS:
        --------
        durations | numpy.ndarray : time steps for the model

        """
        duration_core = self.diamond.age_core - self.diamond.age_kimberlite
        durations = np.arange(0., duration_core+1, self.dt)
        durations[0] = 0.01

        return durations

    def run(self):
        """
        Optimise the model history by fitting the cooling and aggregation
        parameters to the observed data.
        """
        acp = self.aggregate_and_cool_partial()

        res = op.minimize(
            acp,
            x0=(self.T_start0, self.cooling_rate0),
            bounds=(self.T_bounds, self.rate_bounds),
            args=(
                self.get_durations(),
                self.diamond.age_core, self.diamond.age_rim,
                self.diamond.c_NT, self.diamond.r_NT,
                self.diamond.c_agg, self.diamond.r_agg),
            tol=1e-7
            )

        self.model_results = res.x
        self.model_success = res.success
        self.model_status = res.status
        self.model_message = res.message
        self.fitted = True

    def get_history(self):
        """
        Retrieve temperature and aggregation history of the model.

        If the model has been fitted with model_history(), use the fitted
        parameters; otherwise, use the initial guesses to project the
        aggregation and cooling history.

        RETURNS:
        --------
        results | dict : dictionary with keys:
            'T_start' : fitted starting temperature (deg.C)
            'cooling_rate' : fitted cooling rate (deg.C/Myr)
            'NA_core' : final N_A concentration in core (ppm)
            'NA_rim' : final N_A concentration in rim (ppm)
            'NB_core' : final N_B concentration in core (ppm)
            'NB_rim' : final N_B concentration in rim (ppm)
        """

        if self.fitted:
            params = self.model_results
        else:
            params = (self.T_start0, self.cooling_rate0)

        acp = self.aggregate_and_cool_partial()
        NA_core, NA_rim, T_all = acp(
            params,
            self.get_durations(),
            self.diamond.age_core,
            self.diamond.age_rim,
            self.diamond.c_NT,
            self.diamond.r_NT,
            self.diamond.c_agg,
            self.diamond.r_agg,
            return_history=True
            )

        # Calculate NB concentrations
        NB_core = self.diamond.c_NT - NA_core
        NB_rim = self.diamond.r_NT - NA_rim

        if self.fitted:
            T_start = self.model_results[0]
            cooling_rate = self.model_results[1]
        else:
            T_start = self.T_start0
            cooling_rate = self.cooling_rate0

        history = {
            'durations': self.get_durations(),
            'T_start': T_start,
            'cooling_rate': cooling_rate,
            'T_all': T_all,
            'NA_core': NA_core,
            'NA_rim': NA_rim,
            'NB_core': NB_core,
            'NB_rim': NB_rim
        }

        return history

    def plot_T_history(self):
        """Plot the temperature history of the fitted model.
        """
        T_all = self.get_history()['T_all']
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.get_durations(), T_all, 'k-', label='T')

        ax.set_xlabel('Time since core growth (Myr)')
        ax.set_ylabel('Temperature (deg.C)')

    def plot_aggregation_history(self, rim_start=True):
        """Plot the nitrogen aggregation history of the fitted model.

        PARAMS:
        -------
        rim_start | bool : if True, plot rim data starting from rim growth
            time; if False, plot rim data from core growth time (default: True)
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        history = self.get_history()

        if rim_start:
            rim_durations = [
                d for d in history['durations']
                if d >= self.diamond.age_core-self.diamond.age_rim
                ]
            rim_start = len(history['durations'])-len(rim_durations)
            ax.plot(
                history['durations'], history['NA_core'],
                label='$[N_{A}]$ core', linestyle='-', color='blue'
                )
            ax.plot(
                history['durations'], history['NB_core'],
                label='$[N_{B}]$ core', linestyle='--', color='blue'
                )
            ax.plot(
                rim_durations,
                history['NA_rim'][rim_start:],
                label='$[N_{A}]$ rim', linestyle='-', color='orange'
                )
            ax.plot(
                rim_durations,
                history['NB_rim'][rim_start:],
                label='$[N_{B}]$ rim', linestyle='--', color='orange'
                )
        else:
            ax.plot(
                history['durations'], history['NA_core'],
                label='$[N_{A}]$ core', linestyle='-', color='blue'
                )
            ax.plot(
                history['durations'], history['NB_core'],
                label='$[N_{B}]$ core', linestyle='--', color='blue'
                )
            ax.plot(
                history['durations'], history['NA_rim'],
                label='$[N_{A}]$ rim',
                linestyle='-', color='orange'
                )
            ax.plot(
                history['durations'], history['NB_rim'],
                label='$[N_{B}]$ rim',
                linestyle='--', color='orange'
                )

        # set axis labels and legend
        ax.legend(bbox_to_anchor=(1.01, 1))
        ax.set_xlabel('Time since core growth (Myr)')
        ax.set_ylabel('N concentration (ppm)')

    def save_history(self, filename: str):
        """Save the fitted model history to a .csv file.

        The CSV file will have columns for durations, temperature, N_A and N_B
        concentrations in core and rim.

        Modelled initial temperature and cooling rate will be included in the
        filename.

        PARAMS:
        -------
        filename | str : path to save the .csv file
        """
        history = self.get_history()

        # retrieve fitted parameters for filename and remove from dictionary
        T_start = history.pop('T_start')
        cooling_rate = history.pop('cooling_rate')

        # retrieve only filename without extension
        base_filename = os.path.splitext(filename)[0]
        savename = (f"{base_filename}_"
                    f"{T_start:.0f}C_"
                    f"{cooling_rate*1000:.0f}K_Gyr.csv"
                    )

        with open(savename, mode='w', newline='') as file:
            writer = csv.DictWriter(
                file,
                fieldnames=history.keys())

            # write header
            writer.writeheader()

            # write data rows
            for i in range(len(history['durations'])):
                row = {key: history[key][i] for key in history.keys()}
                writer.writerow(row)

    def to_json(self, filepath: str):
        """Save AggregationModel instance to JSON file.

        PARAMS:
        -------
        filepath | str : path to JSON file
        """
        diamond_data = {
                'age_core': self.diamond.age_core,
                'age_rim': self.diamond.age_rim,
                'age_kimberlite': self.diamond.age_kimberlite,
                'c_NT': self.diamond.c_NT,
                'c_agg': self.diamond.c_agg,
                'r_NT': self.diamond.r_NT,
                'r_agg': self.diamond.r_agg
        }

        data = {
            'diamond': diamond_data,
            'cooling_rate0': self.cooling_rate0,
            'T_start0': self.T_start0,
            'rate_bounds': self.rate_bounds,
            'T_bounds': self.T_bounds,
            'dt': self.dt,
            'T_scenario': self.T_scenario,
            'scenario_params': self.scenario_params
            }

        if self.fitted:
            model_results = {'initial_T': self.model_results[0],
                             'cooling_rate': self.model_results[1]}
            data['model_results'] = model_results

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def from_json(cls, filepath: str, diamond: None | Diamond | str = None):
        """Create AggregationModel instance from JSON file.

        PARAMS:
        -------
        filepath | str : path to JSON file
        diamond | (None or snac.diamond.Diamond or str), optional :
            optional diamond data. If the model file located at filepath
            contains diamond data, no further input is required.

            If not, diamond data can be provided as a snac.diamond.Diamond
            object or the path to a diamond data json file. If the model file
            contains diamond data AND diamond data is provided separately,
            the latter takes precedence.

        RETURNS:
        --------
        AggregationModel instance
        """
        diamond = diamond

        with open(filepath, 'r') as f:
            data = json.load(f)

        # if user provides a diamond object, use that;
        # if not, see if diamond is in data
        if diamond is not None:
            if isinstance(diamond, Diamond):
                pass
            elif isinstance(diamond, str):
                diamond = Diamond.from_json(diamond)

        elif 'diamond' in data:
            diamond_data = data['diamond']
            diamond = Diamond(
                age_core=diamond_data['age_core'],
                age_rim=diamond_data['age_rim'],
                age_kimberlite=diamond_data['age_kimberlite'],
                c_NT=diamond_data['c_NT'],
                c_agg=diamond_data['c_agg'],
                r_NT=diamond_data['r_NT'],
                r_agg=diamond_data['r_agg']
            )
        else:
            raise ValueError(
                'Diamond data is required to create AggregationModel.'
                )

        model = cls(
           diamond=diamond,
           cooling_rate0=data['cooling_rate0'],
           T_start0=data['T_start0'],
           rate_bounds=tuple(data['rate_bounds']),
           T_bounds=tuple(data['T_bounds']),
           dt=data['dt'],
           T_scenario=data['T_scenario'],
           scenario_params=data['scenario_params']
        )

        return model

    def __str__(self):
        """Return human-readable string represantation.
        """
        msg = (
            f"AggregationModel:\n"
            f"for {self.diamond}\n\n"
            f"Cooling scenario: {self.T_scenario}\n"
            f"- with additional parameters: {self.scenario_params}\n"
            f"Initial guesses:\n"
            f"- Starting: {self.T_start0} deg.C\n"
            f"- Cooling rate: {1e3*self.cooling_rate0} K/Gyr"
        )

        if self.fitted:
            msg += (
                f"\n\nFitted model results:\n"
                f"- Starting T: {self.model_results[0]:.2f} deg.C\n"
                f"- Cooling rate: {1e3*self.model_results[1]:.2f} K/Gyr"
            )
        return msg
