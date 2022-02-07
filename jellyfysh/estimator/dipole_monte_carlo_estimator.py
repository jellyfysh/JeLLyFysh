# JeLLFysh - a Python application for all-atom event-chain Monte Carlo - https://github.com/jellyfysh
# Copyright (C) 2019, 2022 The JeLLyFysh organization
# (See the AUTHORS.md file for the full list of authors.)
#
# This file is part of JeLLyFysh.
#
# JeLLyFysh is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# JeLLyFysh is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with JeLLyFysh in the LICENSE file.
# If not, see <https://www.gnu.org/licenses/>.
#
# If you use JeLLyFysh in published work, please cite the following reference (see [Hoellmer2020] in References.bib):
# Philipp Hoellmer, Liang Qin, Michael F. Faulkner, A. C. Maggs, and Werner Krauth,
# JeLLyFysh-Version1.0 -- a Python application for all-atom event-chain Monte Carlo,
# Computer Physics Communications, Volume 253, 107168 (2020), https://doi.org/10.1016/j.cpc.2020.107168.
#
"""Module for the DipoleMonteCarloEstimator class."""
from copy import copy
import logging
import random
from typing import List, Sequence, Tuple
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.vectors import random_vector_on_unit_sphere
from jellyfysh.potential import Potential
import jellyfysh.setting as setting
from .estimator import Estimator


class DipoleMonteCarloEstimator(Estimator):
    """
    Estimator which compares the space derivatives along the direction of one of the cartesian axes for separations to
    dipoles randomly distributed in the given region.

    This estimator is implemented for a factor potential between point masses in two composite point objects. Only one
    point mass is active, and the target composite point object is approximated as a dipole. The charges of the point
    masses in the target composite point object should sum up to 0.0.

    The derivative method of the potential should expect exactly one separation and two charges.
    The estimator places random points as the dipole center and constructs a dipole for a random orientation. The point
    masses within this dipole then have opposite charges with the absolute value being the maximum charge which is set
    in the initialize method. The charge of the active point mass is set to 1.0 during the estimation of the bounds.
    The derivative between the active point mass and the dipole is then computed by summing two derivatives between the
    active point mass and the target point masses in the dipole.

    Per default, this estimator only uses the primary image separation, that means that the separations in the given
    region are corrected for periodic boundaries.
    """

    def __init__(self, potential: Potential, dipole_separation: float, prefactor: float = 1.5,
                 empirical_bound: float = float('inf'), number_trials: int = 10000, dipole_charge: float = 1.0,
                 periodic_boundaries: bool = True) -> None:
        """
        The constructor of the InnerPointEstimator class.

        Parameters
        ----------
        potential : potential.Potential
            Potential whose derivative is to be bounded.
        dipole_separation : float
            Separation of the two point masses within the constructed dipole.
        prefactor : float, optional
            A constant which gets multiplied to the bounds.
        empirical_bound : float, optional
            If a bound exceeds this value, this value will be returned instead.
        number_trials : int, optional
            The number of separation samples taken.
        dipole_charge : float, optional
            The absolute value of the charges in the dipole.
        periodic_boundaries : bool
            Whether the separations in the given region should be corrected for periodic boundaries.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the potential derivative method does not expect exactly one separation.
            If the potential derivative method does not expect exactly two charges.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential.__class__.__name__, dipole_separation=dipole_separation,
                           prefactor=prefactor, empirical_bound=empirical_bound, number_trials=number_trials,
                           dipole_charge=dipole_charge, periodic_boundaries=periodic_boundaries)
        super().__init__(potential=potential, prefactor=prefactor, empirical_bound=empirical_bound)
        self._number_trials = number_trials
        self._dipole_separation = dipole_separation
        self._dipole_separation_over_two = dipole_separation / 2
        self._dipole_charge = dipole_charge
        if self._potential.number_separation_arguments != 1:
            raise ConfigurationError("The estimator {0} expects a potential "
                                     "which handles exactly one separation!".format(self.__class__.__name__))
        if self._potential.number_charge_arguments != 2:
            raise ConfigurationError("The estimator {0} expects a potential "
                                     "which handles exactly two charges!".format(self.__class__.__name__))

    def derivative_bound(self, lower_corner: Sequence[float], upper_corner: Sequence[float], direction: int,
                         calculate_lower_bound: bool = False) -> List[float]:
        """
        Estimate an upper and an optional lower bound of the potential's space derivative along the given direction
        between a minimum and maximum corner of a given region for the possible separations.

        This method extends the derivative_bound method of the Estimator class.

        The region is covered randomly by creating dipoles at random positions with a random orientation. The dipole has
        a separation and charges that were set on initialization. The upper (lower) bound is then the maximum (minimum)
        of the differences of the sum of the derivatives with respect to the positions of the two point masses, again
        corrected by the prefactor and the empirical bound.

        Parameters
        ----------
        lower_corner : Sequence[float]
            Lower corner of the region to be estimated. The length of the tuple agrees with system dimensions.
        upper_corner : Sequence[float]
            Upper corner of the region to be estimated. The length of the tuple agrees with system dimensions.
        direction : int
            Direction with respect to which the space derivative is taken and the bound is determined.
        calculate_lower_bound : bool
            Whether the lower bound should be calculated or not.

        Returns
        -------
        List[float]
            The list of the determined upper bound and the optionally determined lower bound.
        """
        lower_corner = copy(lower_corner)
        upper_corner = copy(upper_corner)
        super().derivative_bound(lower_corner, upper_corner, direction, calculate_lower_bound)
        for d in range(setting.dimension):
            lower_corner[d] -= self._dipole_separation_over_two
            upper_corner[d] += self._dipole_separation_over_two

        upper_bound = 0.0
        for _ in range(self._number_trials):
            dipole_center = [random.uniform(lower_corner[d], upper_corner[d]) for d in range(setting.dimension)]
            random_direction = random_vector_on_unit_sphere(setting.dimension)
            position1 = [dipole_center[d] + random_direction[d] * self._dipole_separation_over_two
                         for d in range(setting.dimension)]
            position2 = [dipole_center[d] - random_direction[d] * self._dipole_separation_over_two
                         for d in range(setting.dimension)]
            self._correct_separation(position1)
            self._correct_separation(position2)
            derivative1 = self._potential_derivative(direction, position1, 1.0, self._dipole_charge)
            derivative2 = self._potential_derivative(direction, position2, 1.0, -self._dipole_charge)
            upper_bound = max(upper_bound, abs(derivative1 + derivative2))

        upper_bound *= self._prefactor

        if calculate_lower_bound:
            return [min(self._empirical_bound, upper_bound), max(-self._empirical_bound, -upper_bound)]
        else:
            return [min(self._empirical_bound, upper_bound)]

    def charge_correction_factor(self, active_charge: float, target_charges: Tuple[float] = None) -> float:
        """
        Return a multiplicative correction factor on the estimated bounds based on the given charges of the active point
        mass and the target point masses.

        The charge of the single active point mass should be given as a float. The charges of the point masses in the
        target composite point object should be given as a tuple of floats, and sum up to 0.0.

        This method returns the charge of the active point mass times the maximum absolute value of the charges of the
        target point masses. If the charges of the target point masses is not given, just the charge of the active point
        mass is returned.

        Parameters
        ----------
        active_charge : float
            The charge of the active point mass.
        target_charges : Tuple[float]
            The charges of the target composite point object.

        Returns
        -------
        float
            The multiplicative correction factor.

        Raises
        ------
        AssertionError
            If the active charge is not given as a single float, or if the target charge is not given as a tuple of
            floats.
        AssertionError
            If the charges of the target point masses do not sum up to 0.0.
        """
        if target_charges is not None:
            assert isinstance(active_charge, float) and isinstance(target_charges, tuple)
            assert sum(target_charges) == 0.0
            return active_charge * max(abs(charge_two) for charge_two in target_charges) / self._dipole_charge
        else:
            return active_charge
