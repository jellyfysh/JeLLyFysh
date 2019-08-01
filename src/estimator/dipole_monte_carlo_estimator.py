# JeLLFysh - a Python application for all-atom event-chain Monte Carlo - https://github.com/jellyfysh
# Copyright (C) 2019 The JeLLyFysh organization
# (see the AUTHORS file for the full list of authors)
#
# This file is part of JeLLyFysh.
#
# JeLLyFysh is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either > version 3 of the License, or (at your option) any
# later version.
#
# JeLLyFysh is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with JeLLyFysh in the LICENSE file.
# If not, see <https://www.gnu.org/licenses/>.
#
# If you use JeLLyFysh in published work, please cite the following reference (see [Hoellmer2019] in References.bib):
# Philipp Hoellmer, Liang Qin, Michael F. Faulkner, A. C. Maggs, Werner Krauth
# JeLLyFysh-Version1.0 -- a Python application for all-atom event-chain Monte Carlo,
# arXiv e-prints: 1907.12502 (2019), https://arxiv.org/abs/1907.12502
#
"""Module for the DipoleMonteCarloEstimator class."""
from copy import copy
import logging
import random
from typing import List, Sequence
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.vectors import random_vector_on_unit_sphere
from potential import Potential
import setting
from .estimator import Estimator


class DipoleMonteCarloEstimator(Estimator):
    """
    Simple estimator which compares the derivatives of separations to dipoles randomly distributed in the given region.

    The estimator places random points as the dipole center and constructs a dipole for a random orientation. The point
    masses within this dipole then have opposite charges with the absolute value being the maximum charge which is set
    in the initialize method. This estimator is most probably only useful when the potential uses a charge.
    This estimator is implemented for a potential which gets exactly one separation in the derivative method.
    """

    def __init__(self, potential: Potential, prefactor: float = 1.5, empirical_bound: float = float('inf'),
                 number_trials: int = 10000, dipole_separation: float = 0.05) -> None:
        """
        The constructor of the InnerPointEstimator class.

        Parameters
        ----------
        potential : potential.Potential
            Potential whose derivative is to be bounded.
        prefactor : float, optional
            A constant which gets multiplied to the bounds.
        empirical_bound : float, optional
            If a bound exceeds this value, this value will be returned instead.
        number_trials : int, optional
            The number of separation samples taken.
        dipole_separation : float, optional
            Separation of the two point masses within the constructed dipole.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the potential derivative method does not expect exactly one separation.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential.__class__.__name__, prefactor=prefactor, empirical_bound=empirical_bound,
                           number_trials=number_trials, dipole_separation=dipole_separation)
        super().__init__(potential=potential, prefactor=prefactor, empirical_bound=empirical_bound)
        self._number_trials = number_trials
        self._dipole_separation = dipole_separation
        self._dipole_separation_over_two = dipole_separation / 2
        if self._potential.number_separation_arguments != 1:
            raise ConfigurationError("The estimator {0} expects a potential "
                                     "which handles exactly one separation!".format(self.__class__.__name__))

    def derivative_bound(self, lower_corner: Sequence[float], upper_corner: Sequence[float], direction: int,
                         calculate_lower_bound: bool = False) -> List[float]:
        """
        Estimate an upper and an optional lower bound of potential derivative between a minimum and maximum corner of a
        given region for the possible separations.

        This method extends the derivative_bound method of the Estimator class.
        The region is covered randomly by creating dipoles at random positions with a random orientation.
        The upper (lower) bound is then the maximum (minimum) of the differences of the derivatives with respect to
        the positions of the two point masses, again corrected by the prefactor and the
        empirical bound.

        Parameters
        ----------
        lower_corner : Sequence[float]
            Lower corner of the region to be estimated. The length of the tuple agrees with system dimensions.
        upper_corner : Tuple[float]
            Upper corner of the region to be estimated. The length of the tuple agrees with system dimensions.
        direction : int
            Direction with respect to which the derivative is taken and the bound is determined.
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
            derivative1 = self._derivative(direction, position1)
            derivative2 = self._derivative(direction, position2)
            upper_bound = max(upper_bound, abs(derivative1 - derivative2))

        upper_bound *= self._prefactor

        if calculate_lower_bound:
            return [min(self._empirical_bound, upper_bound), max(-self._empirical_bound, -upper_bound)]
        else:
            return [min(self._empirical_bound, upper_bound)]
