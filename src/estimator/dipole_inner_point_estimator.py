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
"""Module for the DipoleInnerPointEstimator class."""
from copy import copy
import logging
from typing import List, Sequence
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base import vectors
from potential import Potential
import setting
from .estimator import Estimator


class DipoleInnerPointEstimator(Estimator):
    """
    A simple estimator which compares the derivatives of separations to dipoles evenly distributed in the given region.

    The estimator places at each evenly distributed point a dipole center. At this point, it calculates the gradient of
    the derivative and aligns the dipole along the direction of this gradient. The point masses within this dipole then
    have opposite charges with the absolute value being the maximum charge which is set in the initialize method.
    This estimator is most probably only useful when the potential uses a charge.
    This estimator is implemented for a potential which gets exactly one separation in the derivative method.
    """

    def __init__(self, potential: Potential, dipole_separation: float, empirical_bound: float = float('inf'),
                 prefactor: float = 1.5, points_per_side: int = 3) -> None:
        """
        The constructor of the DipoleInnerPointEstimator class.

        Parameters
        ----------
        potential : potential.Potential
            Potential whose derivative is to be bounded.
        prefactor : float, optional
            A constant which gets multiplied to the bounds.
        empirical_bound : float, optional
            If a bound exceeds this value, this value will be returned instead.
        points_per_side : int, optional
            Specifies, into how many parts in each direction the given region for the separations in the
            derivative_bound method  is cut.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the potential derivative method does not expect exactly one separation.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential.__class__.__name__, dipole_separation=dipole_separation,
                           empirical_bound=empirical_bound, prefactor=prefactor, points_per_side=points_per_side)
        super().__init__(potential=potential, empirical_bound=empirical_bound, prefactor=prefactor)
        self._max_index_per_side = points_per_side - 1
        self._dipole_separation = dipole_separation
        self._dipole_separation_over_two = self._dipole_separation / 2
        if self._potential.number_separation_arguments != 1:
            raise ConfigurationError("The estimator {0} expects a potential "
                                     "which handles exactly one separation!".format(self.__class__.__name__))

    def derivative_bound(self, lower_corner: Sequence[float], upper_corner: Sequence[float], direction: int,
                         calculate_lower_bound: bool = False) -> List[float]:
        """
        Estimate an upper and an optional lower bound of potential derivative between a minimum and maximum corner of a
        given region for the possible separations.

        This method extends the derivative_bound method of the Estimator class.
        The region is covered evenly in each direction in a number of steps which is set on initialization. At each
        point, this estimator creates a dipole with the orientation aligned to the gradient of the potential's
        derivative at this point. The upper (lower) bound is then the maximum (minimum) of the differences of the
        derivatives with respect to the positions of the two point masses, again corrected by the prefactor and the
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
        point_indices = [0] * setting.dimension
        delta = self._dipole_separation / 20
        for _ in range((self._max_index_per_side + 1) ** setting.dimension):
            dipole_center = [lower_corner[d] + (upper_corner[d] - lower_corner[d]) * point_indices[d] /
                             self._max_index_per_side for d in range(setting.dimension)]

            gradient = [0.0] * setting.dimension
            for d in range(setting.dimension):
                position1 = dipole_center[:]
                position2 = dipole_center[:]
                position1[d] += delta
                position2[d] -= delta
                gradient[d] = ((self._derivative(direction, position1) - self._derivative(direction, position2))
                               / delta / 2)
            gradient = vectors.normalize(gradient)
            half_dipole_separation_vector = [a * self._dipole_separation_over_two for a in gradient]

            position1 = [dipole_center[d] + half_dipole_separation_vector[d] for d in range(setting.dimension)]
            position2 = [dipole_center[d] - half_dipole_separation_vector[d] for d in range(setting.dimension)]
            derivative1 = self._derivative(direction, position1)
            derivative2 = self._derivative(direction, position2)
            upper_bound = max(upper_bound, abs(derivative1 - derivative2))

            point_indices = self._next_inner_point(point_indices)

        upper_bound *= self._prefactor
        if calculate_lower_bound:
            return [min(self._empirical_bound, upper_bound), max(-self._empirical_bound, -upper_bound)]
        else:
            return [min(self._empirical_bound, upper_bound)]

    def _next_inner_point(self, point):
        point += [0]
        i = None
        for i, component in enumerate(point):
            if component < self._max_index_per_side:
                break
        for j in range(i):
            point[j] = 0
        point[i] += 1
        return point[:-1]
