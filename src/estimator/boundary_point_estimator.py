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
"""Module for the BoundaryPointEstimator class."""
import logging
from typing import List, Sequence
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from potential import Potential
import setting
from .estimator import Estimator


class BoundaryPointEstimator(Estimator):
    """
    A simple estimator which compares the derivatives of separations on the boundary of the given region.

    This methods works only well, when the potential derivative is monotonic. Also this estimator is implemented
    for a potential which gets exactly one separation in the derivative method.
    """

    def __init__(self, potential: Potential, prefactor: float = 1.5, empirical_bound: float = float('inf'),
                 points_per_side: int = 10) -> None:
        """
        The constructor of the BoundaryPointEstimator class.

        Parameters
        ----------
        potential : potential.Potential
            Potential whose derivative is to be bounded.
        prefactor : float, optional
            A constant which gets multiplied to the bounds.
        empirical_bound : float, optional
            If a bound exceeds this value, this value will be returned instead.
        points_per_side : int, optional
            Specifies, into how many parts the boundary of the given region for the separations in the derivative_bound
            method is cut.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the potential derivative method does not expect exactly one separation.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential.__class__.__name__, prefactor=prefactor, empirical_bound=empirical_bound,
                           points_per_side=points_per_side)
        super().__init__(potential=potential, prefactor=prefactor, empirical_bound=empirical_bound)
        self._points_per_side = points_per_side
        if self._potential.number_separation_arguments != 1:
            raise ConfigurationError("The estimator {0} expects a potential "
                                     "which handles exactly one separation!".format(self.__class__.__name__))

    def derivative_bound(self, lower_corner: Sequence[float], upper_corner: Sequence[float], direction: int,
                         calculate_lower_bound: bool = False) -> List[float]:
        """
        Estimate an upper and an optional lower bound of potential derivative between a minimum and maximum corner of a
        given region for the possible separations.

        This method extends the derivative_bound method of the Estimator class.
        The boundary of the region is covered in a number of steps which is set on initialization. The upper (lower)
        bound is then the maximum (minimum) of the derivatives corrected by the prefactor and the empirical bound.

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
        super().derivative_bound(lower_corner, upper_corner, direction, calculate_lower_bound)
        upper_bound = -float('inf')
        lower_bound = float('inf')
        point_indices = [0 for _ in range(setting.dimension - 1)]
        point_indices[0] = -1
        for _ in range((self._points_per_side + 1) ** (setting.dimension - 1)):
            self._increment(point_indices)
            for d in range(setting.dimension):
                point_indices.insert(d, 0)
                point = [point_indices[i] / self._points_per_side * (upper_corner[i] - lower_corner[i])
                         + lower_corner[i] for i in range(setting.dimension)]
                derivative = self._derivative(direction, point)
                upper_bound = max(upper_bound, derivative)
                lower_bound = min(lower_bound, derivative)
                point_indices.pop(d)

                point_indices.insert(d, self._points_per_side)
                point = [point_indices[i] / self._points_per_side * (upper_corner[i] - lower_corner[i])
                         + lower_corner[i] for i in range(setting.dimension)]
                derivative = self._derivative(direction, point)
                upper_bound = max(upper_bound, derivative)
                lower_bound = min(lower_bound, derivative)
                point_indices.pop(d)

        if upper_bound > 0.0:
            upper_bound *= self._prefactor
        else:
            upper_bound /= self._prefactor

        if lower_bound > 0.0:
            lower_bound /= self._prefactor
        else:
            lower_bound *= self._prefactor

        if calculate_lower_bound:
            return [min(self._empirical_bound, upper_bound), max(-self._empirical_bound, lower_bound)]
        else:
            return [min(self._empirical_bound, upper_bound)]

    def _increment(self, index_list):
        i = 0
        index_list[0] += 1
        while index_list[i] > self._points_per_side:
            index_list[i] = 0
            i += 1
            index_list[i] += 1
