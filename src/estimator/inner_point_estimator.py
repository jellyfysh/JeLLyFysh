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
"""Module for the InnerPointEstimator class."""
import logging
from typing import List, Sequence
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from potential import Potential
from .estimator import Estimator


class InnerPointEstimator(Estimator):
    """
    A simple estimator which compares the derivatives of separations evenly distributed in the given region.

    This estimator is implemented for a potential which gets exactly one separation in the derivative method.
    """

    def __init__(self, potential: Potential, empirical_bound: float = float('inf'), prefactor: float = 1.5,
                 points_per_side: int = 10) -> None:
        """
        The constructor of the InnerPointEstimator class.

        If the given region has the dimension D, then the total number of compared points is simply
        points_per_side ** D.

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
                           potential=potential.__class__.__name__, empirical_bound=empirical_bound, prefactor=prefactor,
                           points_per_side=points_per_side)
        super().__init__(potential=potential, empirical_bound=empirical_bound, prefactor=prefactor)
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
        The region is covered evenly in each direction in a number of steps which is set on initialization.
        The upper (lower) bound is then the maximum (minimum) of the derivatives corrected by the prefactor and the
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
        super().derivative_bound(lower_corner, upper_corner, direction, calculate_lower_bound)
        upper_bound = -float('inf')
        lower_bound = float('inf')
        for ix in range(self._points_per_side + 1):
            px = lower_corner[0] + (upper_corner[0] - lower_corner[0]) * ix / self._points_per_side
            for iy in range(self._points_per_side + 1):
                py = lower_corner[1] + (upper_corner[1] - lower_corner[1]) * iy / self._points_per_side
                for iz in range(self._points_per_side + 1):
                    pz = lower_corner[2] + (upper_corner[2] - lower_corner[2]) * iz / self._points_per_side
                    derivative = self._derivative(direction, [px, py, pz])
                    upper_bound = max(upper_bound, derivative)
                    lower_bound = min(lower_bound, derivative)

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
