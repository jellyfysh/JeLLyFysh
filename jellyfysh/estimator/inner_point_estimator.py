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
"""Module for the InnerPointEstimator class."""
import logging
from typing import List, Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.potential import Potential
from .estimator import Estimator


class InnerPointEstimator(Estimator):
    """
    Estimator which compares the space derivatives along the direction of one of the cartesian axes for separations
    evenly distributed in the given region.

    This estimator is implemented for a factor potential between two point masses (of which one is active)
    which gets exactly one separation in the derivative method. The potential may further expect 0 or 2 charges. For the
    latter case, these charges are the ones of the active and the target point mass. The charge of the active point mass
    is set to 1.0 during the estimation of the bounds. The charge of the target point mass can be set on initialization.

    Per default, this estimator only uses the primary image separation, that means that the separations in the given
    region are corrected for periodic boundaries.
    """

    def __init__(self, potential: Potential, prefactor: float = 1.5, empirical_bound: float = float('inf'),
                 points_per_side: int = 10, target_charge: float = None, periodic_boundaries: bool = True) -> None:
        """
        The constructor of the InnerPointEstimator class.

        If the given region has the dimension D, then the total number of compared points is simply
        points_per_side ** D.

        If the potential expects two charges and the target_charge argument is not given, the charge 1.0 is used during
        the estimation of the bounds.

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
        target_charge : float, optional
            The charge of the target point mass.
        periodic_boundaries : bool
            Whether the separations in the given region should be corrected for periodic boundaries.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the potential derivative method does not expect exactly one separation.
            If the potential derivative method does not expect 0 or 2 charges.
            If a value for the target charge was specified but the potential derivative method does not expect any.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential.__class__.__name__, prefactor=prefactor, empirical_bound=empirical_bound,
                           points_per_side=points_per_side, target_charge=target_charge,
                           periodic_boundaries=periodic_boundaries)
        super().__init__(potential=potential, prefactor=prefactor, empirical_bound=empirical_bound,
                         periodic_boundaries=periodic_boundaries)
        self._points_per_side = points_per_side
        if self._potential.number_separation_arguments != 1:
            raise ConfigurationError("The estimator {0} expects a potential "
                                     "which handles exactly one separation!".format(self.__class__.__name__))

        self._target_charge = target_charge
        if self._potential.number_charge_arguments == 0:
            if target_charge is not None:
                raise ConfigurationError("The estimator {0} was initialized with a target charge which is not None,"
                                         " but its potential {1} expects no charges."
                                         .format(self.__class__.__name__, self._potential.__class__.__name__))
            self._charges = tuple()
            self.charge_correction_factor = self._charge_correction_factor_potential_no_charges
        elif self._potential.number_charge_arguments == 2:
            if target_charge is None:
                self._target_charge = 1.0
                self._charges = (1.0, 1.0)
            else:
                self._charges = (1.0, target_charge)
        else:
            raise ConfigurationError("The estimator {0} can only be used with a potential that expects 0 or 2 charges."
                                     .format(self.__class__.__name__))

    def derivative_bound(self, lower_corner: Sequence[float], upper_corner: Sequence[float], direction: int,
                         calculate_lower_bound: bool = False) -> List[float]:
        """
        Estimate an upper and an optional lower bound of the potential's space derivative along the given direction
        between a minimum and maximum corner of a given region for the possible separations.

        This method extends the derivative_bound method of the Estimator class.

        The region is covered evenly in each direction in a number of steps which is set on initialization. The upper
        (lower) bound is then the maximum (minimum) of the derivatives corrected by the prefactor and the empirical
        bound.

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
        super().derivative_bound(lower_corner, upper_corner, direction, calculate_lower_bound)
        upper_bound = -float('inf')
        lower_bound = float('inf')
        for ix in range(self._points_per_side + 1):
            px = lower_corner[0] + (upper_corner[0] - lower_corner[0]) * ix / self._points_per_side
            for iy in range(self._points_per_side + 1):
                py = lower_corner[1] + (upper_corner[1] - lower_corner[1]) * iy / self._points_per_side
                for iz in range(self._points_per_side + 1):
                    pz = lower_corner[2] + (upper_corner[2] - lower_corner[2]) * iz / self._points_per_side
                    separation = [px, py, pz]
                    self._correct_separation(separation)
                    derivative = self._potential_derivative(direction, separation, *self._charges)
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

    def charge_correction_factor(self, active_charge: float, target_charge: float = 1.0) -> float:
        """
        Return a multiplicative correction factor on the estimated bounds based on the given charges of the active point
        mass and the target point mass.

        This method is used when the potential derivative method expects two charges. Both charges should be given as a
        float. Since the factor derivative depends on the product of the charges of the point masses, this product is
        returned here.

        Parameters
        ----------
        active_charge : float
            The charge of the active point mass.
        target_charge : float
            The charge of the target point mass.

        Returns
        -------
        float
            The multiplicative correction factor.

        Raises
        ------
        AssertionError
            If the active or the target charge are given as tuples and not as a single float.
        """
        assert isinstance(active_charge, float) and isinstance(target_charge, float)
        return active_charge * target_charge / self._target_charge

    # noinspection PyMethodMayBeStatic
    def _charge_correction_factor_potential_no_charges(self, active_charge: float, target_charge: float = 1.0) -> float:
        """
        Return a multiplicative correction factor on the estimated bounds based on the given charges of the active point
        mass and the target point mass.

        This method is used instead of charge_correction_factor when the potential derivative method expects zero
        charges. Both charges should be 1.0 and this method just returns 1.0.

        Parameters
        ----------
        active_charge : float
            The charge of the active point mass.
        target_charge : float, optional
            The charge of the target point mass.

        Returns
        -------
        float
            The multiplicative correction factor.

        Raises
        ------
        AssertionError
            If not both charges are 1.0.
        """
        assert active_charge == 1.0 and target_charge == 1.0
        return 1.0
