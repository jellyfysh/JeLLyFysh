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
"""Module for the abstract Estimator class."""
from abc import ABCMeta, abstractmethod
from typing import Sequence, Tuple, Union
from jellyfysh.base import vectors
from jellyfysh.potential import Potential
import jellyfysh.setting as setting


class Estimator(metaclass=ABCMeta):
    """
    Abstract class which determines an upper and an optional lower bound on the factor space derivative along the
    direction of one of the cartesian axes between a minimum and maximum corner of a given region for the possible
    separations.

    The number of possible point masses that are involved in the factor (e.g., pair factors involving two point masses
    or molecular factors involving all point masses of two composite point objects), and the number of active point
    masses within these factors depend on the specific implementation of the estimator.

    The optional lower bound is only relevant if the sign of the factor potential is not fixed, i.e., if it
    is dependant on charges. The estimator decides which charges are used in the potential during the estimation
    of these bounds. The charge_correction_factor method can then be used to correct the estimated bounds by the
    estimator under consideration of the charges of the involved active point masses. If the charges of the target
    point masses are known as well (which is the case for the CellBoundingPotential but not the CellVetoEventHandler),
    these can also be considered.

    On initialization of the estimator, it can be specified whether periodic boundaries should be considered (per
    default they are considered). If so, the estimators only use the primary image separation, that means that the
    separations in the given region are corrected for periodic boundaries.

    Examples for the usage of this class can be found in the CellBoundingPotential and the CellVetoEventHandler.
    """

    def __init__(self, potential: Potential, prefactor: float = 1.0, empirical_bound: float = float('inf'),
                 periodic_boundaries: bool = True) -> None:
        """
        The constructor of the abstract Estimator class.

        The estimator usually relies on some method to sample the separation region. There it will compare all space
        derivatives along the given direction and gain an upper and a lower bound. This bounds can be adjusted with a
        multiplicative prefactor. Also, an empirical bound can be given to exclude physically unreachable too high
        bounds.

        The method self._correct_separation can be used to obtain the separation vector which was possibly corrected
        for periodic boundary conditions.

        Parameters
        ----------
        potential : potential.Potential
            Potential whose derivative is to be bounded.
        prefactor : float, optional
            A constant which gets multiplied to the bounds.
        empirical_bound : float, optional
            If a bound exceeds this value, this value will be returned instead.
        """
        super().__init__()
        self._potential = potential
        self._get_derivative = None
        self._prefactor = prefactor
        self._empirical_bound = empirical_bound
        self._number_charges = self._potential.number_charge_arguments
        if periodic_boundaries:
            self._correct_separation = setting.periodic_boundaries.correct_separation
        else:
            self._correct_separation = lambda separation: None

    @abstractmethod
    def init_arguments(self):
        raise NotImplementedError

    @abstractmethod
    def derivative_bound(self, lower_corner: Sequence[float], upper_corner: Sequence[float], direction: Sequence[float],
                         calculate_lower_bound: bool = False):
        """
        Estimate an upper and an optional lower bound of the potential's space derivative along the given direction
        between a minimum and maximum corner of a given region for the possible separations.

        This method should be extended. This method checks if the corners have the correct dimension and if the lower
        corner is indeed smaller than the upper corner.

        The upper and the optional lower bound should be returned in a sequence. The first entry should be the upper
        bound, the optional second entry the lower bound.

        Note that this class implements the _potential_derivative method that can be used as a replacement for the
        potential's derivative method for convenience. The first argument of the _potential_derivative method is the
        direction. This method then converts the direction into the corresponding velocity, and then calls the
        potential's derivative method. Additional args and kwargs that should be passed to the potential's derivative
        argument should just appear as args and kwargs of the _potential_derivative method.

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
        Sequence[float]
            The sequence of the determined upper bound and the optionally determined lower bound.

        Raises
        ------
        AssertionError
            If the length of the lower or upper corner does not equal the dimension.
            If the lower corner is not smaller than the upper corner in every direction.
        """
        assert len(direction) == setting.dimension
        assert abs(vectors.norm(direction) - 1.0) < 1.0e-13
        assert len(lower_corner) == setting.dimension
        assert len(upper_corner) == setting.dimension
        for i in range(setting.dimension):
            assert lower_corner[i] <= upper_corner[i]

    @property
    def potential(self) -> Potential:
        """
        Return the potential whose derivative gets bounded.

        Returns
        -------
        potential.Potential
            The potential.
        """
        return self._potential

    @abstractmethod
    def charge_correction_factor(self, active_charges: Union[float, Tuple[float]],
                                 target_charges: Union[float, Tuple[float]] = None) -> float:
        """
        Return a multiplicative correction factor on the estimated bounds based on the given charges of the active point
        masses, and optionally based on the charges of the target point masses.

        If there is only a single active and/or target point mass, the single charge should be given as a float.
        Otherwise the charges should be given as tuples.

        Parameters
        ----------
        active_charges : float or Tuple[float]
            The charges of the active point masses.
        target_charges : float or Tuple[float], or None
            The charges of the target point masses.

        Returns
        -------
        float
            The multiplicative correction factor.
        """
        raise NotImplementedError
