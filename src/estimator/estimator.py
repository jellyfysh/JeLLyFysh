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
"""Module for the abstract Estimator class."""
from abc import ABCMeta, abstractmethod
from typing import Sequence
from base.initializer import Initializer
from potential import Potential
import setting


class Estimator(Initializer, metaclass=ABCMeta):
    """
    Abstract class which determines an upper and an optional lower bound on the factor derivative between a minimum
    and maximum corner of a given region for the possible separations.

    The optional lower bound is only relevant if the sign of the potential is not fixed, i.e. if the potential includes
    is dependant on charges. Then the estimator estimates these bounds with the maximum relevant charge in the system.
    Examples for the usage of this class can be found in CellBoundingPotential and CellVetoEventHandler.

    This estimator only uses the primary image separation, that means that the separations in the given region are
    corrected for periodic boundaries.
    """

    def __init__(self, potential: Potential, prefactor: float = 1.0, empirical_bound: float = float('inf')) -> None:
        """
        The constructor of the abstract Estimator class.

        The estimator usually relies on some method to sample the separation region. There it will compare all
        derivatives and gain an upper and a lower bound. This bounds can be adjusted with a multiplicative prefactor.
        Also an empirical bound can be given to exclude physically unreachable too high bounds.

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
        self._derivative = None
        self._number_charges = self._potential.number_charge_arguments

    def initialize(self, max_charge: float) -> None:
        """
        Initializes the estimator with the maximum relevant charge in the system.

        Each charge argument the derivative method of the potential expects, will be set to this charge. To do this
        automatically, use the self._derivative method instead of the self._potential.derivative method. The other
        arguments (except the charges) stay the same. This method will also correct all the separations for periodic
        boundaries.
        Extends the initialize method of the Initializer class. This method should be called once in the beginning of
        the run. Only after a call of this method, other public methods of this class can be called without raising an
        error.

        Parameters
        ----------
        max_charge : float
            The maximum relevant charge in the system for the potential.
        """
        super().initialize()
        self._derivative = self._convert_derivative_method(self._potential.derivative, max_charge)

    @abstractmethod
    def derivative_bound(self, lower_corner: Sequence[float], upper_corner: Sequence[float], direction: int,
                         calculate_lower_bound: bool = False):
        """
        Estimate an upper and an optional lower bound of potential derivative between a minimum and maximum corner of a
        given region for the possible separations.

        This method should be extended. Here we check if the corners have the correct dimension and if the lower corner
        is indeed smaller than the upper corner.
        The upper and the optional lower bound should be returned in a sequence. The first entry should be the upper
        bound, the optional second entry the lower bound.

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
        Sequence[float]
            The sequence of the determined upper bound and the optionally determined lower bound.

        Raises
        ------
        AssertionError
            If the length of the lower or upper corner does not equal the dimension.
            If the lower corner is not smaller than the upper corner in every direction.
        """
        assert len(lower_corner) == setting.dimension
        assert len(upper_corner) == setting.dimension
        for i in range(setting.dimension):
            assert lower_corner[i] <= upper_corner[i]

    def _convert_derivative_method(self, derivative_method, max_charge):
        """
        Convert the derivative method, so that each separation gets corrected for periodic boundaries and that
        each charge is set to the maximum charge.
        """
        # noinspection PyMissingTypeHints
        def _get_derivative(direction, *separations):
            for separation in separations:
                setting.periodic_boundaries.correct_separation(separation)
            return derivative_method(direction, *separations, *[max_charge for _ in range(self._number_charges)])
        return _get_derivative

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
