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
"""Module for the DisplacedEvenPowerPotential class."""
import logging
from typing import Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base import vectors
from .abstracts import MexicanHatPotential


# noinspection PyMethodOverriding
class DisplacedEvenPowerPotential(MexicanHatPotential):
    """
    This class implements the displaced even power pair potential U_ij = k * (|r_ij| - r_0) ** p.

    k is a prefactor, r_ij = r_j - r_i is the separation between the units and p > 0 even is the power.
    This class assumes that i is the active unit. The given potential has the shape of a mexican hat, therefore this
    class inherits from .abstracts.MexicanHatPotential.

    This potential only allows for standard velocities (i.e., velocities parallel to one of the cartesian coordinate
    axes going in the positive direction) of the active unit.
    """

    def __init__(self, equilibrium_separation: float, power: int, prefactor: float = 1.0) -> None:
        """
        The constructor of the DisplacedEvenPowerPotential class.

        Parameters
        ----------
        equilibrium_separation : float
            The absolute value r_0 of the equilibrium separation of the potential.
        power : int
            The power p of the potential.
        prefactor : float, optional
            The prefactor k of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the power p is not larger than 0 or if p is odd.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           equilibrium_separation=equilibrium_separation, power=power, prefactor=prefactor)
        super().__init__(prefactor=prefactor, equilibrium_separation=equilibrium_separation)
        if not (power > 0 and power % 2 == 0):
            raise ConfigurationError("The potential {0} can only be used with "
                                     "a power > 0 divisible by 2!".format(self.__class__.__name__))
        self._equilibrium_separation = equilibrium_separation
        self._equilibrium_separation_squared = equilibrium_separation * equilibrium_separation
        self._power = power
        self._inverse_power = 1.0 / power

    def standard_velocity_derivative(self, direction: int, separation: Sequence[float]) -> float:
        """
        Return the space derivative of the potential along a positive direction parallel to one of the cartesian axes
        for the given separation.

        Parameters
        ----------
        direction : int
            The direction of the derivative.
        separation : Sequence[float]
            The separation vector r_ij.

        Returns
        -------
        float
            The space derivative.
        """
        norm_of_separation = vectors.norm(separation)
        return (- self._power * self._prefactor
                * (norm_of_separation - self._equilibrium_separation) ** (self._power - 1)
                * separation[direction] / norm_of_separation)

    def _potential(self, separation: Sequence[float]) -> float:
        """
        Return the potential for the given separation.

        Parameters
        ----------
        separation : Sequence[float]
            The separation vector r_ij.

        Returns
        -------
        float
            The potential.
        """
        distance_from_minimum = vectors.norm(separation) - self._equilibrium_separation
        return self._prefactor * distance_from_minimum ** self._power

    def _invert_potential_inside_minimum(self, potential: float) -> float:
        """
        Return the absolute value of the separation r where the potential equals the given potential with r <= r_0.

        Parameters
        ----------
        potential : float
            The wanted potential.

        Returns
        -------
        float:
            The absolute value of the separation.
        """
        return self._equilibrium_separation - (potential / self._prefactor) ** self._inverse_power

    def _invert_potential_outside_minimum(self, potential: float) -> float:
        """
        Return the absolute value of the separation r where the potential equals the given potential with r >= r_0.

        Parameters
        ----------
        potential : float
            The wanted potential.

        Returns
        -------
        float:
            The absolute value of the separation.
        """
        return self._equilibrium_separation + (potential / self._prefactor) ** self._inverse_power
