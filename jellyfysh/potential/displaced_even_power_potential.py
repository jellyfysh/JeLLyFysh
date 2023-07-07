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

    Here, k is a prefactor, r_ij = r_j - r_i is the separation between the units, and p > 0 even is the power.
    This class assumes that i is the active unit. The given potential has the shape of a mexican hat, therefore this
    class inherits from .abstracts.MexicanHatPotential.
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
        self.init_arguments = lambda: {"equilibrium_separation": equilibrium_separation, "power": power,
                                       "prefactor": prefactor}
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           equilibrium_separation=equilibrium_separation, power=power, prefactor=prefactor)
        if not (power > 0 and power % 2 == 0):
            raise ConfigurationError("The potential {0} can only be used with "
                                     "a power > 0 divisible by 2!".format(self.__class__.__name__))
        self._power = power
        self._inverse_power = 1.0 / power
        super().__init__(prefactor=prefactor, equilibrium_separation=equilibrium_separation)

    def init_arguments(self):
        raise NotImplementedError

    def gradient(self, separation: Sequence[float]) -> Sequence[float]:
        """
        Return the gradient of the potential evaluated at the given separation.

        Parameters
        ----------
        separation : Sequence[float]
            The separation vector r_ij.

        Returns
        -------
        Sequence[float]
            The gradient with respect to the position r_i of the active unit.
        """
        norm_of_separation = vectors.norm(separation)
        prefactor = (-self._power * self._prefactor
                     * (norm_of_separation - self._equilibrium_separation) ** (self._power - 1) / norm_of_separation)
        return [prefactor * s for s in separation]

    def derivative(self, velocity: Sequence[float], separation: Sequence[float]) -> float:
        """
        Return the directional time derivative along a given velocity vector of the active unit for the given
        separation.

        Parameters
        ----------
        velocity : Sequence[float]
            The velocity of the active unit along which the directional derivative is computed.
        separation : Sequence[float]
            The separation vector r_ij.

        Returns
        -------
        float
            The directional time derivative.

        Raises
        ------
        AssertionError
            If the velocity is zero.
        """
        assert any(entry != 0.0 for entry in velocity)
        norm_of_separation = vectors.norm(separation)
        return (-self._power * self._prefactor
                * (norm_of_separation - self._equilibrium_separation) ** (self._power - 1)
                / norm_of_separation) * vectors.dot(separation, velocity)

    def _potential(self, norm_of_separation: float) -> float:
        """
        Return the potential for the given absolute value of the separation.

        Parameters
        ----------
        norm_of_separation : float
            The absolute value |r_ij| of the separation vector.

        Returns
        -------
        float
            The potential.
        """
        return self._prefactor * (norm_of_separation - self._equilibrium_separation) ** self._power

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
