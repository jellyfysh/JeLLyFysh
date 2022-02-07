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
"""Module for the LennardJonesPotential class."""
import logging
from typing import Sequence
from jellyfysh.base.logging import log_init_arguments
from .abstracts import MexicanHatPotential
from .inverse_power_potential import InversePowerPotential


# noinspection PyMethodOverriding
class LennardJonesPotential(MexicanHatPotential):
    """
    This class implements the Lennard-Jones pair potential U_ij = k * ((s/|r_ij|) ** 12 - (s/|r_ij|)**6).

    k is a prefactor, r_ij = r_j - r_i is the separation between the units and s is the characteristic length.
    This class assumes that i is the active unit. The given potential has the shape of a mexican hat, therefore this
    class inherits from .abstracts.MexicanHatPotential. The equilibrium separation r_0 is s * 2 ** (1/6).

    This potential only allows for standard velocities (i.e., velocities parallel to one of the cartesian coordinate
    axes going in the positive direction) of the active unit.
    """

    def __init__(self, prefactor: float = 1.0, characteristic_length: float = 1.0):
        """
        The constructor of the LennardJonesPotential class.

        Parameters
        ----------
        prefactor : float, optional
            The prefactor k of the potential.
        characteristic_length : float, optional
            The characteristic length s of the potential.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           prefactor=prefactor, characteristic_length=characteristic_length)
        super().__init__(prefactor=prefactor, equilibrium_separation=characteristic_length * 2 ** (1 / 6))
        self._six_power_potential = InversePowerPotential(power=6, prefactor=-prefactor *
                                                          characteristic_length ** 6)
        self._twelve_power_potential = InversePowerPotential(power=12, prefactor=prefactor *
                                                             characteristic_length ** 12)
        self._characteristic_length = characteristic_length

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
        return (self._six_power_potential.standard_velocity_derivative(direction, separation, 1.0, 1.0)
                + self._twelve_power_potential.standard_velocity_derivative(direction, separation, 1.0, 1.0))

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
        return (self._six_power_potential.potential(1.0, separation)
                + self._twelve_power_potential.potential(1.0, separation))

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
        sigma_over_r_six = (1 + (1 + 4 * potential / self._prefactor) ** 0.5) / 2
        return self._characteristic_length / sigma_over_r_six ** (1 / 6)

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
        if potential >= 0.0:
            return float('inf')
        else:
            sigma_over_r_six = (1 - (1 + 4 * potential / self._prefactor) ** 0.5) / 2
            return self._characteristic_length / sigma_over_r_six ** (1 / 6)
