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
from math import sqrt
from typing import Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base import vectors
from .abstracts import MexicanHatPotential


# noinspection PyMethodOverriding
class LennardJonesPotential(MexicanHatPotential):
    """
    This class implements the Lennard-Jones pair potential U_ij = k * ((s/|r_ij|) ** 12 - (s/|r_ij|)**6).

    k is a prefactor, r_ij = r_j - r_i is the separation between the units and s is the characteristic length.
    This class assumes that i is the active unit. The given potential has the shape of a mexican hat, therefore this
    class inherits from .abstracts.MexicanHatPotential. The equilibrium separation r_0 is s * 2 ** (1/6).
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

        Raises
        ------
        base.exceptions.ConfigurationError
            If the prefactor is not greater than 0.0.
            If the characteristic length is not greater than 0.0.
        """
        self.init_arguments = lambda: {"prefactor": prefactor, "characteristic_length": characteristic_length}
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           prefactor=prefactor, characteristic_length=characteristic_length)
        self._characteristic_length = characteristic_length
        self._prefactor_power_twelve = prefactor * characteristic_length ** 12
        self._prefactor_power_six = -prefactor * characteristic_length ** 6
        if not self._characteristic_length > 0.0:
            raise ConfigurationError("The potential {0} can only be used with an characteristic length which is greater"
                                     " than 0.0!".format(self.__class__.__name__))
        super().__init__(prefactor=prefactor, equilibrium_separation=characteristic_length * 2 ** (1 / 6))

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
        separation_norm_squared = vectors.norm_sq(separation)
        prefactor = (6.0 * self._prefactor_power_six / separation_norm_squared ** 4
                     + 12.0 * self._prefactor_power_twelve / separation_norm_squared ** 7)
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
        separation_norm_squared = vectors.norm_sq(separation)
        return ((6.0 * self._prefactor_power_six / separation_norm_squared ** 4
                 + 12.0 * self._prefactor_power_twelve / separation_norm_squared ** 7)
                * vectors.dot(separation, velocity))

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
        return (self._prefactor_power_six / norm_of_separation ** 6
                + self._prefactor_power_twelve / norm_of_separation ** 12)

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
        sigma_over_r_six = (1.0 + sqrt(1.0 + 4.0 * potential / self._prefactor)) / 2.0
        return self._characteristic_length / sigma_over_r_six ** (1.0 / 6.0)

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
            sigma_over_r_six = (1.0 - sqrt(1.0 + 4.0 * potential / self._prefactor)) / 2.0
            return self._characteristic_length / sigma_over_r_six ** (1.0 / 6.0)
