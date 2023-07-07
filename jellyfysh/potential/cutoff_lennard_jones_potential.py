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
"""Module for the CutoffLennardJonesPotential class."""
import logging
from math import inf, sqrt
from typing import Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base import vectors
from .abstracts import MexicanHatPotential


# noinspection PyMethodOverriding
class CutoffLennardJonesPotential(MexicanHatPotential):
    """
    This class implements the Lennard-Jones pair potential U_ij = k * ((s/|r_ij|) ** 12 - (s/|r_ij|)**6) that is cutoff
    at some radius rc (U_ij(|s| > rc) = 0).

    k is a prefactor, r_ij = r_j - r_i is the separation between the units and s is the characteristic length.
    This class assumes that i is the active unit. The given potential has the shape of a mexican hat, therefore this
    class inherits from .abstracts.MexicanHatPotential. The equilibrium separation r_0 is s * 2 ** (1/6).
    """

    def __init__(self, cutoff: float, prefactor: float = 1.0, characteristic_length: float = 1.0):
        """
        The constructor of the CutoffLennardJonesPotential class.

        Parameters
        ----------
        prefactor : float, optional
            The prefactor k of the potential.
        characteristic_length : float, optional
            The characteristic length s of the potential.
        cutoff: float, optional
            The cutoff rc of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the prefactor is not greater than 0.0.
            If the characteristic length is not greater than 0.0.
            If the cutoff is not greater than 0.0.
            If the cutoff is not greater than the equilibrium separation [characteristic_length * 2 ** (1 / 6)].
        """
        self.init_arguments = lambda: {"prefactor": prefactor, "characteristic_length": characteristic_length,
                                       "cutoff": cutoff}
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           prefactor=prefactor, characteristic_length=characteristic_length, cutoff=cutoff)
        self._characteristic_length = characteristic_length
        self._prefactor_power_twelve = prefactor * characteristic_length ** 12
        self._prefactor_power_six = -prefactor * characteristic_length ** 6
        if cutoff is not None:
            if not cutoff > 0.0:
                raise ConfigurationError("The potential {0} can only be used with a cutoff which is greater than 0.0!"
                                         .format(self.__class__.__name__))
            self._cutoff = cutoff
        self._cutoff_squared = self._cutoff * self._cutoff
        if not self._characteristic_length > 0.0:
            raise ConfigurationError("The potential {0} can only be used with an characteristic length which is greater"
                                     " than 0.0!".format(self.__class__.__name__))
        super().__init__(prefactor=prefactor, equilibrium_separation=characteristic_length * 2 ** (1 / 6))
        if not self._cutoff > self._equilibrium_separation:
            raise ConfigurationError("The potential {0} can only be used with a cutoff that is larger than the "
                                     "equilibrium separation of the Lennard-Jones potential (which is given by"
                                     "characteristic_length * 2 ** (1/6)).".format(self.__class__.__name__))

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
        if separation_norm_squared > self._cutoff_squared:
            return [0.0 for _ in range(len(separation))]
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
        if separation_norm_squared > self._cutoff_squared:
            return 0.0
        return ((6.0 * self._prefactor_power_six / separation_norm_squared ** 4
                 + 12.0 * self._prefactor_power_twelve / separation_norm_squared ** 7)
                * vectors.dot(separation, velocity))

    def displacement(self, velocity: Sequence[float], separation: Sequence[float], potential_change: float) -> float:
        """
        Return the required time displacement of the active unit along its velocity where the cumulative event rate of
        the potential equals the given potential change.

        This method determines whether the movement of the active unit lets the separation become smaller ('behind') or
        bigger ('front'), and whether the the absolute value of the separation is larger than rc.

        Parameters
        ----------
        velocity : Sequence[float]
            The velocity of the active unit.
        separation : Sequence[float]
            The separation vector r_ij.
        potential_change : float
            The sampled potential change.

        Returns
        -------
        float
            The required time displacement of the active unit along its velocity where the cumulative event rate equals
            the sampled potential change.

        Raises
        ------
        AssertionError
            If the velocity is zero.
        """
        assert any(entry != 0.0 for entry in velocity)
        velocity_squared = vectors.norm_sq(velocity)
        separation_squared = vectors.norm_sq(separation)
        separation_dot_velocity = vectors.dot(separation, velocity)
        # Active unit is outside of the cutoff.
        if separation_squared > self._cutoff_squared:
            # Norm of separation becomes bigger.
            if separation_dot_velocity <= 0.0:
                return inf
            # Norm of separation becomes smaller.
            else:
                sqrt_term = (separation_dot_velocity * separation_dot_velocity
                             - velocity_squared * (separation_squared - self._cutoff_squared))
                # Active unit can reach cutoff potential.
                if sqrt_term >= 0.0:
                    displacement = (separation_dot_velocity - sqrt(sqrt_term)) / velocity_squared
                    assert separation_dot_velocity - displacement * velocity_squared > 0.0
                    next_displacement = self._displacement_behind_outside_sphere(
                        velocity_squared, self._cutoff_squared,
                        separation_dot_velocity - displacement * velocity_squared, potential_change)
                    total_displacement = displacement + next_displacement
                    final_separation_squared = (separation_squared - 2.0 * separation_dot_velocity * total_displacement
                                                + velocity_squared * total_displacement * total_displacement)
                    if final_separation_squared > self._cutoff_squared:
                        return inf
                    else:
                        return total_displacement
                # Active unit cannot reach cutoff potential.
                else:
                    return inf
        # Active unit is inside of the cutoff.
        else:
            displacement = super().displacement(velocity, separation, potential_change)
            final_separation_squared = (separation_squared - 2.0 * separation_dot_velocity * displacement
                                        + velocity_squared * displacement * displacement)
            if final_separation_squared > self._cutoff_squared:
                return inf
            else:
                return displacement

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
        if norm_of_separation > self._cutoff:
            return 0.0
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
