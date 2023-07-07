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
"""Module for the InversePowerPotential class."""
import logging
from math import inf, sqrt
from typing import Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base import vectors
from .potential import InvertiblePotential


# noinspection PyMethodOverriding
class InversePowerPotential(InvertiblePotential):
    """
    This class implements the inverse power pair potential U_ij = c_i * c_j * k / |r_ij| ** p.

    k is a prefactor, c_i and c_j are the charges of the involved units, r_ij = r_j - r_i is the separation between
    the units and p > 0 is the power. This class assumes that i is the active unit.
    """

    def __init__(self, power: float, prefactor: float) -> None:
        """
        The constructor of the InversePowerPotential.

        Parameters
        ----------
        power : float
            The power p of the potential.
        prefactor : float
            The prefactor k of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the power p is not larger than 0.
        """
        self.init_arguments = lambda: {"power": power, "prefactor": prefactor}
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           power=power, prefactor=prefactor)
        super().__init__(prefactor=prefactor)
        if not power > 0.0:
            raise ConfigurationError("Give a power > 0.0 as the power for the potential {0}."
                                     .format(self.__class__.__name__))
        self._power = power
        self._two_over_power = 2.0 / self._power
        self._power_over_two = self._power / 2.0
        self._power_plus_two = self._power + 2

    def init_arguments(self):
        raise NotImplementedError

    def gradient(self, separation: Sequence[float], charge_one: float, charge_two: float) -> Sequence[float]:
        """
        Return the gradient of the potential evaluated at the given separation and for the given charges.

        Parameters
        ----------
        separation : Sequence[float]
            The separation vector r_ij.
        charge_one : float
            The charge c_i.
        charge_two : float
            The charge c_j.

        Returns
        -------
        Sequence[float]
            The gradient with respect to the position r_i of the active unit.
        """
        prefactor = (self._power / vectors.norm(separation) ** self._power_plus_two * self._prefactor
                     * charge_one * charge_two)
        return [prefactor * s for s in separation]

    def derivative(self, velocity: Sequence[float], separation: Sequence[float], charge_one: float,
                   charge_two: float) -> float:
        """
        Return the directional time derivative along a given velocity vector of the active unit for the given
        separation and charges.

        Parameters
        ----------
        velocity : Sequence[float]
            The velocity of the active unit along which the directional derivative is computed.
        separation : Sequence[float]
            The separation vector r_ij.
        charge_one : float
            The charge c_i.
        charge_two : float
            The charge c_j.

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
        return (self._power / vectors.norm(separation) ** self._power_plus_two
                * self._prefactor * charge_one * charge_two * vectors.dot(separation, velocity))

    def displacement(self, velocity: Sequence[float], separation: Sequence[float], charge_one: float,
                     charge_two: float, potential_change: float) -> float:
        """
        Return the required time displacement of the active unit along its velocity where the cumulative event rate of
        the potential equals the given potential change.

        Parameters
        ----------
        velocity : Sequence[float]
            The velocity of the active unit.
        separation : Sequence[float]
            The separation vector r_ij.
        charge_one : float
            The charge c_i.
        charge_two : float
            The charge c_j.
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
        prefactor_product = self._prefactor * charge_one * charge_two
        return (self._displacement_repulsive(prefactor_product, velocity, separation, potential_change)
                if prefactor_product > 0.0
                else self._displacement_attractive(prefactor_product, velocity, separation, potential_change))

    def potential(self, prefactor_product: float, norm_of_separation: float) -> float:
        """
        Return the potential for the given absolute value of the separation and prefactor product.

        Parameters
        ----------
        prefactor_product : float
            The prefactor product k * c_i * c_j.
        norm_of_separation : float
            The absolute value |r_ij| of the separation vector.

        Returns
        -------
        float
            The potential.
        """
        return prefactor_product / norm_of_separation ** self._power

    def _displacement_repulsive(self, prefactor_product: float, velocity: Sequence[float], separation: Sequence[float],
                                potential_change: float) -> float:
        """Return the time displacement displacement for a repulsive potential between the two units."""
        separation_dot_velocity = vectors.dot(separation, velocity)
        # Norm of separation becomes bigger -> travel downhill.
        if separation_dot_velocity <= 0.0:
            return inf

        # Norm of separation becomes smaller -> travel uphill.
        velocity_squared = vectors.norm_sq(velocity)
        separation_squared = vectors.norm_sq(separation)
        max_displacement = separation_dot_velocity / velocity_squared
        minimum_separation_squared = separation_squared - max_displacement * max_displacement * velocity_squared
        maximum_potential = self.potential(prefactor_product, sqrt(minimum_separation_squared))
        current_potential = self.potential(prefactor_product, sqrt(separation_squared))
        if potential_change < maximum_potential - current_potential:
            norm_sq_of_new_separation = (
                    (prefactor_product / (current_potential + potential_change)) ** self._two_over_power)
            sqrt_term = (separation_dot_velocity * separation_dot_velocity
                         - velocity_squared * (separation_squared - norm_sq_of_new_separation))
            return (separation_dot_velocity - sqrt(sqrt_term)) / velocity_squared
        return inf

    def _displacement_attractive(self, prefactor_product: float, velocity: Sequence[float], separation: Sequence[float],
                                 potential_change: float) -> float:
        """Return the time displacement for an attractive potential between the two units."""
        velocity_squared = vectors.norm_sq(velocity)
        separation_squared = vectors.norm_sq(separation)
        separation_dot_velocity = vectors.dot(separation, velocity)
        total_displacement = 0.0
        # Norm of separation becomes smaller -> travel downhill.
        if separation_dot_velocity > 0.0:
            displacement = separation_dot_velocity / velocity_squared
            separation_squared -= displacement * displacement * velocity_squared
            separation_dot_velocity -= displacement * velocity_squared
            total_displacement += displacement

        # Norm of separation becomes bigger -> travel uphill.
        current_potential = self.potential(prefactor_product, sqrt(separation_squared))
        if current_potential + potential_change >= 0.0:
            return inf
        norm_sq_of_new_separation = (
                (prefactor_product / (current_potential + potential_change)) ** self._two_over_power)
        sqrt_term = (separation_dot_velocity * separation_dot_velocity
                     - velocity_squared * (separation_squared - norm_sq_of_new_separation))
        total_displacement += (separation_dot_velocity + sqrt(sqrt_term)) / velocity_squared
        return total_displacement
