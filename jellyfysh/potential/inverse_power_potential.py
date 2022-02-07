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
from typing import MutableSequence, Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base import vectors
from .abstracts import StandardVelocityInvertiblePotential


# noinspection PyMethodOverriding
class InversePowerPotential(StandardVelocityInvertiblePotential):
    """
    This class implements the inverse power pair potential U_ij = c_i * c_j * k / |r_ij| ** p.

    k is a prefactor, c_i and c_j are the charges of the involved units, r_ij = r_j - r_i is the separation between
    the units and p > 0 is the power. This class assumes that i is the active unit.

    This potential only allows for standard velocities (i.e., velocities parallel to one of the cartesian coordinate
    axes going in the positive direction) of the active unit.
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
        self._infinity = float('inf')

    def standard_velocity_derivative(self, direction: int, separation: Sequence[float], charge_one: float,
                                     charge_two: float) -> float:
        """
        Return the space derivative of the potential along a positive direction parallel to one of the cartesian axes
        for the given separation and charges.

        Parameters
        ----------
        direction : int
            The direction of the derivative.
        separation : Sequence[float]
            The separation vector r_ij.
        charge_one : float
            The charge c_i.
        charge_two : float
            The charge c_j.

        Returns
        -------
        float
            The space derivative.
        """
        return (self._power * separation[direction] / vectors.norm(separation) ** self._power_plus_two
                * self._prefactor * charge_one * charge_two)

    def standard_velocity_displacement(self, direction: int, separation: MutableSequence[float], charge_one: float,
                                       charge_two: float, potential_change: float) -> float:
        """
        Return the required displacement in space of the active unit along the positive direction of motion parallel to
        one of the cartesian axes where the cumulative event rate of the potential equals the given potential change.

        Parameters
        ----------
        direction : int
            The direction of motion of the active unit.
        separation : MutableSequence[float]
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
            The required displacement in space of the active unit along its direction of motion where the cumulative
            event rate equals the sampled potential change.
        """
        charge_product = charge_one * charge_two
        prefactor_product = self._prefactor * charge_product
        return (self._displacement_repulsive(direction, charge_product, potential_change, separation)
                if prefactor_product > 0
                else self._displacement_attractive(direction, charge_product, potential_change, separation))

    def potential(self, charge_product: float, separation: Sequence[float]) -> float:
        """
        Return the potential for the given separation and charge product.

        Parameters
        ----------
        charge_product : float
            The charge product c_i * c_j.
        separation : Sequence[float]
            The separation vector r_ij.

        Returns
        -------
        float
            The potential.
        """
        return charge_product * self._prefactor / vectors.norm_sq(separation) ** self._power_over_two

    def _displacement_repulsive(self, direction: int, charge_product: float, potential_change: float,
                                separation: Sequence[float]) -> float:
        """Return the displacement in space for a repulsive potential between the two units."""
        # Active unit is in front of target unit -> travel downhill
        if separation[direction] <= 0.0:
            return self._infinity

        # Active unit behind of target unit -> travel uphill
        maximum_potential = self.potential(charge_product,
                                           vectors.copy_vector_with_replaced_component(separation, direction, 0.0))
        current_potential = self.potential(charge_product, separation)
        if potential_change < maximum_potential - current_potential:
            norm_sq_of_new_separation_vector = (
                    (charge_product * self._prefactor / (current_potential + potential_change)) ** self._two_over_power)
            return vectors.displacement_until_new_norm_sq_component_positive(
                separation, norm_sq_of_new_separation_vector, direction)
        return self._infinity

    def _displacement_attractive(self, direction: int, charge_product: float, potential_change: float,
                                 separation: MutableSequence[float]) -> float:
        """Return the displacement in space for an attractive potential between the two units."""
        current_displacement = 0.0
        # Active unit behind of target unit -> travel downhill
        if separation[direction] > 0.0:
            current_displacement += separation[direction]
            separation[direction] = 0.0
        current_potential = self.potential(charge_product, separation)
        # Active unit in front of target unit -> travel downhill
        if current_potential + potential_change >= 0.0:
            return self._infinity
        norm_sq_of_new_separation_vector = (
                (charge_product * self._prefactor / (current_potential + potential_change)) ** self._two_over_power)
        current_displacement += vectors.displacement_until_new_norm_sq_component_negative(
            separation, norm_sq_of_new_separation_vector, direction)
        return current_displacement
