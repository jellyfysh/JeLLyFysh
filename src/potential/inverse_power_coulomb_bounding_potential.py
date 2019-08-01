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
"""Module for the InversePowerCoulombBoundingPotential."""
import logging
from typing import MutableSequence, Sequence
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base import vectors
from setting import hypercubic_setting as setting
from .potential import InvertiblePotential
from .inverse_power_potential import InversePowerPotential


# noinspection PyMethodOverriding
class InversePowerCoulombBoundingPotential(InvertiblePotential):
    """
    This class implements the pair potential U_ij = c_i * c_j * k / |r_ij,0|.

    k is a prefactor, c_i and c_j are the charges of the involved units and r_ij,0 = nearest(r_j - r_i) is the
    shortest separation between the units, possible corrected for periodic boundaries. With a correct prefactor, this
    potential can bound the merged image Coulomb potential. This class assumes that i is the active unit.
    Any constant k > 1.5836 is appropriate for a bounding potential in a hypercubic box. This class is only
    implemented for a hypercubic setting.
    """

    def __init__(self, prefactor: float = 1.5837) -> None:
        """
        The constructor of the InversePowerCoulombBoundingPotential class.

        Parameters
        ----------
        prefactor : float, optional
            The prefactor k of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the hypercubic setting is not initialized.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           prefactor=prefactor)
        if not setting.initialized():
            raise ConfigurationError("Potential {0} can only be used in a hypercubic setting."
                                     .format(self.__class__.__name__))
        super().__init__(prefactor=prefactor)
        self._inverse_power_potential = InversePowerPotential(1, prefactor)

    def _potential(self, charge_product: float, separation: Sequence[float]) -> float:
        """Return the potential for the given separation and charge product."""
        return self._inverse_power_potential.potential(charge_product, separation)

    def derivative(self, direction: int, separation: Sequence[float], charge_one: float, charge_two: float) -> float:
        """
        Return the derivative of the potential along a direction.

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
            The derivative.
        """
        return self._inverse_power_potential.derivative(direction, separation, charge_one, charge_two)

    def displacement(self, direction: int, separation: MutableSequence[float], charge_one: float, charge_two: float,
                     potential_change: float) -> float:
        """
        Return the displacement of the active unit i until the cumulative event rate of the potential equals the given
        potential change.

        This method computes the cumulative event rate when displacing the active unit by the system length. For any
        leftover of the potential change, the displacement is then computed separately by using an instance of an
        inverse power potential. The leftover displacement must then be smaller than a system length.

        Parameters
        ----------
        direction : int
            The direction of motion of the active unit i.
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
            The displacement of the active unit i where the cumulative event rate equals the sampled potential change.

        Raises
        ------
        AssertionError
            If the leftover displacement is larger than or equal to a system length.
        """
        charge_product = charge_one * charge_two
        prefactor_product = self._prefactor * charge_product

        potential_zero = self._potential(charge_product,
                                         vectors.copy_vector_with_replaced_component(separation, direction, 0.0))
        separation_half_length = vectors.copy_vector_with_replaced_component(separation, direction,
                                                                             setting.system_length_over_two)
        potential_half_length = self._potential(charge_product, separation_half_length)

        potential_change_per_system_length = abs(potential_zero - potential_half_length)
        multiple_displacement, potential_change = divmod(potential_change, potential_change_per_system_length)
        multiple_displacement *= setting.system_length

        remaining_displacement = self._inverse_power_potential.displacement(direction, list(separation),
                                                                            charge_one, charge_two, potential_change)

        # Displacing active leaf unit by remaining_displacement would carry it into interaction with next periodic image
        # Place active leaf unit so that separation in direction of motion is L/2 and calculate again
        if separation[direction] - remaining_displacement < -setting.system_length_over_two:
            current_potential = self._inverse_power_potential.potential(charge_product, separation)
            if separation[direction] <= 0:
                potential_change -= max(0.0, potential_half_length - current_potential)
            elif prefactor_product > 0:
                potential_change -= potential_zero - current_potential
            else:
                raise RuntimeError('Remaining energy change greater than potential change per system length.')
            remaining_displacement = setting.system_length_over_two + separation[direction]
            separation[direction] = setting.system_length_over_two
            remaining_displacement += self._inverse_power_potential.displacement(direction, list(separation),
                                                                                 charge_one, charge_two,
                                                                                 potential_change)
        assert remaining_displacement < setting.system_length

        return multiple_displacement + remaining_displacement
