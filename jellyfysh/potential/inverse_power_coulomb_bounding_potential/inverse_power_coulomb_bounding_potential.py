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
"""Module for the InversePowerCoulombBoundingPotential."""
import logging
from typing import MutableSequence, Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.vectors import permutation_3d
from jellyfysh.potential.abstracts import StandardVelocityInvertiblePotential
from jellyfysh.setting import hypercubic_setting as setting
# noinspection PyUnresolvedReferences
from ._inverse_power_coulomb_bounding_potential import ffi, lib

# Directly import C functions used in performance relevant parts of the code.
_lib_derivative = lib.derivative
_lib_displacement = lib.displacement


# noinspection PyMethodOverriding
class InversePowerCoulombBoundingPotential(StandardVelocityInvertiblePotential):
    """
    This class implements the pair potential U_ij = c_i * c_j * k / |r_ij,0|.

    Here, k is a prefactor, c_i and c_j are the charges of the involved units and r_ij,0 = nearest(r_j - r_i) is the
    shortest separation between the units, possibly corrected for periodic boundaries. With a correct prefactor, this
    potential can bound the merged image coulomb potential. This class assumes that i is the active unit.
    Any constant k > 1.5836 is appropriate for a bounding potential in a hypercubic box. This class is only
    implemented for a hypercubic setting in three dimensions.

    This potential only allows for standard velocities (i.e., velocities parallel to one of the cartesian coordinate
    axes going in the positive direction) of the active unit.

    The standard_velocity_derivative and standard_velocity_displacement functions use C code that is stored in the files
    inverse_power_coulomb_bounding_potential.c and inverse_power_coulomb_bounding_potential.h. The cffi package is used
    call the C code. The executable module inverse_power_coulomb_bounding_potential.py can be used to compile the C code
    and to create the necessary files.
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
        base.exceptions.ConfigurationError
            If the dimension does not equal three.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           prefactor=prefactor)
        if not setting.initialized():
            raise ConfigurationError("Potential {0} can only be used in a hypercubic setting."
                                     .format(self.__class__.__name__))
        if not setting.dimension == 3:
            raise ConfigurationError("The potential {0} can only be used in 3 dimensions."
                                     .format(self.__class__.__name__))
        super().__init__(prefactor=prefactor)

    def standard_velocity_derivative(self, direction: int, separation: Sequence[float], charge_one: float,
                                     charge_two: float) -> float:
        """
        Return the space derivative of the potential along a positive direction parallel to one of the cartesian axes
        for the given separations and charges.

        Note that the derivative function written in C always computes the space derivative in x direction. The
        separation vector is therefore permuted before the function is called.

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
        return _lib_derivative(self._prefactor * charge_one * charge_two, *permutation_3d(separation, direction))

    def standard_velocity_displacement(self, direction: int, separation: MutableSequence[float], charge_one: float,
                                       charge_two: float, potential_change: float) -> float:
        """
        Return the required displacement in space of the active unit along the positive direction of motion parallel to
        one of the cartesian axes where the cumulative event rate of the potential equals the given potential change.

        Note that the displacement function written in C always computes the displacement in x direction. The separation
        vector is therefore permuted before the function is called.

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
            The required displacement in space of the active unit along its direction of motion where the cumulative
            event rate equals the sampled potential change.
        """
        return _lib_displacement(self._prefactor * charge_one * charge_two, *permutation_3d(separation, direction),
                                 potential_change, setting.system_length)
