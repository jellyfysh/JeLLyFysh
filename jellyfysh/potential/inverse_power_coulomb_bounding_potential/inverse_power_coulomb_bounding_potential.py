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
from typing import Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.potential import InvertiblePotential
from jellyfysh.setting import hypercubic_setting as setting
# noinspection PyUnresolvedReferences
from ._inverse_power_coulomb_bounding_potential import ffi, lib

# Directly import C functions used in performance relevant parts of the code.
_lib_derivative = lib.derivative
_lib_gradient = lib.gradient
_lib_displacement = lib.displacement


# noinspection PyMethodOverriding
class InversePowerCoulombBoundingPotential(InvertiblePotential):
    """
    This class implements the pair potential U_ij = c_i * c_j * k / |r_ij,0|.

    Here, k is a prefactor, c_i and c_j are the charges of the involved units and r_ij,0 = nearest(r_j - r_i) is the
    shortest separation between the units, possibly corrected for periodic boundaries. With a correct prefactor, this
    potential can bound the merged image coulomb potential. This class assumes that i is the active unit.
    Any constant k > 1.5836 is appropriate for a bounding potential in a hypercubic box. This class is only
    implemented for a hypercubic setting in three dimensions.

    The derivative and displacement functions use C code that is stored in the files
    inverse_power_coulomb_bounding_potential.c and inverse_power_coulomb_bounding_potential.h. The cffi package is used
    call the C code. The executable module inverse_power_coulomb_bounding_potential_build.py can be used to compile the
    C code and to create the necessary files.
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
        self.init_arguments = lambda: {"prefactor": prefactor}
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           prefactor=prefactor)
        if not setting.initialized():
            raise ConfigurationError("Potential {0} can only be used in a hypercubic setting."
                                     .format(self.__class__.__name__))
        if not setting.dimension == 3:
            raise ConfigurationError("The potential {0} can only be used in 3 dimensions."
                                     .format(self.__class__.__name__))
        super().__init__(prefactor=prefactor)

    def init_arguments(self):
        raise NotImplementedError

    def gradient(self, separation: Sequence[float], charge_one: float, charge_two: float) -> Sequence[float]:
        """
        Return the gradient of the potential evaluated at the given separation and for the given charges.

        Parameters
        ----------
        separation : Sequence[float]
            The separation vector r_ij,0.
        charge_one : float
            The charge c_i.
        charge_two : float
            The charge c_j.

        Returns
        -------
        Sequence[float]
            The gradient with respect to the position r_i of the active unit.

        Raises
        ------
        AssertionError
            If the given separation vector is not the shortest separation with periodic boundary conditions.
        """
        assert all(abs(s) <= setting.system_length_over_two for s in separation)
        # The _lib_gradient C function returns a struct that contains a member called gradient of type double[3]. This
        # array can be converted to a Python list by calling list on it.
        gradient = _lib_gradient(self._prefactor * charge_one * charge_two, separation)
        return [gradient.gx, gradient.gy, gradient.gz]

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
        return _lib_derivative(self._prefactor * charge_one * charge_two, velocity, separation)

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
            If the given separation vector is not the shortest separation with periodic boundary conditions.
        """
        assert any(entry != 0.0 for entry in velocity)
        assert all(abs(s) <= setting.system_length_over_two for s in separation)
        return _lib_displacement(self._prefactor * charge_one * charge_two, velocity, separation, potential_change,
                                 setting.system_length)
