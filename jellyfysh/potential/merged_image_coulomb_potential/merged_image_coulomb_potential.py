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
"""Module for the MergedImageCoulombPotential class."""
from copy import deepcopy
import logging
from typing import Any, Mapping, MutableMapping, Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.vectors import permutation_3d
from jellyfysh.potential.abstracts import StandardVelocityPotential
from jellyfysh.setting import hypercubic_setting as setting
# noinspection PyUnresolvedReferences
from ._merged_image_coulomb_potential import ffi, lib

# Directly import C functions used in performance relevant parts of the code.
_lib_derivative = lib.derivative


class MergedImageCoulombPotential(StandardVelocityPotential):
    r"""
    This class implements the merged image Coulomb pair potential
    U_ij = k * c_i * c_j * \sum_{\vec{n}\in\mathbb{Z}^3} 1/ (|\vec{r_ij}+\vec{n}L|).

    k is a prefactor, c_i and c_j are the charges of the involved units and r_ij = r_j - r_i is the separation between
    the units. This class assumes that i is the active unit.
    L is the side length of the three-dimensional cubic simulation box with periodic boundary conditions. This class is
    only implemented for a hypercubic setting in three dimensions.
    The conditionally convergent sum can be consistently defined in terms of tin-foil boundary conditions. Then,
    the sum is absolutely convergent. Ewald summation splits the sum up partly in position space and partly in Fourier
    space (see [Faulkner2018] in References.bib). The summation has three parameters, namely the cutoff in Fourier
    space, the cutoff in position space and a convergence factor alpha, which balances the converging speeds of the two
    sums.

    This potential only allows for standard velocities (i.e., velocities parallel to one of the cartesian coordinate
    axes going in the positive direction) of the active unit.

    The standard_velocity_derivative method uses C code that is stored in the files merged_image_coulomb_potential.c and
    merged_image_coulomb_potential.h. The cffi package is used to call the C code. The executable module
    merged_image_coulomb_potential_build.py can be used to compile the C code and to create the necessary files.
    """

    def __init__(self, alpha: float = 3.45, fourier_cutoff: int = 6, position_cutoff: int = 2,
                 prefactor: float = 1.0) -> None:
        """
        The constructor of the MergedImageCoulombPotential class.

        The default values are optimized so that the result with machine precision is computed in the shortest time.

        Parameters
        ----------
        alpha : float, optional
            The convergence factor alpha of the Ewald summation.
        fourier_cutoff : int, optional
            The cutoff in Fourier space of the Ewald summation.
        position_cutoff : int, optional
            The cutoff in position space of the Ewald summation.
        prefactor : float, optional
            The prefactor k of the potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the dimension does not equal three.
        base.exceptions.ConfigurationError
            If the hypercubic setting is not initialized.
        base.exceptions.ConfigurationError
            If the convergence factor alpha is negative or zero.
        base.exceptions.ConfigurationError
            If the cutoff in Fourier space is negative.
        base.exceptions.ConfigurationError
            If the cutoff in position space is negative.
        MemoryError
            If the C code fails to allocate memory.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           alpha=alpha, fourier_cutoff=fourier_cutoff, position_cutoff=position_cutoff,
                           prefactor=prefactor)
        super().__init__(prefactor=prefactor)
        if not setting.dimension == 3:
            raise ConfigurationError("The potential {0} can only be used in 3 dimensions."
                                     .format(self.__class__.__name__))
        if not setting.initialized():
            raise ConfigurationError("The potential {0} can only be used in a hypercubic setting."
                                     .format(self.__class__.__name__))
        if alpha <= 0.0:
            raise ConfigurationError("The argument converge_factor must be > 0.0 in the class {0}."
                                     .format(self.__class__.__name__))
        if fourier_cutoff < 0:
            raise ConfigurationError("The argument fourier_cutoff must be >= 0 in the class {0}."
                                     .format(self.__class__.__name__))
        if position_cutoff < 0:
            raise ConfigurationError("The argument position_cutoff must be >= 0 in the class {0}."
                                     .format(self.__class__.__name__))
        # Store the parameters of the construct_merged_image_coulomb_potential function for the __setstate__ method
        self._fourier_cutoff = fourier_cutoff
        self._position_cutoff = position_cutoff
        self._alpha = alpha
        self._system_length = setting.system_length
        c_potential = lib.construct_merged_image_coulomb_potential(self._fourier_cutoff, self._position_cutoff,
                                                                   self._alpha, self._system_length)
        if c_potential == ffi.NULL:
            raise MemoryError("Could not allocate memory for the class {0}.".format(self.__class__.__name__))
        # ffi.gc takes care of calling the destructor function on the created c_potential
        # This is done when this class is garbage collected
        self._potential = ffi.gc(c_potential, lib.destroy_merged_image_coulomb_potential,
                                 size=lib.estimated_size(c_potential))

    # noinspection PyMethodOverriding
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
        separation = permutation_3d(separation, direction)
        return self._prefactor * charge_one * charge_two * _lib_derivative(self._potential, *separation)

    def __copy__(self) -> 'MergedImageCoulombPotential':
        """
        Create a shallow copy of this class.

        This class just updates the __dict__ of the new instance with the __dict__ of self.
        Note that this is the default behavior of the copy function. However, if __copy__ is not defined, the
        __getstate__ and __setstate__ methods are used (if they are defined). This is not desired here because
        __setstate__ explicitly constructs a new C potential.

        Returns
        -------
        MergedImageCoulombPotential
            The shallow copy.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    # noinspection PyDefaultArgument
    def __deepcopy__(self, memodict: MutableMapping[int, Any] = {}) -> 'MergedImageCoulombPotential':
        """
        Create a deep copy of this class.

        This class copies every attribute using the copy.deepcopy function (while filling the memory dictionary) except
        for the _potential attribute, which points to the C potential. Here, the corresponding C function is used.

        Parameters
        ----------
        memodict : Mapping[int, Any]
            The memo dictionary.
        Returns
        -------
        MergedImageCoulombPotential
            The deep copy.
        """
        cls = self.__class__
        copied_class = cls.__new__(cls)
        # see https://stackoverflow.com/questions/1500718/
        memodict[id(self)] = copied_class
        for key, value in self.__dict__.items():
            if key != "_potential":
                # noinspection PyArgumentList
                setattr(copied_class, key, deepcopy(value, memodict))
            else:
                # For the C potential, we have to copy on C level
                copied_c_potential = lib.copy_merged_image_coulomb_potential(self._potential)
                if copied_c_potential == ffi.NULL:
                    raise MemoryError("Could not copy the C potential of the class {0}."
                                      .format(self.__class__.__name__))
                setattr(copied_class, key,
                        ffi.gc(copied_c_potential, lib.destroy_merged_image_coulomb_potential,
                               size=lib.estimated_size(copied_c_potential)))
        return copied_class

    def __getstate__(self) -> Mapping[str, Any]:
        """
        Return a state of this class that can be pickled.

        This method removes _potential from the dictionary self.__dict__ so that it can be pickled.

        Returns
        -------
        Mapping[str, Any]
            The state that can be pickled.
        """
        state = self.__dict__.copy()
        del state["_potential"]
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """
        Use the state dictionary to initialize this class.

        This method creates the self._potential attribute that was deleted in the __getstate__ method.

        Parameters
        ----------
        state : Mapping[str, Any]
            The state.

        Raises
        ------
        MemoryError
               If the C code fails to allocate memory.
        """
        self.__dict__.update(state)
        c_potential = lib.construct_merged_image_coulomb_potential(self._fourier_cutoff, self._position_cutoff,
                                                                   self._alpha, self._system_length)
        if c_potential == ffi.NULL:
            raise MemoryError("Could not allocate memory for the class {0}.".format(self.__class__.__name__))
        self._potential = ffi.gc(c_potential, lib.destroy_merged_image_coulomb_potential,
                                 size=lib.estimated_size(c_potential))
