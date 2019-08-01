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
"""Module for the MergedImageCoulombPotential class."""
import logging
import math
from typing import Sequence
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.vectors import permutation_3d
from setting import hypercubic_setting as setting
from .potential import Potential


# noinspection PyMethodOverriding
class MergedImageCoulombPotential(Potential):
    r"""
    This class implements the merged image Coulomb pair potential
    U_ij = k * c_i * c_j * \sum_{\vec{n}\in\mathbb{Z}^3} 1/ (|\vec{r_ij}+\vec{n}\vec{L}|).

    k is a prefactor, c_i and c_j are the charges of the involved units and r_ij = r_j - r_i is the separation between
    the units. This class assumes that i is the active unit.
    \vec{L} are the sides of the three-dimensional simulation box with periodic boundary conditions. This class is only
    implemented for a hypercubic setting in three dimensions.
    The conditionally convergent sum can be consistently defined in terms of tin-foil boundary conditions. Then,
    the sum is absolutely convergent. Ewald summation splits the sum up partly in position space and partly in Fourier
    space (see [Faulkner2018] in References.bib). The summation has three parameters, namely the cutoff in Fourier
    space, the cutoff in position space and a convergence factor alpha, which balances the converging speeds of the two
    sums.
    """

    def __init__(self, alpha: float = 3.45, fourier_cutoff: int = 6,
                 position_cutoff: int = 2, prefactor: float = 1.0) -> None:
        """
        The constructor of the MergedImageCoulombPotential class.

        The default values are optimized so that the result has machine precision.

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
            If the cutoff in Fourier space is negative.
        base.exceptions.ConfigurationError
            If the cutoff in position space is negative.
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

        pi_sq = math.pi * math.pi
        self._fourier_cutoff = fourier_cutoff
        self._position_cutoff = position_cutoff

        self._alpha = alpha / setting.system_length
        self._alpha_sq = self._alpha * self._alpha
        self._fourier_cutoff_sq = self._fourier_cutoff * self._fourier_cutoff
        self._position_cutoff_sq = self._position_cutoff * self._position_cutoff
        self._two_alpha_over_root_pi = 2 * self._alpha / math.sqrt(math.pi)
        self.two_pi_over_length = 2 * math.pi / setting.system_length
        length_sq = setting.system_length * setting.system_length

        fourier_list = [[[0.0 for _ in range(self._fourier_cutoff + 1)] for _ in
                        range(self._fourier_cutoff + 1)] for _ in range(self._fourier_cutoff + 1)]
        for k in range(0, self._fourier_cutoff + 1):
            for j in range(0, self._fourier_cutoff + 1):
                for i in range(1, self._fourier_cutoff + 1):
                    if j == 0 and k == 0:
                        coefficient = 1.0
                    elif k == 0:
                        coefficient = 2.0
                    elif j == 0:
                        coefficient = 2.0
                    else:
                        coefficient = 4.0

                    norm_sq = i * i + j * j + k * k
                    fourier_list[i][j][k] = 4 * i * coefficient * (math.exp(
                        - pi_sq * norm_sq / self._alpha_sq / length_sq) / norm_sq / length_sq)

        self._fourier_array = tuple(
            [tuple([tuple(fourier_list[i][j]) for j in range(self._fourier_cutoff + 1)]) for i in
             range(self._fourier_cutoff + 1)])

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
        separation = permutation_3d(separation, direction)
        return self._prefactor * charge_one * charge_two * (self._derivative_position_space(*separation)
                                                            + self._derivative_fourier_space(*separation))

    def _derivative_position_space(self, separation_x: float, separation_y: float, separation_z: float) -> float:
        """Return the part of the Ewald sum of the derivative in position space in x direction."""
        derivative = 0.0

        for k in range(- self._position_cutoff, self._position_cutoff + 1):
            vector_z_sq = (separation_z + k * setting.system_length) * (separation_z + k * setting.system_length)
            cutoff_y = int((self._position_cutoff_sq - k * k) ** 0.5)
            for j in range(- cutoff_y, cutoff_y + 1):
                vector_y_sq = (separation_y + j * setting.system_length) * (separation_y + j * setting.system_length)
                cutoff_x = int((self._position_cutoff_sq - j * j - k * k) ** 0.5)
                for i in range(- cutoff_x, cutoff_x + 1):
                    vector_x = separation_x + i * setting.system_length
                    vector_sq = vector_x * vector_x + vector_y_sq + vector_z_sq
                    vector_norm = vector_sq ** 0.5
                    derivative += (vector_x * (
                            self._two_alpha_over_root_pi * math.exp(- self._alpha_sq * vector_sq) + math.erfc(
                                self._alpha * vector_norm) / vector_norm) / vector_sq)

        return derivative

    def _derivative_fourier_space(self, separation_x: float, separation_y: float, separation_z: float) -> float:
        """Return the part of the Ewald sum of the derivative in fourier space in x direction."""
        derivative = 0.0

        delta_cos_x = math.cos(self.two_pi_over_length * separation_x)
        delta_sin_x = math.sin(self.two_pi_over_length * separation_x)
        delta_cos_y = math.cos(self.two_pi_over_length * separation_y)
        delta_sin_y = math.sin(self.two_pi_over_length * separation_y)
        delta_cos_z = math.cos(self.two_pi_over_length * separation_z)
        delta_sin_z = math.sin(self.two_pi_over_length * separation_z)

        cos_x = delta_cos_x
        sin_x = delta_sin_x
        cos_y = 1.0
        sin_y = 0.0
        cos_z = 1.0
        sin_z = 0.0

        for i in range(1, self._fourier_cutoff + 1):
            cutoff_y = int((self._fourier_cutoff_sq - i * i) ** 0.5)
            for j in range(0, cutoff_y + 1):
                cutoff_z = int((self._fourier_cutoff_sq - i * i - j * j) ** 0.5)
                for k in range(0, cutoff_z + 1):
                    derivative += self._fourier_array[i][j][k] * sin_x * cos_y * cos_z

                    if k != cutoff_z:
                        store_cos_z = cos_z
                        cos_z = store_cos_z * delta_cos_z - sin_z * delta_sin_z
                        sin_z = sin_z * delta_cos_z + store_cos_z * delta_sin_z
                    elif j != cutoff_y:
                        store_cos_y = cos_y
                        cos_y = store_cos_y * delta_cos_y - sin_y * delta_sin_y
                        sin_y = sin_y * delta_cos_y + store_cos_y * delta_sin_y
                        cos_z = 1.0
                        sin_z = 0.0
                    elif i != self._fourier_cutoff:
                        store_cos_x = cos_x
                        cos_x = store_cos_x * delta_cos_x - sin_x * delta_sin_x
                        sin_x = sin_x * delta_cos_x + store_cos_x * delta_sin_x
                        cos_y = 1.0
                        sin_y = 0.0
                        cos_z = 1.0
                        sin_z = 0.0

        return derivative
