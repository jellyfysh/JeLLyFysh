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
"""Module for the BendingPotential class."""
import logging
from math import acos, sqrt
from typing import Sequence, Tuple
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base import vectors
from .potential import Potential


# noinspection PyMethodOverriding
class BendingPotential(Potential):
    """
    This class implements the three-unit bending potential U_ijk = k/2 * (phi_ijk - phi_0)^2.

    Here, k is a prefactor, phi_ijk denotes the angle formed by the vectors r_i - r_j and r_k - r_j, and phi_0 is an
    equilibrium angle.
    """

    def __init__(self, equilibrium_angle: float, prefactor: float = 1.0):
        """
        The constructor of the BendingPotential class.

        Parameters
        ----------
        equilibrium_angle : float
            The equilibrium angle phi_0 of the potential.
        prefactor : float, optional
            The prefactor k of the potential.
        """
        self.init_arguments = lambda: {"equilibrium_angle": equilibrium_angle, "prefactor": prefactor}
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           equilibrium_angle=equilibrium_angle, prefactor=prefactor)
        super().__init__(prefactor=prefactor)
        self._equilibrium_angle = equilibrium_angle

    def init_arguments(self):
        raise NotImplementedError

    def gradient(self, separation_one: Sequence[float],
                 separation_two: Sequence[float]) -> Tuple[Sequence[float], Sequence[float], Sequence[float]]:
        """
        Return the gradient of the potential evaluated at the given separations.

        This method returns three gradients with respect to the positions r_i, r_j, and r_k, respectively.

        Parameters
        ----------
        separation_one : Sequence[float]
            The separation vector r_i - r_j.
        separation_two : Sequence[float]
            The separation vector r_k - r_j.

        Returns
        -------
        (Sequence[float], Sequence[float], Sequence[float])
            The gradients with respect to the positions r_i, r_j, and r_k, respectively.
        """
        separation_one_norm_squared = vectors.norm_sq(separation_one)
        separation_two_norm_squared = vectors.norm_sq(separation_two)
        separation_one_norm = sqrt(separation_one_norm_squared)
        separation_two_norm = sqrt(separation_two_norm_squared)
        cos_angle = vectors.dot(separation_one, separation_two) / separation_one_norm / separation_two_norm
        angle = acos(cos_angle)
        d_potential_by_d_angle = self._prefactor * (angle - self._equilibrium_angle)
        d_angle_by_d_cos_angle = -1.0 / sqrt(1.0 - cos_angle * cos_angle)
        constant_factor_one = (d_potential_by_d_angle * d_angle_by_d_cos_angle
                               / separation_one_norm / separation_two_norm)
        constant_factor_two = (d_potential_by_d_angle * d_angle_by_d_cos_angle
                               * cos_angle / separation_one_norm_squared)
        constant_factor_three = (d_potential_by_d_angle * d_angle_by_d_cos_angle
                                 * cos_angle / separation_two_norm_squared)
        d_potential_by_d_xi = [s_two * constant_factor_one - s_one * constant_factor_two
                               for s_one, s_two in zip(separation_one, separation_two)]
        d_potential_by_d_xk = [s_one * constant_factor_one - s_two * constant_factor_three
                               for s_one, s_two in zip(separation_one, separation_two)]
        d_potential_by_d_xj = [-d_xi - d_xk for d_xi, d_xk in zip(d_potential_by_d_xi, d_potential_by_d_xk)]
        return d_potential_by_d_xi, d_potential_by_d_xj, d_potential_by_d_xk

    def derivative(self, velocity: Sequence[float], separation_one: Sequence[float],
                   separation_two: Sequence[float]) -> Tuple[float, float, float]:
        """
        Return the directional time derivative along the given velocity vector of the active unit for the given
        separations.

        This method returns three derivatives that correspond to the units i, j, and k being active with the given
        velocity, respectively.
        
        Parameters
        ----------
        velocity : Sequence[float]
            The velocity of the active unit along which the directional time derivative is computed.
        separation_one : Sequence[float]
            The separation vector r_i - r_j.
        separation_two : Sequence[float]
            The separation vector r_k - r_j.

        Returns
        -------
        (float, float, float)
            The directional time derivatives with respect to the units i, j, and k.

        Raises
        ------
        AssertionError
            If the velocity is zero.
        """
        assert any(entry != 0.0 for entry in velocity)
        separation_one_norm_squared = vectors.norm_sq(separation_one)
        separation_two_norm_squared = vectors.norm_sq(separation_two)
        separation_one_norm = sqrt(separation_one_norm_squared)
        separation_two_norm = sqrt(separation_two_norm_squared)
        cos_angle = vectors.dot(separation_one, separation_two) / separation_one_norm / separation_two_norm
        angle = acos(cos_angle)
        d_potential_by_d_angle = self._prefactor * (angle - self._equilibrium_angle)
        d_angle_by_d_cos_angle = -1.0 / sqrt(1.0 - cos_angle * cos_angle)
        separation_one_dot_velocity = vectors.dot(separation_one, velocity)
        separation_two_dot_velocity = vectors.dot(separation_two, velocity)
        velocity_times_d_cos_angle_by_d_xi = (separation_two_dot_velocity / separation_one_norm / separation_two_norm
                                              - separation_one_dot_velocity * cos_angle / separation_one_norm_squared)
        velocity_times_d_cos_angle_by_d_xk = (separation_one_dot_velocity / separation_one_norm / separation_two_norm
                                              - separation_two_dot_velocity * cos_angle / separation_two_norm_squared)
        velocity_times_d_cos_angle_by_d_xj = -velocity_times_d_cos_angle_by_d_xi - velocity_times_d_cos_angle_by_d_xk
        return (d_potential_by_d_angle * d_angle_by_d_cos_angle * velocity_times_d_cos_angle_by_d_xi,
                d_potential_by_d_angle * d_angle_by_d_cos_angle * velocity_times_d_cos_angle_by_d_xj,
                d_potential_by_d_angle * d_angle_by_d_cos_angle * velocity_times_d_cos_angle_by_d_xk)
