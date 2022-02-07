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
import math
from typing import Sequence, Tuple
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base import vectors
import jellyfysh.setting as setting
from .abstracts import StandardVelocityPotential


# noinspection PyMethodOverriding
class BendingPotential(StandardVelocityPotential):
    """
    This class implements the three-unit bending potential U_ijk = k/2 * (phi_ijk - phi_0).

    k is a prefactor, phi_ijk denotes the angle formed by the vectors r_i - r_j and r_k - r_j and phi_0 is an
    equilibrium angle. This class computes the derivative with respect to all units.

    This potential only allows for standard velocities (i.e., velocities parallel to one of the cartesian coordinate
    axes going in the positive direction) of the active unit.
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
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           equilibrium_angle=equilibrium_angle, prefactor=prefactor)
        super().__init__(prefactor=prefactor)
        self._equilibrium_angle = equilibrium_angle

    def standard_velocity_derivative(self, direction: int, separation_one: Sequence[float],
                                     separation_two: Sequence[float]) -> Tuple[float, float, float]:
        """
        Return the space derivative with respect to i, j, and k of the potential along a positive direction parallel to
        one of the cartesian axes for the given separations .

        Parameters
        ----------
        direction : int
            The direction of the derivative.
        separation_one : Sequence[float]
            The separation vector r_i - r_j.
        separation_two : Sequence[float]
            The separation vector r_k - r_j.

        Returns
        -------
        (float, float, float)
            The space derivatives with respect to i, j, and k.
        """
        separation_norm_one = vectors.norm(separation_one)
        separation_norm_two = vectors.norm(separation_two)
        cosine_of_angle = (sum(separation_one[i] * separation_two[i] for i in range(setting.dimension))
                           / separation_norm_one / separation_norm_two)
        angle = math.acos(cosine_of_angle)

        d_potential_by_d_angle = self._prefactor * (angle - self._equilibrium_angle)
        d_angle_by_d_cosine_of_angle = -1.0 / math.sin(angle)
        d_cosine_by_d_separation_one = (
                separation_two[direction] / separation_norm_one / separation_norm_two - cosine_of_angle *
                separation_one[direction] / separation_norm_one ** 2)
        d_cosine_by_d_separation_two = (
                separation_one[direction] / separation_norm_one / separation_norm_two - cosine_of_angle *
                separation_two[direction] / separation_norm_two ** 2)

        d_potential_by_d_separation_one = (
                d_potential_by_d_angle * d_angle_by_d_cosine_of_angle * d_cosine_by_d_separation_one)
        d_potential_by_d_separation_two = (
                d_potential_by_d_angle * d_angle_by_d_cosine_of_angle * d_cosine_by_d_separation_two)

        return (d_potential_by_d_separation_one, - d_potential_by_d_separation_one - d_potential_by_d_separation_two,
                d_potential_by_d_separation_two)

    def derivative(self, velocity: Sequence[float], separation_one: Sequence[float],
                   separation_two: Sequence[float]) -> Tuple[float, float, float]:
        """
        Return the directional time derivative along a given velocity vector of the active unit for certain separations
        and charges.

        This potential only allows for standard velocities (i.e., velocities parallel to one of the cartesian coordinate
        axes going in the positive direction) of the active unit.

        This method first computes the space derivative with respect to the units i, j, and k of the potential along the
        positive direction parallel to one of the cartesian axes of the active unit for the given separations. Each
        derivative is then multiplied by the absolute value of the velocity for the returned time derivatives.
        
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
            If the velocity is not in the positive direction parallel to one of the cartesian axes.
        """
        direction, speed = self._analyse_velocity(velocity)
        # noinspection PyTypeChecker
        return tuple(derivative * speed
                     for derivative in self.standard_velocity_derivative(direction, separation_one, separation_two))
