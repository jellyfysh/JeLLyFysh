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
"""Module for the BendingPotential class."""
import logging
import math
from typing import Sequence, Tuple
from base.logging import log_init_arguments
from base import vectors
import setting
from .potential import Potential


# noinspection PyMethodOverriding
class BendingPotential(Potential):
    r"""
    This class implements the three-unit bending potential U_ijk = k/2 * (\phi_ijk - \phi_0).

    k is a prefactor, \phi_ijk denotes the angle formed by the vectors r_i - r_j and r_k - r_j and \phi_0 is an
    equilibrium angle. This class computes the derivative with respect to all units.
    """

    def __init__(self, equilibrium_angle: float, prefactor: float = 1.0):
        r"""
        The constructor of the BendingPotential class.

        Parameters
        ----------
        equilibrium_angle : float
            The equilibrium angle \phi_0 of the potential.
        prefactor : float, optional
            The prefactor k of the potential.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           equilibrium_angle=equilibrium_angle, prefactor=prefactor)
        super().__init__(prefactor=prefactor)
        self._equilibrium_angle = equilibrium_angle

    def derivative(self, direction: int, separation_one: Sequence[float],
                   separation_two: Sequence[float]) -> Tuple[float, float, float]:
        """
        Return the derivative of the potential along a direction with respect to i, j, and k.
        
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
            The derivatives with respect to i, j, and k.
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
