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
"""Module for the HardSpherePotential class."""
import logging
from math import sqrt
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.vectors import dot, norm_sq
from jellyfysh.potential import InvertiblePotential
from typing import Sequence


# noinspection PyMethodOverriding
class HardSpherePotential(InvertiblePotential):
    """
    This class implements the invertible d-dimensional hard sphere potential.

    The potential between the two involved units is infinite if two d-dimensional spheres with radius r centered at the
    units overlap, and vanishes otherwise: U_ij = inf if |r_ij| < 2 * r, U_ij = 0 otherwise. Here, r_ij = r_j - r_i is
    the separation between the units. This class assumes that i is the active unit.
    """

    _inf = float("inf")

    def __init__(self, radius: float) -> None:
        """
        The constructor of the HardDiskPotential class.

        Parameters
        ----------
        radius : float
            The radius r of the d-dimensional hard spheres.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the radius r is not larger than 0.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, radius=radius)
        if not radius > 0.0:
            raise ConfigurationError("The class {0} can only be used with a radius bigger than 0.0."
                                     .format(self.__class__.__name__))
        super().__init__()
        self._diameter_squared = 4.0 * radius * radius

    def displacement(self, velocity: Sequence[float], separation: Sequence[float]) -> float:
        """
        Return the required time displacement of the active unit along its velocity where the cumulative event rate of
        the potential equals the given potential change.

        Since this potential becomes infinite at |r_ij| = 2 * r, this method returns the required time displacement up
        to this separation without needing a sampled potential change.

        Parameters
        ----------
        velocity : Sequence[float]
            The velocity of the active hard sphere.
        separation : Sequence[float]
            The separation vector |r_ij| between the active and target sphere.

        Returns
        -------
            The required time displacement of the active sphere along its velocity where the potential becomes infinite.

        Raises
        ------
        AssertionError
            If the norm of the velocity is zero.
            If the given separation yields overlapping spheres.
        """
        velocity_squared = norm_sq(velocity)
        assert velocity_squared > 0.0
        separation_squared = norm_sq(separation)
        assert separation_squared - self._diameter_squared > -1.0e-13
        velocity_dot_separation = dot(velocity, separation)
        square_root_term = (velocity_dot_separation * velocity_dot_separation
                            - velocity_squared * (separation_squared - self._diameter_squared))
        return ((velocity_dot_separation - sqrt(square_root_term)) / velocity_squared
                if square_root_term >= 0.0 and velocity_dot_separation >= 0.0
                else HardSpherePotential._inf)

    def derivative(self, velocity: Sequence[float], separation: Sequence[float]) -> float:
        """
        Return the directional time derivative along a given velocity vector of the active unit for the given
        separation.

        This derivative vanishes everywhere, except at |r_ij| = 2 * r. However, this derivative should not be used
        within ECMC. Thus, this method raises an error.

        Parameters
        ----------
        velocity : Sequence[float]
            The velocity of the active sphere along which the directional derivative is computed.
        separation : Sequence[float]
            The separation vector |r_ij| between the active and target sphere.

        Returns
        -------
            The directional time derivative.

        Raises
        ------
        NotImplementedError
            If this method is called.
        """
        raise NotImplementedError("The derivative method of the class {0} should not be used."
                                  .format(self.__class__.__name__))
