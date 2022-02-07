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
"""Module for the HardDipolePotential class."""
import logging
from math import sqrt
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.vectors import dot, norm_sq
from jellyfysh.potential import InvertiblePotential
from typing import Sequence


# noinspection PyMethodOverriding
class HardDipolePotential(InvertiblePotential):
    """
    This class implements the invertible hard dipole potential.

    The potential between the two involved units only vanishes if the absolute value of their separation lies in between
    a minimum value r and a maximum value R: U_ij = 0 if r <= |r_ij| <= R, U_ij = inf otherwise. Here, r_ij = r_j - r_i
    is the separation between the units. This class assumes that i is the active unit.
    """

    def __init__(self, minimum_separation: float, maximum_separation: float) -> None:
        """
        The constructor of the HardDipolePotential class.

        Parameters
        ----------
        minimum_separation : float
            The minimum separation r.
        maximum_separation : float
            The maximum separation R.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the minimum separation r is not larger than 0.
            If the maximum separation R is not larger than 0.
            If the minimum separation r is not smaller than the maximum separation R.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           minimum_separation=minimum_separation, maximum_separation=maximum_separation)
        if not minimum_separation > 0.0:
            raise ConfigurationError("The class {0} can only be used with a minimum separation bigger than 0.0."
                                     .format(self.__class__.__name__))
        if not maximum_separation:
            raise ConfigurationError("The class {0} can only be used with a maximum separation bigger than 0.0."
                                     .format(self.__class__.__name__))
        if not minimum_separation < maximum_separation:
            raise ConfigurationError("The class {0} can only be used with a minimum separation that is smaller than the"
                                     "maximum separation.".format(self.__class__.__name__))
        super().__init__()
        self._minimum_separation_squared = minimum_separation * minimum_separation
        self._maximum_separation_squared = maximum_separation * maximum_separation

    def displacement(self, velocity: Sequence[float], separation: Sequence[float]) -> float:
        """
        Return the required time displacement of the active unit along its velocity where the cumulative event rate of
        the potential equals the given potential change.

        Since this potential becomes infinite at either |r_ij| = r or |r_ij| = R, this method returns the required time
        displacement up to these separations without needing a sampled potential change.

        Parameters
        ----------
        velocity : Sequence[float]
            The velocity of the active unit.
        separation : Sequence[float]
            The separation vector |r_ij| between the active and target unit.

        Returns
        -------
            The required time displacement of the active unit along its velocity where the potential becomes infinite.

        Raises
        ------
        AssertionError
            If the norm of the velocity is zero.
            If the given separation does not lie within the allowed region.
        """
        velocity_squared = norm_sq(velocity)
        assert velocity_squared > 0.0
        separation_squared = norm_sq(separation)
        assert separation_squared - self._minimum_separation_squared > -1.0e-13
        assert self._maximum_separation_squared - separation_squared > -1.0e-13
        velocity_dot_separation = dot(velocity, separation)
        if velocity_dot_separation >= 0.0:
            minimum_square_root_term = (velocity_dot_separation * velocity_dot_separation
                                        - velocity_squared * (separation_squared - self._minimum_separation_squared))
            if minimum_square_root_term >= 0.0:
                return (velocity_dot_separation - sqrt(minimum_square_root_term)) / velocity_squared
        maximum_square_root_term = (velocity_dot_separation * velocity_dot_separation
                                    - velocity_squared * (separation_squared - self._maximum_separation_squared))
        assert maximum_square_root_term >= 0.0
        return (velocity_dot_separation + sqrt(maximum_square_root_term)) / velocity_squared

    def derivative(self, velocity: Sequence[float], separation: Sequence[float]) -> float:
        """
        Return the directional time derivative along a given velocity vector of the active unit for the given
        separation.

        This derivative vanishes everywhere, except at |r_ij| = r and |r_ij| = R. However, this derivative should not be
        used within ECMC. Thus, this method raises an error.

        Parameters
        ----------
        velocity : Sequence[float]
            The velocity of the active unit along which the directional derivative is computed.
        separation : Sequence[float]
            The separation vector |r_ij| between the active and target unit.

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
