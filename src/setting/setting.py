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
"""Module for the abstract setting package setter class."""
from abc import ABCMeta, abstractmethod
from typing import Sequence
from types import ModuleType
import setting
from .periodic_boundaries import PeriodicBoundaries, PeriodicBoundariesNotImplemented


# noinspection PyProtectedMember
class Setting(metaclass=ABCMeta):
    """
    An abstract class which sets all attributes in the setting package accordingly.
    """
    def __init__(self, module: ModuleType, beta: float, dimension: int,
                 periodic_boundaries: PeriodicBoundaries = PeriodicBoundariesNotImplemented(),
                 similar_modules: Sequence[ModuleType] = ()):
        """
        The constructor of the Setting setter class.

        This constructor sets all attributes in the setting package. It first sets the initialized module and the
        similar modules. Therefore the inverse temperature beta, the dimension the random position function and the
        periodic boundaries are also set in all these modules, when calling their setter function in this constructor
        afterwards.
        Per default, the periodic boundaries are an instance of a class, where each method raises an exception.

        Parameters
        ----------
        module : ModuleType
            The module which gets initialized.
        beta : float
            The inverse temperature beta.
        dimension : int
            The dimension.
        periodic_boundaries : setting.periodic_boundaries.PeriodicBoundaries, optional
            The periodic boundaries.
        similar_modules : Sequence[ModuleType]
            Similar modules which also get initialized.
        """
        setting._set_initialized_setting_module(module)
        setting._set_similar_modules(similar_modules)
        setting._set_beta(beta)
        setting._set_dimension(dimension)
        setting._set_random_position_function(self.random_position)
        setting._set_periodic_boundaries(periodic_boundaries)

    @staticmethod
    @abstractmethod
    def random_position() -> Sequence[float]:
        """
        Return a random position.

        The random position function which will be set in the setting package, the initialized module and all similar
        modules.

        Returns
        -------
        Sequence[float]
            The random position.
        """
        raise NotImplementedError
