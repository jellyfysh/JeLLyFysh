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
"""Module for the abstract Lifting class."""
from abc import ABCMeta, abstractmethod
import random
from typing import Any
from base.exceptions import LiftingSchemeError


class Lifting(metaclass=ABCMeta):
    """
    Abstract class for a lifting scheme used in the event handlers.

    Event handlers with more than two independent units require a lifting scheme to compute their our-states. For this,
    the event handler prepares factor derivatives of relevant time-sliced units. Then it can fill the derivative table
    stored within this class via the insert method. The get_active_identifier method, which should be implemented by the
    inheriting class, then determines the next active identifier based on the derivative table. The reset method
    deletes the derivative table.
    This class is suited for a single active independent unit.
    """

    def __init__(self):
        """The constructor of the abstract Lifting class."""
        self._negative_lifting_rates = []
        self._associated_identifiers = []
        self._random_position = 0.0
        self._sum_positive_lifting_rates = 0.0
        self._active_recorded = False

    def reset(self) -> None:
        """
        Reset the lifting scheme by deleting the derivative table.
        """
        self._negative_lifting_rates = []
        self._associated_identifiers = []
        self._random_position = 0.0
        self._sum_positive_lifting_rates = 0.0
        self._active_recorded = False

    def insert(self, lifting_rate: float, associated_identifier: Any, is_active: bool) -> None:
        """
        Insert a factor derivative of a unit into the derivative table.

        The associated identifier is the global state identifier of the unit. The precise format depends on the used
        state handler. This method also needs to know if the identifier corresponds to an active unit.

        Parameters
        ----------
        lifting_rate : float
            The factor derivative of the unit.
        associated_identifier : Any
            The global state identifier of the unit.
        is_active : bool
            Whether the unit is active or not.

        Raises
        ------
        AssertionError
            If the factor derivative is negative and the unit is active.
        """
        if lifting_rate > 0.0:
            self._sum_positive_lifting_rates += lifting_rate
            if is_active:
                self._active_recorded = True
                self._random_position += random.uniform(0.0, lifting_rate)
            elif not self._active_recorded:
                self._random_position += lifting_rate

        else:
            assert not is_active
            self._negative_lifting_rates.append(-lifting_rate)
            self._associated_identifiers.append(associated_identifier)

    @abstractmethod
    def get_active_identifier(self) -> Any:
        """
        Get the next active global state identifier based on the derivative table.

        Returns
        -------
        Any
            The next active global state identifier.

        Raises
        ------
        base.exceptions.LiftingSchemeError
            If the active independent unit has not been recorded yet.
        """
        if not self._active_recorded:
            raise LiftingSchemeError("Active unit has not been recorded.")
