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
"""Module for the abstract PhysicalState class."""
from abc import ABCMeta, abstractmethod
from typing import Any, Iterable, Sequence
from base.initializer import Initializer


class PhysicalState(Initializer, metaclass=ABCMeta):
    """
    Abstract class for a global physical state.

    The global physical state stores positions and charges. Each position and charge should be accessible via the get
    method based on a global state identifier. Only the position should be changeable via the set method. Finally,
    the class provides a method which generates all global state identifiers.
    The precise format of the global state identifiers depends on the implementation of this class. Therefore we use the
    type Any for them throughout this class. The same is true for the format of the global physical state which are
    returned to the state handler.
    """
    def __init__(self):
        """
        The constructor of the abstract PhysicalState class.
        """
        super().__init__()

    @abstractmethod
    def initialize(self, global_physical_state: Any) -> None:
        """
        Initialize the state handler with the given full global physical state.

        Extends the initialize method of the Initializer class. This method must be extended and used once in the
        beginning of the run to initialize the physical state. Only after a call of this method, other public methods of
        this class can be called without raising an error. The precise format of the argument is specified by the used
        input handler.

        Parameters
        ----------
        global_physical_state : Any
            The full global physical state.
        """
        super().initialize()

    @abstractmethod
    def get(self, identifier: Any) -> Any:
        """
        Return the global physical state for a global state identifier.

        Parameters
        ----------
        identifier : Any
            The global state identifier.

        Returns
        -------
        Any
            The global physical state information.
        """
        raise NotImplementedError

    @abstractmethod
    def set(self, identifier: Any, position: Sequence[float]):
        """
        Store the given position for the global state identifier.

        Parameters
        ----------
        identifier : Any
            The global state identifier.
        position : Sequence[float]
            The position.
        """
        raise NotImplementedError

    @abstractmethod
    def yield_identifiers(self) -> Iterable[Any]:
        """
        Generate all global state identifier.

        Yields
        ------
        Any
            The global state identifiers.
        """
        raise NotImplementedError
