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
"""Module for the abstract CellOccupancy class."""
from abc import ABCMeta, abstractmethod
from typing import Any, Iterable, Sequence, Tuple
from activator.internal_state import InternalState
from .cells import Cells


class CellOccupancy(InternalState, metaclass=ABCMeta):
    """
    Abstract class for a general cell-occupancy system.

    A cell-occupancy system maps identifiers of the global state onto cell identifiers in the underlying cell system.
    It should implement the methods of the abstract InternalState class in addition to the methods listed here.
    This cell-occupancy class already relies on the global state identifiers to be sequences of integers of different
    lengths. This is fulfilled, for example, in the TreeStateHandler. Therefore global state identifiers will be of type
    Sequence[int] in this class.
    The cell-occupancy system stores one global state identifier per cell identifier. All other global state identifiers
    are treated as surplus identifiers. Also identifier of active units are not stored in the cell-occupancy system
    itself. Surplus identifiers and active unit identifiers should not be returned in the __getitem__ method.
    """

    def __init__(self, cells: Cells, cell_level: int) -> None:
        """
        The constructor of the CellOccupancy class.

        Parameters
        ----------
        cells : activator.internal_state.cell_occupancy.cells.Cells
            The underlying cell system.
        cell_level : int
            The length of the global state identifiers which should be stored in this internal state.
        """
        super().__init__()
        self._cells = cells
        self._cell_level = cell_level

    @abstractmethod
    def initialize(self, extracted_global_state: Any) -> None:
        """
        Initialize the internal state based on the full extracted global state from the state handler.

        Extends the initialize method of the InternalState class. Use this method once in the beginning of the run to
        initialize the internal state. Only after a call of this method, other public methods of this class can be
        called without raising an error. The precise format of the argument is specified by the activator class which
        uses the internal state.

        Parameters
        ----------
        extracted_global_state : Any
            The full extracted global state from the state handler.
        """
        super().initialize(extracted_global_state)

    @abstractmethod
    def yield_surplus(self) -> Iterable[Sequence[int]]:
        """
        Generate the surplus identifiers.

        Yields
        ------
        Sequence[int]
            Surplus identifier.
        """
        raise NotImplementedError

    @abstractmethod
    def yield_active_cells(self) -> Iterable[Tuple[Any, Sequence[int]]]:
        """
        Generate the cell identifiers and the global state identifiers of the active units.

        The precise format of the cell identifiers is given by the underlying cell system.

        Yields
        ------
        (Any, Sequence[int])
            The cell identifier, the global state identifier of the active unit.
        """
        raise NotImplementedError

    @property
    def cells(self) -> Cells:
        """
        Return the underlying cell system.

        Returns
        -------
        activator.internal_state.cell_occupancy.cells.Cells
            The underlying cell system.
        """
        return self._cells

    @property
    def cell_level(self) -> int:
        """
        Return the length of the global state identifiers stored in this internal state.

        Returns
        -------
        int
            The cell level.
        """
        return self._cell_level
