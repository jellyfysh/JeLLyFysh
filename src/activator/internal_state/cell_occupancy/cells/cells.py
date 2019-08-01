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
"""Module for the abstract Cells class."""
from abc import ABCMeta, abstractmethod
from typing import AbstractSet, Any, Iterable, MutableSequence, Sequence, Union


class Cells(metaclass=ABCMeta):
    """
    Abstract class for a general cell geometry.

    A cell system is a decomposition of simulating box and provides several methods to access its geometry.
    Generally, cells are accessed by identifiers. The specific form of these (integer, tuple...) is left open so they
    are of type Any in this class.
    """

    def __init__(self):
        """
        The (currently empty) constructor of the abstract Cells class.
        """
        pass

    @abstractmethod
    def yield_cells(self) -> Iterable[Any]:
        """
        Generate all cell identifiers.

        Yields
        ------
        Any
            Cell identifier.
        """
        raise NotImplementedError

    @abstractmethod
    def excluded_cells(self, cell: Any) -> AbstractSet[Any]:
        """
        Excluded cells with respect to given cell.

        Each cell might have excluded cells (for example nearby cells). These should be returned in this method.

        Parameters
        ----------
        cell : Any
            The cell whose excluded cells should be returned.

        Returns
        -------
        AbstractSet[Any]
            Set of excluded cells.
        """
        raise NotImplementedError

    @abstractmethod
    def position_to_cell(self, position: Sequence[float]) -> Any:
        """
        Map a given position onto a cell.

        Parameters
        ----------
        position : Sequence[float]
            The position which should be mapped onto a cell.

        Returns
        -------
        Any
            The cell identifier which contains the position.
        """
        raise NotImplementedError

    @abstractmethod
    def cell_min(self, cell: Any) -> MutableSequence[float]:
        """
        Return the minimum position in each direction which belongs to the given cell.

        Parameters
        ----------
        cell : Any
            The cell whose minimum position should be returned.

        Returns
        -------
        MutableSequence[float]
            The minimum position in each directions which belongs to the cell
        """
        raise NotImplementedError

    @abstractmethod
    def cell_max(self, cell: Any) -> MutableSequence[float]:
        """
        Return the maximum position in each direction which belongs to the given cell.

        Parameters
        ----------
        cell : Any
            The cell whose maximum position should be returned.

        Returns
        -------
        MutableSequence[float]
            The maximum position in each directions which belongs to the cell
        """
        raise NotImplementedError

    @abstractmethod
    def successor(self, cell: Any, direction: int) -> Union[Any, None]:
        """
        Return the successor cell of a given cell in a given direction .

        In a non periodic setting, the successor might be None.

        Parameters
        ----------
        cell : Any
            The cell whose successor should be returned.
        direction : int
            The direction in which the successor should be returned. Should be larger than 0.

        Returns
        -------
        Any or None
            The successor cell in the given direction.
        """
        raise NotImplementedError
