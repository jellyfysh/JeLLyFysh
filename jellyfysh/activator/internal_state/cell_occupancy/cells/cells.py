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
"""Module for the abstract Cell and Cells classes."""
from abc import ABCMeta, abstractmethod
from typing import Any, Iterable, Optional, Sequence, Set
from jellyfysh.base.exceptions import ConfigurationError
import jellyfysh.setting as setting


class Cell(object):
    """
    Abstract class for a general cell.

    A cell is determined by a minimum and maximum position that belong to it. Here, the minimum position is guaranteed
    to be smaller than the maximum position in each dimension. A cell is also associated with an identifier that can
    be used by a cell system (see Cells class below). The specific form of these (integer, tuple...) is left open so
    they are of type Any in this class.
    """

    def __init__(self, identifier: Any, cell_min: Sequence[float], cell_max: Sequence[float]) -> None:
        """
        The constructor of the abstract Cell class.

        Raises
        ------
        ConfigurationError
            If the minimum position is not strictly smaller than the maximum position.
        """
        if any(cell_min[index] >= cell_max[index] for index in range(setting.dimension)):
            raise ConfigurationError("The components of the minimum position {0} are not strictly smaller than the "
                                     "maximum position{1}.".format(cell_min, cell_max))
        self._cell_max = cell_max
        self._identifier = identifier
        self._cell_min = cell_min

    @property
    def identifier(self) -> Any:
        """
        Return the identifier of the cell.

        Returns
        -------
        Any
            The identifier.
        """
        return self._identifier

    @property
    def cell_min(self) -> Sequence[float]:
        """
        Return the minimum position that belongs to the cell.

        Returns
        -------
        Sequence[float]
            The minimum position.
        """
        return self._cell_min

    @property
    def cell_max(self) -> Sequence[float]:
        """
        Return the maximum position that belongs to the cell.

        Returns
        -------
        Sequence[float]
            The maximum position.
        """
        return self._cell_max


class Cells(metaclass=ABCMeta):
    """
    Abstract class for a general cell system consisting of many cells.

    This class decomposes the simulation box into several cells. It gives access to the cells themselves, defines
    methods that determine relations between cells, and maps positions onto the corresponding cell.

    Note that in this version of JeLLyFysh, a cell system does not define a cell separation. Only a periodic cell system
    (see 'PeriodicCells' class) defines a cell separation via the 'relative_cell' method.
    """

    def __init__(self):
        """
        The (currently empty) constructor of the abstract Cells class.
        """
        pass

    @abstractmethod
    def yield_cells(self) -> Iterable[Cell]:
        """
        Generate all cells of the cell system.

        Yields
        ------
        Cell
            The cells.
        """
        raise NotImplementedError

    @abstractmethod
    def position_to_cell(self, position: Sequence[float]) -> Cell:
        """
        Map a given position onto the corresponding cell.

        Parameters
        ----------
        position : Sequence[float]
            The position.

        Returns
        -------
        Cell
            The cell that contains the position.
        """
        raise NotImplementedError

    @abstractmethod
    def nearby_cells(self, cell: Cell) -> Set[Cell]:
        """
        Return the set of nearby cells in the cell system of the given cell.

        The criterion for nearby cells should be defined in the inheriting class.

        Parameters
        ----------
        cell : Cell
            The cell whose nearby cells are returned.

        Returns
        -------
        Set[Cell]
            The set of nearby cells.
        """
        raise NotImplementedError

    @abstractmethod
    def neighbor_cell(self, cell: Cell, direction: int, positive: bool) -> Optional[Cell]:
        """
        Return the neighbor cell of the given cell in the given positive or negative direction.

        The direction indicates the axis along which the neighbor is returned and should satisfy
        0 <= direction < setting.dimension. If the positive bool is True, this method returns the neighbor in the
        positive direction. Otherwise, it returns the neighbor in the negative direction.
        In a non periodic setting, the neighbor might be None.

        Parameters
        ----------
        cell : Cell
            The cell whose neighbor is returned
        direction : int
            The direction in which the neighbor is returned.
        positive : bool
            Whether this method returns the neighbor along the positive or negative direction.

        Returns
        -------
        Cell or None
            The neighbor cell in the given positive or negative direction, or None if the neighbor cell does not exist.
        """
        raise NotImplementedError
