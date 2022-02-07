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
"""Module for the abstract PeriodicCells class."""
from abc import ABCMeta, abstractmethod
from .cells import Cell, Cells


class PeriodicCells(Cells, metaclass=ABCMeta):
    """
    Abstract class for a general periodic cell system consisting of many cells.

    This class decomposes the simulation box into several cells similar to the Cells class. However, this class should
    explicitly consider periodic boundary conditions. A periodic cell system implements all methods of the underlying
    Cells class.

    Additionally, it defines methods that are specific to a periodic system with translational invariance. This class
    singles out a certain zero cell. The 'relative_cell' method then determines the cell, which has the same distance
    (under consideration of periodic boundary conditions) to the zero cell as the two cells that appear as the arguments
    of this method. This method is thus used to determine a cell separation. The 'translate' does the inverse, that is,
    it gets a relative cell which has a certain distance to the zero cell. The method then returns the cell that has the
    same distance to the cell that appears as an other argument of the method.
    """

    def __init__(self) -> None:
        """
        The (currently empty) constructor of the abstract PeriodicCells class.
        """
        super().__init__()

    @property
    @abstractmethod
    def zero_cell(self) -> Cell:
        """
        Return the zero cell of the periodic cell system.

        Returns
        -------
        Cell
            The zero cell.
        """
        raise NotImplementedError

    @abstractmethod
    def relative_cell(self, cell: Cell, reference_cell: Cell) -> Cell:
        """
        Return the cell that has the same distance to the zero cell as the first cell in the argument list to the
        second reference cell.

        This method takes periodic boundaries into account. The 'translate' method is the inverse method.

        Parameters
        ----------
        cell : Cell
            The cell that gets mapped onto the returned cell, which has the same distance to the zero cell as this cell
            to the reference cell.
        reference_cell : Cell
            The reference cell that gets mapped onto the zero cell.

        Returns
        -------
        Cell
            The cell that has the same distance to the zero cell as the first cell in the argument list to the second
            reference cell.
        """
        raise NotImplementedError

    @abstractmethod
    def translate(self, cell: Cell, relative_cell: Cell) -> Cell:
        """
        Return the cell that has the same distance to the first cell in the argument list as the second relative cell
        the zero cell.

        This method takes periodic boundaries into account. The 'relative_cell' method is the inverse method.

        Parameters
        ----------
        cell : Cell
            The cell that gets mapped onto the zero cell and to which the returned cell has the same distance as the
            second cell in the argument list to the zero cell.
        relative_cell : Cell
            The cell that gets mapped onto the returned cell, which has the same distance to the first cell in the
            argument list as this cell to the zero cell.

        Returns
        -------
        Cell
            The cell that has the same distance to the first cell in the argument list as the second relative cell
            the zero cell.
        """
        raise NotImplementedError
