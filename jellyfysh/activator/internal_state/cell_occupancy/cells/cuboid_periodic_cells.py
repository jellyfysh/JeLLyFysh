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
"""Module for the CuboidPeriodicCells class."""
import itertools
import logging
from typing import Iterable, Sequence
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.setting import hypercuboid_setting as setting
from .cells import Cell
from .cuboid_cells import CuboidCells
from .periodic_cells import PeriodicCells


class CuboidPeriodicCells(CuboidCells, PeriodicCells):
    """
    This class constructs and stores a cuboid cell system that takes periodic boundary conditions and translational
    invariance into account.

    This class uses the CuboidCells class to construct and store the cuboid cell system. Also the same definition for
    nearby cells is used. However, this class overwrites some methods to explicitly take periodic boundary conditions
    into account. Also, all methods of the abstract PeriodicCells class are defined. Here, this class defines the cell
    that has the origin [0.0 * setting.dimension] of the simulation box as the minimum position (i.e., the cell with the
    cell identifier list [0 * setting.dimension]) as the zero cell.
    """

    def __init__(self, cells_per_side: Sequence[int], neighbor_layers: int = 1) -> None:
        """
        The constructor of the CuboidPeriodicCells class.

        This constructor gets exactly the same arguments as the constructor of the CuboidCells base class and just
        passes all arguments through.

        Parameters
        ----------
        cells_per_side : Sequence[int]
            The number of cells per side of the simulation box that this class uses to construct the cuboid cell system.
            If fewer numbers than the dimension of the simulation are given, the first number is reused.
        neighbor_layers : int, optional
            The number of cells in each direction that are considered as nearby in this cell system.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the hypercuboid_setting is not initialized.
        base.exceptions.ConfigurationError
            If zero or too many cells per side are given for the chosen dimension.
        base.exceptions.ConfigurationError
            If the number of neighbor layers is smaller than zero.
        """
        logger = logging.getLogger(__name__)
        log_init_arguments(logger.debug, self.__class__.__name__, cells_per_side=cells_per_side,
                           neighbor_layers=neighbor_layers)
        super().__init__(cells_per_side=cells_per_side, neighbor_layers=neighbor_layers)

    def _yield_nearby_cells(self, cell: Cell) -> Iterable[Cell]:
        """
        Generate all nearby cells of the given cell.

        Each cell has the cells in its neighbored layers (in all directions) as its nearby cells. Therefore, each cell
        has at most (self._neighbor_layers * 2 + 1) ** setting.dimension nearby cells.

        Parameters
        ----------
        cell : Cell
            The cell.

        Yields
        -------
        Cell
            The nearby cells.
        """
        for cell_identifier in itertools.product(
                *[range(cell.identifier[direction] - self._neighbor_layers,
                        cell.identifier[direction] + self._neighbor_layers + 1)
                  for direction in range(setting.dimension)]):
            corrected_identifier = []
            for direction in range(setting.dimension):
                corrected_identifier.append(cell_identifier[direction] % self._cells_per_side[direction])
            cell_index = sum(
                corrected_identifier[index] * self._cumulative_product[index] for index in range(setting.dimension))
            yield self._cells[cell_index]

    def neighbor_cell(self, cell: Cell, direction: int, positive: bool) -> Cell:
        """
        Return the neighbor cell of the given cell in the given positive or negative direction.

        The direction indicates the axis along which the neighbor is returned and should satisfy
        0 <= direction < setting.dimension. If the positive bool is True, this method returns the neighbor in the
        positive direction. Otherwise, it returns the neighbor in the negative direction.

        This method overwrites the 'neighbor_cell' method of the CuboidCells base class in order to explicitly take
        periodic boundary conditions into account. In the periodic cell system of this class, the desired neighbor cell
        always exists.

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
        Cell
            The neighbor cell in the given positive or negative direction.
        """
        assert 0 <= direction < setting.dimension
        if positive:
            neighbor_index = sum(
                cell.identifier[index] * self._cumulative_product[index] if index != direction
                else ((cell.identifier[index] + 1) % self._cells_per_side[index]) * self._cumulative_product[index]
                for index in range(setting.dimension))
        else:
            neighbor_index = sum(
                cell.identifier[index] * self._cumulative_product[index] if index != direction
                else ((cell.identifier[index] - 1) % self._cells_per_side[index]) * self._cumulative_product[index]
                for index in range(setting.dimension))
        return self._cells[neighbor_index]

    @property
    def zero_cell(self) -> Cell:
        """
        Return the zero cell of the periodic cell system.

        This class defines the cell that has the origin [0.0 * setting.dimension] of the simulation box as the minimum
        position (i.e., the cell with the cell identifier list [0 * setting.dimension]) as the zero cell.

        Returns
        -------
        Cell
            The zero cell.
        """
        return self._cells[0]

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
        relative_position = [setting.periodic_boundaries.correct_position_entry(
            (cell.cell_max[index] + cell.cell_min[index]) / 2.0 - reference_cell.cell_min[index], index)
                             for index in range(setting.dimension)]
        return self.position_to_cell(relative_position)

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
        translated_position = [setting.periodic_boundaries.correct_position_entry(
            (cell.cell_max[index] + cell.cell_min[index]) / 2.0 + relative_cell.cell_min[index], index)
                               for index in range(setting.dimension)]
        return self.position_to_cell(translated_position)
