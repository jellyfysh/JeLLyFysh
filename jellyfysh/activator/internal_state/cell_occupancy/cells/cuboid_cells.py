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
"""Module for the CuboidCells class."""
import itertools
import logging
import math
import struct
from typing import Iterable, Optional, Sequence, Set
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.setting import hypercuboid_setting as setting
from .cells import Cell, Cells


class CuboidCells(Cells):
    """
    This class constructs and stores a cuboid cell system.

    It decomposes the simulation box into cuboids. The side lengths of the cuboids are determined by the number of cells
    per side of the simulation box that is set on initialization. Each cuboid cell is identified by a list of integers
    with a length equal to the dimension of the simulation box. Each entry in the list then gives the number of the cell
    along the given direction. Here, the cell that has the origin [0.0 * setting.dimension] as the minimum position has
    the cell identifier list [0 * setting.dimension].

    In this class, each cell has the cells in its neighbored layers as its nearby cells (used in the 'cells_nearby'
    method). The number of neighbor layers is set on initialization of this class. With this, each cell has at most
    (neighbor_layers * 2 + 1) ** setting.dimension nearby cells.

    Note that in this version of JeLLyFysh, a cell system does not define a cell separation. Only a periodic cell system
    (see 'CuboidPeriodicCells' class) defines a cell separation via the 'relative_cell' method.

    This class can only be used if the hypercuboid setting is initialized.
    """

    def __init__(self, cells_per_side: Sequence[int], neighbor_layers: int = 1) -> None:
        """
        The constructor of the CuboidCells class.

        The cells are identified by storing the index of the cell in each possible direction. This is similar to matrix
        indices except that the zero index is at the most left position in each dimension.
        The cells are also stored in a list in the self._cells attribute in this class. To map the list of cell
        identifiers in each direction onto the index in the list, the self._cumulative_product attribute stores the
        multiplicative factors for each of the cell identifiers in the list. To get the list index, simply multiply the
        factor with the cell identifier and sum over all directions.

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
        self.init_arguments = lambda: {"cells_per_side": cells_per_side, "neighbor_layers": neighbor_layers}
        logger = logging.getLogger(__name__)
        log_init_arguments(logger.debug, self.__class__.__name__, cells_per_side=cells_per_side,
                           neighbor_layers=neighbor_layers)
        super().__init__()
        if not setting.initialized():
            raise ConfigurationError("The class {0} can only be used in a hypercuboid setting."
                                     .format(self.__class__.__name__))
        if not 0 < len(cells_per_side) <= setting.dimension:
            raise ConfigurationError("The number of cells per side in the class {0} is either zero or exceeds the "
                                     "chosen dimension.".format(self.__class__.__name__))
        if not neighbor_layers >= 0:
            raise ConfigurationError("The number of neighbor layers has to be greater than or equal to zero.")

        self._neighbor_layers = neighbor_layers
        self._cells_per_side = [cells_per_side[i] if i < len(cells_per_side) else cells_per_side[0]
                                for i in range(setting.dimension)]
        self._cell_side_lengths = [setting.system_lengths[index] / self._cells_per_side[index]
                                   for index in range(setting.dimension)]

        self._cells = []
        self._cumulative_product = [1]
        for d in range(setting.dimension - 1):
            self._cumulative_product.append(self._cumulative_product[d] * self._cells_per_side[d])
        number_of_cells = self._cumulative_product[-1] * self._cells_per_side[-1]

        # This list stores the cell identifier of the cell whose minimum and maximum positions are currently computed.
        cell_identifier_list = [0 for _ in range(setting.dimension)]
        for summed_cell_identifier in range(number_of_cells):
            assert summed_cell_identifier == sum(cell_identifier_list[index] * self._cumulative_product[index]
                                                 for index in range(setting.dimension))
            cell_min = []
            cell_max = []
            for index in range(setting.dimension):
                lower_position = cell_identifier_list[index] * self._cell_side_lengths[index]
                upper_position = (cell_identifier_list[index] + 1) * self._cell_side_lengths[index]
                # If the lower position is 0.0, the following method to determine the exact float position fails because
                # int() truncates towards 0.0. Instead we can simply use the 0.0 lower position.
                if lower_position > 0.0:
                    # Find the smallest lower position that is still within the cell by slowly decreasing the position.
                    while int(lower_position / self._cell_side_lengths[index]) == cell_identifier_list[index]:
                        lower_position = _next_float_down(lower_position)
                    while int(lower_position / self._cell_side_lengths[index]) < cell_identifier_list[index]:
                        lower_position = _next_float_up(lower_position)
                cell_min.append(lower_position)
                # Find the greatest upper position that is still within the cell by slowly increasing the position.
                while int(upper_position / self._cell_side_lengths[index]) == cell_identifier_list[index]:
                    upper_position = _next_float_up(upper_position)
                while int(upper_position / self._cell_side_lengths[index]) > cell_identifier_list[index]:
                    upper_position = _next_float_down(upper_position)
                cell_max.append(upper_position)
            self._cells.append(Cell(tuple(cell_identifier_list), tuple(cell_min), tuple(cell_max)))

            # Find the first identifier that can be increased by one in the cell identifier list.
            d = 0
            for d in range(setting.dimension):
                if cell_identifier_list[d] + 1 < self._cells_per_side[d]:
                    break
            # Increase the found identifier and reset all identifiers in smaller dimensions to 0.
            cell_identifier_list[d] += 1
            for smaller_d in range(d):
                cell_identifier_list[smaller_d] = 0

        self._nearby_cells = {}
        for cell in self._cells:
            self._nearby_cells[cell] = set(nearby_cell for nearby_cell in self._yield_nearby_cells(cell))

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
            cell_identifier_valid = True
            for direction in range(setting.dimension):
                if not 0 <= cell_identifier[direction] < self._cells_per_side[direction]:
                    cell_identifier_valid = False
            if not cell_identifier_valid:
                continue
            cell_index = sum(
                cell_identifier[index] * self._cumulative_product[index] for index in range(setting.dimension))
            yield self._cells[cell_index]

    def init_arguments(self):
        raise NotImplementedError

    def yield_cells(self) -> Iterable[Cell]:
        """
        Generate all cells of the cell system.

        Yields
        ------
        Cell
            The cells.
        """
        yield from self._cells

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

        Raises
        ------
        AssertionError
            If the given position lies outside of the simulation box.
        """
        assert all(0.0 <= position[index] <= setting.system_lengths[index] for index in range(setting.dimension))
        return self._cells[sum(int(position[index] / self._cell_side_lengths[index]) * self._cumulative_product[index]
                               for index in range(setting.dimension))]

    def nearby_cells(self, cell: Cell) -> Set[Cell]:
        """
        Return the set of nearby cells in the cell system of the given cell.

        Each cell has the cells in its neighbored layers (in all directions) as its nearby cells. Therefore, each cell
        has at most (self._neighbor_layers * 2 + 1) ** setting.dimension nearby cells.

        Parameters
        ----------
        cell : Cell
            The cell whose nearby cells are returned.

        Returns
        -------
        Set[Cell]
            The set of nearby cells.
        """
        return self._nearby_cells[cell]

    def neighbor_cell(self, cell: Cell, direction: int, positive: bool) -> Optional[Cell]:
        """
        Return the neighbor cell of the given cell in the given positive or negative direction.

        The direction indicates the axis along which the neighbor is returned and should satisfy
        0 <= direction < setting.dimension. If the positive bool is True, this method returns the neighbor in the
        positive direction. Otherwise, it returns the neighbor in the negative direction.

        In this class, the desired neighbor cell may not exist if a cuboid cell is at the border of the simulation box.
        Then, this method returns None.

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
        assert 0 <= direction < setting.dimension
        if positive:
            if cell.identifier[direction] + 1 >= self._cells_per_side[direction]:
                return None
            neighbor_index = sum(
                cell.identifier[index] * self._cumulative_product[index] if index != direction
                else (cell.identifier[index] + 1) * self._cumulative_product[index]
                for index in range(setting.dimension))
        else:
            if cell.identifier[direction] - 1 < 0:
                return None
            neighbor_index = sum(
                cell.identifier[index] * self._cumulative_product[index] if index != direction
                else (cell.identifier[index] - 1) * self._cumulative_product[index]
                for index in range(setting.dimension))
        return self._cells[neighbor_index]


def _next_float_up(x: float) -> float:
    """Get the next float that is larger than x by using bit manipulation
    (see https://stackoverflow.com/questions/10420848)."""
    # NaNs and positive infinity map to themselves.
    if math.isnan(x) or (math.isinf(x) and x > 0):
        return x

    # 0.0 and -0.0 both map to the smallest +ve float.
    if x == 0.0:
        x = 0.0

    n = struct.unpack('<q', struct.pack('<d', x))[0]
    if n >= 0:
        n += 1
    else:
        n -= 1
    return struct.unpack('<d', struct.pack('<q', n))[0]


def _next_float_down(x: float) -> float:
    """Get the next float that is smaller than x by using bit manipulation
    (see https://stackoverflow.com/questions/10420848)."""
    return -_next_float_up(-x)
