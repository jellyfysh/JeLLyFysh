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
"""Module for CuboidPeriodicCells class."""
import logging
import sys
from typing import Iterable, List, Set, Sequence
from base.logging import log_init_arguments
from base.exceptions import ConfigurationError
from setting import hypercuboid_setting as setting
from .periodic_cells import PeriodicCells


class CuboidPeriodicCells(PeriodicCells):
    """
    This class stores cuboid periodic cells.

    The class divides the simulation box into cuboids and provides all methods needed for the abstract PeriodicCells
    class. Each cell is identified by an integer. This class can only be used in a hypercuboid setting.
    """

    def __init__(self, cells_per_side: Sequence[int], origin: Sequence[float] = None, neighbor_layers: int = 1) -> None:
        """
        The constructor of the CuboidPeriodicCells class.

        Parameters
        ----------
        cells_per_side : Sequence[int]
            Sequence of wanted numbers of cells per side. If fewer numbers than the dimension are given,
            the first number is reused. Too many numbers are ignored.
        origin : Sequence[float] or None, optional
            The origin (or offset) of the cell system (minimum corner of the 0th cell). If None, it is set to
            [0.0]*dimension. Other values should be used with care.
        neighbor_layers : int, optional
            Number of cells in each direction, which should be returned by the excluded_cells method of this class.
            Therefore (neighbor_layers * 2 + 1) ** dimension cells are excluded per cell.

        Raises
        ------
        base.exceptions.ConfigurationError
            If hypercuboid_setting is not initialized.
        """
        logger = logging.getLogger(__name__)
        log_init_arguments(logger.debug, self.__class__.__name__, cells_per_side=cells_per_side,
                           origin=origin, neighbor_layers=neighbor_layers)
        if not setting.initialized():
            raise ConfigurationError("The class {0} can only be used in a hypercuboid setting."
                                     .format(self.__class__.__name__))
        super().__init__()

        if origin is None:
            self._origin = [0.0 for _ in range(setting.dimension)]
        else:
            self._origin = origin
            logger.warning("Cell system origin is set non-zero. Please care about the consequence "
                           "of translation and relative cell.")

        self._cells_per_side = [cells_per_side[i] if i < len(cells_per_side) and cells_per_side[i] is not None
                                else cells_per_side[0]
                                for i in range(setting.dimension)]
        self._cell_side_length = [setting.system_lengths[index] / number_cells
                                  for index, number_cells in enumerate(self._cells_per_side)]

        self._cumulative_product = [1]
        for d in range(setting.dimension - 1):
            self._cumulative_product.append(self._cumulative_product[d] * self._cells_per_side[d])

        self._max_cell = self._cumulative_product[-1] * self._cells_per_side[-1]

        # Calculate the corner with minimum coordinates for each cell (not yet the final solution)
        self._cell_center = []
        self._cell_max = []
        self._cell_min = []

        cell_tuple = [0 for _ in range(setting.dimension)]

        for _ in range(self._max_cell):
            upper_corner = []
            lower_corner = []
            cell_center = []

            for d in range(setting.dimension):
                cell_base = self._cell_side_length[d]

                lower_index = cell_tuple[d]
                upper_index = (cell_tuple[d] + 1) % cells_per_side[d]
                lower_base = setting.periodic_boundaries.correct_position_entry(
                    lower_index * cell_base + self._origin[d], d)
                upper_base = setting.periodic_boundaries.correct_position_entry(
                    upper_index * cell_base + self._origin[d], d)
                bit = self._last_bit(setting.system_lengths[d])

                while int((lower_base - self._origin[d]) % setting.system_lengths[d] / cell_base) < lower_index:
                    lower_base += bit
                while int((lower_base - bit - self._origin[d]) % setting.system_lengths[d] / cell_base) == lower_index:
                    lower_base -= bit
                lower_corner.append(lower_base)

                while int((upper_base - self._origin[d]) % setting.system_lengths[d] / cell_base) == upper_index:
                    upper_base -= bit
                while int((upper_base + bit - self._origin[d]) % setting.system_lengths[d] / cell_base) == lower_index:
                    upper_base += bit
                upper_corner.append(upper_base)

                if lower_corner[d] < upper_corner[d]:
                    cell_center.append((lower_corner[d] + upper_corner[d]) / 2)
                else:
                    cell_center.append((lower_corner[d] + upper_corner[d] + setting.system_lengths[d])
                                       / 2 % setting.system_lengths[d])

            self._cell_min.append(lower_corner)
            self._cell_max.append(upper_corner)
            self._cell_center.append(cell_center)

            d = 0
            for d in range(setting.dimension):
                if cell_tuple[d] + 1 < self._cells_per_side[d]:
                    break

            cell_tuple[d] += 1
            for d1 in range(d):
                cell_tuple[d1] = 0

        # Calculate neighbors

        self._successors = []

        for cell in range(self._max_cell):
            cell_center = self._cell_center[cell]
            self._successors.append([])

            for d in range(setting.dimension):
                neighbor_center = cell_center.copy()
                neighbor_center[d] = setting.periodic_boundaries.correct_position_entry(
                    neighbor_center[d] + self._cell_side_length[d], d)
                self._successors[-1].append(self.position_to_cell(neighbor_center))

        # Calculate excluded cells

        self._excluded_cells = []

        for cell in range(self._max_cell):
            cell_center = self._cell_center[cell]
            self._excluded_cells.append(set())

            for nearby in range((neighbor_layers * 2 + 1) ** setting.dimension):
                cell_tuple = [a - neighbor_layers for a in self._rebase(nearby, (neighbor_layers * 2 + 1),
                                                                        setting.dimension)]
                nearby_center = [setting.periodic_boundaries.correct_position_entry(
                    cell_center[d] + self._cell_side_length[d] * cell_tuple[d], d) for d in range(setting.dimension)]
                nearby_cell = self.position_to_cell(nearby_center)
                if nearby_cell not in self._excluded_cells[-1]:
                    self._excluded_cells[-1].add(nearby_cell)

    @staticmethod
    def _last_bit(value):
        return value * sys.float_info.epsilon

    @staticmethod
    def _rebase(number, base, figures):
        result = []
        while number:
            result.append(number % base)
            number //= base
        while len(result) < figures:
            result.append(0)
        return result

    def yield_cells(self) -> Iterable[int]:
        """
        Generate all cell identifiers.

        Overwrites the yield_cells method of the abstract Cells class.

        Yields
        ------
        int
            Cell identifier.
        """
        yield from range(self._max_cell)

    def excluded_cells(self, cell: int) -> Set[int]:
        """
        Return cell identifiers of excluded cells around given cell.

        The number of cells to go in each direction relative to the given cell is specified in the constructor by
        neighbor_layers. There are (neighbor_layer * 2 + 1) ** dimension excluded cells in total.
        Overwrites the excluded_cells method of the abstract Cells class.

        Parameters
        ----------
        cell : int
            The cell whose excluded cells should be returned.

        Returns
        -------
        Set[int]
            Set of excluded cells.
        """
        return self._excluded_cells[cell]

    def successor(self, cell: int, direction: int) -> int:
        """
        Return the successor cell of a given cell in a given direction.

        This method corrects for periodic boundaries.
        Overwrites the successor method of the abstract Cells class.

        Parameters
        ----------
        cell : int
            The cell whose successor should be returned.
        direction : int
            The direction in which the successor should be returned. Should be larger than 0.

        Returns
        -------
        int
            The successor cell in the given direction.
        """
        return self._successors[cell][direction]

    def cell_min(self, cell: int) -> List[float]:
        """
        Return the minimum position in each direction which belongs to the given cell.

        Overwrites the cell_min method of the abstract Cells class.

        Parameters
        ----------
        cell : int
            The cell whose minimum position should be returned.

        Returns
        -------
        List[float]
            The minimum position in each directions which belongs to the cell
        """
        return self._cell_min[cell]

    def cell_max(self, cell: int) -> List[float]:
        """
        Return the maximum position in each direction which belongs to the given cell.

        Overwrites the cell_max method of the abstract Cells class.

        Parameters
        ----------
        cell : int
            The cell whose maximum position should be returned.

        Returns
        -------
        List[float]
            The maximum position in each directions which belongs to the cell
        """
        return self._cell_max[cell]

    def relative_cell(self, cell: int, reference_cell: int) -> int:
        """
        Return the cell identifier with the same distance to the origin cell as the cell to the reference cell.

        This is the inverse method of the translate method. This method takes periodic boundaries into account.
        Overwrites the relative_cell method of the abstract PeriodicCells class.

        Parameters
        ----------
        cell : int
            The cell which gets mapped onto a cell with the same distance to the origin cell as the cell to the
            reference_cell.
        reference_cell : int
            The cell which gets mapped onto the origin cell.

        Returns
        -------
        int
            The cell identifier with the same distance to the origin cell as the cell to the reference cell.
        """
        center = self._cell_center[cell]
        reference_lower_bound = self._cell_min[reference_cell]
        relative_position = [center[d] - reference_lower_bound[d] for d in range(setting.dimension)]
        setting.periodic_boundaries.correct_position(relative_position)
        return self.position_to_cell(relative_position)

    def translate(self, cell: int, relative_cell: int) -> int:
        """
        Return the cell identifier with the same distance to the given cell as the given relative cell to the origin.

        This is the inverse method of the relative_cell method. This method takes periodic boundaries into account.
        Overwrites the translate method of the abstract PeriodicCells class.

        Parameters
        ----------
        cell : int
            The cell from which the resulting cell has the same distance as the relative_cell to the origin cell.
        relative_cell : int
            The cell whose distance to the origin cell matters.

        Returns
        -------
        int
            The cell identifier with the same distance to the cell as the relative_cell to the origin cell.
        """
        origin = self._cell_center[cell]
        displacement = self._cell_min[relative_cell]
        return self.position_to_cell([setting.periodic_boundaries.correct_position_entry(
            origin[d] + displacement[d], d) for d in range(setting.dimension)])

    def position_to_cell(self, position: Sequence[float]) -> int:
        """
        Map a given position onto a cell.

        Overwrites the position_to_cell method of the abstract Cells class.

        Parameters
        ----------
        position : Sequence[float]
            The position which should be mapped onto a cell.

        Returns
        -------
        int
            The cell identifier which contains the position.
        """
        return sum(int(setting.periodic_boundaries.correct_position_entry(position[d] - self._origin[d], d)
                       / self._cell_side_length[d]) * self._cumulative_product[d]
                   for d in range(setting.dimension))

    @property
    def zero_cell(self) -> int:
        """
        Return the cell identifier which is located at the origin.

        Returns
        -------
        int
            The cell identifier which is located at the origin.
        """
        return 0
