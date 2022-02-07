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
import os
import sys
from unittest import TestCase, main
from jellyfysh.activator.internal_state.cell_occupancy.cells.cuboid_cells import CuboidCells
from jellyfysh.base.exceptions import ConfigurationError
import jellyfysh.setting as setting
from jellyfysh.setting import hypercuboid_setting
from jellyfysh.setting import hypercubic_setting
_unittest_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))
_unittest_directory_added_to_path = False
if _unittest_directory not in sys.path:
    sys.path.append(_unittest_directory)
    _unittest_directory_added_to_path = True
# noinspection PyUnresolvedReferences
from expanded_test_case import ExpandedTestCase


def tearDownModule():
    if _unittest_directory_added_to_path:
        sys.path.remove(_unittest_directory)


# Inherit explicitly from TestCase class for Test functionality in PyCharm.
class TestCuboidCells(ExpandedTestCase, TestCase):
    def tearDown(self) -> None:
        setting.reset()

    def setUpTwoDimensionalCells(self):
        hypercuboid_setting.HypercuboidSetting(beta=1.0, dimension=2, system_lengths=[1.0, 2.0])
        # Set these values so that the setting package is fully initialized.
        setting.set_number_of_node_levels(1)
        setting.set_number_of_root_nodes(1)
        setting.set_number_of_nodes_per_root_node(1)
        self._cells = CuboidCells(cells_per_side=[4, 5], neighbor_layers=2)
        self._all_cell_identifiers = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2), (3, 2),
            (0, 3), (1, 3), (2, 3), (3, 3),
            (0, 4), (1, 4), (2, 4), (3, 4)
        ]
        self._all_cell_min_positions = [
            [0.0, 0.0], [1.0 / 4.0, 0.0], [2.0 / 4.0, 0.0], [3.0 / 4.0, 0.0],
            [0.0, 2.0 / 5.0], [1.0 / 4.0, 2.0 / 5.0], [2.0 / 4.0, 2.0 / 5.0], [3.0 / 4.0, 2.0 / 5.0],
            [0.0, 4.0 / 5.0], [1.0 / 4.0, 4.0 / 5.0], [2.0 / 4.0, 4.0 / 5.0], [3.0 / 4.0, 4.0 / 5.0],
            [0.0, 6.0 / 5.0], [1.0 / 4.0, 6.0 / 5.0], [2.0 / 4.0, 6.0 / 5.0], [3.0 / 4.0, 6.0 / 5.0],
            [0.0, 8.0 / 5.0], [1.0 / 4.0, 8.0 / 5.0], [2.0 / 4.0, 8.0 / 5.0], [3.0 / 4.0, 8.0 / 5.0]
        ]
        self._all_cell_max_positions = [
            [1.0 / 4.0, 2.0 / 5.0], [2.0 / 4.0, 2.0 / 5.0], [3.0 / 4.0, 2.0 / 5.0], [4.0 / 4.0, 2.0 / 5.0],
            [1.0 / 4.0, 4.0 / 5.0], [2.0 / 4.0, 4.0 / 5.0], [3.0 / 4.0, 4.0 / 5.0], [4.0 / 4.0, 4.0 / 5.0],
            [1.0 / 4.0, 6.0 / 5.0], [2.0 / 4.0, 6.0 / 5.0], [3.0 / 4.0, 6.0 / 5.0], [4.0 / 4.0, 6.0 / 5.0],
            [1.0 / 4.0, 8.0 / 5.0], [2.0 / 4.0, 8.0 / 5.0], [3.0 / 4.0, 8.0 / 5.0], [4.0 / 4.0, 8.0 / 5.0],
            [1.0 / 4.0, 10.0 / 5.0], [2.0 / 4.0, 10.0 / 5.0], [3.0 / 4.0, 10.0 / 5.0], [4.0 / 4.0, 10.0 / 5.0]
        ]

    def test_two_dimensional_yield_cells(self):
        self.setUpTwoDimensionalCells()
        all_cells = list(self._cells.yield_cells())
        self.assertEqual(len(all_cells), 20)
        for index in range(20):
            cell_identifier = self._all_cell_identifiers[index]
            cell_min_position = self._all_cell_min_positions[index]
            cell_max_positions = self._all_cell_max_positions[index]
            found_index = None
            for cell_index, cell in enumerate(all_cells):
                if cell.identifier == cell_identifier:
                    self.assertIsNone(found_index)
                    found_index = cell_index
            self.assertIsNotNone(found_index)
            compare_cell = all_cells[found_index]
            self.assertEqual(cell_identifier, compare_cell.identifier)
            self.assertAlmostEqualSequence(cell_min_position, compare_cell.cell_min, places=13)
            self.assertAlmostEqualSequence(cell_max_positions, compare_cell.cell_max, places=13)

    def test_two_dimensional_position_to_cell(self):
        self.setUpTwoDimensionalCells()
        # Simply test some random positions. Here, we simply compare the identifiers of the returned cells. Their
        # cell_min and cell_max attributes where checked in the 'test_two_dimensional_yield_cells' method.
        self.assertEqual(self._cells.position_to_cell([1.0e-13, 1.0e-13]).identifier, (0, 0))
        self.assertEqual(self._cells.position_to_cell([0.25 + 1.0e-13, 0.4 - 1.0e-13]).identifier, (1, 0))
        self.assertEqual(self._cells.position_to_cell([0.5 + 1.0e-13, 0.2]).identifier, (2, 0))
        self.assertEqual(self._cells.position_to_cell([0.75 + 1.0e-13, 0.01]).identifier, (3, 0))
        self.assertEqual(self._cells.position_to_cell([0.25 - 1.0e-13, 0.4 + 1.0e-13]).identifier, (0, 1))
        self.assertEqual(self._cells.position_to_cell([0.5 - 1.0e-13, 0.8 - 1.0e-13]).identifier, (1, 1))
        self.assertEqual(self._cells.position_to_cell([0.75 - 1.0e-13, 0.53]).identifier, (2, 1))
        self.assertEqual(self._cells.position_to_cell([1.0 - 1.0e-13, 0.6]).identifier, (3, 1))
        self.assertEqual(self._cells.position_to_cell([0.21, 0.8 + 1.0e-13]).identifier, (0, 2))
        self.assertEqual(self._cells.position_to_cell([0.27, 1.2 - 1.0e-13]).identifier, (1, 2))
        self.assertEqual(self._cells.position_to_cell([0.52, 1.01]).identifier, (2, 2))
        self.assertEqual(self._cells.position_to_cell([0.99, 0.96]).identifier, (3, 2))
        self.assertEqual(self._cells.position_to_cell([0.17, 1.2 + 1.0e-13]).identifier, (0, 3))
        self.assertEqual(self._cells.position_to_cell([0.31, 1.6 - 1.0e-13]).identifier, (1, 3))
        self.assertEqual(self._cells.position_to_cell([0.69, 1.45]).identifier, (2, 3))
        self.assertEqual(self._cells.position_to_cell([0.76, 1.54]).identifier, (3, 3))
        self.assertEqual(self._cells.position_to_cell([0.08, 1.6 + 1.0e-13]).identifier, (0, 4))
        self.assertEqual(self._cells.position_to_cell([0.48, 2.0 - 1.0e-13]).identifier, (1, 4))
        self.assertEqual(self._cells.position_to_cell([0.74, 1.77]).identifier, (2, 4))
        self.assertEqual(self._cells.position_to_cell([0.82, 1.83]).identifier, (3, 4))

    def test_two_dimensional_position_to_cell_raises_error_if_position_outside_cell(self):
        self.setUpTwoDimensionalCells()
        with self.assertRaises(AssertionError):
            self._cells.position_to_cell([-1.0e-13, 1.5])
        with self.assertRaises(AssertionError):
            self._cells.position_to_cell([0.1, -1.0e-13])
        with self.assertRaises(AssertionError):
            self._cells.position_to_cell([1.0 + 1.0e-13, 1.5])
        with self.assertRaises(AssertionError):
            self._cells.position_to_cell([0.1, 2.0 + 1.0e-13])

    def test_two_dimensional_nearby_cells(self):
        self.setUpTwoDimensionalCells()
        for cell in self._cells.yield_cells():
            if cell.identifier == (0, 0):
                expected_nearby_cells = {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (1, 0):
                expected_nearby_cells = {(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1),
                                      (0, 2), (1, 2), (2, 2), (3, 2)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (2, 0):
                expected_nearby_cells = {(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1),
                                      (0, 2), (1, 2), (2, 2), (3, 2)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (3, 0):
                expected_nearby_cells = {(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue

            if cell.identifier == (0, 1):
                expected_nearby_cells = {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2),
                                      (0, 3), (1, 3), (2, 3)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (1, 1):
                expected_nearby_cells = {(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1),
                                      (0, 2), (1, 2), (2, 2), (3, 2), (0, 3), (1, 3), (2, 3), (3, 3)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (2, 1):
                expected_nearby_cells = {(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1),
                                      (0, 2), (1, 2), (2, 2), (3, 2), (0, 3), (1, 3), (2, 3), (3, 3)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (3, 1):
                expected_nearby_cells = {(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2),
                                      (1, 3), (2, 3), (3, 3)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue

            if cell.identifier == (0, 2):
                expected_nearby_cells = {(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2),
                                      (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (1, 2):
                expected_nearby_cells = {(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1),
                                      (0, 2), (1, 2), (2, 2), (3, 2), (0, 3), (1, 3), (2, 3), (3, 3),
                                      (0, 4), (1, 4), (2, 4), (3, 4)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (2, 2):
                expected_nearby_cells = {(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1),
                                      (0, 2), (1, 2), (2, 2), (3, 2), (0, 3), (1, 3), (2, 3), (3, 3),
                                      (0, 4), (1, 4), (2, 4), (3, 4)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (3, 2):
                expected_nearby_cells = {(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2),
                                      (1, 3), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue

            if cell.identifier == (0, 3):
                expected_nearby_cells = {(0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3),
                                      (0, 4), (1, 4), (2, 4)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (1, 3):
                expected_nearby_cells = {(0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2),
                                      (0, 3), (1, 3), (2, 3), (3, 3), (0, 4), (1, 4), (2, 4), (3, 4)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (2, 3):
                expected_nearby_cells = {(0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (3, 2),
                                      (0, 3), (1, 3), (2, 3), (3, 3), (0, 4), (1, 4), (2, 4), (3, 4)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (3, 3):
                expected_nearby_cells = {(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3),
                                      (1, 4), (2, 4), (3, 4)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue

            if cell.identifier == (0, 4):
                expected_nearby_cells = {(0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (1, 4):
                expected_nearby_cells = {(0, 2), (1, 2), (2, 2), (3, 2), (0, 3), (1, 3), (2, 3), (3, 3),
                                      (0, 4), (1, 4), (2, 4), (3, 4)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (2, 4):
                expected_nearby_cells = {(0, 2), (1, 2), (2, 2), (3, 2), (0, 3), (1, 3), (2, 3), (3, 3),
                                      (0, 4), (1, 4), (2, 4), (3, 4)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (3, 4):
                expected_nearby_cells = {(1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue

            self.fail("Unexpected cell returned by yield_cells method.")

    def test_two_dimensional_nearby_cells_zero_neighbor_layers(self):
        hypercuboid_setting.HypercuboidSetting(beta=1.0, dimension=2, system_lengths=[1.0, 2.0])
        # Set these values so that the setting package is fully initialized.
        setting.set_number_of_node_levels(1)
        setting.set_number_of_root_nodes(1)
        setting.set_number_of_nodes_per_root_node(1)
        self._cells = CuboidCells(cells_per_side=[4, 5], neighbor_layers=0)

        for cell in self._cells.yield_cells():
            nearby_cells = self._cells.nearby_cells(cell)
            self.assertEqual(len(nearby_cells), 1)
            self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), {cell.identifier})

    def test_two_dimensional_neighbor_cell(self):
        self.setUpTwoDimensionalCells()
        for cell in self._cells.yield_cells():
            if cell.identifier == (0, 0):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (1, 0))
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, False))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (0, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, False))
                continue
            if cell.identifier == (1, 0):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (2, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (0, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (1, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, False))
                continue
            if cell.identifier == (2, 0):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (3, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (1, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (2, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, False))
                continue
            if cell.identifier == (3, 0):
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (2, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (3, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, False))
                continue

            if cell.identifier == (0, 1):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (1, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, False))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (0, 2))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (0, 0))
                continue
            if cell.identifier == (1, 1):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (2, 1))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (0, 1))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (1, 2))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (1, 0))
                continue
            if cell.identifier == (2, 1):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (3, 1))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (1, 1))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (2, 2))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (2, 0))
                continue
            if cell.identifier == (3, 1):
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (2, 1))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (3, 2))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (3, 0))
                continue

            if cell.identifier == (0, 2):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (1, 2))
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, False))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (0, 3))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (0, 1))
                continue
            if cell.identifier == (1, 2):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (2, 2))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (0, 2))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (1, 3))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (1, 1))
                continue
            if cell.identifier == (2, 2):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (3, 2))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (1, 2))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (2, 3))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (2, 1))
                continue
            if cell.identifier == (3, 2):
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (2, 2))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (3, 3))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (3, 1))
                continue

            if cell.identifier == (0, 3):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (1, 3))
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, False))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (0, 4))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (0, 2))
                continue
            if cell.identifier == (1, 3):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (2, 3))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (0, 3))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (1, 4))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (1, 2))
                continue
            if cell.identifier == (2, 3):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (3, 3))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (1, 3))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (2, 4))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (2, 2))
                continue
            if cell.identifier == (3, 3):
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (2, 3))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (3, 4))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (3, 2))
                continue

            if cell.identifier == (0, 4):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (1, 4))
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, False))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (0, 3))
                continue
            if cell.identifier == (1, 4):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (2, 4))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (0, 4))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (1, 3))
                continue
            if cell.identifier == (2, 4):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (3, 4))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (1, 4))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (2, 3))
                continue
            if cell.identifier == (3, 4):
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (2, 4))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (3, 3))
                continue
            self.fail("Unexpected cell returned by yield_cells method.")

    def test_two_dimensional_cell_min_position_belongs_to_cell(self):
        self.setUpTwoDimensionalCells()
        for cell in self._cells.yield_cells():
            self.assertIs(self._cells.position_to_cell(cell.cell_min), cell)

    def test_two_dimensional_cell_max_position_belongs_to_cell(self):
        self.setUpTwoDimensionalCells()
        for cell in self._cells.yield_cells():
            self.assertIs(self._cells.position_to_cell(cell.cell_max), cell)

    def setUpThreeDimensionalCells(self):
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        # Set these values so that the setting package is fully initialized.
        setting.set_number_of_node_levels(1)
        setting.set_number_of_root_nodes(1)
        setting.set_number_of_nodes_per_root_node(1)
        self._cells = CuboidCells(cells_per_side=[4, 2, 2], neighbor_layers=1)

        self._all_cell_identifiers = [
            (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0),
            (0, 1, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0),
            (0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1),
            (0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1)
        ]
        self._all_cell_min_positions = [
            [0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0], [0.75, 0.0, 0.0],
            [0.0, 0.5, 0.0], [0.25, 0.5, 0.0], [0.5, 0.5, 0.0], [0.75, 0.5, 0.0],
            [0.0, 0.0, 0.5], [0.25, 0.0, 0.5], [0.5, 0.0, 0.5], [0.75, 0.0, 0.5],
            [0.0, 0.5, 0.5], [0.25, 0.5, 0.5], [0.5, 0.5, 0.5], [0.75, 0.5, 0.5]
        ]
        self._all_cell_max_positions = [
            [0.25, 0.5, 0.5], [0.5, 0.5, 0.5], [0.75, 0.5, 0.5], [1.0, 0.5, 0.5],
            [0.25, 1.0, 0.5], [0.5, 1.0, 0.5], [0.75, 1.0, 0.5], [1.0, 1.0, 0.5],
            [0.25, 0.5, 1.0], [0.5, 0.5, 1.0], [0.75, 0.5, 1.0], [1.0, 0.5, 1.0],
            [0.25, 1.0, 1.0], [0.5, 1.0, 1.0], [0.75, 1.0, 1.0], [1.0, 1.0, 1.0],
        ]

    def test_three_dimensional_yield_cells(self):
        self.setUpThreeDimensionalCells()
        all_cells = list(self._cells.yield_cells())
        self.assertEqual(len(all_cells), 16)
        for index in range(16):
            cell_identifier = self._all_cell_identifiers[index]
            cell_min_position = self._all_cell_min_positions[index]
            cell_max_positions = self._all_cell_max_positions[index]
            found_index = None
            for cell_index, cell in enumerate(all_cells):
                if cell.identifier == cell_identifier:
                    self.assertIsNone(found_index)
                    found_index = cell_index
            self.assertIsNotNone(found_index)
            compare_cell = all_cells[found_index]
            self.assertEqual(cell_identifier, compare_cell.identifier)
            self.assertAlmostEqualSequence(cell_min_position, compare_cell.cell_min, places=13)
            self.assertAlmostEqualSequence(cell_max_positions, compare_cell.cell_max, places=13)

    def test_three_dimensional_position_to_cell(self):
        self.setUpThreeDimensionalCells()
        # Simply test some random positions. Here, we simply compare the identifiers of the returned cells. Their
        # cell_min and cell_max attributes where checked in the 'test_two_dimensional_yield_cells' method.
        self.assertEqual(self._cells.position_to_cell([1.0e-13, 1.0e-13, 1.0e-13]).identifier, (0, 0, 0))
        self.assertEqual(self._cells.position_to_cell([0.25 + 1.0e-13, 0.5 - 1.0e-13, 0.5 - 1.0e-13]).identifier,
                         (1, 0, 0))
        self.assertEqual(self._cells.position_to_cell([0.5 + 1.0e-13, 0.23, 0.43]).identifier, (2, 0, 0))
        self.assertEqual(self._cells.position_to_cell([0.75 + 1.0e-13, 0.38, 0.02]).identifier, (3, 0, 0))
        self.assertEqual(self._cells.position_to_cell([0.25 - 1.0e-13, 0.5 + 1.0e-13, 0.17]).identifier, (0, 1, 0))
        self.assertEqual(self._cells.position_to_cell([0.5 - 1.0e-13, 1.0 - 1.0e-13, 0.23]).identifier,
                         (1, 1, 0))
        self.assertEqual(self._cells.position_to_cell([0.75 - 1.0e-13, 0.82, 0.37]).identifier, (2, 1, 0))
        self.assertEqual(self._cells.position_to_cell([1.0 - 1.0e-13, 0.90, 0.09]).identifier, (3, 1, 0))
        self.assertEqual(self._cells.position_to_cell([0.18, 0.43, 0.5 + 1.0e-13]).identifier, (0, 0, 1))
        self.assertEqual(self._cells.position_to_cell([0.33, 0.38, 1.0 - 1.0e-13]).identifier, (1, 0, 1))
        self.assertEqual(self._cells.position_to_cell([0.6, 0.43, 0.93]).identifier, (2, 0, 1))
        self.assertEqual(self._cells.position_to_cell([0.81, 0.49, 0.52]).identifier, (3, 0, 1))
        self.assertEqual(self._cells.position_to_cell([0.02, 0.81, 0.67]).identifier, (0, 1, 1))
        self.assertEqual(self._cells.position_to_cell([0.44, 0.55, 0.73]).identifier, (1, 1, 1))
        self.assertEqual(self._cells.position_to_cell([0.67, 0.82, 0.87]).identifier, (2, 1, 1))
        self.assertEqual(self._cells.position_to_cell([0.99, 0.40, 0.59]).identifier, (3, 0, 1))

    def test_three_dimensional_position_to_cell_raises_error_if_position_outside_cell(self):
        self.setUpThreeDimensionalCells()
        with self.assertRaises(AssertionError):
            self._cells.position_to_cell([-1.0e-13, 0.4, 0.3])
        with self.assertRaises(AssertionError):
            self._cells.position_to_cell([0.1, -1.0e-13, 0.3])
        with self.assertRaises(AssertionError):
            self._cells.position_to_cell([0.1, 0.4, -1.0e-13])
        with self.assertRaises(AssertionError):
            self._cells.position_to_cell([1.0 + 1.0e-13, 0.4, 0.3])
        with self.assertRaises(AssertionError):
            self._cells.position_to_cell([0.1, 1.0 + 1.0e-13, 0.3])
        with self.assertRaises(AssertionError):
            self._cells.position_to_cell([0.1, 0.4, 1.0 + 1.0e-13])

    def test_three_dimensional_nearby_cells(self):
        self.setUpThreeDimensionalCells()

        for cell in self._cells.yield_cells():
            if cell.identifier == (0, 0, 0):
                expected_nearby_cells = {(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                                         (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (1, 0, 0):
                expected_nearby_cells = {(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0),
                                         (0, 0, 1), (1, 0, 1), (2, 0, 1), (0, 1, 1), (1, 1, 1), (2, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (2, 0, 0):
                expected_nearby_cells = {(1, 0, 0), (2, 0, 0), (3, 0, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0),
                                         (1, 0, 1), (2, 0, 1), (3, 0, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (3, 0, 0):
                expected_nearby_cells = {(2, 0, 0), (3, 0, 0), (2, 1, 0), (3, 1, 0),
                                         (2, 0, 1), (3, 0, 1), (2, 1, 1), (3, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue

            if cell.identifier == (0, 1, 0):
                expected_nearby_cells = {(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                                         (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (1, 1, 0):
                expected_nearby_cells = {(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0),
                                         (0, 0, 1), (1, 0, 1), (2, 0, 1), (0, 1, 1), (1, 1, 1), (2, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (2, 1, 0):
                expected_nearby_cells = {(1, 0, 0), (2, 0, 0), (3, 0, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0),
                                         (1, 0, 1), (2, 0, 1), (3, 0, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (3, 1, 0):
                expected_nearby_cells = {(2, 0, 0), (3, 0, 0), (2, 1, 0), (3, 1, 0),
                                         (2, 0, 1), (3, 0, 1), (2, 1, 1), (3, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue

            if cell.identifier == (0, 0, 1):
                expected_nearby_cells = {(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                                         (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (1, 0, 1):
                expected_nearby_cells = {(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0),
                                         (0, 0, 1), (1, 0, 1), (2, 0, 1), (0, 1, 1), (1, 1, 1), (2, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (2, 0, 1):
                expected_nearby_cells = {(1, 0, 0), (2, 0, 0), (3, 0, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0),
                                         (1, 0, 1), (2, 0, 1), (3, 0, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (3, 0, 1):
                expected_nearby_cells = {(2, 0, 0), (3, 0, 0), (2, 1, 0), (3, 1, 0),
                                         (2, 0, 1), (3, 0, 1), (2, 1, 1), (3, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue

            if cell.identifier == (0, 1, 1):
                expected_nearby_cells = {(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                                         (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (1, 1, 1):
                expected_nearby_cells = {(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0),
                                         (0, 0, 1), (1, 0, 1), (2, 0, 1), (0, 1, 1), (1, 1, 1), (2, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (2, 1, 1):
                expected_nearby_cells = {(1, 0, 0), (2, 0, 0), (3, 0, 0), (1, 1, 0), (2, 1, 0), (3, 1, 0),
                                         (1, 0, 1), (2, 0, 1), (3, 0, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue
            if cell.identifier == (3, 1, 1):
                expected_nearby_cells = {(2, 0, 0), (3, 0, 0), (2, 1, 0), (3, 1, 0),
                                         (2, 0, 1), (3, 0, 1), (2, 1, 1), (3, 1, 1)}
                nearby_cells = self._cells.nearby_cells(cell)
                self.assertEqual(len(nearby_cells), len(expected_nearby_cells))
                self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), expected_nearby_cells)
                continue

            self.fail("Unexpected cell returned by yield_cells method.")

    def test_three_dimensional_nearby_cells_zero_neighbor_layers(self):
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        # Set these values so that the setting package is fully initialized.
        setting.set_number_of_node_levels(1)
        setting.set_number_of_root_nodes(1)
        setting.set_number_of_nodes_per_root_node(1)
        self._cells = CuboidCells(cells_per_side=[4, 2, 2], neighbor_layers=0)

        for cell in self._cells.yield_cells():
            nearby_cells = self._cells.nearby_cells(cell)
            self.assertEqual(len(nearby_cells), 1)
            self.assertEqual(set(nearby_cell.identifier for nearby_cell in nearby_cells), {cell.identifier})

    def test_three_dimensional_neighbor_cell(self):
        self.setUpThreeDimensionalCells()
        for cell in self._cells.yield_cells():
            if cell.identifier == (0, 0, 0):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (1, 0, 0))
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, False))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (0, 1, 0))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, False))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, True).identifier, (0, 0, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, False))
                continue
            if cell.identifier == (1, 0, 0):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (2, 0, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (0, 0, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (1, 1, 0))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, False))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, True).identifier, (1, 0, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, False))
                continue
            if cell.identifier == (2, 0, 0):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (3, 0, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (1, 0, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (2, 1, 0))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, False))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, True).identifier, (2, 0, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, False))
                continue
            if cell.identifier == (3, 0, 0):
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (2, 0, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (3, 1, 0))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, False))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, True).identifier, (3, 0, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, False))
                continue

            if cell.identifier == (0, 1, 0):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (1, 1, 0))
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, False))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (0, 0, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, True).identifier, (0, 1, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, False))
                continue
            if cell.identifier == (1, 1, 0):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (2, 1, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (0, 1, 0))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (1, 0, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, True).identifier, (1, 1, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, False))
                continue
            if cell.identifier == (2, 1, 0):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (3, 1, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (1, 1, 0))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (2, 0, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, True).identifier, (2, 1, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, False))
                continue
            if cell.identifier == (3, 1, 0):
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (2, 1, 0))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (3, 0, 0))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, True).identifier, (3, 1, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, False))
                continue

            if cell.identifier == (0, 0, 1):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (1, 0, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, False))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (0, 1, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, False))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, False).identifier, (0, 0, 0))
                continue
            if cell.identifier == (1, 0, 1):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (2, 0, 1))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (0, 0, 1))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (1, 1, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, False))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, False).identifier, (1, 0, 0))
                continue
            if cell.identifier == (2, 0, 1):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (3, 0, 1))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (1, 0, 1))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (2, 1, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, False))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, False).identifier, (2, 0, 0))
                continue
            if cell.identifier == (3, 0, 1):
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (2, 0, 1))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, True).identifier, (3, 1, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, False))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, False).identifier, (3, 0, 0))
                continue

            if cell.identifier == (0, 1, 1):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (1, 1, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, False))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (0, 0, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, False).identifier, (0, 1, 0))
                continue
            if cell.identifier == (1, 1, 1):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (2, 1, 1))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (0, 1, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (1, 0, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, False).identifier, (1, 1, 0))
                continue
            if cell.identifier == (2, 1, 1):
                self.assertEqual(self._cells.neighbor_cell(cell, 0, True).identifier, (3, 1, 1))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (1, 1, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (2, 0, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, False).identifier, (2, 1, 0))
                continue
            if cell.identifier == (3, 1, 1):
                self.assertIsNone(self._cells.neighbor_cell(cell, 0, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 0, False).identifier, (2, 1, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 1, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 1, False).identifier, (3, 0, 1))
                self.assertIsNone(self._cells.neighbor_cell(cell, 2, True))
                self.assertEqual(self._cells.neighbor_cell(cell, 2, False).identifier, (3, 1, 0))
                continue
            self.fail("Unexpected cell returned by yield_cells method.")

    def test_three_dimensional_cell_min_position_belongs_to_cell(self):
        self.setUpThreeDimensionalCells()
        for cell in self._cells.yield_cells():
            self.assertIs(self._cells.position_to_cell(cell.cell_min), cell)

    def test_three_dimensional_cell_max_position_belongs_to_cell(self):
        self.setUpThreeDimensionalCells()
        for cell in self._cells.yield_cells():
            self.assertIs(self._cells.position_to_cell(cell.cell_max), cell)

    def test_cuboid_cells_raises_error_if_hypercuboid_setting_is_not_initialized(self):
        with self.assertRaises(ConfigurationError):
            CuboidCells(cells_per_side=[4, 2, 2], neighbor_layers=1)

    def test_cuboid_cells_raises_error_if_empty_cells_per_side_is_given(self):
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        # Set these values so that the setting package is fully initialized.
        setting.set_number_of_node_levels(1)
        setting.set_number_of_root_nodes(1)
        setting.set_number_of_nodes_per_root_node(1)
        with self.assertRaises(ConfigurationError):
            self._cells = CuboidCells(cells_per_side=[], neighbor_layers=1)

    def test_cuboid_cells_raises_error_if_too_many_cells_per_side_are_given(self):
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        # Set these values so that the setting package is fully initialized.
        setting.set_number_of_node_levels(1)
        setting.set_number_of_root_nodes(1)
        setting.set_number_of_nodes_per_root_node(1)
        with self.assertRaises(ConfigurationError):
            self._cells = CuboidCells(cells_per_side=[3, 3, 3, 3], neighbor_layers=1)

    def test_cuboid_cells_raises_error_if_negative_neighbor_layers_is_given(self):
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        # Set these values so that the setting package is fully initialized.
        setting.set_number_of_node_levels(1)
        setting.set_number_of_root_nodes(1)
        setting.set_number_of_nodes_per_root_node(1)
        with self.assertRaises(ConfigurationError):
            self._cells = CuboidCells(cells_per_side=[3, 3, 3, 3], neighbor_layers=-1)


if __name__ == '__main__':
    main()
