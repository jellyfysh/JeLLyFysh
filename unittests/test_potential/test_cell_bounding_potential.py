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
import contextlib
import os
from unittest import TestCase, main, mock
from activator.internal_state.cell_occupancy.cells import Cells
from estimator import Estimator
from potential.cell_bounding_potential import CellBoundingPotential
import setting


# We hardcode a simple cell class and its methods
def mock_cell_min(cell_index):
    if cell_index == 0:
        return [0, 0]
    elif cell_index == 1:
        return [1/4, 0]
    elif cell_index == 2:
        return [2/4, 0]
    elif cell_index == 3:
        return [3/4, 0]
    elif cell_index == 4:
        return [0, 1/4]
    elif cell_index == 5:
        return [1/4, 1/4]
    elif cell_index == 6:
        return [2/4, 1/4]
    elif cell_index == 7:
        return [3/4, 1/4]
    elif cell_index == 8:
        return [0, 2/4]
    elif cell_index == 9:
        return [1/4, 2/4]
    elif cell_index == 10:
        return [2/4, 2/4]
    elif cell_index == 11:
        return [3/4, 2/4]
    elif cell_index == 12:
        return [0, 3/4]
    elif cell_index == 13:
        return [1/4, 3/4]
    elif cell_index == 14:
        return [2/4, 3/4]
    elif cell_index == 15:
        return [3/4, 3/4]
    raise RuntimeError("Not allowed cell index!")


# Consider periodic boundaries
def mock_cell_max(cell_index):
    if cell_index == 0:
        return [1/4, 1/4]
    elif cell_index == 1:
        return [2/4, 1/4]
    elif cell_index == 2:
        return [3/4, 1/4]
    elif cell_index == 3:
        return [0, 1/4]
    elif cell_index == 4:
        return [1/4, 2/4]
    elif cell_index == 5:
        return [2/4, 2/4]
    elif cell_index == 6:
        return [3/4, 2/4]
    elif cell_index == 7:
        return [0, 2/4]
    elif cell_index == 8:
        return [1/4, 3/4]
    elif cell_index == 9:
        return [2/4, 3/4]
    elif cell_index == 10:
        return [3/4, 3/4]
    elif cell_index == 11:
        return [0, 3/4]
    elif cell_index == 12:
        return [1/4, 0]
    elif cell_index == 13:
        return [2/4, 0]
    elif cell_index == 14:
        return [3/4, 0]
    elif cell_index == 15:
        return [0, 0]
    raise RuntimeError("Not allowed cell index!")


def mock_excluded_cells(cell_index):
    if cell_index == 0:
        return [0, 1, 3, 4, 5, 7, 12, 13, 15]
    raise RuntimeError("Not allowed cell index")


# We hardcode some return values of the estimator and can by the same token check if the arguments are correct
def mock_derivative_bound(lower_corner, upper_corner, direction, calculate_lower_bound):
    if not calculate_lower_bound:
        # Between cells 2 and 0
        if lower_corner == [1/4, -1/4] and upper_corner == [3/4, 1/4] and direction == 0:
            return [0.2]
        # Between cells 6 and 0
        elif lower_corner == [1/4, 0] and upper_corner == [3/4, 2/4] and direction == 0:
            return [-0.6]
        # Between cells 8 and 0
        elif lower_corner == [-1/4, 1/4] and upper_corner == [1/4, 3/4] and direction == 0:
            return [0.8]
        # Between cells 9 and 0
        elif lower_corner == [0, 1/4] and upper_corner == [2/4, 3/4] and direction == 0:
            return [-0.9]
        # Between cells 10 and 0
        elif lower_corner == [1/4, 1/4] and upper_corner == [3/4, 3/4] and direction == 0:
            return [0.1]
        # Between cells 11 and 0
        elif lower_corner == [2/4, 1/4] and upper_corner == [1, 3/4] and direction == 0:
            return [-0.11]
        # Between cells 14 and 0
        elif lower_corner == [1/4, 2/4] and upper_corner == [3/4, 1] and direction == 0:
            return [0.14]

        # Between cells 2 and 0
        if lower_corner == [1 / 4, -1 / 4] and upper_corner == [3 / 4, 1 / 4] and direction == 1:
            return [-0.3]
        # Between cells 6 and 0
        elif lower_corner == [1 / 4, 0] and upper_corner == [3 / 4, 2 / 4] and direction == 1:
            return [0.7]
        # Between cells 8 and 0
        elif lower_corner == [-1 / 4, 1 / 4] and upper_corner == [1 / 4, 3 / 4] and direction == 1:
            return [-0.9]
        # Between cells 9 and 0
        elif lower_corner == [0, 1 / 4] and upper_corner == [2 / 4, 3 / 4] and direction == 1:
            return [1.0]
        # Between cells 10 and 0
        elif lower_corner == [1 / 4, 1 / 4] and upper_corner == [3 / 4, 3 / 4] and direction == 1:
            return [-0.2]
        # Between cells 11 and 0
        elif lower_corner == [2 / 4, 1 / 4] and upper_corner == [1, 3 / 4] and direction == 1:
            return [0.21]
        # Between cells 14 and 0
        elif lower_corner == [1 / 4, 2 / 4] and upper_corner == [3 / 4, 1] and direction == 1:
            return [-0.24]
        raise RuntimeError("Not allowed arguments!")
    else:
        # Between cells 2 and 0
        if lower_corner == [1 / 4, -1 / 4] and upper_corner == [3 / 4, 1 / 4] and direction == 0:
            return [0.2, -0.3]
        # Between cells 6 and 0
        elif lower_corner == [1 / 4, 0] and upper_corner == [3 / 4, 2 / 4] and direction == 0:
            return [0.6, 0.5]
        # Between cells 8 and 0
        elif lower_corner == [-1 / 4, 1 / 4] and upper_corner == [1 / 4, 3 / 4] and direction == 0:
            return [0.8, -0.8]
        # Between cells 9 and 0, not tested below -> set to None
        elif lower_corner == [0, 1 / 4] and upper_corner == [2 / 4, 3 / 4] and direction == 0:
            return [None, None]
        # Between cells 10 and 0
        elif lower_corner == [1 / 4, 1 / 4] and upper_corner == [3 / 4, 3 / 4] and direction == 0:
            return [None, None]
        # Between cells 11 and 0
        elif lower_corner == [2 / 4, 1 / 4] and upper_corner == [1, 3 / 4] and direction == 0:
            return [None, None]
        # Between cells 14 and 0
        elif lower_corner == [1 / 4, 2 / 4] and upper_corner == [3 / 4, 1] and direction == 0:
            return [None, None]

        # Between cells 2 and 0
        if lower_corner == [1 / 4, -1 / 4] and upper_corner == [3 / 4, 1 / 4] and direction == 1:
            return [0.3, 0.4]
        # Between cells 6 and 0
        elif lower_corner == [1 / 4, 0] and upper_corner == [3 / 4, 2 / 4] and direction == 1:
            return [-0.7, 0.7]
        # Between cells 8 and 0
        elif lower_corner == [-1 / 4, 1 / 4] and upper_corner == [1 / 4, 3 / 4] and direction == 1:
            return [0.9, 0.1]
        # Between cells 9 and 0
        elif lower_corner == [0, 1 / 4] and upper_corner == [2 / 4, 3 / 4] and direction == 1:
            return [None, None]
        # Between cells 10 and 0
        elif lower_corner == [1 / 4, 1 / 4] and upper_corner == [3 / 4, 3 / 4] and direction == 1:
            return [None, None]
        # Between cells 11 and 0
        elif lower_corner == [2 / 4, 1 / 4] and upper_corner == [1, 3 / 4] and direction == 1:
            return [None, None]
        # Between cells 14 and 0
        elif lower_corner == [1 / 4, 2 / 4] and upper_corner == [3 / 4, 1] and direction == 1:
            return [None, None]
        raise RuntimeError("Not allowed arguments!")


class TestCellBoundingPotential(TestCase):
    def setUp(self) -> None:
        setting.HypercubicSetting(beta=1.0, dimension=2, system_length=1.0)
        setting.set_number_of_node_levels(1)
        self._estimator_without_charge_mock = mock.MagicMock(spec_set=Estimator)
        self._estimator_without_charge_mock.derivative_bound.side_effect = mock_derivative_bound
        self._estimator_with_charge_mock = mock.MagicMock(spec_set=Estimator)
        self._estimator_with_charge_mock.derivative_bound.side_effect = mock_derivative_bound
        # Mock a 4 * 4 cell system
        cells_mock = mock.MagicMock(spec_set=Cells)
        cells_mock.yield_cells.side_effect = [iter([i for i in range(16)]) for _ in range(2)]
        cells_mock.cell_min.side_effect = mock_cell_min
        cells_mock.cell_max.side_effect = mock_cell_max
        cells_mock.excluded_cells = mock_excluded_cells
        self._cell_bounding_potential_without_charge = CellBoundingPotential(
            estimator=self._estimator_without_charge_mock)
        self._cell_bounding_potential_with_charge = CellBoundingPotential(estimator=self._estimator_with_charge_mock)

        # Redirect stdout to the null device while initializing the estimator
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                self._cell_bounding_potential_without_charge.initialize(cells_mock, 1.0, False)
                self._cell_bounding_potential_with_charge.initialize(cells_mock, 2.0, True)

    def tearDown(self) -> None:
        setting.reset()

    def test_initialize_of_estimator_without_charge_correctly_called(self):
        self._estimator_without_charge_mock.initialize.assert_called_once_with(1.0)

    def test_initialize_of_estimator_with_charge_correctly_called(self):
        self._estimator_with_charge_mock.initialize.assert_called_once_with(2.0)

    def test_displacement_derivative_cell_separation_two_direction_zero_without_charge(self):
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            0, 2, 1.0, 1.0, 0.3), 0.3 / 0.2)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            0, 2, 1.0, 1.0), 0.2)

    def test_displacement_derivative_cell_separation_two_direction_one_without_charge(self):
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            1, 2, 1.0, 1.0, 0.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            1, 2, 1.0, 1.0), -0.3)

    def test_displacement_derivative_cell_separation_six_direction_zero_without_charge(self):
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            0, 6, 1.0, 1.0, 0.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            0, 6, 1.0, 1.0), -0.6)

    def test_displacement_derivative_cell_separation_six_direction_one_without_charge(self):
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            1, 6, 1.0, 1.0, 1.3), 1.3 / 0.7)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            1, 6, 1.0, 1.0), 0.7)

    def test_displacement_derivative_cell_separation_eight_direction_zero_without_charge(self):
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            0, 8, 1.0, 1.0, 33.0), 33.0 / 0.8)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            0, 8, 1.0, 1.0), 0.8)

    def test_displacement_derivative_cell_separation_eight_direction_one_without_charge(self):
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            1, 8, 1.0, 1.0, 0.00001), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            1, 8, 1.0, 1.0), -0.9)

    def test_displacement_derivative_cell_separation_nine_direction_zero_without_charge(self):
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            0, 9, 1.0, 1.0, 11.0), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            0, 9, 1.0, 1.0), -0.9)

    def test_displacement_derivative_cell_separation_nine_direction_one_without_charge(self):
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            1, 9, 1.0, 1.0, 11.0), 11.0 / 1.0)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            1, 9, 1.0, 1.0), 1.0)

    def test_displacement_derivative_cell_separation_ten_direction_zero_without_charge(self):
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            0, 10, 1.0, 1.0, 0.3), 0.3 / 0.1)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            0, 10, 1.0, 1.0), 0.1)

    def test_displacement_derivative_cell_separation_ten_direction_one_without_charge(self):
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            1, 10, 1.0, 1.0, 11.0), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            1, 10, 1.0, 1.0), -0.2)

    def test_displacement_derivative_cell_separation_eleven_direction_zero_without_charge(self):
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            0, 11, 1.0, 1.0, 0.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            0, 11, 1.0, 1.0), -0.11)

    def test_displacement_derivative_cell_separation_eleven_direction_one_without_charge(self):
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            1, 11, 1.0, 1.0, 0.1), 0.1 / 0.21)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            1, 11, 1.0, 1.0), 0.21)

    def test_displacement_derivative_cell_separation_fourteen_direction_zero_without_charge(self):
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            0, 14, 1.0, 1.0, 0.1111), 0.1111 / 0.14)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            0, 14, 1.0, 1.0), 0.14)

    def test_displacement_derivative_cell_separation_fourteen_direction_one_without_charge(self):
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            1, 14, 1.0, 1.0, 0.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            1, 14, 1.0, 1.0), -0.24)

    def test_displacement_of_estimator_initialized_without_charge_with_non_simple_charges_raises_error(self):
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_without_charge.displacement(0, 2, 1.0, -1.0, 0.3)
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_without_charge.displacement(0, 2, 0.3, 1.0, 0.3)

    def test_displacement_derivative_cell_separation_two_direction_zero_with_charge_equal_charge(self):
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            0, 2, 1.0, 1.0, 0.3), 0.3 / 0.2 * 4.0)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            0, 2, 1.0, 1.0), 0.2 / 4.0)

    def test_displacement_derivative_cell_separation_two_direction_zero_with_charge_opposite_charge(self):
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            0, 2, 1.0, -1.0, 0.3), 0.3 / 0.3 * 4.0)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            0, 2, 1.0, -1.0), 0.3 / 4.0)

    def test_displacement_derivative_cell_separation_two_direction_one_with_charge_equal_charge(self):
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            1, 2, 1.0, 0.5, 0.1), 0.1 / (0.3 * 0.5) * 4.0)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            1, 2, 1.0, 0.5), 0.3 * 0.5 / 4.0)

    def test_displacement_derivative_cell_separation_two_direction_one_with_charge_opposite_charge(self):
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            1, 2, 1.0, -0.5, 0.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            1, 2, 1.0, -0.5), -0.4 * 0.5 / 4.0)

    def test_displacement_derivative_cell_separation_six_direction_zero_with_charge_equal_charge(self):
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            0, 6, 1.0, 1.1, 0.4), 0.4 / (0.6 * 1.1) * 4.0)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            0, 6, 1.0, 1.0), 0.6 * 1.1 / 4.0)

    def test_displacement_derivative_cell_separation_six_direction_zero_with_charge_opposite_charge(self):
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            0, 6, -2.0, 2.0, 1.1), float('inf'))
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            0, 6, -2.0, 2.0), -0.5 * 4.0 / 4.0)

    def test_displacement_derivative_cell_separation_six_direction_one_with_charge_equal_charge(self):
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            1, 6, 0.3, 0.3, 1.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            1, 6, 0.3, 0.3), -0.7 * 0.3 * 0.3 / 4.0)

    def test_displacement_derivative_cell_separation_six_direction_one_with_charge_opposite_charge(self):
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            1, 6, -0.3, 0.3, 1.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            1, 6, -0.3, 0.3), -0.7 * 0.3 * 0.3 / 4.0)

    def test_displacement_derivative_cell_separation_eight_direction_zero_with_charge_equal_charge(self):
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            0, 8, 1.0, -1.0, 33.0), 33.0 / 0.8 * 4.0)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            0, 8, 1.0, -1.0), 0.8 / 4.0)

    def test_displacement_derivative_cell_separation_eight_direction_zero_with_charge_opposite_charge(self):
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            0, 8, 1.0, -1.0, 33.0), 33.0 / 0.8 * 4.0)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            0, 8, 1.0, -1.0), 0.8 / 4.0)

    def test_displacement_derivative_cell_separation_eight_direction_one_with_charge_equal_charge(self):
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            1, 8, -0.1, -1.0, 0.00001), 0.00001 / (0.9 * 0.1) * 4.0)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            1, 8, -0.1, -1.0), 0.9 * 0.1 / 4.0)

    def test_displacement_derivative_cell_separation_eight_direction_one_with_charge_opposite_charge(self):
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            1, 8, 0.1, -1.0, 0.00001), float('inf'))
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            1, 8, 0.1, -1.0), -0.1 * 0.1 / 4.0)


if __name__ == '__main__':
    main()
