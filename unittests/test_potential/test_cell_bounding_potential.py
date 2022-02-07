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
import contextlib
import os
from unittest import TestCase, main, mock
from jellyfysh.activator.internal_state.cell_occupancy.cells.cuboid_cells import CuboidCells
from jellyfysh.activator.internal_state.cell_occupancy.cells.cuboid_periodic_cells import CuboidPeriodicCells
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.estimator import Estimator
from jellyfysh.potential.cell_bounding_potential import CellBoundingPotential
import jellyfysh.setting as setting
from jellyfysh.setting import hypercubic_setting


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


def sequences_almost_equal(sequence_one, sequence_two, precision=1.0e-13):
    if len(sequence_one) == len(sequence_two):
        return all(abs(sequence_one[index] - sequence_two[index]) < precision for index in range(len(sequence_one)))
    return False


# We hardcode some return values of the estimator and can by the same token check if the arguments are correct
def mock_derivative_bound(lower_corner, upper_corner, direction, calculate_lower_bound):
    if not calculate_lower_bound:
        if direction == 0:
            # Used by self._cell_bounding_potential_without_charge
            # Between cells 2 and 0
            if sequences_almost_equal(lower_corner, [1/4, -1/4]) and sequences_almost_equal(upper_corner, [3/4, 1/4]):
                return [0.2]
            # Between cells 6 and 0
            elif sequences_almost_equal(lower_corner, [1/4, 0]) and sequences_almost_equal(upper_corner, [3/4, 2/4]):
                return [-0.6]
            # Between cells 8 and 0
            elif sequences_almost_equal(lower_corner, [-1/4, 1/4]) and sequences_almost_equal(upper_corner, [1/4, 3/4]):
                return [0.8]
            # Between cells 9 and 0
            elif sequences_almost_equal(lower_corner, [0, 1/4]) and sequences_almost_equal(upper_corner, [2/4, 3/4]):
                return [-0.9]
            # Between cells 10 and 0
            elif sequences_almost_equal(lower_corner, [1/4, 1/4]) and sequences_almost_equal(upper_corner, [3/4, 3/4]):
                return [0.1]
            # Between cells 11 and 0
            elif sequences_almost_equal(lower_corner, [2/4, 1/4]) and sequences_almost_equal(upper_corner, [1, 3/4]):
                return [-0.11]
            # Between cells 14 and 0
            elif sequences_almost_equal(lower_corner, [1/4, 2/4]) and sequences_almost_equal(upper_corner, [3/4, 1]):
                return [0.14]
        elif direction == 1:
            # Between cells 2 and 0
            if sequences_almost_equal(lower_corner, [1/4, -1/4]) and sequences_almost_equal(upper_corner, [3/4, 1/4]):
                return [-0.3]
            # Between cells 6 and 0
            elif sequences_almost_equal(lower_corner, [1/4, 0]) and sequences_almost_equal(upper_corner, [3/4, 2/4]):
                return [0.7]
            # Between cells 8 and 0
            elif sequences_almost_equal(lower_corner, [-1/4, 1/4]) and sequences_almost_equal(upper_corner, [1/4, 3/4]):
                return [-0.9]
            # Between cells 9 and 0
            elif sequences_almost_equal(lower_corner, [0, 1/4]) and sequences_almost_equal(upper_corner, [2/4, 3/4]):
                return [1.0]
            # Between cells 10 and 0
            elif sequences_almost_equal(lower_corner, [1/4, 1/4]) and sequences_almost_equal(upper_corner, [3/4, 3/4]):
                return [-0.2]
            # Between cells 11 and 0
            elif sequences_almost_equal(lower_corner, [2/4, 1/4]) and sequences_almost_equal(upper_corner, [1, 3/4]):
                return [0.21]
            # Between cells 14 and 0
            elif sequences_almost_equal(lower_corner, [1/4, 2/4]) and sequences_almost_equal(upper_corner, [3/4, 1]):
                return [-0.24]
    else:
        if direction == 0:
            # Used by self._cell_bounding_potential_with_charge
            # Between cells 2 and 0
            if sequences_almost_equal(lower_corner, [1/4, -1/4]) and sequences_almost_equal(upper_corner, [3/4, 1/4]):
                return [0.2, -0.3]
            # Between cells 6 and 0
            elif sequences_almost_equal(lower_corner, [1/4, 0]) and sequences_almost_equal(upper_corner, [3/4, 2/4]):
                return [0.6, 0.5]
            # Between cells 8 and 0
            elif sequences_almost_equal(lower_corner, [-1/4, 1/4]) and sequences_almost_equal(upper_corner, [1/4, 3/4]):
                return [0.8, -0.8]
            # Between cells 9 and 0, not tested below -> set to None
            elif sequences_almost_equal(lower_corner, [0, 1/4]) and sequences_almost_equal(upper_corner, [2/4, 3/4]):
                return [None, None]
            # Between cells 10 and 0
            elif sequences_almost_equal(lower_corner, [1/4, 1/4]) and sequences_almost_equal(upper_corner, [3/4, 3/4]):
                return [None, None]
            # Between cells 11 and 0
            elif sequences_almost_equal(lower_corner, [2/4, 1/4]) and sequences_almost_equal(upper_corner, [1, 3/4]):
                return [None, None]
            # Between cells 14 and 0
            elif sequences_almost_equal(lower_corner, [1/4, 2/4]) and sequences_almost_equal(upper_corner, [3/4, 1]):
                return [None, None]
        elif direction == 1:
            # Between cells 2 and 0
            if sequences_almost_equal(lower_corner, [1/4, -1/4]) and sequences_almost_equal(upper_corner, [3/4, 1/4]):
                return [0.3, 0.4]
            # Between cells 6 and 0
            elif sequences_almost_equal(lower_corner, [1/4, 0]) and sequences_almost_equal(upper_corner, [3/4, 2/4]):
                return [-0.7, 0.7]
            # Between cells 8 and 0
            elif sequences_almost_equal(lower_corner, [-1/4, 1/4]) and sequences_almost_equal(upper_corner, [1/4, 3/4]):
                return [0.9, 0.1]
            # Between cells 9 and 0
            elif sequences_almost_equal(lower_corner, [0, 1/4]) and sequences_almost_equal(upper_corner, [2/4, 3/4]):
                return [None, None]
            # Between cells 10 and 0
            elif sequences_almost_equal(lower_corner, [1/4, 1/4]) and sequences_almost_equal(upper_corner, [3/4, 3/4]):
                return [None, None]
            # Between cells 11 and 0
            elif sequences_almost_equal(lower_corner, [2/4, 1/4]) and sequences_almost_equal(upper_corner, [1, 3/4]):
                return [None, None]
            # Between cells 14 and 0
            elif sequences_almost_equal(lower_corner, [1/4, 2/4]) and sequences_almost_equal(upper_corner, [3/4, 1]):
                return [None, None]
    raise RuntimeError("Not allowed arguments!")


# We hardcode some return values of the estimator and can by the same token check if the arguments are correct
def mock_charge_correction_factor(active_charges, target_charges):
    if active_charges == 1.0 and target_charges == 1.0:
        return 1.0
    elif active_charges == 1.0 and target_charges == -1.0:
        return -1.0
    elif active_charges == 1.0 and target_charges == 0.5:
        return 0.5
    elif active_charges == 1.0 and target_charges == -0.5:
        return -0.5
    elif active_charges == 1.0 and target_charges == 1.1:
        return 1.1
    elif active_charges == -2.0 and target_charges == 2.0:
        return -4.0
    elif active_charges == 0.3 and target_charges == 0.3:
        return 0.09
    elif active_charges == -0.3 and target_charges == 0.3:
        return -0.09
    elif active_charges == -0.1 and target_charges == -1.0:
        return 0.1
    elif active_charges == 0.1 and target_charges == -1.0:
        return -0.1
    raise RuntimeError("Not allowed arguments!")


class TestCellBoundingPotential(TestCase):
    def setUp(self) -> None:
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=2, system_length=1.0)
        setting.set_number_of_node_levels(1)
        setting.set_number_of_nodes_per_root_node(1)
        setting.set_number_of_root_nodes(1)
        self._estimator_mock = mock.MagicMock(spec_set=Estimator)
        self._estimator_mock.derivative_bound.side_effect = mock_derivative_bound
        self._estimator_mock.charge_correction_factor = mock_charge_correction_factor
        # self._cells has the cell 0 with cell_min at the origin as the zero cell.
        self._cells = CuboidPeriodicCells(cells_per_side=[4, 4], neighbor_layers=1)
        self._cell_bounding_potential_without_charge = CellBoundingPotential(
            estimator=self._estimator_mock)
        self._cell_bounding_potential_with_charge = CellBoundingPotential(estimator=self._estimator_mock)

        # Redirect stdout to the null device while initializing the estimator
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                self._cell_bounding_potential_without_charge.initialize(self._cells, False)
                self._cell_bounding_potential_with_charge.initialize(self._cells, True)

    def tearDown(self) -> None:
        setting.reset()

    def test_displacement_derivative_cell_separation_two_direction_zero_without_charge(self):
        # Cell separation 2 (because the zero cell 0 has its cell_min at the origin).
        # The time displacement depends on the value of the non-vanishing component of the velocity.
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, 1.0, 0.3), 0.3 / 0.2)
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [3.0, 0.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, 1.0, 0.3), 0.3 / 0.2 / 3.0)
        # The derivative does not depend on the absolute value of the velocity but only on the non-vanishing component.
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, 1.0), 0.2)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [3.0, 0.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, 1.0), 0.2 * 3.0)

    def test_displacement_derivative_cell_separation_two_direction_one_without_charge(self):
        # Cell separation 2.
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, 1.0, 0.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [0.0, 3.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, 1.0, 0.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, 1.0), -0.3)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [0.0, 3.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, 1.0), -0.3 * 3.0)

    def test_displacement_derivative_cell_separation_six_direction_zero_without_charge(self):
        # Cell separation 6.
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.3]), 1.0, 1.0, 0.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [3.0, 0.0], self._cells.position_to_cell([0.6, 0.3]), 1.0, 1.0, 0.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.3]), 1.0, 1.0), -0.6)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [3.0, 0.0], self._cells.position_to_cell([0.6, 0.3]), 1.0, 1.0), -0.6 * 3.0)

    def test_displacement_derivative_cell_separation_six_direction_one_without_charge(self):
        # Cell separation 6.
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.3]), 1.0, 1.0, 1.3), 1.3 / 0.7)
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [0.0, 3.0], self._cells.position_to_cell([0.6, 0.3]), 1.0, 1.0, 1.3), 1.3 / 0.7 / 3.0)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.3]), 1.0, 1.0), 0.7)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [0.0, 3.0], self._cells.position_to_cell([0.6, 0.3]), 1.0, 1.0), 0.7 * 3.0)

    def test_displacement_derivative_cell_separation_eight_direction_zero_without_charge(self):
        # Cell separation 8.
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [1.0, 0.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, 1.0, 33.0), 33.0 / 0.8)
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [3.0, 0.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, 1.0, 33.0), 33.0 / 0.8 / 3.0)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [1.0, 0.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, 1.0), 0.8)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [3.0, 0.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, 1.0), 0.8 * 3.0)

    def test_displacement_derivative_cell_separation_eight_direction_one_without_charge(self):
        # Cell separation 8.
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [0.0, 1.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, 1.0, 0.00001), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [0.0, 3.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, 1.0, 0.00001), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [0.0, 1.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, 1.0), -0.9)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [0.0, 3.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, 1.0), -0.9 * 3.0)

    def test_displacement_derivative_cell_separation_nine_direction_zero_without_charge(self):
        # Cell separation 9.
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [1.0, 0.0], self._cells.position_to_cell([0.3, 0.6]), 1.0, 1.0, 11.0), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [3.0, 0.0], self._cells.position_to_cell([0.3, 0.6]), 1.0, 1.0, 11.0), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [1.0, 0.0], self._cells.position_to_cell([0.3, 0.6]), 1.0, 1.0), -0.9)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [3.0, 0.0], self._cells.position_to_cell([0.3, 0.6]), 1.0, 1.0), -0.9 * 3.0)

    def test_displacement_derivative_cell_separation_nine_direction_one_without_charge(self):
        # Cell separation 9.
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [0.0, 1.0], self._cells.position_to_cell([0.3, 0.6]), 1.0, 1.0, 11.0), 11.0 / 1.0)
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [0.0, 3.0], self._cells.position_to_cell([0.3, 0.6]), 1.0, 1.0, 11.0), 11.0 / 1.0 / 3.0)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [0.0, 1.0], self._cells.position_to_cell([0.3, 0.6]), 1.0, 1.0), 1.0)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [0.0, 3.0], self._cells.position_to_cell([0.3, 0.6]), 1.0, 1.0), 1.0 * 3.0)

    def test_displacement_derivative_cell_separation_ten_direction_zero_without_charge(self):
        # Cell separation 10.
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.6]), 1.0, 1.0, 0.3), 0.3 / 0.1)
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [3.0, 0.0], self._cells.position_to_cell([0.6, 0.6]), 1.0, 1.0, 0.3), 0.3 / 0.1 / 3.0)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.6]), 1.0, 1.0), 0.1)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [3.0, 0.0], self._cells.position_to_cell([0.6, 0.6]), 1.0, 1.0), 0.1 * 3.0)

    def test_displacement_derivative_cell_separation_ten_direction_one_without_charge(self):
        # Cell separation 10.
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.6]), 1.0, 1.0, 11.0), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [0.0, 3.0], self._cells.position_to_cell([0.6, 0.6]), 1.0, 1.0, 11.0), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.6]), 1.0, 1.0), -0.2)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [0.0, 3.0], self._cells.position_to_cell([0.6, 0.6]), 1.0, 1.0), -0.2 * 3.0)

    def test_displacement_derivative_cell_separation_eleven_direction_zero_without_charge(self):
        # Cell separation 11.
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [1.0, 0.0], self._cells.position_to_cell([0.8, 0.6]), 1.0, 1.0, 0.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [3.0, 0.0], self._cells.position_to_cell([0.8, 0.6]), 1.0, 1.0, 0.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [1.0, 0.0], self._cells.position_to_cell([0.8, 0.6]), 1.0, 1.0), -0.11)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [3.0, 0.0], self._cells.position_to_cell([0.8, 0.6]), 1.0, 1.0), -0.11 * 3.0)

    def test_displacement_derivative_cell_separation_eleven_direction_one_without_charge(self):
        # Cell separation 11.
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [0.0, 1.0], self._cells.position_to_cell([0.8, 0.6]), 1.0, 1.0, 0.1), 0.1 / 0.21)
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [0.0, 3.0], self._cells.position_to_cell([0.8, 0.6]), 1.0, 1.0, 0.1), 0.1 / 0.21 / 3.0)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [0.0, 1.0], self._cells.position_to_cell([0.8, 0.6]), 1.0, 1.0), 0.21)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [0.0, 3.0], self._cells.position_to_cell([0.8, 0.6]), 1.0, 1.0), 0.21 * 3.0)

    def test_displacement_derivative_cell_separation_fourteen_direction_zero_without_charge(self):
        # Cell separation 14.
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.8]), 1.0, 1.0, 0.1111), 0.1111 / 0.14)
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [3.0, 0.0], self._cells.position_to_cell([0.6, 0.8]), 1.0, 1.0, 0.1111), 0.1111 / 0.14 / 3.0)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.8]), 1.0, 1.0), 0.14)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [3.0, 0.0], self._cells.position_to_cell([0.6, 0.8]), 1.0, 1.0), 0.14 * 3.0)

    def test_displacement_derivative_cell_separation_fourteen_direction_one_without_charge(self):
        # Cell separation 14.
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.8]), 1.0, 1.0, 0.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.displacement(
            [0.0, 3.0], self._cells.position_to_cell([0.6, 0.8]), 1.0, 1.0, 0.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.8]), 1.0, 1.0), -0.24)
        self.assertEqual(self._cell_bounding_potential_without_charge.derivative(
            [0.0, 3.0], self._cells.position_to_cell([0.6, 0.8]), 1.0, 1.0), -0.24 * 3.0)

    def test_displacement_of_estimator_initialized_without_charge_with_non_simple_charges_raises_error(self):
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_without_charge.displacement(
                [1.0, 0.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, -1.0, 0.3)
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_without_charge.displacement(
                [1.0, 0.0], self._cells.position_to_cell([0.6, 0.8]), 0.3, 1.0, 0.3)

    def test_displacement_derivative_cell_separation_two_direction_zero_with_charge_equal_charge(self):
        # Cell separation 2.
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, 1.0, 0.3), 0.3 / 0.2)
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [3.1, 0.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, 1.0, 0.3), 0.3 / 0.2 / 3.1)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, 1.0), 0.2)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [3.1, 0.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, 1.0), 0.2 * 3.1)

    def test_displacement_derivative_cell_separation_two_direction_zero_with_charge_opposite_charge(self):
        # Cell separation 2.
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, -1.0, 0.3), 0.3 / 0.3)
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [3.1, 0.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, -1.0, 0.3), 0.3 / 0.3 / 3.1)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, -1.0), 0.3)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [3.1, 0.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, -1.0), 0.3 * 3.1)

    def test_displacement_derivative_cell_separation_two_direction_one_with_charge_equal_charge(self):
        # Cell separation 2.
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, 0.5, 0.1), 0.1 / (0.3 * 0.5))
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [0.0, 3.1], self._cells.position_to_cell([0.6, 0.1]), 1.0, 0.5, 0.1), 0.1 / (0.3 * 0.5) / 3.1)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, 0.5), 0.3 * 0.5)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [0.0, 3.1], self._cells.position_to_cell([0.6, 0.1]), 1.0, 0.5), 0.3 * 0.5 * 3.1)

    def test_displacement_derivative_cell_separation_two_direction_one_with_charge_opposite_charge(self):
        # Cell separation 2.
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, -0.5, 0.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [0.0, 3.1], self._cells.position_to_cell([0.6, 0.1]), 1.0, -0.5, 0.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.1]), 1.0, -0.5), -0.4 * 0.5)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [0.0, 3.1], self._cells.position_to_cell([0.6, 0.1]), 1.0, -0.5), -0.4 * 0.5 * 3.1)

    def test_displacement_derivative_cell_separation_six_direction_zero_with_charge_equal_charge(self):
        # Cell separation 6.
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.3]), 1.0, 1.1, 0.4), 0.4 / (0.6 * 1.1))
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [3.1, 0.0], self._cells.position_to_cell([0.6, 0.3]), 1.0, 1.1, 0.4), 0.4 / (0.6 * 1.1) / 3.1)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.3]), 1.0, 1.1), 0.6 * 1.1)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [3.1, 0.0], self._cells.position_to_cell([0.6, 0.3]), 1.0, 1.1), 0.6 * 1.1 * 3.1)

    def test_displacement_derivative_cell_separation_six_direction_zero_with_charge_opposite_charge(self):
        # Cell separation 6.
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.3]), -2.0, 2.0, 1.1), float('inf'))
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [3.1, 0.0], self._cells.position_to_cell([0.6, 0.3]), -2.0, 2.0, 1.1), float('inf'))
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [1.0, 0.0], self._cells.position_to_cell([0.6, 0.3]), -2.0, 2.0), -0.5 * 4.0)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [3.1, 0.0], self._cells.position_to_cell([0.6, 0.3]), -2.0, 2.0), -0.5 * 4.0 * 3.1)

    def test_displacement_derivative_cell_separation_six_direction_one_with_charge_equal_charge(self):
        # Cell separation 6.
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.3]), 0.3, 0.3, 1.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [0.0, 3.1], self._cells.position_to_cell([0.6, 0.3]), 0.3, 0.3, 1.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.3]), 0.3, 0.3), -0.7 * 0.3 * 0.3)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [0.0, 3.1], self._cells.position_to_cell([0.6, 0.3]), 0.3, 0.3), -0.7 * 0.3 * 0.3 * 3.1)

    def test_displacement_derivative_cell_separation_six_direction_one_with_charge_opposite_charge(self):
        # Cell separation 6.
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.3]), -0.3, 0.3, 1.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [0.0, 3.1], self._cells.position_to_cell([0.6, 0.3]), -0.3, 0.3, 1.3), float('inf'))
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [0.0, 1.0], self._cells.position_to_cell([0.6, 0.3]), -0.3, 0.3), -0.7 * 0.3 * 0.3)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [0.0, 3.1], self._cells.position_to_cell([0.6, 0.3]), -0.3, 0.3), -0.7 * 0.3 * 0.3 * 3.1)

    def test_displacement_derivative_cell_separation_eight_direction_zero_with_charge_equal_charge(self):
        # Cell separation 8.
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [1.0, 0.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, -1.0, 33.0), 33.0 / 0.8)
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [3.1, 0.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, -1.0, 33.0), 33.0 / 0.8 / 3.1)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [1.0, 0.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, -1.0), 0.8)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [3.1, 0.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, -1.0), 0.8 * 3.1)

    def test_displacement_derivative_cell_separation_eight_direction_zero_with_charge_opposite_charge(self):
        # Cell separation 8.
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [1.0, 0.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, -1.0, 33.0), 33.0 / 0.8)
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [3.1, 0.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, -1.0, 33.0), 33.0 / 0.8 / 3.1)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [1.0, 0.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, -1.0), 0.8)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [3.1, 0.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, -1.0), 0.8 * 3.1)

    def test_displacement_derivative_cell_separation_eight_direction_one_with_charge_equal_charge(self):
        # Cell separation 8.
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [0.0, 1.0], self._cells.position_to_cell([0.1, 0.6]), -0.1, -1.0, 0.00001), 0.00001 / (0.9 * 0.1))
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [0.0, 3.1], self._cells.position_to_cell([0.1, 0.6]), -0.1, -1.0, 0.00001), 0.00001 / (0.9 * 0.1) / 3.1)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [0.0, 1.0], self._cells.position_to_cell([0.1, 0.6]), -0.1, -1.0), 0.9 * 0.1)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [0.0, 3.1], self._cells.position_to_cell([0.1, 0.6]), -0.1, -1.0), 0.9 * 0.1 * 3.1)

    def test_displacement_derivative_cell_separation_eight_direction_one_with_charge_opposite_charge(self):
        # Cell separation 8.
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [0.0, 1.0], self._cells.position_to_cell([0.1, 0.6]), 0.1, -1.0, 0.00001), float('inf'))
        self.assertEqual(self._cell_bounding_potential_with_charge.displacement(
            [0.0, 3.1], self._cells.position_to_cell([0.1, 0.6]), 0.1, -1.0, 0.00001), float('inf'))
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [0.0, 1.0], self._cells.position_to_cell([0.1, 0.6]), 0.1, -1.0), -0.1 * 0.1)
        self.assertEqual(self._cell_bounding_potential_with_charge.derivative(
            [0.0, 3.1], self._cells.position_to_cell([0.1, 0.6]), 0.1, -1.0), -0.1 * 0.1 * 3.1)

    def test_number_separation_arguments_is_one(self):
        self.assertEqual(self._cell_bounding_potential_with_charge.number_separation_arguments, 1)
        self.assertEqual(self._cell_bounding_potential_without_charge.number_separation_arguments, 1)

    def test_number_charge_arguments_is_two(self):
        self.assertEqual(self._cell_bounding_potential_with_charge.number_charge_arguments, 2)
        self.assertEqual(self._cell_bounding_potential_without_charge.number_charge_arguments, 2)

    def test_potential_change_argument_required(self):
        self.assertTrue(self._cell_bounding_potential_with_charge.potential_change_required)
        self.assertTrue(self._cell_bounding_potential_without_charge.potential_change_required)

    def test_velocity_zero_raises_error(self):
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_with_charge.displacement(
                [0.0, 0.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, -1.0, 33.0)
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_with_charge.derivative(
                [0.0, 0.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, -1.0)
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_without_charge.displacement(
                [0.0, 0.0], self._cells.position_to_cell([0.6, 0.8]), 1.0, 1.0, 0.1111)
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_without_charge.derivative(
                [0.0, 0.0], self._cells.position_to_cell([0.6, 0.8]), 1.0, 1.0)

    def test_negative_velocity_along_axis_raises_error(self):
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_with_charge.displacement(
                [-1.0, 0.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, -1.0, 33.0)
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_with_charge.derivative(
                [0.0, -1.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, -1.0)
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_without_charge.displacement(
                [-1.0, 0.0], self._cells.position_to_cell([0.6, 0.8]), 1.0, 1.0, 0.1111)
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_without_charge.derivative(
                [0.0, -1.0], self._cells.position_to_cell([0.6, 0.8]), 1.0, 1.0)

    def test_velocity_not_parallel_to_axis_raises_error(self):
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_with_charge.displacement(
                [1.0, 2.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, -1.0, 33.0)
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_with_charge.derivative(
                [1.0, 2.0], self._cells.position_to_cell([0.1, 0.6]), 1.0, -1.0)
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_without_charge.displacement(
                [1.0, 2.0], self._cells.position_to_cell([0.6, 0.8]), 1.0, 1.0, 0.1111)
        with self.assertRaises(AssertionError):
            self._cell_bounding_potential_without_charge.derivative(
                [1.0, 2.0], self._cells.position_to_cell([0.6, 0.8]), 1.0, 1.0)

    def test_initialize_with_not_periodic_cells_raises_error(self):
        cells = CuboidCells(cells_per_side=[4, 4], neighbor_layers=1)
        cell_bounding_potential = CellBoundingPotential(estimator=self._estimator_mock)
        # Redirect stdout to the null device while initializing the estimator
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            with self.assertRaises(ConfigurationError):
                cell_bounding_potential.initialize(cells, False)


if __name__ == '__main__':
    main()
