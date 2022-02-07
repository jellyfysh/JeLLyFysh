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
from unittest import TestCase, main, mock
import os
from jellyfysh.activator.internal_state.cell_occupancy.cells import PeriodicCells
from jellyfysh.base.node import Node
from jellyfysh.base.unit import Unit
from jellyfysh.estimator import Estimator
from jellyfysh.event_handler.leaf_unit_cell_veto_event_handler import LeafUnitCellVetoEventHandler
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


# We hardcode some return values of the estimator and can by the same token check if the arguments are correct
def mock_derivative_bound(lower_corner, upper_corner, direction, calculate_lower_bound):
    if calculate_lower_bound:
        # return upper_bound, lower_bound, max(upper_bound, -lower_bound) goes into Walker table
        # Between cells 2 and 0
        if lower_corner == [1/4, -1/4] and upper_corner == [3/4, 1/4] and direction == 0:
            # Inserts 0.3 in Walker table
            return [0.2, -0.3]
        # Between cells 6 and 0
        elif lower_corner == [1/4, 0] and upper_corner == [3/4, 2/4] and direction == 0:
            # Inserts 0.7 in Walker table
            return [-0.6, -0.7]
        # Between cells 8 and 0
        elif lower_corner == [-1/4, 1/4] and upper_corner == [1/4, 3/4] and direction == 0:
            # Inserts 0.8 in Walker table
            return [0.8, 0.2]
        # Between cells 9 and 0
        elif lower_corner == [0, 1/4] and upper_corner == [2/4, 3/4] and direction == 0:
            # Inserts 1.1 in Walker table
            return [-0.9, -1.1]
        # Between cells 10 and 0
        elif lower_corner == [1/4, 1/4] and upper_corner == [3/4, 3/4] and direction == 0:
            # Inserts 0.1 in Walker table
            return [0.1, -0.05]
        # Between cells 11 and 0
        elif lower_corner == [2/4, 1/4] and upper_corner == [1, 3/4] and direction == 0:
            # Inserts 0.2 in Walker table
            return [-0.11, -0.2]
        # Between cells 14 and 0
        elif lower_corner == [1/4, 2/4] and upper_corner == [3/4, 1] and direction == 0:
            # Inserts 0.3 in Walker table
            return [0.3, 0.1]

        # Between cells 2 and 0
        if lower_corner == [1 / 4, -1 / 4] and upper_corner == [3 / 4, 1 / 4] and direction == 1:
            # Inserts 0.4 in Walker table
            return [-0.3, -0.4]
        # Between cells 6 and 0
        elif lower_corner == [1 / 4, 0] and upper_corner == [3 / 4, 2 / 4] and direction == 1:
            # Inserts 0.7 in Walker table
            return [0.7, 0.7]
        # Between cells 8 and 0
        elif lower_corner == [-1 / 4, 1 / 4] and upper_corner == [1 / 4, 3 / 4] and direction == 1:
            # Inserts 0.9 in Walker table
            return [-0.9, -0.9]
        # Between cells 9 and 0
        elif lower_corner == [0, 1 / 4] and upper_corner == [2 / 4, 3 / 4] and direction == 1:
            # Inserts 1.0 in Walker table
            return [1.0, 0.5]
        # Between cells 10 and 0
        elif lower_corner == [1 / 4, 1 / 4] and upper_corner == [3 / 4, 3 / 4] and direction == 1:
            # Inserts 0.3 in Walker table
            return [-0.2, -0.3]
        # Between cells 11 and 0
        elif lower_corner == [2 / 4, 1 / 4] and upper_corner == [1, 3 / 4] and direction == 1:
            # Inserts 0.4 in Walker table
            return [0.21, -0.4]
        # Between cells 14 and 0
        elif lower_corner == [1 / 4, 2 / 4] and upper_corner == [3 / 4, 1] and direction == 1:
            # Inserts 0.5 in Walker table
            return [-0.24, -0.5]
        raise RuntimeError("Not allowed arguments!")
    raise RuntimeError("Not allowed arguments!")


"""
Direction of motion = 0 --> Total rate 3.5, number items 7, mean rate 3.5/7 = 0.5
Large rates: 0.7 (6), 0.8 (8), 1.1 (9)
Small rates: 0.3 (2), 0.1 (10), 0.2 (11), 0.3 (14)
Table:
0.2 (9)  | 0.3 (9)  | 0.4 (9)  | 0.3 (8) |         | 0.2 (6) |
0.3 (14) | 0.2 (11) | 0.1 (10) | 0.2 (9) | 0.5 (8) | 0.3 (2) | 0.5 (6)

Direction of motion = 1 --> Total rate 4.2, number items 7, mean rate 4.2/7 = 0.6
Large rates: 0.7 (6), 0.9 (8), 1.0 (9) 
Small rates: 0.4 (2), 0.3 (10), 0.4 (11), 0.5 (14)
Table:
0.1 (9)  | 0.2 (9)  | 0.3 (9)  | 0.2 (8) | 0.2 (8) | 0.1 (6) |
0.5 (14) | 0.4 (11) | 0.3 (10) | 0.4 (9) | 0.4 (2) | 0.5 (8) | 0.6 (6)
"""


class TestLeafUnitCellVetoEventHandler(TestCase):
    def setUp(self) -> None:
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=2, system_length=1.0)
        # Set this for the initial extraction of charge
        setting.set_number_of_node_levels(1)

        self._estimator_without_charge_mock = mock.MagicMock(spec_set=Estimator)
        self._estimator_without_charge_mock.derivative_bound.side_effect = mock_derivative_bound
        self._estimator_with_charge_mock = mock.MagicMock(spec_set=Estimator)
        self._estimator_with_charge_mock.derivative_bound.side_effect = mock_derivative_bound
        # Mock a 4 * 4 cell system
        cells_mock = mock.MagicMock(spec_set=PeriodicCells)
        cells_mock.yield_cells.side_effect = [iter([i for i in range(16)]) for _ in range(4)]
        cells_mock.cell_min.side_effect = mock_cell_min
        cells_mock.cell_max.side_effect = mock_cell_max
        cells_mock.excluded_cells = mock_excluded_cells
        self._estimator_without_charge_mock.potential.number_separation_arguments = 1
        self._estimator_without_charge_mock.potential.number_charge_arguments = 0
        self._estimator_with_charge_mock.potential.number_separation_arguments = 1
        self._estimator_with_charge_mock.potential.number_charge_arguments = 2
        self._event_handler_without_charge = LeafUnitCellVetoEventHandler(estimator=self._estimator_without_charge_mock)
        self._event_handler_with_charge = LeafUnitCellVetoEventHandler(estimator=self._estimator_with_charge_mock,
                                                                       charge="charge")

        # relevant id length will be set later in each test
        root_cnodes = [Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3], charge={"charge": 2.0}))]
        # Redirect stdout to the null device while initializing the estimator
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                self._event_handler_without_charge.initialize(cells_mock, None)
                self._event_handler_with_charge.initialize(cells_mock, None)

    def tearDown(self) -> None:
        setting.reset()


if __name__ == '__main__':
    main()
