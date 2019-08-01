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
from unittest import TestCase, main, mock
from activator.internal_state.cell_occupancy.cells import Cells, PeriodicCells
from base.exceptions import ConfigurationError
from base.node import Node
from base.unit import Unit
from event_handler.two_leaf_unit_cell_bounding_potential_event_handler import \
    TwoLeafUnitCellBoundingPotentialEventHandler
from potential import Potential, InvertiblePotential
from potential.cell_bounding_potential import CellBoundingPotential
import setting


@mock.patch("event_handler.abstracts.event_handler_with_bounding_potential.random.uniform")
@mock.patch("event_handler.two_leaf_unit_cell_bounding_potential_event_handler.random.expovariate")
class TestTwoLeafUnitCellBoundingPotentialEventHandler(TestCase):
    def setUp(self) -> None:
        setting.HypercubicSetting(beta=1.0, dimension=2, system_length=1.0)
        self._potential_mock_without_charge = mock.MagicMock(spec_set=Potential)
        self._potential_mock_without_charge.number_separation_arguments = 1
        self._potential_mock_without_charge.number_charge_arguments = 0
        self._cell_bounding_potential_mock_without_charge = mock.MagicMock(spec_set=CellBoundingPotential)
        self._cell_bounding_potential_mock_without_charge.number_separation_arguments = 1
        self._cell_bounding_potential_mock_without_charge.number_charge_arguments = 0
        self._event_handler_without_charge = TwoLeafUnitCellBoundingPotentialEventHandler(
            potential=self._potential_mock_without_charge,
            bounding_potential=self._cell_bounding_potential_mock_without_charge)

        self._potential_mock_with_charge = mock.MagicMock(spec_set=Potential)
        self._potential_mock_with_charge.number_separation_arguments = 1
        self._potential_mock_with_charge.number_charge_arguments = 2
        self._cell_bounding_potential_mock_with_charge = mock.MagicMock(spec_set=CellBoundingPotential)
        self._cell_bounding_potential_mock_with_charge.number_separation_arguments = 1
        self._cell_bounding_potential_mock_with_charge.number_charge_arguments = 2
        self._event_handler_with_charge = TwoLeafUnitCellBoundingPotentialEventHandler(
            potential=self._potential_mock_with_charge,
            bounding_potential=self._cell_bounding_potential_mock_with_charge, charge="charge")
        self._cells_mock = mock.MagicMock(spec_set=PeriodicCells)
        root_cnodes = [Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3], charge={"charge": 2.0}))]
        self._event_handler_without_charge.initialize(self._cells_mock, root_cnodes)
        self._event_handler_with_charge.initialize(self._cells_mock, root_cnodes)

    def tearDown(self) -> None:
        setting.reset()

    def test_initialize_of_cell_bounding_potential_called_without_charge(self, _, __):
        self._cell_bounding_potential_mock_without_charge.initialize.assert_called_once_with(
            self._cells_mock, 1.0, False)

    def test_initialize_of_cell_bounding_potential_called_with_charge(self, _, __):
        self._cell_bounding_potential_mock_with_charge.initialize.assert_called_once_with(
            self._cells_mock, 2.0, True)

    def test_initialize_with_initialized_potential(self, _, __):
        bounding_potential_mock = mock.MagicMock(spec_set=CellBoundingPotential)
        bounding_potential_mock.number_separation_arguments = 1
        bounding_potential_mock.number_charge_arguments = 2
        event_handler = TwoLeafUnitCellBoundingPotentialEventHandler(
            potential=self._potential_mock_without_charge,
            bounding_potential=self._cell_bounding_potential_mock_without_charge)
        event_handler.initialize_with_initialized_potential(self._cells_mock, bounding_potential_mock)
        self.assertIs(event_handler.bounding_potential, bounding_potential_mock)

    def _setUpSendEventTime(self, random_expovariate_mock, cell_bounding_potential_mock, position_one, position_two):
        # We mimic a 4 x 4 cell system
        # Active cell is 4, Target cell is 12
        def position_to_cell(position):
            if position == position_one:
                return 4
            elif position == position_two:
                return 12
            raise AttributeError
        # Mock does not copy the arguments when recording the calls
        # Since position_to_cell receives the position of the active leaf unit which gets changed later,
        # we cannot check the argument by setting side_effect to an iterable here and using has_calls later
        # Therefore we use this method, in which we compare to the expected positions
        self._cells_mock.position_to_cell.side_effect = position_to_cell
        # Excluded cells of cell 4
        self._cells_mock.excluded_cells.return_value = [0, 1, 3, 4, 5, 7, 8, 9, 11]
        self._cells_mock.relative_cell.return_value = 8
        cell_bounding_potential_mock.displacement.return_value = 0.3
        random_expovariate_mock.return_value = 2

    def _setUpSendOutStateAccept(self, random_uniform_mock, cell_bounding_potential_mock, potential_mock):
        cell_bounding_potential_mock.derivative.return_value = 0.7
        potential_mock.derivative.return_value = 0.5
        random_uniform_mock.return_value = 0.4
        # Return again the active cell
        self._cells_mock.position_to_cell.side_effect = [4]

    def _setUpSendOutStateReject(self, random_uniform_mock, cell_bounding_potential_mock, potential_mock):
        cell_bounding_potential_mock.derivative.return_value = 0.7
        potential_mock.derivative.return_value = 0.5
        random_uniform_mock.return_value = 0.6
        # Return again the active cell
        self._cells_mock.position_to_cell.side_effect = [4]

    def test_send_event_time_without_charge(self, random_expovariate_mock, _):
        self._setUpSendEventTime(random_expovariate_mock, self._cell_bounding_potential_mock_without_charge,
                                 [0.5, 0.9], [0.1, 0.3])
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.3), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))

        event_time = self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        # Arguments are checked in _setUpSendEventTime
        self.assertEqual(self._cells_mock.position_to_cell.call_count, 2)
        self._cells_mock.excluded_cells.assert_called_once_with(4)
        self._cells_mock.relative_cell.assert_called_once_with(12, 4)
        self._cell_bounding_potential_mock_without_charge.displacement.assert_called_once_with(
            1, 8, 2)
        self.assertAlmostEqual(event_time, 1.6)

    def test_send_out_state_without_charge_accept(self, random_expovariate_mock, random_uniform_mock):
        self._setUpSendEventTime(random_expovariate_mock, self._cell_bounding_potential_mock_without_charge,
                                 [0.5, 0.9], [0.1, 0.3])
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateAccept(random_uniform_mock, self._cell_bounding_potential_mock_without_charge,
                                      self._potential_mock_without_charge)
        out_state = self._event_handler_without_charge.send_out_state()
        new_position = (0.9 + (1.6 - 1.3) * 1.0) % 1.0
        self._cell_bounding_potential_mock_without_charge.derivative.assert_called_once_with(1, 8)
        self._potential_mock_without_charge.derivative.assert_called_once_with(
            1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - new_position + 0.5) % 1.0 - 0.5])
        random_uniform_mock.assert_called_once_with(0, 0.7)
        self._cells_mock.position_to_cell.assert_has_calls([mock.call([0.5, new_position])])

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [0.2, (0.8 + 0.2) % 1.0])
        self.assertEqual(first_cnode.weight, 1)
        self.assertIsNone(first_cnode.value.velocity)
        self.assertIsNone(first_cnode.value.time_stamp)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 2))
        self.assertEqual(first_child.value.position, [0.5, (0.9 + (1.6 - 1.3) * 1.0) % 1.0])
        self.assertEqual(first_child.weight, 0.5)
        self.assertIsNone(first_child.value.velocity)
        self.assertIsNone(first_child.value.time_stamp)
        self.assertIsNone(first_child.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (3,))
        self.assertEqual(second_cnode.value.position, [0.5, 0.6])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(second_cnode.value.time_stamp, 1.6)
        self.assertIsNone(second_cnode.value.charge)
        self.assertEqual(len(second_cnode.children), 1)
        second_child = second_cnode.children[0]
        self.assertIs(second_child.parent, second_cnode)
        self.assertEqual(second_child.children, [])
        self.assertEqual(second_child.value.identifier, (3, 1))
        self.assertEqual(second_child.value.position, [0.1, 0.3])
        self.assertEqual(second_child.weight, 0.5)
        self.assertEqual(second_child.value.velocity, [0.0, 1.0])
        self.assertEqual(second_child.value.time_stamp, 1.6)
        self.assertIsNone(second_child.value.charge)

    def test_send_out_state_without_charge_reject(self, random_expovariate_mock, random_uniform_mock):
        self._setUpSendEventTime(random_expovariate_mock, self._cell_bounding_potential_mock_without_charge,
                                 [0.5, 0.9], [0.1, 0.3])
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateReject(random_uniform_mock, self._cell_bounding_potential_mock_without_charge,
                                      self._potential_mock_without_charge)
        out_state = self._event_handler_without_charge.send_out_state()
        new_position = (0.9 + (1.6 - 1.3) * 1.0) % 1.0
        self._cell_bounding_potential_mock_without_charge.derivative.assert_called_once_with(1, 8)
        self._potential_mock_without_charge.derivative.assert_called_once_with(
            1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - new_position + 0.5) % 1.0 - 0.5])
        random_uniform_mock.assert_called_once_with(0, 0.7)
        self._cells_mock.position_to_cell.assert_has_calls([mock.call([0.5, new_position])])

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [0.2, (0.8 + 0.2) % 1.0])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(first_cnode.value.time_stamp, 1.6)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 2))
        self.assertEqual(first_child.value.position, [0.5, (0.9 + (1.6 - 1.3) * 1.0) % 1.0])
        self.assertEqual(first_child.weight, 0.5)
        self.assertEqual(first_child.value.velocity, [0.0, 1.0])
        self.assertEqual(first_child.value.time_stamp, 1.6)
        self.assertIsNone(first_child.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (3,))
        self.assertEqual(second_cnode.value.position, [0.5, 0.6])
        self.assertEqual(second_cnode.weight, 1)
        self.assertIsNone(second_cnode.value.velocity)
        self.assertIsNone(second_cnode.value.time_stamp)
        self.assertIsNone(second_cnode.value.charge)
        self.assertEqual(len(second_cnode.children), 1)
        second_child = second_cnode.children[0]
        self.assertIs(second_child.parent, second_cnode)
        self.assertEqual(second_child.children, [])
        self.assertEqual(second_child.value.identifier, (3, 1))
        self.assertEqual(second_child.value.position, [0.1, 0.3])
        self.assertEqual(second_child.weight, 0.5)
        self.assertIsNone(second_child.value.velocity)
        self.assertIsNone(second_child.value.time_stamp)
        self.assertIsNone(second_child.value.charge)

    def test_send_event_time_with_charge(self, random_expovariate_mock, _):
        self._setUpSendEventTime(random_expovariate_mock, self._cell_bounding_potential_mock_with_charge,
                                 [0.5, 0.9], [0.5, 0.6])
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[2.0, 0.0], time_stamp=1.3,
                                 charge={"charge": 3.4}), weight=1)

        event_time = self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        self.assertEqual(self._cells_mock.position_to_cell.call_count, 2)
        self._cells_mock.excluded_cells.assert_called_once_with(4)
        self._cells_mock.relative_cell.assert_called_once_with(12, 4)
        self._cell_bounding_potential_mock_with_charge.displacement.assert_called_once_with(
            0, 8, -1.2, 3.4, 2)
        self.assertAlmostEqual(event_time, 1.45)

    def test_send_out_state_with_charge_accept(self, random_expovariate_mock, random_uniform_mock):
        self._setUpSendEventTime(random_expovariate_mock, self._cell_bounding_potential_mock_with_charge,
                                 [0.5, 0.9], [0.5, 0.6])
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[2.0, 0.0], time_stamp=1.3,
                                 charge={"charge": 3.4}), weight=1)

        self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateAccept(random_uniform_mock, self._cell_bounding_potential_mock_with_charge,
                                      self._potential_mock_with_charge)
        out_state = self._event_handler_with_charge.send_out_state()
        new_position = (0.5 + (1.45 - 1.3) * 2.0) % 1.0
        self._cell_bounding_potential_mock_with_charge.derivative.assert_called_once_with(
            0, 8, -1.2, 3.4)
        self._potential_mock_with_charge.derivative.assert_called_once_with(
            0, [(0.5 - new_position + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4)
        random_uniform_mock.assert_called_once_with(0, 0.7)
        self._cells_mock.position_to_cell.assert_has_calls([mock.call([new_position, 0.9])])

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.children, [])
        self.assertEqual(first_cnode.value.identifier, (3,))
        self.assertEqual(first_cnode.value.position, [0.5, 0.6])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.charge, {"charge": -1.2})
        self.assertEqual(first_cnode.value.velocity, [2.0, 0.0])
        self.assertEqual(first_cnode.value.time_stamp, 1.45)
        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.children, [])
        self.assertEqual(second_cnode.value.identifier, (1,))
        self.assertEqual(second_cnode.value.position, [(0.5 + (1.45 - 1.3) * 2.0) % 1.0, 0.9])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.charge, {"charge": 3.4})
        self.assertIsNone(second_cnode.value.velocity)
        self.assertIsNone(second_cnode.value.time_stamp)

    def test_send_out_state_with_charge_reject(self, random_expovariate_mock, random_uniform_mock):
        self._setUpSendEventTime(random_expovariate_mock, self._cell_bounding_potential_mock_with_charge,
                                 [0.5, 0.9], [0.5, 0.6])
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[2.0, 0.0], time_stamp=1.3,
                                 charge={"charge": 3.4}), weight=1)

        self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateReject(random_uniform_mock, self._cell_bounding_potential_mock_with_charge,
                                      self._potential_mock_with_charge)
        out_state = self._event_handler_with_charge.send_out_state()
        new_position = (0.5 + (1.45 - 1.3) * 2.0) % 1.0
        self._cell_bounding_potential_mock_with_charge.derivative.assert_called_once_with(
            0, 8, -1.2, 3.4)
        self._potential_mock_with_charge.derivative.assert_called_once_with(
            0, [(0.5 - new_position + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4)
        random_uniform_mock.assert_called_once_with(0, 0.7)
        self._cells_mock.position_to_cell.assert_has_calls([mock.call([new_position, 0.9])])

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.children, [])
        self.assertEqual(first_cnode.value.identifier, (3,))
        self.assertEqual(first_cnode.value.position, [0.5, 0.6])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.charge, {"charge": -1.2})
        self.assertIsNone(first_cnode.value.velocity)
        self.assertIsNone(first_cnode.value.time_stamp)
        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.children, [])
        self.assertEqual(second_cnode.value.identifier, (1,))
        self.assertEqual(second_cnode.value.position, [(0.5 + (1.45 - 1.3) * 2.0) % 1.0, 0.9])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.charge, {"charge": 3.4})
        self.assertEqual(second_cnode.value.velocity, [2.0, 0.0])
        self.assertEqual(second_cnode.value.time_stamp, 1.45)

    def test_send_event_time_leaf_units_in_same_composite_object(self, random_expovariate_mock, _):
        self._setUpSendEventTime(random_expovariate_mock, self._cell_bounding_potential_mock_without_charge,
                                 [0.5, 0.9], [0.1, 0.3])
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.2), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 0), position=[0.1, 0.3]), weight=0.5))
        event_time = self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        # Arguments are checked in _setUpSendEventTime
        self.assertEqual(self._cells_mock.position_to_cell.call_count, 2)
        self._cells_mock.excluded_cells.assert_called_once_with(4)
        self._cells_mock.relative_cell.assert_called_once_with(12, 4)
        self._cell_bounding_potential_mock_without_charge.displacement.assert_called_once_with(
            1, 8, 2)
        self.assertAlmostEqual(event_time, 1.5)

    def test_send_out_state_leaf_units_in_same_composite_object_accept(self, random_expovariate_mock,
                                                                       random_uniform_mock):
        self._setUpSendEventTime(random_expovariate_mock, self._cell_bounding_potential_mock_without_charge,
                                 [0.5, 0.9], [0.1, 0.3])
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.2), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 0), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateAccept(random_uniform_mock, self._cell_bounding_potential_mock_without_charge,
                                      self._potential_mock_without_charge)
        out_state = self._event_handler_without_charge.send_out_state()
        new_position = (0.9 + (1.5 - 1.2) * 1.0) % 1.0
        self._cell_bounding_potential_mock_without_charge.derivative.assert_called_once_with(1, 8)
        self._potential_mock_without_charge.derivative.assert_called_once_with(
            1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - new_position + 0.5) % 1.0 - 0.5])
        random_uniform_mock.assert_called_once_with(0, 0.7)
        self._cells_mock.position_to_cell.assert_has_calls([mock.call([0.5, new_position])])

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [0.2, (0.8 + (1.5 - 1.2) * 0.5) % 1.0])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(first_cnode.value.time_stamp, 1.5)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 1))
        self.assertEqual(first_child.value.position, [0.5, (0.9 + (1.5 - 1.2) * 1.0) % 1.0])
        self.assertEqual(first_child.weight, 0.5)
        self.assertIsNone(first_child.value.velocity)
        self.assertIsNone(first_child.value.time_stamp)
        self.assertIsNone(first_child.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (0,))
        self.assertEqual(second_cnode.value.position, [0.2, (0.8 + (1.5 - 1.2) * 0.5) % 1.0])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(second_cnode.value.time_stamp, 1.5)
        self.assertIsNone(second_cnode.value.charge)
        self.assertEqual(len(second_cnode.children), 1)
        second_child = second_cnode.children[0]
        self.assertIs(second_child.parent, second_cnode)
        self.assertEqual(second_child.children, [])
        self.assertEqual(second_child.value.identifier, (0, 0))
        self.assertEqual(second_child.value.position, [0.1, 0.3])
        self.assertEqual(second_child.weight, 0.5)
        self.assertEqual(second_child.value.velocity, [0.0, 1.0])
        self.assertEqual(second_child.value.time_stamp, 1.5)
        self.assertIsNone(second_child.value.charge)

    def test_send_out_state_leaf_units_in_same_composite_object_reject(self, random_expovariate_mock,
                                                                       random_uniform_mock):
        self._setUpSendEventTime(random_expovariate_mock, self._cell_bounding_potential_mock_without_charge,
                                 [0.5, 0.9], [0.1, 0.3])
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.2), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 0), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateReject(random_uniform_mock, self._cell_bounding_potential_mock_without_charge,
                                      self._potential_mock_without_charge)
        out_state = self._event_handler_without_charge.send_out_state()
        new_position = (0.9 + (1.5 - 1.2) * 1.0) % 1.0
        self._cell_bounding_potential_mock_without_charge.derivative.assert_called_once_with(1, 8)
        self._potential_mock_without_charge.derivative.assert_called_once_with(
            1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - new_position + 0.5) % 1.0 - 0.5])
        random_uniform_mock.assert_called_once_with(0, 0.7)
        self._cells_mock.position_to_cell.assert_has_calls([mock.call([0.5, new_position])])

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [0.2, (0.8 + (1.5 - 1.2) * 0.5) % 1.0])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(first_cnode.value.time_stamp, 1.5)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 1))
        self.assertEqual(first_child.value.position, [0.5, (0.9 + (1.5 - 1.2) * 1.0) % 1.0])
        self.assertEqual(first_child.weight, 0.5)
        self.assertEqual(first_child.value.velocity, [0.0, 1.0])
        self.assertEqual(first_child.value.time_stamp, 1.5)
        self.assertIsNone(first_child.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (0,))
        self.assertEqual(second_cnode.value.position, [0.2, (0.8 + (1.5 - 1.2) * 0.5) % 1.0])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(second_cnode.value.time_stamp, 1.5)
        self.assertIsNone(second_cnode.value.charge)
        self.assertEqual(len(second_cnode.children), 1)
        second_child = second_cnode.children[0]
        self.assertIs(second_child.parent, second_cnode)
        self.assertEqual(second_child.children, [])
        self.assertEqual(second_child.value.identifier, (0, 0))
        self.assertEqual(second_child.value.position, [0.1, 0.3])
        self.assertEqual(second_child.weight, 0.5)
        self.assertIsNone(second_child.value.velocity)
        self.assertIsNone(second_child.value.time_stamp)
        self.assertIsNone(second_child.value.charge)

    def _setUpSendOutStateActiveCellChanged(self, random_uniform_mock, cell_bounding_potential_mock, potential_mock):
        cell_bounding_potential_mock.derivative.return_value = 0.7
        potential_mock.derivative.return_value = 0.5
        random_uniform_mock.return_value = 0.4
        # Return not the active cell
        self._cells_mock.position_to_cell.side_effect = [5]

    def test_active_cell_changed_returns_none(self, random_expovariate_mock, random_uniform_mock):
        self._setUpSendEventTime(random_expovariate_mock, self._cell_bounding_potential_mock_with_charge,
                                 [0.5, 0.9], [0.5, 0.6])
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[2.0, 0.0], time_stamp=1.3,
                                 charge={"charge": 3.4}), weight=1)

        self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])
        self._setUpSendOutStateActiveCellChanged(random_uniform_mock, self._cell_bounding_potential_mock_with_charge,
                                                 self._potential_mock_with_charge)
        out_state = self._event_handler_with_charge.send_out_state()
        self.assertIsNone(out_state)
        self._potential_mock_with_charge.derivative.assert_not_called()
        random_uniform_mock.assert_not_called()
        self._cell_bounding_potential_mock_with_charge.derivative.assert_not_called()
        self._cells_mock.position_to_cell.assert_has_calls([mock.call([(0.5 + (1.45 - 1.3) * 2.0) % 1.0, 0.9])])

    def _setUpSendEventTimeInfiniteDisplacement(self, random_expovariate_mock, cell_bounding_potential_mock,
                                                position_one, position_two):
        # We mimic a 4 x 4 cell system
        # Active cell is 4, Target cell is 12
        def position_to_cell(position):
            if position == position_one:
                return 4
            elif position == position_two:
                return 12
            raise AttributeError
        # Mock does not copy the arguments when recording the calls
        # Since position_to_cell receives the position of the active leaf unit which gets changed later,
        # we cannot check the argument by setting side_effect to an iterable here and using has_calls later
        # Therefore we use this method, in which we compare to the expected positions
        self._cells_mock.position_to_cell.side_effect = position_to_cell
        # Excluded cells of cell 4
        self._cells_mock.excluded_cells.return_value = [0, 1, 3, 4, 5, 7, 8, 9, 11]
        self._cells_mock.relative_cell.return_value = 8
        cell_bounding_potential_mock.displacement.return_value = float('inf')
        random_expovariate_mock.return_value = 2

    def test_infinite_displacement_returns_none(self, random_expovariate_mock, random_uniform_mock):
        self._setUpSendEventTimeInfiniteDisplacement(random_expovariate_mock,
                                                     self._cell_bounding_potential_mock_with_charge,
                                                     [0.5, 0.9], [0.5, 0.6])
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[2.0, 0.0], time_stamp=1.3,
                                 charge={"charge": 3.4}), weight=1)

        self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])
        self._setUpSendOutStateAccept(random_uniform_mock, self._cell_bounding_potential_mock_with_charge,
                                      self._potential_mock_with_charge)
        out_state = self._event_handler_with_charge.send_out_state()
        self.assertIsNone(out_state)
        self._potential_mock_with_charge.derivative.assert_not_called()
        random_uniform_mock.assert_not_called()
        self._cell_bounding_potential_mock_with_charge.derivative.assert_not_called()

    def test_non_active_cell_in_excluded_cells_raises_error(self, _, __):
        in_state_one = Node(Unit(identifier=(3,), position=[0.1, 0.05], velocity=[1.0, 0.0],
                                 time_stamp=0.3), weight=1)
        in_state_two = Node(Unit(identifier=(17,), position=[0.3, 0.01]), weight=1)
        # First index is active cell, second index is target cell
        self._cells_mock.position_to_cell.side_effect = [0, 1]
        # Excluded cells for 4 times 4 system, where 0 is the active cell
        self._cells_mock.excluded_cells.return_value = [0, 1, 3, 4, 5, 7, 12, 13, 15]
        with self.assertRaises(AssertionError):
            self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

    def test_wrong_type_of_bounding_potential_raises_error(self, _, __):
        bounding_potential_mock = mock.MagicMock(spec_set=InvertiblePotential)
        bounding_potential_mock.number_separation_arguments = 1
        bounding_potential_mock.number_charge_arguments = 0
        with self.assertRaises(ConfigurationError):
            TwoLeafUnitCellBoundingPotentialEventHandler(potential=self._potential_mock_without_charge,
                                                         bounding_potential=bounding_potential_mock)

    def test_potential_with_wrong_number_of_separation_arguments_raises_error(self, _, __):
        potential_mock = mock.MagicMock(spec_set=Potential)
        potential_mock.number_separation_arguments = 2
        potential_mock.number_charge_arguments = 0
        with self.assertRaises(ConfigurationError):
            TwoLeafUnitCellBoundingPotentialEventHandler(potential=potential_mock,
                                                         bounding_potential=self._cell_bounding_potential_mock_without_charge)

    def test_bounding_potential_with_wrong_number_of_separation_arguments_raises_error(self, _, __):
        bounding_potential_mock = mock.MagicMock(spec_set=CellBoundingPotential)
        bounding_potential_mock.number_separation_arguments = 2
        bounding_potential_mock.number_charge_arguments = 0
        with self.assertRaises(ConfigurationError):
            TwoLeafUnitCellBoundingPotentialEventHandler(potential=self._potential_mock_without_charge,
                                                         bounding_potential=bounding_potential_mock)

    def test_potential_with_wrong_number_of_charge_arguments_raises_error(self, _, __):
        potential_mock = mock.MagicMock(spec_set=Potential)
        potential_mock.number_separation_arguments = 1
        potential_mock.number_charge_arguments = 1
        with self.assertRaises(ConfigurationError):
            TwoLeafUnitCellBoundingPotentialEventHandler(potential=potential_mock,
                                                         bounding_potential=self._cell_bounding_potential_mock_with_charge,
                                                         charge="charge")

    def test_bounding_potential_with_wrong_number_of_charge_arguments_raises_error(self, _, __):
        bounding_potential_mock = mock.MagicMock(spec_set=CellBoundingPotential)
        bounding_potential_mock.number_separation_arguments = 1
        bounding_potential_mock.number_charge_arguments = 1
        with self.assertRaises(ConfigurationError):
            TwoLeafUnitCellBoundingPotentialEventHandler(potential=self._potential_mock_with_charge,
                                                         bounding_potential=bounding_potential_mock,
                                                         charge="charge")

    def test_missing_charges_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTime(random_expovariate_mock, self._cell_bounding_potential_mock_with_charge,
                                 [0.5, 0.9], [0.5, 0.6])
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"other_charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9],
                                 velocity=[2.0, 0.0], time_stamp=1.3,
                                 charge={"other_charge": 3.4}), weight=1)
        with self.assertRaises(KeyError):
            self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])

    def test_no_leaf_unit_active_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTime(random_expovariate_mock, self._cell_bounding_potential_mock_without_charge,
                                 [0.5, 0.6], [0.5, 0.9])
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9]), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

    def test_both_leaf_units_active_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTime(random_expovariate_mock, self._cell_bounding_potential_mock_without_charge,
                                 [0.5, 0.6], [0.5, 0.9])
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6], velocity=[1.0, 0.0],
                                 time_stamp=0.0), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[1.0, 0.0],
                                 time_stamp=0.0), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

    def test_more_than_two_leaf_units_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTime(random_expovariate_mock, self._cell_bounding_potential_mock_without_charge,
                                 [0.5, 0.6], [0.3, 0.8])
        in_state_one = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[1.0, 0.0],
                                 time_stamp=0.7), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                         velocity=[2.0, 0.0], time_stamp=0.7), weight=0.5))
        in_state_one.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.8]), weight=0.5))
        in_state_two = Node(Unit(identifier=(2,), position=[0.2, 0.4], velocity=[1.0, 0.0],
                                 time_stamp=0.7), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(2, 0), position=[0.5, 0.6],
                                         velocity=[2.0, 0.0], time_stamp=0.7), weight=0.5))
        in_state_two.add_child(Node(Unit(identifier=(2, 1), position=[0.3, 0.8]), weight=0.5))
        with self.assertRaises(AssertionError):
            self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

    def test_no_periodic_cells_raises_error(self, _, __):
        event_handler = TwoLeafUnitCellBoundingPotentialEventHandler(
            potential=self._potential_mock_without_charge,
            bounding_potential=self._cell_bounding_potential_mock_without_charge)
        cells_mock = mock.MagicMock(spec_set=Cells)
        with self.assertRaises(ConfigurationError):
            event_handler.initialize(cells_mock, "All units without charge")


if __name__ == '__main__':
    main()
