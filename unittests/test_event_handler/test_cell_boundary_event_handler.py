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
from activator.internal_state.cell_occupancy.cells import Cells
from base.node import Node
from base.unit import Unit
from event_handler.cell_boundary_event_handler import CellBoundaryEventHandler
import setting


class TestCellBoundaryEventHandler(TestCase):
    def setUp(self) -> None:
        setting.HypercubicSetting(beta=1.0, dimension=2, system_length=1.0)
        self._cells_mock = mock.MagicMock(spec_set=Cells)
        self._event_handler = CellBoundaryEventHandler()

    def tearDown(self) -> None:
        setting.reset()

    def _setUpCellNoCompositeObjectsLeafUnitActive(self):
        self._event_handler.initialize(cells=self._cells_mock, cell_level=1)
        self._cells_mock.position_to_cell.return_value = 3
        self._cells_mock.successor.return_value = 4
        self._cells_mock.cell_min.return_value = [0.2, 0.2]

    def test_send_event_time_cell_no_composite_objects_leaf_unit_active(self):
        self._setUpCellNoCompositeObjectsLeafUnitActive()
        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[0.5, 0.0], time_stamp=0.3), weight=1)
        event_time = self._event_handler.send_event_time([in_state])
        self._cells_mock.position_to_cell.assert_called_once_with([0.1, 0.2])
        self._cells_mock.successor.assert_called_once_with(3, 0)
        self._cells_mock.cell_min.assert_called_once_with(4)
        self.assertAlmostEqual(event_time, 0.5, places=13)

    def test_send_out_state_cell_no_composite_objects_leaf_unit_active(self):
        self._setUpCellNoCompositeObjectsLeafUnitActive()
        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[0.5, 0.0], time_stamp=0.3), weight=1)
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.children, [])
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertEqual(cnode.value.position, [0.2, 0.2])
        self.assertEqual(cnode.weight, 1)
        self.assertEqual(cnode.value.charge, {"charge": 1.0})
        self.assertEqual(cnode.value.velocity, [0.5, 0.0])
        self.assertAlmostEqual(cnode.value.time_stamp, 0.5, places=13)

    def _setUpCellForLeafUnitsLeafUnitActive(self):
        self._event_handler.initialize(cells=self._cells_mock, cell_level=2)
        self._cells_mock.position_to_cell.return_value = 5
        self._cells_mock.successor.return_value = 6
        self._cells_mock.cell_min.return_value = [0.5, 0.0]

    def test_send_event_time_cell_for_leaf_units_leaf_unit_active(self):
        self._setUpCellForLeafUnitsLeafUnitActive()
        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                             time_stamp=1.2), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self._cells_mock.position_to_cell.assert_called_once_with([0.5, 0.9])
        self._cells_mock.successor.assert_called_once_with(5, 1)
        self._cells_mock.cell_min.assert_called_once_with(6)
        self.assertAlmostEqual(event_time, 1.4)

    def test_send_out_state_cell_for_leaf_units_leaf_unit_active(self):
        self._setUpCellForLeafUnitsLeafUnitActive()
        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                             time_stamp=1.2), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertEqual(cnode.value.position, [0.2, 0.9])
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [0.0, 0.5])
        self.assertAlmostEqual(cnode.value.time_stamp, 1.4, places=13)

        self.assertEqual(len(cnode.children), 1)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (0, 2))
        self.assertEqual(child_cnode.value.position, [0.5, 0.0])
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [0.0, 1.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, 1.4, places=13)

    def _setUpCellForCompositeObjectsLeafUnitActive(self):
        self._event_handler.initialize(cells=self._cells_mock, cell_level=1)
        self._cells_mock.position_to_cell.return_value = 5
        self._cells_mock.successor.return_value = 6
        self._cells_mock.cell_min.return_value = [0.5, 0.0]

    def test_send_event_time_cell_for_composite_objects_leaf_unit_active(self):
        self._setUpCellForCompositeObjectsLeafUnitActive()
        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                             time_stamp=1.4), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self._cells_mock.position_to_cell.assert_called_once_with([0.2, 0.8])
        self._cells_mock.successor.assert_called_once_with(5, 1)
        self._cells_mock.cell_min.assert_called_once_with(6)
        self.assertAlmostEqual(event_time, 1.8)

    def test_send_out_state_cell_for_composite_objects_leaf_unit_active(self):
        self._setUpCellForCompositeObjectsLeafUnitActive()
        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                             time_stamp=1.4), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertEqual(cnode.value.position, [0.2, 0.0])
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [0.0, 0.5])
        self.assertAlmostEqual(cnode.value.time_stamp, 1.8, places=13)

        self.assertEqual(len(cnode.children), 1)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (0, 2))
        self.assertEqual(child_cnode.value.position, [0.5, (0.9 + 0.5) % 1.0])
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [0.0, 1.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, 1.8, places=13)

    def _setUpCellForCompositeObjectsRootUnitActive(self):
        self._event_handler.initialize(cells=self._cells_mock, cell_level=1)
        self._cells_mock.position_to_cell.return_value = 1
        self._cells_mock.successor.return_value = 13
        self._cells_mock.cell_min.return_value = [0.5, 0.55]

    def test_send_event_time_cell_for_composite_objects_root_unit_active(self):
        self._setUpCellForCompositeObjectsRootUnitActive()
        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[2.0, 0.0],
                             time_stamp=0.7), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[2.0, 0.0], time_stamp=0.7), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.8],
                                     velocity=[2.0, 0.0], time_stamp=0.7), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self._cells_mock.position_to_cell.assert_called_once_with([0.2, 0.4])
        self._cells_mock.successor.assert_called_once_with(1, 0)
        self._cells_mock.cell_min.assert_called_once_with(13)
        self.assertAlmostEqual(event_time, 0.85)

    def test_send_out_state_cell_for_composite_objects_root_unit_active(self):
        self._setUpCellForCompositeObjectsRootUnitActive()
        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[2.0, 0.0],
                             time_stamp=0.7), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[2.0, 0.0], time_stamp=0.7), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.8],
                                     velocity=[2.0, 0.0], time_stamp=0.7), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertEqual(cnode.value.position, [0.5, 0.4])
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [2.0, 0.0])
        self.assertAlmostEqual(cnode.value.time_stamp, 0.85, places=13)

        self.assertEqual(len(cnode.children), 2)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (1, 0))
        self.assertEqual(child_cnode.value.position, [0.8, 0.6])
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [2.0, 0.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, 0.85, places=13)

        child_cnode = cnode.children[1]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (1, 1))
        self.assertEqual(child_cnode.value.position, [(0.3 + (0.85 - 0.7) * 2.0) % 1.0, 0.8])
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [2.0, 0.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, 0.85, places=13)

    def test_more_than_one_in_state_raises_error(self):
        self._setUpCellNoCompositeObjectsLeafUnitActive()
        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[0.5, 0.0], time_stamp=0.3), weight=1)
        in_state_two = Node(Unit(identifier=(3,), position=[0.2, 0.3], charge={"charge": 1.0},
                                 velocity=[0.5, 0.0], time_stamp=0.3), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler.send_event_time([in_state, in_state_two])

    def test_in_state_not_active_raises_error(self):
        self._setUpCellNoCompositeObjectsLeafUnitActive()
        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2],
                             charge={"charge": 1.0}), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler.send_event_time([in_state])

    def test_more_than_one_velocity_component_raises_error(self):
        self._setUpCellNoCompositeObjectsLeafUnitActive()
        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[1.0, 1.0], time_stamp=0.0), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler.send_event_time([in_state])


if __name__ == '__main__':
    main()
