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
from jellyfysh.activator.internal_state.cell_occupancy.cells.cuboid_periodic_cells import CuboidPeriodicCells
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
from jellyfysh.base.unit import Unit
from jellyfysh.event_handler.cell_boundary_event_handler import CellBoundaryEventHandler
import jellyfysh.setting as setting
from jellyfysh.setting import hypercubic_setting
_unittest_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_unittest_directory_added_to_path = False
if _unittest_directory not in sys.path:
    sys.path.append(_unittest_directory)
    _unittest_directory_added_to_path = True
# noinspection PyUnresolvedReferences
from expanded_test_case import ExpandedTestCase


def tearDownModule():
    if _unittest_directory_added_to_path:
        sys.path.remove(_unittest_directory)


class TestCellBoundaryEventHandler(ExpandedTestCase, TestCase):
    def setUp(self) -> None:
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=2, system_length=1.0)
        self._event_handler = CellBoundaryEventHandler()

    def tearDown(self) -> None:
        setting.reset()

    def _setUpCellNoCompositeObjectsLeafUnitActive(self):
        setting.set_number_of_root_nodes(6)
        setting.set_number_of_nodes_per_root_node(1)
        setting.set_number_of_node_levels(1)
        cells = CuboidPeriodicCells(cells_per_side=[5, 4])
        self._event_handler.initialize(cells=cells, cell_level=1)

    def test_send_event_time_cell_no_composite_objects_leaf_unit_active(self):
        self._setUpCellNoCompositeObjectsLeafUnitActive()
        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[0.5, 0.0], time_stamp=Time.from_float(0.3)), weight=1)
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(0.5), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[0.0, 0.3], time_stamp=Time.from_float(0.3)), weight=1)
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(0.3 + 0.05 / 0.3), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[-0.5, 0.0], time_stamp=Time.from_float(0.3)), weight=1)
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(0.5), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[0.0, -0.3], time_stamp=Time.from_float(0.3)), weight=1)
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(0.3 + 0.2 / 0.3), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[0.5, 0.3], time_stamp=Time.from_float(0.3)), weight=1)
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(0.3 + 0.05 / 0.3), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[0.5, -0.3], time_stamp=Time.from_float(0.3)), weight=1)
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(0.5), places=13)

    def test_send_out_state_cell_no_composite_objects_leaf_unit_active(self):
        self._setUpCellNoCompositeObjectsLeafUnitActive()
        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[0.5, 0.0], time_stamp=Time.from_float(0.3)), weight=1)
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.children, [])
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2, 0.2], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertEqual(cnode.value.charge, {"charge": 1.0})
        self.assertEqual(cnode.value.velocity, [0.5, 0.0])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(0.5), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[0.0, 0.3], time_stamp=Time.from_float(0.3)), weight=1)
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.children, [])
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.1, 0.25], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertEqual(cnode.value.charge, {"charge": 1.0})
        self.assertEqual(cnode.value.velocity, [0.0, 0.3])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(0.3 + 0.05 / 0.3), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[-0.5, 0.0], time_stamp=Time.from_float(0.3)), weight=1)
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.children, [])
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertAlmostEqualSequence(cnode.value.position, [1.0, 0.2], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertEqual(cnode.value.charge, {"charge": 1.0})
        self.assertEqual(cnode.value.velocity, [-0.5, 0.0])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(0.5), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[0.0, -0.3], time_stamp=Time.from_float(0.3)), weight=1)
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.children, [])
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.1, 1.0], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertEqual(cnode.value.charge, {"charge": 1.0})
        self.assertEqual(cnode.value.velocity, [0.0, -0.3])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(0.3 + 0.2 / 0.3), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[0.5, 0.3], time_stamp=Time.from_float(0.3)), weight=1)
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.children, [])
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.1 + 0.05 / 0.3 * 0.5, 0.25], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertEqual(cnode.value.charge, {"charge": 1.0})
        self.assertEqual(cnode.value.velocity, [0.5, 0.3])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(0.3 + 0.05 / 0.3), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[0.5, -0.3], time_stamp=Time.from_float(0.3)), weight=1)
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.children, [])
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2, 0.2 - 0.2 * 0.3], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertEqual(cnode.value.charge, {"charge": 1.0})
        self.assertEqual(cnode.value.velocity, [0.5, -0.3])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(0.5), places=13)

    def _setUpCellForLeafUnitsLeafUnitActive(self):
        setting.set_number_of_root_nodes(6)
        setting.set_number_of_nodes_per_root_node(3)
        setting.set_number_of_node_levels(2)
        cells = CuboidPeriodicCells(cells_per_side=[5, 4])
        self._event_handler.initialize(cells=cells, cell_level=2)

    def test_send_event_time_cell_for_leaf_units_leaf_unit_active(self):
        self._setUpCellForLeafUnitsLeafUnitActive()
        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                             time_stamp=Time.from_float(1.2)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.0, 1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(1.4), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.4, 0.0],
                             time_stamp=Time.from_float(1.2)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.8, 0.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(1.3 + 0.1 / 0.8), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, -0.5],
                             time_stamp=Time.from_float(1.2)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.0, -1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(1.45), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[-0.4, 0.0],
                             time_stamp=Time.from_float(1.2)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[-0.8, 0.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(1.3 + 0.1 / 0.8), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[-0.4, -0.5],
                             time_stamp=Time.from_float(1.2)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[-0.8, -1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(1.3 + 0.1 / 0.8), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[-0.4, 0.5],
                             time_stamp=Time.from_float(1.2)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[-0.8, 1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(1.4), places=13)

    def test_send_out_state_cell_for_leaf_units_leaf_unit_active(self):
        self._setUpCellForLeafUnitsLeafUnitActive()
        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                             time_stamp=Time.from_float(1.2)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.0, 1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2, 0.9], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [0.0, 0.5])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(1.4), places=13)
        self.assertEqual(len(cnode.children), 1)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (0, 2))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.5, 0.0], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [0.0, 1.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(1.4), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.4, 0.0],
                             time_stamp=Time.from_float(1.2)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.8, 0.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2 + (0.1 + 0.1 / 0.8) * 0.4, 0.8], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [0.4, 0.0])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(1.3 + 0.1 / 0.8), places=13)
        self.assertEqual(len(cnode.children), 1)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (0, 2))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.6, 0.9], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [0.8, 0.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(1.3 + 0.1 / 0.8), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, -0.5],
                             time_stamp=Time.from_float(1.2)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.0, -1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2, 0.8 - 0.25 * 0.5], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [0.0, -0.5])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(1.45), places=13)
        self.assertEqual(len(cnode.children), 1)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (0, 2))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.5, 0.75], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [0.0, -1.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(1.45), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[-0.4, 0.0],
                             time_stamp=Time.from_float(1.2)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[-0.8, 0.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2 - (0.1 + 0.1 / 0.8) * 0.4, 0.8], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [-0.4, 0.0])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(1.3 + 0.1 / 0.8), places=13)
        self.assertEqual(len(cnode.children), 1)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (0, 2))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.4, 0.9], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [-0.8, 0.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(1.3 + 0.1 / 0.8), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[-0.4, -0.5],
                             time_stamp=Time.from_float(1.2)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[-0.8, -1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(cnode.value.position,
                                       [0.2 - (0.1 + 0.1 / 0.8) * 0.4, 0.8 - (0.1 + 0.1 / 0.8) * 0.5], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [-0.4, -0.5])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(1.3 + 0.1 / 0.8), places=13)
        self.assertEqual(len(cnode.children), 1)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (0, 2))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.4, 0.9 - 0.1 / 0.8], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [-0.8, -1.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(1.3 + 0.1 / 0.8), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[-0.4, 0.5],
                             time_stamp=Time.from_float(1.2)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[-0.8, 1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2 - 0.2 * 0.4, 0.8 + 0.2 * 0.5], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [-0.4, 0.5])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(1.4), places=13)
        self.assertEqual(len(cnode.children), 1)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (0, 2))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.5 - 0.1 * 0.8, 0.0], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [-0.8, 1.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(1.4), places=13)

    def _setUpCellForCompositeObjectsLeafUnitActive(self):
        setting.set_number_of_root_nodes(6)
        setting.set_number_of_nodes_per_root_node(3)
        setting.set_number_of_node_levels(2)
        cells = CuboidPeriodicCells(cells_per_side=[5, 4])
        self._event_handler.initialize(cells=cells, cell_level=1)

    def test_send_event_time_cell_for_composite_objects_leaf_unit_active(self):
        self._setUpCellForCompositeObjectsLeafUnitActive()
        in_state = Node(Unit(identifier=(0,), position=[0.2 + 1.0e-13, 0.75 - 1.0e-13], velocity=[0.0, 0.5],
                             time_stamp=Time.from_float(1.3)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.0, 1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(1.3 + 1.0e-13 / 0.5), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2 + 1.0e-13, 0.75 - 1.0e-13], velocity=[0.4, 0.0],
                             time_stamp=Time.from_float(1.3)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.8, 0.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(1.3 + (0.2 - 1.0e-13) / 0.4), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2 + 1.0e-13, 0.75 - 1.0e-13], velocity=[0.0, -0.5],
                             time_stamp=Time.from_float(1.3)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.0, -1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(1.3 + (0.25 - 1.0e-13) / 0.5), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2 + 1.0e-13, 0.75 - 1.0e-13], velocity=[-0.4, 0.0],
                             time_stamp=Time.from_float(1.3)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[-0.8, 0.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(1.3 + 1.0e-13 / 0.4), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2 + 1.0e-13, 0.75 - 1.0e-13], velocity=[-0.4, -0.5],
                             time_stamp=Time.from_float(1.3)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[-0.8, -1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(1.3 + 1.0e-13 / 0.4), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2 + 1.0e-13, 0.75 - 1.0e-13], velocity=[-0.4, 0.5],
                             time_stamp=Time.from_float(1.3)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[-0.8, 1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(1.3 + 1.0e-13 / 0.4), places=13)

    def test_send_out_state_cell_for_composite_objects_leaf_unit_active(self):
        self._setUpCellForCompositeObjectsLeafUnitActive()
        in_state = Node(Unit(identifier=(0,), position=[0.2 + 1.0e-13, 0.75 - 1.0e-13], velocity=[0.0, 0.5],
                             time_stamp=Time.from_float(1.3)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.0, 1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2 + 1.0e-13, 0.75], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [0.0, 0.5])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(1.3 + 1.0e-13 / 0.5), places=13)
        self.assertEqual(len(cnode.children), 1)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (0, 2))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.5, 0.9 + 1.0e-13 / 0.5], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [0.0, 1.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(1.3 + 1.0e-13 / 0.5), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2 + 1.0e-13, 0.75 - 1.0e-13], velocity=[0.4, 0.0],
                             time_stamp=Time.from_float(1.3)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.8, 0.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.4, 0.75 - 1.0e-13], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [0.4, 0.0])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(1.3 + (0.2 - 1.0e-13) / 0.4), places=13)
        self.assertEqual(len(cnode.children), 1)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (0, 2))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.5 + 0.8 * (0.2 - 1.0e-13) / 0.4, 0.9], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [0.8, 0.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(1.3 + (0.2 - 1.0e-13) / 0.4), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2 + 1.0e-13, 0.75 - 1.0e-13], velocity=[0.0, -0.5],
                             time_stamp=Time.from_float(1.3)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[0.0, -1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2 + 1.0e-13, 0.5], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [0.0, -0.5])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(1.3 + (0.25 - 1.0e-13) / 0.5), places=13)
        self.assertEqual(len(cnode.children), 1)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (0, 2))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.5, 0.9 - 1.0 * (0.25 - 1.0e-13) / 0.5], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [0.0, -1.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(1.3 + (0.25 - 1.0e-13) / 0.5), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2 + 1.0e-13, 0.75 - 1.0e-13], velocity=[-0.4, 0.0],
                             time_stamp=Time.from_float(1.3)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[-0.8, 0.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2, 0.75 - 1.0e-13], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [-0.4, 0.0])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(1.3 + 1.0e-13 / 0.4), places=13)
        self.assertEqual(len(cnode.children), 1)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (0, 2))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.5 - 0.8 * 1.0e-13 / 0.4, 0.9], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [-0.8, 0.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(1.3 + 1.0e-13 / 0.4), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2 + 1.0e-13, 0.75 - 1.0e-13], velocity=[-0.4, -0.5],
                             time_stamp=Time.from_float(1.3)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[-0.8, -1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2, 0.75 - 1.0e-13 - 0.5 * 1.0e-13 / 0.4], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [-0.4, -0.5])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(1.3 + 1.0e-13 / 0.4), places=13)
        self.assertEqual(len(cnode.children), 1)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (0, 2))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.5 - 0.8 * 1.0e-13 / 0.4,
                                                                    0.9 - 1.0 * 1.0e-13 / 0.4], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [-0.8, -1.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(1.3 + 1.0e-13 / 0.4), places=13)

        in_state = Node(Unit(identifier=(0,), position=[0.2 + 1.0e-13, 0.75 - 1.0e-13], velocity=[-0.4, 0.5],
                             time_stamp=Time.from_float(1.3)), weight=1)
        in_state.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                     velocity=[-0.8, 1.0], time_stamp=Time.from_float(1.3)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2, 0.75 - 1.0e-13 + 0.5 * 1.0e-13 / 0.4], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [-0.4, 0.5])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(1.3 + 1.0e-13 / 0.4), places=13)
        self.assertEqual(len(cnode.children), 1)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (0, 2))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.5 - 0.8 * 1.0e-13 / 0.4,
                                                                    0.9 + 1.0 * 1.0e-13 / 0.4], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [-0.8, 1.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(1.3 + 1.0e-13 / 0.4), places=13)

    def _setUpCellForCompositeObjectsRootUnitActive(self):
        setting.set_number_of_root_nodes(6)
        setting.set_number_of_nodes_per_root_node(3)
        setting.set_number_of_node_levels(2)
        cells = CuboidPeriodicCells(cells_per_side=[5, 4])
        self._event_handler.initialize(cells=cells, cell_level=1)

    def test_send_event_time_cell_for_composite_objects_root_unit_active(self):
        self._setUpCellForCompositeObjectsRootUnitActive()
        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[2.0, 0.0],
                             time_stamp=Time.from_float(0.7)), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[2.0, 0.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.1],
                                     velocity=[2.0, 0.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(0.8), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[0.0, 3.0],
                             time_stamp=Time.from_float(0.7)), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[0.0, 3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.1],
                                     velocity=[0.0, 3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(0.7 + 0.1 / 3.0), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[-2.0, 0.0],
                             time_stamp=Time.from_float(0.7)), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[-2.0, 0.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.1],
                                     velocity=[-2.0, 0.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(0.7), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[0.0, -3.0],
                             time_stamp=Time.from_float(0.7)), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[0.0, -3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.1],
                                     velocity=[0.0, -3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(0.7 + 0.15 / 3.0), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[2.0, 3.0],
                             time_stamp=Time.from_float(0.7)), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[2.0, 3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.1],
                                     velocity=[2.0, 3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(0.7 + 0.1 / 3.0), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[2.0, -3.0],
                             time_stamp=Time.from_float(0.7)), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[2.0, -3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.1],
                                     velocity=[2.0, -3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state])
        self.assertAlmostEqual(event_time, Time.from_float(0.7 + 0.15 / 3.0), places=13)

    def test_send_out_state_cell_for_composite_objects_root_unit_active(self):
        self._setUpCellForCompositeObjectsRootUnitActive()
        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[2.0, 0.0],
                             time_stamp=Time.from_float(0.7)), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[2.0, 0.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.1],
                                     velocity=[2.0, 0.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.4, 0.4], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [2.0, 0.0])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(0.8), places=13)
        self.assertEqual(len(cnode.children), 2)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (1, 0))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.7, 0.6], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [2.0, 0.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(0.8), places=13)
        child_cnode = cnode.children[1]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (1, 1))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.5, 0.1], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [2.0, 0.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(0.8), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[0.0, 3.0],
                             time_stamp=Time.from_float(0.7)), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[0.0, 3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.1],
                                     velocity=[0.0, 3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2, 0.5], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [0.0, 3.0])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(0.7 + 0.1 / 3.0), places=13)
        self.assertEqual(len(cnode.children), 2)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (1, 0))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.5, 0.7], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [0.0, 3.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(0.7 + 0.1 / 3.0), places=13)
        child_cnode = cnode.children[1]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (1, 1))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.3, 0.2], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [0.0, 3.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(0.7 + 0.1 / 3.0), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[-2.0, 0.0],
                             time_stamp=Time.from_float(0.7)), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[-2.0, 0.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.1],
                                     velocity=[-2.0, 0.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2, 0.4], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [-2.0, 0.0])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(0.7), places=13)
        self.assertEqual(len(cnode.children), 2)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (1, 0))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.5, 0.6], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [-2.0, 0.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(0.7), places=13)
        child_cnode = cnode.children[1]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (1, 1))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.3, 0.1], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [-2.0, 0.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(0.7), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[0.0, -3.0],
                             time_stamp=Time.from_float(0.7)), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[0.0, -3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.1],
                                     velocity=[0.0, -3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2, 0.25], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [0.0, -3.0])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(0.7 + 0.15 / 3.0), places=13)
        self.assertEqual(len(cnode.children), 2)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (1, 0))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.5, 0.45], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [0.0, -3.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(0.7 + 0.15 / 3.0), places=13)
        child_cnode = cnode.children[1]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (1, 1))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.3, 0.95], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [0.0, -3.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(0.7 + 0.15 / 3.0), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[2.0, 3.0],
                             time_stamp=Time.from_float(0.7)), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[2.0, 3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.1],
                                     velocity=[2.0, 3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.2 + 0.2 / 3.0, 0.5], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [2.0, 3.0])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(0.7 + 0.1 / 3.0), places=13)
        self.assertEqual(len(cnode.children), 2)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (1, 0))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.5 + 0.2 / 3.0, 0.7], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [2.0, 3.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(0.7 + 0.1 / 3.0), places=13)
        child_cnode = cnode.children[1]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (1, 1))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.3 + 0.2 / 3.0, 0.2], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [2.0, 3.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(0.7 + 0.1 / 3.0), places=13)

        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[2.0, -3.0],
                             time_stamp=Time.from_float(0.7)), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[2.0, -3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.1],
                                     velocity=[2.0, -3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        self._event_handler.send_event_time([in_state])
        out_state = self._event_handler.send_out_state()
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertAlmostEqualSequence(cnode.value.position, [0.3, 0.25], places=13)
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.velocity, [2.0, -3.0])
        self.assertAlmostEqual(cnode.value.time_stamp, Time.from_float(0.7 + 0.15 / 3.0), places=13)
        self.assertEqual(len(cnode.children), 2)
        child_cnode = cnode.children[0]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (1, 0))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.6, 0.45], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [2.0, -3.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(0.7 + 0.15 / 3.0), places=13)
        child_cnode = cnode.children[1]
        self.assertIs(child_cnode.parent, cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (1, 1))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.4, 0.95], places=13)
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertIsNone(child_cnode.value.charge)
        self.assertEqual(child_cnode.value.velocity, [2.0, -3.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(0.7 + 0.15 / 3.0), places=13)

    def test_more_than_one_in_state_raises_error(self):
        self._setUpCellNoCompositeObjectsLeafUnitActive()
        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[0.5, 0.0], time_stamp=Time.from_float(0.3)), weight=1)
        in_state_two = Node(Unit(identifier=(3,), position=[0.2, 0.3], charge={"charge": 1.0},
                                 velocity=[0.5, 0.0], time_stamp=Time.from_float(0.3)), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler.send_event_time([in_state, in_state_two])

    def test_in_state_not_active_raises_error(self):
        self._setUpCellNoCompositeObjectsLeafUnitActive()
        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2],
                             charge={"charge": 1.0}), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler.send_event_time([in_state])

    def test_more_than_one_active_cnode_on_cell_level_raises_error(self):
        self._setUpCellForLeafUnitsLeafUnitActive()
        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[2.0, 3.0],
                             time_stamp=Time.from_float(0.7)), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[2.0, 3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.1],
                                     velocity=[2.0, 3.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        with self.assertRaises(AssertionError):
            self._event_handler.send_event_time([in_state])

    def test_no_velocity_component_unequal_zero_raises_error(self):
        self._setUpCellNoCompositeObjectsLeafUnitActive()
        in_state = Node(Unit(identifier=(1,), position=[0.1, 0.2], charge={"charge": 1.0},
                             velocity=[0.0, 0.0], time_stamp=Time.from_float(0.3)), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler.send_event_time([in_state])

    def test_initialize_with_non_periodic_cell_system_raises_error(self):
        setting.set_number_of_root_nodes(6)
        setting.set_number_of_nodes_per_root_node(1)
        setting.set_number_of_node_levels(1)
        cells = CuboidCells(cells_per_side=[5, 4])
        with self.assertRaises(ConfigurationError):
            self._event_handler.initialize(cells=cells, cell_level=1)

    def test_number_send_event_time_arguments_one(self):
        self.assertEqual(self._event_handler.number_send_event_time_arguments, 1)

    def test_number_send_out_state_arguments_zero(self):
        self.assertEqual(self._event_handler.number_send_out_state_arguments, 0)


if __name__ == '__main__':
    main()
