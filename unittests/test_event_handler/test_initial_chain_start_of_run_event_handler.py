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
from unittest import TestCase, main
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
from jellyfysh.base.unit import Unit
from jellyfysh.event_handler.initial_chain_start_of_run_event_handler import InitialChainStartOfRunEventHandler
import jellyfysh.setting as setting
from jellyfysh.setting import hypercubic_setting


class TestInitialChainStartOfRunEventHandler(TestCase):
    def setUp(self) -> None:
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        self._event_handler_one = InitialChainStartOfRunEventHandler(initial_direction_of_motion=0,
                                                                     speed=1.0,
                                                                     initial_active_identifier=[0])
        self._event_handler_two = InitialChainStartOfRunEventHandler(initial_direction_of_motion=2,
                                                                     speed=0.5,
                                                                     initial_active_identifier=[0, 1])

    def tearDown(self) -> None:
        setting.reset()

    def test_number_send_event_time_arguments_zero(self):
        self.assertEqual(self._event_handler_one.number_send_event_time_arguments, 0)
        self.assertEqual(self._event_handler_two.number_send_event_time_arguments, 0)

    def test_number_send_out_state_arguments_one(self):
        self.assertEqual(self._event_handler_one.number_send_out_state_arguments, 1)
        self.assertEqual(self._event_handler_two.number_send_out_state_arguments, 1)

    def test_send_event_time_one(self):
        event_time, identifier = self._event_handler_one.send_event_time()
        self.assertAlmostEqual(event_time, Time.from_float(0.0), places=13)
        # List will be unpacked in the mediator
        self.assertEqual(identifier, [(0,)])

    def test_send_event_time_two(self):
        event_time, identifier = self._event_handler_two.send_event_time()
        self.assertAlmostEqual(event_time, Time.from_float(0.0), places=13)
        self.assertEqual(identifier, [(0, 1)])

    def test_send_out_state_one_leaf_unit_active(self):
        # Call this to set event time in event handler
        self._event_handler_one.send_event_time()

        cnode = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3]), weight=1)
        out_state = self._event_handler_one.send_out_state(cnode)
        self.assertEqual(len(out_state), 1)
        out_state_cnode = out_state[0]
        self.assertEqual(out_state_cnode.children, [])
        self.assertIsNone(out_state_cnode.parent)
        self.assertEqual(out_state_cnode.value.identifier, (0,))
        self.assertEqual(out_state_cnode.value.position, [0.1, 0.2, 0.3])
        self.assertEqual(out_state_cnode.weight, 1)
        self.assertEqual(out_state_cnode.value.velocity, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(out_state_cnode.value.time_stamp, Time.from_float(0.0), places=13)
        self.assertIsNone(out_state_cnode.value.charge)

    def test_send_out_state_one_root_unit_active(self):
        self._event_handler_one.send_event_time()

        cnode = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3]), weight=1)
        cnode.add_child(Node(Unit(identifier=(0, 0), position=[0.4, 0.5, 0.6],
                                  charge={"charge": 1.0}), weight=0.5))
        cnode.add_child(Node(Unit(identifier=(0, 1), position=[0.7, 0.8, 0.9],
                                  charge={"charge": -1.0}), weight=0.5))
        out_state = self._event_handler_one.send_out_state(cnode)
        self.assertEqual(len(out_state), 1)
        out_state_cnode = out_state[0]
        self.assertIsNone(out_state_cnode.parent)
        self.assertEqual(out_state_cnode.value.identifier, (0,))
        self.assertEqual(out_state_cnode.value.position, [0.1, 0.2, 0.3])
        self.assertEqual(out_state_cnode.weight, 1)
        self.assertEqual(out_state_cnode.value.velocity, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(out_state_cnode.value.time_stamp, Time.from_float(0.0), places=13)
        self.assertIsNone(out_state_cnode.value.charge)

        self.assertEqual(len(out_state_cnode.children), 2)
        first_child_cnode = out_state_cnode.children[0]
        self.assertIs(first_child_cnode.parent, out_state_cnode)
        self.assertEqual(first_child_cnode.children, [])
        self.assertEqual(first_child_cnode.value.identifier, (0, 0))
        self.assertEqual(first_child_cnode.value.position, [0.4, 0.5, 0.6])
        self.assertEqual(first_child_cnode.weight, 0.5)
        self.assertEqual(first_child_cnode.value.velocity, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(first_child_cnode.value.time_stamp, Time.from_float(0.0), places=13)
        self.assertEqual(first_child_cnode.value.charge, {"charge": 1.0})

        second_child_cnode = out_state_cnode.children[1]
        self.assertIs(second_child_cnode.parent, out_state_cnode)
        self.assertEqual(second_child_cnode.children, [])
        self.assertEqual(second_child_cnode.value.identifier, (0, 1))
        self.assertEqual(second_child_cnode.value.position, [0.7, 0.8, 0.9])
        self.assertEqual(second_child_cnode.weight, 0.5)
        self.assertEqual(second_child_cnode.value.velocity, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(second_child_cnode.value.time_stamp, Time.from_float(0.0), places=13)
        self.assertEqual(second_child_cnode.value.charge, {"charge": -1.0})

    def test_send_out_state_two(self):
        self._event_handler_two.send_event_time()

        cnode = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3]), weight=1)
        cnode.add_child(Node(Unit(identifier=(0, 1), position=[0.4, 0.5, 0.6],
                                  charge={"charge": 1.0}), weight=0.5))

        out_state = self._event_handler_two.send_out_state(cnode)
        self.assertEqual(len(out_state), 1)
        out_state_cnode = out_state[0]
        self.assertIsNone(out_state_cnode.parent)
        self.assertEqual(out_state_cnode.value.identifier, (0,))
        self.assertEqual(out_state_cnode.value.position, [0.1, 0.2, 0.3])
        self.assertEqual(out_state_cnode.weight, 1)
        self.assertEqual(out_state_cnode.value.velocity, [0.0, 0.0, 0.25])
        self.assertAlmostEqual(out_state_cnode.value.time_stamp, Time.from_float(0.0), places=13)
        self.assertIsNone(out_state_cnode.value.charge)

        self.assertEqual(len(out_state_cnode.children), 1)
        first_child_cnode = out_state_cnode.children[0]
        self.assertIs(first_child_cnode.parent, out_state_cnode)
        self.assertEqual(first_child_cnode.children, [])
        self.assertEqual(first_child_cnode.value.identifier, (0, 1))
        self.assertEqual(first_child_cnode.value.position, [0.4, 0.5, 0.6])
        self.assertEqual(first_child_cnode.weight, 0.5)
        self.assertEqual(first_child_cnode.value.velocity, [0.0, 0.0, 0.5])
        self.assertAlmostEqual(first_child_cnode.value.time_stamp, Time.from_float(0.0), places=13)
        self.assertEqual(first_child_cnode.value.charge, {"charge": 1.0})

    def test_initial_direction_of_motion_exceeds_dimension_raises_error(self):
        with self.assertRaises(ConfigurationError):
            InitialChainStartOfRunEventHandler(initial_direction_of_motion=3, speed=1.0, initial_active_identifier=[0])

    def test_speed_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            InitialChainStartOfRunEventHandler(initial_direction_of_motion=0, speed=0.0, initial_active_identifier=[0])


if __name__ == '__main__':
    main()
