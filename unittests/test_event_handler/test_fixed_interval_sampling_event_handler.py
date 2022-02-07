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
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
from jellyfysh.base.unit import Unit
from jellyfysh.event_handler.fixed_interval_sampling_event_handler import FixedIntervalSamplingEventHandler
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


class TestFixedIntervalSamplingEventHandler(ExpandedTestCase, TestCase):
    def setUp(self) -> None:
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        self._event_handler_one = FixedIntervalSamplingEventHandler(sampling_interval=0.5,
                                                                    output_handler="some_output_handler")
        self._event_handler_two = FixedIntervalSamplingEventHandler(sampling_interval=1.3, output_handler="output",
                                                                    first_event_time_zero=True)

    def tearDown(self) -> None:
        setting.reset()

    def test_number_send_event_time_arguments_zero(self):
        self.assertEqual(self._event_handler_one.number_send_event_time_arguments, 0)
        self.assertEqual(self._event_handler_two.number_send_event_time_arguments, 0)

    def test_number_send_out_state_arguments_one(self):
        self.assertEqual(self._event_handler_one.number_send_out_state_arguments, 1)
        self.assertEqual(self._event_handler_two.number_send_out_state_arguments, 1)

    def test_send_event_time_event_handler_one(self):
        self.assertAlmostEqual(self._event_handler_one.send_event_time(), Time.from_float(0.5), places=13)
        self.assertAlmostEqual(self._event_handler_one.send_event_time(), Time.from_float(1.0), places=13)

    def test_send_event_time_event_handler_two(self):
        self.assertAlmostEqual(self._event_handler_two.send_event_time(), Time.from_float(0.0), places=13)
        self.assertAlmostEqual(self._event_handler_two.send_event_time(), Time.from_float(1.3), places=13)
        self.assertAlmostEqual(self._event_handler_two.send_event_time(), Time.from_float(2.6), places=13)

    def test_send_out_state_event_handler_one_general_velocity(self):
        # Call this to update the event time
        self._event_handler_one.send_event_time()

        active_cnode_one = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                                     velocity=[0.1, 0.2, 0.3], time_stamp=Time.from_float(0.0)), weight=1)
        active_cnode_two = Node(Unit(identifier=(3,), position=[0.4, 0.5, 0.6],
                                     velocity=[1.0, 0.6, 0.5], time_stamp=Time.from_float(0.2)), weight=1)
        out_state = self._event_handler_one.send_out_state([active_cnode_one, active_cnode_two])
        self.assertEqual(len(out_state), 2)

        first_active_cnode = out_state[0]
        self.assertEqual(len(first_active_cnode.children), 0)
        self.assertIsNone(first_active_cnode.parent)
        self.assertEqual(first_active_cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(first_active_cnode.value.position, [0.15, 0.3, 0.45], places=13)
        self.assertEqual(first_active_cnode.weight, 1)
        self.assertEqual(first_active_cnode.value.velocity, [0.1, 0.2, 0.3])
        self.assertAlmostEqual(first_active_cnode.value.time_stamp, Time.from_float(0.5), places=13)
        self.assertIsNone(first_active_cnode.value.charge)

        second_active_cnode = out_state[1]
        self.assertEqual(len(second_active_cnode.children), 0)
        self.assertIsNone(second_active_cnode.parent)
        self.assertEqual(second_active_cnode.value.identifier, (3,))
        self.assertAlmostEqualSequence(second_active_cnode.value.position, [0.7, 0.68, 0.75], places=13)
        self.assertEqual(second_active_cnode.weight, 1)
        self.assertEqual(second_active_cnode.value.velocity, [1.0, 0.6, 0.5])
        self.assertAlmostEqual(second_active_cnode.value.time_stamp, Time.from_float(0.5), places=13)
        self.assertIsNone(second_active_cnode.value.charge)

        self._event_handler_one.send_event_time()
        active_cnode_one = Node(Unit(identifier=(0,), position=[0.8, 0.4, 0.3],
                                     velocity=[1.0, 2.0, 3.0], time_stamp=Time.from_float(0.7)), weight=1)
        active_cnode_two = Node(Unit(identifier=(3,), position=[0.7, 0.2, 0.3],
                                     velocity=[-1.0, -0.6, -0.5], time_stamp=Time.from_float(0.2)), weight=1)
        out_state = self._event_handler_one.send_out_state([active_cnode_one, active_cnode_two])
        self.assertEqual(len(out_state), 2)

        first_active_cnode = out_state[0]
        self.assertEqual(len(first_active_cnode.children), 0)
        self.assertIsNone(first_active_cnode.parent)
        self.assertEqual(first_active_cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(first_active_cnode.value.position, [0.1, 0.0, 0.2], places=13)
        self.assertEqual(first_active_cnode.weight, 1)
        self.assertEqual(first_active_cnode.value.velocity, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(first_active_cnode.value.time_stamp, Time.from_float(1.0), places=13)
        self.assertIsNone(first_active_cnode.value.charge)

        second_active_cnode = out_state[1]
        self.assertEqual(len(second_active_cnode.children), 0)
        self.assertIsNone(second_active_cnode.parent)
        self.assertEqual(second_active_cnode.value.identifier, (3,))
        self.assertAlmostEqualSequence(second_active_cnode.value.position, [0.9, 0.72, 0.9], places=13)
        self.assertEqual(second_active_cnode.weight, 1)
        self.assertEqual(second_active_cnode.value.velocity, [-1.0, -0.6, -0.5])
        self.assertAlmostEqual(second_active_cnode.value.time_stamp, Time.from_float(1.0), places=13)
        self.assertIsNone(second_active_cnode.value.charge)

    def test_send_out_state_event_handler_two_cartesian_velocity(self):
        self._event_handler_two.send_event_time()

        cnode = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                          velocity=[0.5, 0.0, 0.0], time_stamp=Time.from_float(0.0)), weight=1)
        cnode.add_child(Node(Unit(identifier=(0, 0), position=[0.7, 0.8, 0.9], charge={"charge": 1.0},
                                  velocity=[1.0, 0.0, 0.0], time_stamp=Time.from_float(0.0)), weight=0.5))
        out_state = self._event_handler_two.send_out_state([cnode])
        self.assertEqual(len(out_state), 1)

        out_state_cnode = out_state[0]
        self.assertIsNone(out_state_cnode.parent)
        self.assertEqual(out_state_cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(out_state_cnode.value.position, [0.1, 0.2, 0.3], places=13)
        self.assertEqual(out_state_cnode.weight, 1)
        self.assertEqual(out_state_cnode.value.velocity, [0.5, 0.0, 0.0])
        self.assertAlmostEqual(out_state_cnode.value.time_stamp, Time.from_float(0.0), places=13)
        self.assertIsNone(out_state_cnode.value.charge)
        self.assertEqual(len(out_state_cnode.children), 1)

        child_cnode = out_state_cnode.children[0]
        self.assertEqual(len(child_cnode.children), 0)
        self.assertIs(child_cnode.parent, out_state_cnode)
        self.assertEqual(child_cnode.value.identifier, (0, 0))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.7, 0.8, 0.9], places=13)
        self.assertEqual(child_cnode.value.velocity, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(0.0), places=13)
        self.assertEqual(child_cnode.value.charge, {"charge": 1.0})

        self._event_handler_two.send_event_time()

        cnode = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                          velocity=[0.5, 0.0, 0.0], time_stamp=Time.from_float(1.0)), weight=1)
        cnode.add_child(Node(Unit(identifier=(0, 0), position=[0.7, 0.8, 0.9], charge={"charge": 1.0},
                                  velocity=[1.0, 0.0, 0.0], time_stamp=Time.from_float(1.0)), weight=0.5))
        out_state = self._event_handler_two.send_out_state([cnode])
        self.assertEqual(len(out_state), 1)

        out_state_cnode = out_state[0]
        self.assertIsNone(out_state_cnode.parent)
        self.assertEqual(out_state_cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(out_state_cnode.value.position, [0.25, 0.2, 0.3], places=13)
        self.assertEqual(out_state_cnode.weight, 1)
        self.assertEqual(out_state_cnode.value.velocity, [0.5, 0.0, 0.0])
        self.assertAlmostEqual(out_state_cnode.value.time_stamp, Time.from_float(1.3), places=13)
        self.assertIsNone(out_state_cnode.value.charge)
        self.assertEqual(len(out_state_cnode.children), 1)

        child_cnode = out_state_cnode.children[0]
        self.assertEqual(len(child_cnode.children), 0)
        self.assertIs(child_cnode.parent, out_state_cnode)
        self.assertEqual(child_cnode.value.identifier, (0, 0))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.0, 0.8, 0.9], places=13)
        self.assertEqual(child_cnode.value.velocity, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(1.3), places=13)
        self.assertEqual(child_cnode.value.charge, {"charge": 1.0})

        self._event_handler_two.send_event_time()

        cnode = Node(Unit(identifier=(0,), position=[0.7, 0.6, 0.4],
                          velocity=[0.0, 1.0, 0.0], time_stamp=Time.from_float(1.7)), weight=1)
        cnode.add_child(Node(Unit(identifier=(0, 0), position=[0.8, 0.1, 0.2], charge={"charge": 1.0},
                                  velocity=[0.0, 1.0, 0.0], time_stamp=Time.from_float(1.6)), weight=0.5))
        cnode.add_child(Node(Unit(identifier=(0, 1), position=[0.7, 0.3, 0.4], charge={"charge": 1.0},
                                  velocity=[0.0, 1.0, 0.0], time_stamp=Time.from_float(1.8)), weight=0.5))
        out_state = self._event_handler_two.send_out_state([cnode])
        self.assertEqual(len(out_state), 1)

        out_state_cnode = out_state[0]
        self.assertIsNone(out_state_cnode.parent)
        self.assertEqual(out_state_cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(out_state_cnode.value.position, [0.7, 0.5, 0.4], places=13)
        self.assertEqual(out_state_cnode.weight, 1)
        self.assertEqual(out_state_cnode.value.velocity, [0.0, 1.0, 0.0])
        self.assertAlmostEqual(out_state_cnode.value.time_stamp, Time.from_float(2.6), places=13)
        self.assertIsNone(out_state_cnode.value.charge)
        self.assertEqual(len(out_state_cnode.children), 2)

        child_cnode = out_state_cnode.children[0]
        self.assertEqual(len(child_cnode.children), 0)
        self.assertIs(child_cnode.parent, out_state_cnode)
        self.assertEqual(child_cnode.value.identifier, (0, 0))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.8, 0.1, 0.2], places=13)
        self.assertEqual(child_cnode.value.velocity, [0.0, 1.0, 0.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(2.6), places=13)
        self.assertEqual(child_cnode.value.charge, {"charge": 1.0})

        child_cnode = out_state_cnode.children[1]
        self.assertEqual(len(child_cnode.children), 0)
        self.assertIs(child_cnode.parent, out_state_cnode)
        self.assertEqual(child_cnode.value.identifier, (0, 1))
        self.assertAlmostEqualSequence(child_cnode.value.position, [0.7, 0.1, 0.4], places=13)
        self.assertEqual(child_cnode.value.velocity, [0.0, 1.0, 0.0])
        self.assertAlmostEqual(child_cnode.value.time_stamp, Time.from_float(2.6), places=13)
        self.assertEqual(child_cnode.value.charge, {"charge": 1.0})

    def test_output_handler(self):
        self.assertEqual(self._event_handler_one.output_handler, "some_output_handler")
        self.assertEqual(self._event_handler_two.output_handler, "output")

    def test_sampling_interval_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            FixedIntervalSamplingEventHandler(sampling_interval=0.0, output_handler="output")

    def test_sampling_interval_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            FixedIntervalSamplingEventHandler(sampling_interval=-0.1, output_handler="output")


if __name__ == '__main__':
    main()
