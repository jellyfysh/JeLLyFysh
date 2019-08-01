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
from unittest import TestCase, main
from base.exceptions import ConfigurationError
from base.node import Node
from base.unit import Unit
from event_handler.final_time_end_of_run_event_handler import FinalTimeEndOfRunEventHandler
import setting


class TestFinalTimeEndOfRunEventHandler(TestCase):
    def setUp(self) -> None:
        setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        self._event_handler_one = FinalTimeEndOfRunEventHandler(end_of_run_time=0.5,
                                                                output_handler="some_output_handler")
        self._event_handler_two = FinalTimeEndOfRunEventHandler(end_of_run_time=1.3, output_handler="output")

    def tearDown(self) -> None:
        setting.reset()

    def test_number_send_event_time_arguments_one(self):
        self.assertEqual(self._event_handler_one.number_send_event_time_arguments, 0)

    def test_number_send_event_time_arguments_two(self):
        self.assertEqual(self._event_handler_two.number_send_event_time_arguments, 0)

    def test_number_send_out_state_arguments_one(self):
        self.assertEqual(self._event_handler_one.number_send_out_state_arguments, 1)

    def test_number_send_out_state_arguments_two(self):
        self.assertEqual(self._event_handler_one.number_send_out_state_arguments, 1)

    def test_send_event_time_one(self):
        self.assertEqual(self._event_handler_one.send_event_time(), 0.5)

    def test_send_event_time_two(self):
        self.assertEqual(self._event_handler_two.send_event_time(), 1.3)

    def test_send_out_state_one(self):
        # Call this to update the event time
        self._event_handler_one.send_event_time()

        active_cnode_one = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                                     velocity=[0.1, 0.2, 0.3], time_stamp=0.0), weight=1)
        active_cnode_two = Node(Unit(identifier=(3,), position=[0.4, 0.5, 0.6],
                                     velocity=[1.0, 0.6, 0.5], time_stamp=0.2), weight=1)
        out_state = self._event_handler_one.send_out_state([active_cnode_one, active_cnode_two])
        self.assertEqual(len(out_state), 2)

        first_active_cnode = out_state[0]
        self.assertEqual(len(first_active_cnode.children), 0)
        self.assertIsNone(first_active_cnode.parent)
        self.assertEqual(first_active_cnode.value.identifier, (0,))
        self.assertEqual(first_active_cnode.value.position,
                         [0.1 + 0.5 * 0.1, 0.2 + 0.5 * 0.2, 0.3 + 0.5 * 0.3])
        self.assertEqual(first_active_cnode.weight, 1)
        self.assertEqual(first_active_cnode.value.velocity, [0.1, 0.2, 0.3])
        self.assertEqual(first_active_cnode.value.time_stamp, 0.5)
        self.assertIsNone(first_active_cnode.value.charge)

        second_active_cnode = out_state[1]
        self.assertEqual(len(second_active_cnode.children), 0)
        self.assertIsNone(second_active_cnode.parent)
        self.assertEqual(second_active_cnode.value.identifier, (3,))
        self.assertEqual(second_active_cnode.value.position,
                         [0.4 + 0.3 * 1.0, 0.5 + 0.3 * 0.6, 0.6 + 0.3 * 0.5])
        self.assertEqual(second_active_cnode.weight, 1)
        self.assertEqual(second_active_cnode.value.velocity, [1.0, 0.6, 0.5])
        self.assertEqual(second_active_cnode.value.time_stamp, 0.5)
        self.assertIsNone(second_active_cnode.value.charge)

    def test_send_out_state_two(self):
        self._event_handler_two.send_event_time()

        cnode = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                          velocity=[0.5, 0.0, 0.0], time_stamp=1.0), weight=1)
        cnode.add_child(Node(Unit(identifier=(0, 0), position=[0.7, 0.8, 0.9], charge={"charge": 1.0},
                                  velocity=[1.0, 0.0, 0.0], time_stamp=1.0), weight=0.5))
        out_state = self._event_handler_two.send_out_state([cnode])
        self.assertEqual(len(out_state), 1)

        out_state_cnode = out_state[0]
        self.assertIsNone(out_state_cnode.parent)
        self.assertEqual(out_state_cnode.value.identifier, (0,))
        self.assertEqual(out_state_cnode.value.position, [0.1 + 0.5 * 0.3, 0.2, 0.3])
        self.assertEqual(out_state_cnode.weight, 1)
        self.assertEqual(out_state_cnode.value.velocity, [0.5, 0.0, 0.0])
        self.assertEqual(out_state_cnode.value.time_stamp, 1.3)
        self.assertIsNone(out_state_cnode.value.charge)
        self.assertEqual(len(out_state_cnode.children), 1)

        child_cnode = out_state_cnode.children[0]
        self.assertEqual(len(child_cnode.children), 0)
        self.assertIs(child_cnode.parent, out_state_cnode)
        self.assertEqual(child_cnode.value.identifier, (0, 0))
        self.assertEqual(child_cnode.value.position, [0.0, 0.8, 0.9])
        self.assertEqual(child_cnode.value.velocity, [1.0, 0.0, 0.0])
        self.assertEqual(child_cnode.value.time_stamp, 1.3)
        self.assertEqual(child_cnode.value.charge, {"charge": 1.0})

    def test_output_handler_one(self):
        self.assertEqual(self._event_handler_one.output_handler, "some_output_handler")

    def test_output_handler_two(self):
        self.assertEqual(self._event_handler_two.output_handler, "output")

    def test_end_of_run_time_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            FinalTimeEndOfRunEventHandler(end_of_run_time=0.0, output_handler="output")

    def test_sampling_interval_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            FinalTimeEndOfRunEventHandler(end_of_run_time=-1.3, output_handler="output")


if __name__ == '__main__':
    main()
