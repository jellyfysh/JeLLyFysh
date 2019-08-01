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
from event_handler.same_active_periodic_direction_end_of_chain_event_handler \
    import SameActivePeriodicDirectionEndOfChainEventHandler
import setting


class TestSameActivePeriodicDirectionEndOfChainEventHandler(TestCase):
    def setUp(self) -> None:
        setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        self._event_handler = SameActivePeriodicDirectionEndOfChainEventHandler(chain_length=0.7)

    def tearDown(self) -> None:
        setting.reset()

    def test_number_send_event_time_arguments(self):
        self.assertEqual(self._event_handler.number_send_event_time_arguments, 0)

    def test_number_send_out_state_arguments(self):
        self.assertEqual(self._event_handler.number_send_out_state_arguments, 2)

    def test_send_event_time(self):
        event_time, new_active_identifiers = self._event_handler.send_event_time()
        self.assertEqual(event_time, 0.7)
        # First list gets unpacked by mediator
        self.assertEqual(new_active_identifiers, [[]])
        event_time, new_active_identifiers = self._event_handler.send_event_time()
        self.assertEqual(event_time, 1.4)
        # First list gets unpacked by mediator
        self.assertEqual(new_active_identifiers, [[]])

    def test_send_out_state_leaf_units_initial_direction_zero(self):
        # Call this to update the event time
        self._event_handler.send_event_time()
        active_cnode_one = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                                     velocity=[1.0, 0.0, 0.0], time_stamp=0.0), weight=1)
        active_cnode_two = Node(Unit(identifier=(3,), position=[0.7, 0.8, 0.9],
                                     velocity=[1.0, 0.0, 0.0], time_stamp=0.3), weight=1)
        out_state = self._event_handler.send_out_state([active_cnode_one, active_cnode_two], [])

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.children, [])
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [0.1 + 0.7, 0.2, 0.3])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.0, 1.0, 0.0])
        self.assertEqual(first_cnode.value.time_stamp, 0.7)
        self.assertIsNone(first_cnode.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.children, [])
        self.assertEqual(second_cnode.value.identifier, (3,))
        self.assertEqual(second_cnode.value.position, [(0.7 + (0.7 - 0.3)*1.0) % 1.0, 0.8, 0.9])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [0.0, 1.0, 0.0])
        self.assertEqual(second_cnode.value.time_stamp, 0.7)
        self.assertIsNone(second_cnode.value.charge)

    def test_send_out_state_leaf_units_initial_direction_one(self):
        # Call this to update the event time
        self._event_handler.send_event_time()
        active_cnode_one = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                                     velocity=[0.0, 1.0, 0.0], time_stamp=0.0), weight=1)
        active_cnode_two = Node(Unit(identifier=(3,), position=[0.7, 0.8, 0.9],
                                     velocity=[0.0, 1.0, 0.0], time_stamp=0.3), weight=1)
        out_state = self._event_handler.send_out_state([active_cnode_one, active_cnode_two], [])

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.children, [])
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [0.1, 0.2 + 0.7, 0.3])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.0, 0.0, 1.0])
        self.assertEqual(first_cnode.value.time_stamp, 0.7)
        self.assertIsNone(first_cnode.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.children, [])
        self.assertEqual(second_cnode.value.identifier, (3,))
        self.assertEqual(second_cnode.value.position, [0.7, (0.8 + (0.7 - 0.3)*1.0) % 1.0, 0.9])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [0.0, 0.0, 1.0])
        self.assertEqual(second_cnode.value.time_stamp, 0.7)
        self.assertIsNone(second_cnode.value.charge)

    def test_send_out_state_leaf_units_initial_direction_two(self):
        # Call this to update the event time
        self._event_handler.send_event_time()
        active_cnode_one = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                                     velocity=[0.0, 0.0, 1.0], time_stamp=0.0), weight=1)
        active_cnode_two = Node(Unit(identifier=(3,), position=[0.7, 0.8, 0.9],
                                     velocity=[0.0, 0.0, 1.0], time_stamp=0.3), weight=1)
        out_state = self._event_handler.send_out_state([active_cnode_one, active_cnode_two], [])

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.children, [])
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [0.1, 0.2, (0.3 + 0.7) % 1.0])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [1.0, 0.0, 0.0])
        self.assertEqual(first_cnode.value.time_stamp, 0.7)
        self.assertIsNone(first_cnode.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.children, [])
        self.assertEqual(second_cnode.value.identifier, (3,))
        self.assertEqual(second_cnode.value.position, [0.7, 0.8, (0.9 + (0.7 - 0.3)*1.0) % 1.0])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [1.0, 0.0, 0.0])
        self.assertEqual(second_cnode.value.time_stamp, 0.7)
        self.assertIsNone(second_cnode.value.charge)

    def test_send_out_state_leaf_unit_branch(self):
        self._event_handler.send_event_time()

        cnode = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                          velocity=[0.25, 0.0, 0.0], time_stamp=0.5), weight=1)
        cnode.add_child(Node(Unit(identifier=(0, 0), position=[0.7, 0.8, 0.9], charge={"charge": 1.0},
                                  velocity=[0.5, 0.0, 0.0], time_stamp=0.4), weight=0.5))
        out_state = self._event_handler.send_out_state([cnode], [])

        self.assertEqual(len(out_state), 1)
        out_state_cnode = out_state[0]
        self.assertIsNone(out_state_cnode.parent)
        self.assertEqual(out_state_cnode.value.identifier, (0,))
        self.assertEqual(out_state_cnode.value.position, [0.1 + (0.7 - 0.5) * 0.25, 0.2, 0.3])
        self.assertEqual(out_state_cnode.weight, 1)
        self.assertEqual(out_state_cnode.value.velocity, [0.0, 0.25, 0.0])
        self.assertEqual(out_state_cnode.value.time_stamp, 0.7)
        self.assertIsNone(out_state_cnode.value.charge)

        self.assertEqual(len(out_state_cnode.children), 1)
        child_cnode = out_state_cnode.children[0]
        self.assertIs(child_cnode.parent, out_state_cnode)
        self.assertEqual(child_cnode.children, [])
        self.assertEqual(child_cnode.value.identifier, (0, 0))
        self.assertEqual(child_cnode.value.position, [(0.7 + (0.7 - 0.4) * 0.5) % 1.0, 0.8, 0.9])
        self.assertEqual(child_cnode.weight, 0.5)
        self.assertEqual(child_cnode.value.velocity, [0.0, 0.5, 0.0])
        self.assertEqual(child_cnode.value.time_stamp, 0.7)
        self.assertEqual(child_cnode.value.charge, {"charge": 1.0})

    def test_send_out_state_root_unit(self):
        self._event_handler.send_event_time()

        cnode = Node(Unit(identifier=(1,), position=[0.1, 0.2, 0.3], velocity=[0.0, 2.0, 0.0],
                          time_stamp=0.0), weight=1)
        cnode.add_child(Node(Unit(identifier=(1, 0), position=[0.4, 0.5, 0.6],
                                  velocity=[0.0, 2.0, 0.0], time_stamp=0.0,
                                  charge={"charge": 1.0}), weight=0.5))
        cnode.add_child(Node(Unit(identifier=(1, 1), position=[0.7, 0.8, 0.9],
                                  velocity=[0.0, 2.0, 0.0], time_stamp=0.0,
                                  charge={"charge": -1.0}), weight=0.5))

        out_state = self._event_handler.send_out_state([cnode], [])
        self.assertEqual(len(out_state), 1)
        out_state_cnode = out_state[0]
        self.assertIsNone(out_state_cnode.parent)
        self.assertEqual(out_state_cnode.value.identifier, (1,))
        self.assertEqual(out_state_cnode.value.position, [0.1, (0.2 + 0.7 * 2.0) % 1.0, 0.3])
        self.assertEqual(out_state_cnode.weight, 1)
        self.assertEqual(out_state_cnode.value.velocity, [0.0, 0.0, 2.0])
        self.assertEqual(out_state_cnode.value.time_stamp, 0.7)
        self.assertIsNone(out_state_cnode.value.charge)

        self.assertEqual(len(out_state_cnode.children), 2)
        first_child_cnode = out_state_cnode.children[0]
        self.assertIs(first_child_cnode.parent, out_state_cnode)
        self.assertEqual(first_child_cnode.children, [])
        self.assertEqual(first_child_cnode.value.identifier, (1, 0))
        self.assertEqual(first_child_cnode.value.position, [0.4, (0.5 + 0.7 * 2.0) % 1.0, 0.6])
        self.assertEqual(first_child_cnode.weight, 0.5)
        self.assertEqual(first_child_cnode.value.velocity, [0.0, 0.0, 2.0])
        self.assertEqual(first_child_cnode.value.time_stamp, 0.7)
        self.assertEqual(first_child_cnode.value.charge, {"charge": 1.0})

        second_child_cnode = out_state_cnode.children[1]
        self.assertIs(second_child_cnode.parent, out_state_cnode)
        self.assertEqual(second_child_cnode.children, [])
        self.assertEqual(second_child_cnode.value.identifier, (1, 1))
        self.assertEqual(second_child_cnode.value.position, [0.7, (0.8 + 0.7 * 2.0) % 1.0, 0.9])
        self.assertEqual(second_child_cnode.weight, 0.5)
        self.assertEqual(second_child_cnode.value.velocity, [0.0, 0.0, 2.0])
        self.assertEqual(second_child_cnode.value.time_stamp, 0.7)
        self.assertEqual(second_child_cnode.value.charge, {"charge": -1.0})

    def test_different_velocities_raises_error(self):
        self._event_handler.send_event_time()
        active_cnode_one = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                                     velocity=[1.0, 0.0, 0.0], time_stamp=0.0), weight=1)
        active_cnode_two = Node(Unit(identifier=(3,), position=[0.4, 0.5, 0.6],
                                     velocity=[0.5, 0.0, 0.0], time_stamp=0.2), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler.send_out_state([active_cnode_one, active_cnode_two], [])

    def test_more_than_one_velocity_component_raises_error(self):
        self._event_handler.send_event_time()
        active_cnode_one = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                                     velocity=[1.0, 0.2, 0.3], time_stamp=0.0), weight=1)
        active_cnode_two = Node(Unit(identifier=(3,), position=[0.4, 0.5, 0.6],
                                     velocity=[1.0, 0.6, 0.5], time_stamp=0.2), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler.send_out_state([active_cnode_one, active_cnode_two], [])

    def test_new_active_nodes_in_send_out_state_raises_error(self):
        self._event_handler.send_event_time()
        cnode_with_active_unit = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                                           velocity=[1.0, 0.0, 0.0], time_stamp=0.0), weight=1)
        cnode_with_non_active_unit = Node(Unit(identifier=(3,), position=[0.4, 0.5, 0.6]), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler.send_out_state([cnode_with_active_unit], [cnode_with_non_active_unit])

    def test_active_node_not_moving_raises_error(self):
        self._event_handler.send_event_time()
        cnode_with_active_unit = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                                           velocity=[1.0, 0.0, 0.0], time_stamp=0.0), weight=1)
        cnode_with_non_active_unit = Node(Unit(identifier=(3,), position=[0.4, 0.5, 0.6]), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler.send_out_state([cnode_with_active_unit, cnode_with_non_active_unit], [])

    def test_chain_length_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            SameActivePeriodicDirectionEndOfChainEventHandler(chain_length=0.0)

    def test_chain_length_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            SameActivePeriodicDirectionEndOfChainEventHandler(chain_length=-0.2)


if __name__ == '__main__':
    main()
