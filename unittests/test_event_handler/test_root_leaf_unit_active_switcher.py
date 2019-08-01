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
from base.exceptions import ConfigurationError
from base.node import Node
from base.unit import Unit
from event_handler.root_leaf_unit_active_switcher import RootLeafUnitActiveSwitcher
import setting


class TestRootLeafUnitActiveSwitcher(TestCase):
    def setUp(self) -> None:
        setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        setting.set_number_of_node_levels(2)
        self._event_handler_to_root_unit_motion = RootLeafUnitActiveSwitcher(
            chain_length=0.5, aim_mode="root_unit_active")
        self._event_handler_to_leaf_unit_motion = RootLeafUnitActiveSwitcher(
            chain_length=0.7, aim_mode="leaf_unit_active")

    def tearDown(self) -> None:
        setting.reset()

    def test_send_event_time_to_active_root_unit(self):
        cnode = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                          velocity=[0.5, 0.0, 0.0], time_stamp=1.0), weight=1)
        cnode.add_child(Node(Unit(identifier=(0, 0), position=[0.7, 0.8, 0.9], charge={"charge": 1.0},
                                  velocity=[1.0, 0.0, 0.0], time_stamp=1.0), weight=0.5))
        self.assertEqual(self._event_handler_to_root_unit_motion.send_event_time([cnode]), 1.5)

    def test_send_event_time_to_active_leaf_unit(self):
        cnode = Node(Unit(identifier=(1,), position=[0.1, 0.2, 0.3],
                          velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=1)
        cnode.add_child(Node(Unit(identifier=(1, 0), position=[0.7, 0.8, 0.9], charge={"charge": 1.0},
                                  velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=0.5))
        cnode.add_child(Node(Unit(identifier=(1, 1), position=[0.5, 0.4, 0.3], charge={"charge": -1.0},
                                  velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=0.5))
        self.assertEqual(self._event_handler_to_leaf_unit_motion.send_event_time([cnode]), 2.2)

    def test_send_out_state_to_active_root_unit(self):
        cnode = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                          velocity=[0.5, 0.0, 0.0], time_stamp=1.0), weight=1)
        cnode.add_child(Node(Unit(identifier=(0, 0), position=[0.7, 0.8, 0.9], charge={"charge": 1.0},
                                  velocity=[1.0, 0.0, 0.0], time_stamp=1.0), weight=0.5))
        # Call this to update the event time
        self._event_handler_to_root_unit_motion.send_event_time([cnode])

        cnode.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.6, 0.1], charge={"charge": -1.0}),
                             weight=0.5))
        out_state = self._event_handler_to_root_unit_motion.send_out_state([cnode])
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.weight, 1)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertEqual(cnode.value.position, [0.1 + 0.5 * 0.5, 0.2, 0.3])
        self.assertEqual(cnode.value.velocity, [1.0, 0.0, 0.0])
        self.assertEqual(cnode.value.time_stamp, 1.5)
        self.assertEqual(len(cnode.children), 2)
        first_child = cnode.children[0]
        self.assertIs(first_child.parent, cnode)
        self.assertEqual(first_child.weight, 0.5)
        self.assertEqual(first_child.value.identifier, (0, 0))
        self.assertEqual(first_child.value.position, [(0.7 + 0.5) % 1.0, 0.8, 0.9])
        self.assertEqual(first_child.value.velocity, [1.0, 0.0, 0.0])
        self.assertEqual(first_child.value.time_stamp, 1.5)
        self.assertEqual(len(first_child.children), 0)
        second_child = cnode.children[1]
        self.assertIs(second_child.parent, cnode)
        self.assertEqual(second_child.weight, 0.5)
        self.assertEqual(second_child.value.identifier, (0, 1))
        self.assertEqual(second_child.value.position, [0.5, 0.6, 0.1])
        self.assertEqual(second_child.value.velocity, [1.0, 0.0, 0.0])
        self.assertEqual(second_child.value.time_stamp, 1.5)
        self.assertEqual(len(second_child.children), 0)

    @mock.patch("event_handler.root_leaf_unit_active_switcher.random.choice")
    def test_send_out_state_to_active_leaf_unit(self, random_choice_mock):
        cnode = Node(Unit(identifier=(1,), position=[0.1, 0.2, 0.3],
                          velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=1)
        first_child = Node(Unit(identifier=(1, 0), position=[0.7, 0.8, 0.9], charge={"charge": 1.0},
                                velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=0.5)
        cnode.add_child(first_child)
        second_child = Node(Unit(identifier=(1, 1), position=[0.5, 0.4, 0.3], charge={"charge": -1.0},
                                 velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=0.5)
        cnode.add_child(second_child)
        # Call this to update the event time
        self._event_handler_to_leaf_unit_motion.send_event_time([cnode])
        random_choice_mock.return_value = second_child

        out_state = self._event_handler_to_leaf_unit_motion.send_out_state([cnode])
        random_choice_mock.assert_called_once_with([first_child, second_child])
        self.assertEqual(len(out_state), 1)
        cnode = out_state[0]
        self.assertIsNone(cnode.parent)
        self.assertEqual(cnode.weight, 1)
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertEqual(cnode.value.position, [(0.1 + (2.2 - 1.5) * 1.0) % 1.0, 0.2, 0.3])
        self.assertEqual(cnode.value.velocity, [0.5, 0.0, 0.0])
        self.assertEqual(cnode.value.time_stamp, 2.2)
        self.assertEqual(len(cnode.children), 2)
        first_child = cnode.children[0]
        self.assertIs(first_child.parent, cnode)
        self.assertEqual(first_child.weight, 0.5)
        self.assertEqual(first_child.value.identifier, (1, 0))
        self.assertEqual(first_child.value.position, [(0.7 + (2.2 - 1.5) * 1.0) % 1.0, 0.8, 0.9])
        self.assertIsNone(first_child.value.velocity)
        self.assertIsNone(first_child.value.time_stamp)
        self.assertEqual(len(first_child.children), 0)
        second_child = cnode.children[1]
        self.assertIs(second_child.parent, cnode)
        self.assertEqual(second_child.weight, 0.5)
        self.assertEqual(second_child.value.identifier, (1, 1))
        self.assertEqual(second_child.value.position, [(0.5 + (2.2 - 1.5) * 1.0) % 1.0, 0.4, 0.3])
        self.assertEqual(second_child.value.velocity, [1.0, 0.0, 0.0])
        self.assertEqual(second_child.value.time_stamp, 2.2)
        self.assertEqual(len(second_child.children), 0)

    def test_send_event_time_to_active_root_unit_two_root_nodes_raises_error(self):
        cnode_one = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                              velocity=[0.5, 0.0, 0.0], time_stamp=1.0), weight=1)
        cnode_one.add_child(Node(Unit(identifier=(0, 0), position=[0.7, 0.8, 0.9], charge={"charge": 1.0},
                                      velocity=[1.0, 0.0, 0.0], time_stamp=1.0), weight=0.5))
        cnode_two = Node(Unit(identifier=(5,), position=[0.4, 0.5, 0.6],
                              velocity=[0.5, 0.0, 0.0], time_stamp=1.0), weight=1)
        cnode_two.add_child(Node(Unit(identifier=(5, 0), position=[0.3, 0.2, 0.1], charge={"charge": 1.0},
                                      velocity=[1.0, 0.0, 0.0], time_stamp=1.0), weight=0.5))
        with self.assertRaises(AssertionError):
            self._event_handler_to_root_unit_motion.send_event_time([cnode_one, cnode_two])

    def test_send_event_time_to_active_leaf_unit_two_root_nodes_raises_error(self):
        cnode_one = Node(Unit(identifier=(1,), position=[0.1, 0.2, 0.3],
                              velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=1)
        cnode_one.add_child(Node(Unit(identifier=(1, 0), position=[0.7, 0.8, 0.9], charge={"charge": 1.0},
                                      velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=0.5))
        cnode_one.add_child(Node(Unit(identifier=(1, 1), position=[0.5, 0.4, 0.3], charge={"charge": -1.0},
                                      velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=0.5))
        cnode_two = Node(Unit(identifier=(2,), position=[0.0, 0.0, 0.0],
                              velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=1)
        cnode_two.add_child(Node(Unit(identifier=(2, 7), position=[0.7, 0.7, 0.7], charge={"charge": 1.0},
                                      velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=0.5))
        cnode_two.add_child(Node(Unit(identifier=(2, 13), position=[0.3, 0.3, 0.3], charge={"charge": -1.0},
                                      velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=0.5))
        with self.assertRaises(AssertionError):
            self._event_handler_to_leaf_unit_motion.send_event_time([cnode_one, cnode_two])

    def test_send_out_state_to_active_root_unit_two_root_nodes_raises_error(self):
        cnode = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                          velocity=[0.5, 0.0, 0.0], time_stamp=1.0), weight=1)
        cnode.add_child(Node(Unit(identifier=(0, 0), position=[0.7, 0.8, 0.9], charge={"charge": 1.0},
                                  velocity=[1.0, 0.0, 0.0], time_stamp=1.0), weight=0.5))
        # Call this to update the event time
        self._event_handler_to_root_unit_motion.send_event_time([cnode])

        cnode.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.6, 0.1], charge={"charge": -1.0}),
                             weight=0.5))
        with self.assertRaises(AssertionError):
            self._event_handler_to_root_unit_motion.send_out_state([cnode, cnode])

    def test_send_out_state_to_active_root_unit_root_unit_active_raises_error(self):
        cnode = Node(Unit(identifier=(0,), position=[0.1, 0.2, 0.3],
                          velocity=[1.0, 0.0, 0.0], time_stamp=1.0), weight=1)
        cnode.add_child(Node(Unit(identifier=(0, 0), position=[0.7, 0.8, 0.9], charge={"charge": 1.0},
                                  velocity=[1.0, 0.0, 0.0], time_stamp=1.0), weight=0.5))
        # Call this to update the event time
        self._event_handler_to_root_unit_motion.send_event_time([cnode])

        cnode.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.6, 0.1], charge={"charge": -1.0},
                                  velocity=[1.0, 0.0, 0.0], time_stamp=1.0), weight=0.5))
        with self.assertRaises(AssertionError):
            self._event_handler_to_root_unit_motion.send_out_state([cnode])

    def test_send_out_state_to_active_leaf_unit_two_root_nodes_raises_error(self):
        cnode = Node(Unit(identifier=(1,), position=[0.1, 0.2, 0.3],
                          velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=1)
        first_child = Node(Unit(identifier=(1, 0), position=[0.7, 0.8, 0.9], charge={"charge": 1.0},
                                velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=0.5)
        cnode.add_child(first_child)
        second_child = Node(Unit(identifier=(1, 1), position=[0.5, 0.4, 0.3], charge={"charge": -1.0},
                                 velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=0.5)
        cnode.add_child(second_child)
        # Call this to update the event time
        self._event_handler_to_leaf_unit_motion.send_event_time([cnode])
        with self.assertRaises(AssertionError):
            self._event_handler_to_leaf_unit_motion.send_out_state([cnode, cnode])

    def test_send_out_state_to_active_leaf_unit_different_leaf_unit_velocities_raises_error(self):
        cnode = Node(Unit(identifier=(1,), position=[0.1, 0.2, 0.3],
                          velocity=[1.5, 0.0, 0.0], time_stamp=1.5), weight=1)
        first_child = Node(Unit(identifier=(1, 0), position=[0.7, 0.8, 0.9], charge={"charge": 1.0},
                                velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=0.5)
        cnode.add_child(first_child)
        second_child = Node(Unit(identifier=(1, 1), position=[0.5, 0.4, 0.3], charge={"charge": -1.0},
                                 velocity=[2.0, 0.0, 0.0], time_stamp=1.5), weight=0.5)
        cnode.add_child(second_child)
        # Call this to update the event time
        self._event_handler_to_leaf_unit_motion.send_event_time([cnode])
        with self.assertRaises(AssertionError):
            self._event_handler_to_leaf_unit_motion.send_out_state([cnode])

    def test_send_out_state_to_active_leaf_unit_leaf_unit_active_raises_error(self):
        cnode = Node(Unit(identifier=(1,), position=[0.1, 0.2, 0.3],
                          velocity=[0.5, 0.0, 0.0], time_stamp=1.5), weight=1)
        first_child = Node(Unit(identifier=(1, 0), position=[0.7, 0.8, 0.9], charge={"charge": 1.0},
                                velocity=[1.0, 0.0, 0.0], time_stamp=1.5), weight=0.5)
        cnode.add_child(first_child)
        second_child = Node(Unit(identifier=(1, 1), position=[0.5, 0.4, 0.3], charge={"charge": -1.0}),
                            weight=0.5)
        cnode.add_child(second_child)
        # Call this to update the event time
        self._event_handler_to_leaf_unit_motion.send_event_time([cnode])
        with self.assertRaises(AssertionError):
            self._event_handler_to_leaf_unit_motion.send_out_state([cnode])

    def test_wrong_aim_mode_raises_error(self):
        with self.assertRaises(ConfigurationError):
            RootLeafUnitActiveSwitcher(chain_length=0.3, aim_mode="some_aim_mode")

    def test_chain_length_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            RootLeafUnitActiveSwitcher(chain_length=-0.5, aim_mode="leaf_unit_active")

    def test_chain_length_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            RootLeafUnitActiveSwitcher(chain_length=0.0, aim_mode="leaf_unit_active")

    def test_number_of_node_levels_one_raises_error(self):
        setting.reset()
        setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        setting.set_number_of_node_levels(1)
        with self.assertRaises(ConfigurationError):
            self._event_handler_to_root_unit_motion = RootLeafUnitActiveSwitcher(
                chain_length=0.5, aim_mode="leaf_unit_active")


if __name__ == '__main__':
    main()
