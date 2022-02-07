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
from jellyfysh.base.node import Node
from jellyfysh.base.particle import Particle
from jellyfysh.base.time import Time
from jellyfysh.base.unit import Unit
import jellyfysh.setting as setting
from jellyfysh.setting import hypercubic_setting
from jellyfysh.state_handler.tree_state_handler import TreeStateHandler
from jellyfysh.state_handler.lifting_state.tree_lifting_state import TreeLiftingState
from jellyfysh.state_handler.physical_state.tree_physical_state import TreePhysicalState


# noinspection PyArgumentEqualDefault
class TestTreeStateHandler(TestCase):
    def setUp(self) -> None:
        # Build up two dipoles to test everything
        self._root_nodes = [Node(), Node()]
        self._root_nodes[0].add_child(Node(Particle(position=[0, 0], charge={"e": 1})))
        self._root_nodes[0].add_child(Node(Particle(position=[0.1, 0.05], charge={"e": -1})))
        self._root_nodes[1].add_child(Node(Particle(position=[0.9, 0.8], charge={"e": 1})))
        self._root_nodes[1].add_child(Node(Particle(position=[0.9, 0.75], charge={"e": -1})))
        for node in self._root_nodes:
            position = [sum(leaf_node.value.position[index] * leaf_node.weight
                            for leaf_node in node.children) for index in range(2)]
            node.value = Particle(position=position)
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        setting.number_of_node_levels = 2
        setting.number_of_root_nodes = 2
        setting.number_of_nodes_per_root_node = 2
        # Test the combination used throughout the application in JFV 1.0.0.0
        physical_state = TreePhysicalState()
        lifting_state = TreeLiftingState()
        self._state_handler = TreeStateHandler(physical_state, lifting_state)

    def tearDown(self):
        setting.reset()

    def test_initialize_with_extract_from_global_state(self):
        self._state_handler.initialize(self._root_nodes)

        # Branch of root node should include all children
        cnode = self._state_handler.extract_from_global_state((0,))
        self.assertEqual(cnode.value.position, [0.05, 0.025])
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.identifier, (0,))
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.velocity)
        self.assertIsNone(cnode.value.time_stamp)
        self.assertEqual(len(cnode.children), 2)
        first_child = cnode.children[0]
        self.assertEqual(first_child.value.position, [0, 0])
        self.assertEqual(first_child.value.charge, {"e": 1})
        self.assertEqual(first_child.value.identifier, (0, 0))
        self.assertEqual(first_child.weight, 0.5)
        self.assertIsNone(first_child.value.velocity)
        self.assertIsNone(first_child.value.time_stamp)
        self.assertEqual(len(first_child.children), 0)
        second_child = cnode.children[1]
        self.assertEqual(second_child.value.position, [0.1, 0.05])
        self.assertEqual(second_child.value.charge, {"e": -1})
        self.assertEqual(second_child.value.identifier, (0, 1))
        self.assertEqual(second_child.weight, 0.5)
        self.assertIsNone(second_child.value.velocity)
        self.assertIsNone(second_child.value.time_stamp)
        self.assertEqual(len(second_child.children), 0)

        cnode = self._state_handler.extract_from_global_state((1,))
        self.assertEqual(cnode.value.position, [0.9, 0.775])
        self.assertIsNone(cnode.value.charge)
        self.assertEqual(cnode.value.identifier, (1,))
        self.assertEqual(cnode.weight, 1)
        self.assertIsNone(cnode.value.velocity)
        self.assertIsNone(cnode.value.time_stamp)
        self.assertEqual(len(cnode.children), 2)
        first_child = cnode.children[0]
        self.assertEqual(first_child.value.position, [0.9, 0.8])
        self.assertEqual(first_child.value.charge, {"e": 1})
        self.assertEqual(first_child.value.identifier, (1, 0))
        self.assertEqual(first_child.weight, 0.5)
        self.assertIsNone(first_child.value.velocity)
        self.assertIsNone(first_child.value.time_stamp)
        self.assertEqual(len(first_child.children), 0)
        second_child = cnode.children[1]
        self.assertEqual(second_child.value.position, [0.9, 0.75])
        self.assertEqual(second_child.value.charge, {"e": -1})
        self.assertEqual(second_child.value.identifier, (1, 1))
        self.assertEqual(second_child.weight, 0.5)
        self.assertIsNone(second_child.value.velocity)
        self.assertIsNone(second_child.value.time_stamp)
        self.assertEqual(len(second_child.children), 0)

    def test_extract_from_global_state_of_leaf_nodes(self):
        self._state_handler.initialize(self._root_nodes)

        # Should return root node with only one child
        parent_cnode = self._state_handler.extract_from_global_state((0, 0))
        self.assertEqual(parent_cnode.value.position, [0.05, 0.025])
        self.assertIsNone(parent_cnode.value.charge)
        self.assertEqual(parent_cnode.value.identifier, (0,))
        self.assertEqual(parent_cnode.weight, 1)
        self.assertIsNone(parent_cnode.value.velocity)
        self.assertIsNone(parent_cnode.value.time_stamp)
        self.assertEqual(len(parent_cnode.children), 1)
        cnode = parent_cnode.children[0]
        self.assertEqual(cnode.value.position, [0, 0])
        self.assertEqual(cnode.value.charge, {"e": 1})
        self.assertEqual(cnode.value.identifier, (0, 0))
        self.assertEqual(cnode.weight, 0.5)
        self.assertIsNone(cnode.value.velocity)
        self.assertIsNone(cnode.value.time_stamp)
        self.assertEqual(len(cnode.children), 0)

        parent_cnode = self._state_handler.extract_from_global_state((0, 1))
        self.assertEqual(parent_cnode.value.position, [0.05, 0.025])
        self.assertIsNone(parent_cnode.value.charge)
        self.assertEqual(parent_cnode.value.identifier, (0,))
        self.assertEqual(parent_cnode.weight, 1)
        self.assertIsNone(parent_cnode.value.velocity)
        self.assertIsNone(parent_cnode.value.time_stamp)
        self.assertEqual(len(parent_cnode.children), 1)
        cnode = parent_cnode.children[0]
        self.assertEqual(cnode.value.position, [0.1, 0.05])
        self.assertEqual(cnode.value.charge, {"e": -1})
        self.assertEqual(cnode.value.identifier, (0, 1))
        self.assertEqual(cnode.weight, 0.5)
        self.assertIsNone(cnode.value.velocity)
        self.assertIsNone(cnode.value.time_stamp)
        self.assertEqual(len(cnode.children), 0)

        parent_cnode = self._state_handler.extract_from_global_state((1, 0))
        self.assertEqual(parent_cnode.value.position, [0.9, 0.775])
        self.assertIsNone(parent_cnode.value.charge,)
        self.assertEqual(parent_cnode.value.identifier, (1,))
        self.assertEqual(parent_cnode.weight, 1)
        self.assertIsNone(parent_cnode.value.velocity)
        self.assertIsNone(parent_cnode.value.time_stamp)
        self.assertEqual(len(parent_cnode.children), 1)
        cnode = parent_cnode.children[0]
        self.assertEqual(cnode.value.position, [0.9, 0.8])
        self.assertEqual(cnode.value.charge, {"e": 1})
        self.assertEqual(cnode.value.identifier, (1, 0))
        self.assertEqual(cnode.weight, 0.5)
        self.assertIsNone(cnode.value.velocity)
        self.assertIsNone(cnode.value.time_stamp)
        self.assertEqual(len(cnode.children), 0)

        parent_cnode = self._state_handler.extract_from_global_state((1, 1))
        self.assertEqual(parent_cnode.value.position, [0.9, 0.775])
        self.assertIsNone(parent_cnode.value.charge)
        self.assertEqual(parent_cnode.value.identifier, (1,))
        self.assertEqual(parent_cnode.weight, 1)
        self.assertIsNone(parent_cnode.value.velocity)
        self.assertIsNone(parent_cnode.value.time_stamp)
        self.assertEqual(len(parent_cnode.children), 1)
        cnode = parent_cnode.children[0]
        self.assertEqual(cnode.value.position, [0.9, 0.75])
        self.assertEqual(cnode.value.charge, {"e": -1})
        self.assertEqual(cnode.value.identifier, (1, 1))
        self.assertEqual(cnode.weight, 0.5)
        self.assertIsNone(cnode.value.velocity)
        self.assertIsNone(cnode.value.time_stamp)
        self.assertEqual(len(cnode.children), 0)

    def test_extract_from_global_state_copied_position(self):
        self._state_handler.initialize(self._root_nodes)
        parent_cnode = self._state_handler.extract_from_global_state((1, 1))
        self.assertEqual(parent_cnode.value.position, [0.9, 0.775])
        self.assertIsNone(parent_cnode.value.charge)
        self.assertEqual(parent_cnode.value.identifier, (1,))
        self.assertEqual(parent_cnode.weight, 1)
        self.assertIsNone(parent_cnode.value.velocity)
        self.assertIsNone(parent_cnode.value.time_stamp)
        self.assertEqual(len(parent_cnode.children), 1)
        parent_cnode.value.position[0] -= 0.1
        parent_cnode = self._state_handler.extract_from_global_state((1, 1))
        self.assertEqual(parent_cnode.value.position, [0.9, 0.775])
        self.assertIsNone(parent_cnode.value.charge)
        self.assertEqual(parent_cnode.value.identifier, (1,))
        self.assertEqual(parent_cnode.weight, 1)
        self.assertIsNone(parent_cnode.value.velocity)
        self.assertIsNone(parent_cnode.value.time_stamp)
        self.assertEqual(len(parent_cnode.children), 1)

    def test_extract_global_state(self):
        self._state_handler.initialize(self._root_nodes)

        all_root_cnodes = self._state_handler.extract_global_state()
        self.assertEqual(len(all_root_cnodes), 2)
        unit = all_root_cnodes[0].value
        self.assertEqual(unit.position, [0.05, 0.025])
        self.assertIsNone(unit.charge)
        self.assertEqual(unit.identifier, (0,))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[0].children), 2)

        unit = all_root_cnodes[0].children[0].value
        self.assertEqual(unit.position, [0, 0])
        self.assertEqual(unit.charge, {"e": 1})
        self.assertEqual(unit.identifier, (0, 0))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[0].children[0].children), 0)

        unit = all_root_cnodes[0].children[1].value
        self.assertEqual(unit.position, [0.1, 0.05])
        self.assertEqual(unit.charge, {"e": -1})
        self.assertEqual(unit.identifier, (0, 1))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[0].children[1].children), 0)

        unit = all_root_cnodes[1].value
        self.assertEqual(unit.position, [0.9, 0.775])
        self.assertIsNone(unit.charge)
        self.assertEqual(unit.identifier, (1,))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[1].children), 2)

        unit = all_root_cnodes[1].children[0].value
        self.assertEqual(unit.position, [0.9, 0.8])
        self.assertEqual(unit.charge, {"e": 1})
        self.assertEqual(unit.identifier, (1, 0))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[1].children[0].children), 0)

        unit = all_root_cnodes[1].children[1].value
        self.assertEqual(unit.position, [0.9, 0.75])
        self.assertEqual(unit.charge, {"e": -1})
        self.assertEqual(unit.identifier, (1, 1))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[1].children[1].children), 0)

    def test_extract_global_state_not_copied_position(self):
        self._state_handler.initialize(self._root_nodes)
        all_root_cnodes = self._state_handler.extract_global_state()
        position = all_root_cnodes[0].value.position
        position[0] += 0.1
        new_all_root_cnodes = self._state_handler.extract_global_state()
        self.assertEqual(new_all_root_cnodes[0].value.position, position)

    def test_insert_into_global_state_branch_of_leaf_unit(self):
        self._state_handler.initialize(self._root_nodes)
        branch = Node(Unit(identifier=(0,), position=[0.2, 0.3], charge=None,
                           velocity=[0.5, 0], time_stamp=Time(0.0, 0.3)), weight=1)
        branch.add_child(Node(Unit(identifier=(0, 1), position=[0.4, 0.6],
                                   charge={"e": -1}, velocity=[1, 0], time_stamp=Time(0.0, 0.3)), weight=0.5))
        self._state_handler.insert_into_global_state([branch])
        all_root_cnodes = self._state_handler.extract_global_state()

        self.assertEqual(len(all_root_cnodes), 2)
        unit = all_root_cnodes[0].value
        self.assertEqual(unit.position, [0.2, 0.3])
        self.assertIsNone(unit.charge)
        self.assertEqual(unit.identifier, (0,))
        self.assertEqual(unit.velocity, [0.5, 0])
        self.assertEqual(unit.time_stamp, Time(0.0, 0.3))
        self.assertEqual(len(all_root_cnodes[0].children), 2)

        unit = all_root_cnodes[0].children[0].value
        self.assertEqual(unit.position, [0, 0])
        self.assertEqual(unit.charge, {"e": 1})
        self.assertEqual(unit.identifier, (0, 0))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[0].children[0].children), 0)

        unit = all_root_cnodes[0].children[1].value
        self.assertEqual(unit.position, [0.4, 0.6])
        self.assertEqual(unit.charge, {"e": -1})
        self.assertEqual(unit.identifier, (0, 1))
        self.assertEqual(unit.velocity, [1, 0])
        self.assertEqual(unit.time_stamp, Time(0.0, 0.3))
        self.assertEqual(len(all_root_cnodes[0].children[1].children), 0)

        unit = all_root_cnodes[1].value
        self.assertEqual(unit.position, [0.9, 0.775])
        self.assertIsNone(unit.charge)
        self.assertEqual(unit.identifier, (1,))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[1].children), 2)

        unit = all_root_cnodes[1].children[0].value
        self.assertEqual(unit.position, [0.9, 0.8])
        self.assertEqual(unit.charge, {"e": 1})
        self.assertEqual(unit.identifier, (1, 0))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[1].children[0].children), 0)

        unit = all_root_cnodes[1].children[1].value
        self.assertEqual(unit.position, [0.9, 0.75])
        self.assertEqual(unit.charge, {"e": -1})
        self.assertEqual(unit.identifier, (1, 1))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[1].children[1].children), 0)

    def test_insert_into_global_state_root_branch(self):
        self._state_handler.initialize(self._root_nodes)
        branch = Node(Unit(identifier=(1,), position=[0.5, 0.6], charge=None,
                           velocity=[1, 1], time_stamp=Time(0.0, 0.2)), weight=1)
        # There are no consistency checks within the state handler
        branch.add_child(Node(Unit(identifier=(1, 0), position=[0.4, 0.6], charge={"e": 1},
                                   velocity=[-0.3, 2.1], time_stamp=Time(0.0, 0.53)), weight=0.5))
        branch.add_child(Node(Unit(identifier=(1, 1), position=[0.1, 0.2], charge={"e": -1},
                                   velocity=[0.7, 0.7], time_stamp=Time(0.0, 0.1)), weight=0.5))
        self._state_handler.insert_into_global_state([branch])

        all_root_cnodes = self._state_handler.extract_global_state()
        self.assertEqual(len(all_root_cnodes), 2)
        unit = all_root_cnodes[0].value
        self.assertEqual(unit.position, [0.05, 0.025])
        self.assertIsNone(unit.charge)
        self.assertEqual(unit.identifier, (0,))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[0].children), 2)

        unit = all_root_cnodes[0].children[0].value
        self.assertEqual(unit.position, [0, 0])
        self.assertEqual(unit.charge, {"e": 1})
        self.assertEqual(unit.identifier, (0, 0))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[0].children[0].children), 0)

        unit = all_root_cnodes[0].children[1].value
        self.assertEqual(unit.position, [0.1, 0.05])
        self.assertEqual(unit.charge, {"e": -1})
        self.assertEqual(unit.identifier, (0, 1))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[0].children[1].children), 0)

        unit = all_root_cnodes[1].value
        self.assertEqual(unit.position, [0.5, 0.6])
        self.assertIsNone(unit.charge)
        self.assertEqual(unit.identifier, (1,))
        self.assertEqual(unit.velocity, [1, 1])
        self.assertEqual(unit.time_stamp, Time(0.0, 0.2))
        self.assertEqual(len(all_root_cnodes[1].children), 2)

        unit = all_root_cnodes[1].children[0].value
        self.assertEqual(unit.position, [0.4, 0.6])
        self.assertEqual(unit.charge, {"e": 1})
        self.assertEqual(unit.identifier, (1, 0))
        self.assertEqual(unit.velocity, [-0.3, 2.1])
        self.assertEqual(unit.time_stamp, Time(0.0, 0.53))
        self.assertEqual(len(all_root_cnodes[1].children[0].children), 0)

        unit = all_root_cnodes[1].children[1].value
        self.assertEqual(unit.position, [0.1, 0.2])
        self.assertEqual(unit.charge, {"e": -1})
        self.assertEqual(unit.identifier, (1, 1))
        self.assertEqual(unit.velocity, [0.7, 0.7])
        self.assertEqual(unit.time_stamp, Time(0.0, 0.1))
        self.assertEqual(len(all_root_cnodes[1].children[1].children), 0)

    def test_insert_into_global_state_two_branches(self):
        self._state_handler.initialize(self._root_nodes)
        branch_one = Node(Unit(identifier=(0,), position=[0.2, 0.3], charge=None,
                               velocity=[0.5, 0], time_stamp=Time(0.0, 0.3)), weight=1)
        branch_one.add_child(Node(Unit(identifier=(0, 1), position=[0.4, 0.6],
                                       charge={"e": -1}, velocity=[1, 0], time_stamp=Time(0.0, 0.3)), weight=0.5))
        branch_two = Node(Unit(identifier=(1,), position=[0.5, 0.6], charge=None,
                               velocity=[1, 1], time_stamp=Time(0.0, 0.2)), weight=1)
        # There are no consistency checks within the state handler
        branch_two.add_child(Node(Unit(identifier=(1, 0), position=[0.4, 0.6], charge={"e": 1},
                                       velocity=[-0.3, 2.1], time_stamp=Time(0.0, 0.53)), weight=0.5))
        branch_two.add_child(Node(Unit(identifier=(1, 1), position=[0.1, 0.2], charge={"e": -1},
                                       velocity=[0.7, 0.7], time_stamp=Time(0.0, 0.1)), weight=0.5))
        self._state_handler.insert_into_global_state([branch_one, branch_two])
        all_root_cnodes = self._state_handler.extract_global_state()

        self.assertEqual(len(all_root_cnodes), 2)
        unit = all_root_cnodes[0].value
        self.assertEqual(unit.position, [0.2, 0.3])
        self.assertIsNone(unit.charge)
        self.assertEqual(unit.identifier, (0,))
        self.assertEqual(unit.velocity, [0.5, 0])
        self.assertEqual(unit.time_stamp, Time(0.0, 0.3))
        self.assertEqual(len(all_root_cnodes[0].children), 2)

        unit = all_root_cnodes[0].children[0].value
        self.assertEqual(unit.position, [0, 0])
        self.assertEqual(unit.charge, {"e": 1})
        self.assertEqual(unit.identifier, (0, 0))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[0].children[0].children), 0)

        unit = all_root_cnodes[0].children[1].value
        self.assertEqual(unit.position, [0.4, 0.6])
        self.assertEqual(unit.charge, {"e": -1})
        self.assertEqual(unit.identifier, (0, 1))
        self.assertEqual(unit.velocity, [1, 0])
        self.assertEqual(unit.time_stamp, Time(0.0, 0.3))
        self.assertEqual(len(all_root_cnodes[0].children[1].children), 0)

        unit = all_root_cnodes[1].value
        self.assertEqual(unit.position, [0.5, 0.6])
        self.assertIsNone(unit.charge)
        self.assertEqual(unit.identifier, (1,))
        self.assertEqual(unit.velocity, [1, 1])
        self.assertEqual(unit.time_stamp, Time(0.0, 0.2))
        self.assertEqual(len(all_root_cnodes[1].children), 2)

        unit = all_root_cnodes[1].children[0].value
        self.assertEqual(unit.position, [0.4, 0.6])
        self.assertEqual(unit.charge, {"e": 1})
        self.assertEqual(unit.identifier, (1, 0))
        self.assertEqual(unit.velocity, [-0.3, 2.1])
        self.assertEqual(unit.time_stamp, Time(0.0, 0.53))
        self.assertEqual(len(all_root_cnodes[1].children[0].children), 0)

        unit = all_root_cnodes[1].children[1].value
        self.assertEqual(unit.position, [0.1, 0.2])
        self.assertEqual(unit.charge, {"e": -1})
        self.assertEqual(unit.identifier, (1, 1))
        self.assertEqual(unit.velocity, [0.7, 0.7])
        self.assertEqual(unit.time_stamp, Time(0.0, 0.1))
        self.assertEqual(len(all_root_cnodes[1].children[1].children), 0)

    def test_insert_into_global_state_velocity_and_time_stamp_to_none(self):
        self._state_handler.initialize(self._root_nodes)
        branch = Node(Unit(identifier=(0,), position=[0.2, 0.3], charge=None,
                           velocity=[0.5, 0], time_stamp=Time(0.0, 0.3)), weight=1)
        branch.add_child(Node(Unit(identifier=(0, 1), position=[0.4, 0.6], charge={"e": -1},
                                   velocity=[1, 0], time_stamp=Time(0.0, 0.3)), weight=0.5))
        self._state_handler.insert_into_global_state([branch])
        branch = Node(Unit(identifier=(0,), position=[0.1, 0.2], charge=None,
                           velocity=None, time_stamp=None), weight=1)
        branch.add_child(Node(Unit(identifier=(0, 1), position=[0.3, 0.4],
                                   charge={"e": -1}, velocity=None, time_stamp=None), weight=0.5))
        branch.add_child(Node(Unit(identifier=(0, 0), position=[0.5, 0.6],
                                   charge={"e": 1}, velocity=None, time_stamp=None), weight=0.5))
        self._state_handler.insert_into_global_state([branch])

        all_root_cnodes = self._state_handler.extract_global_state()

        self.assertEqual(len(all_root_cnodes), 2)
        unit = all_root_cnodes[0].value
        self.assertEqual(unit.position, [0.1, 0.2])
        self.assertIsNone(unit.charge)
        self.assertEqual(unit.identifier, (0,))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[0].children), 2)

        unit = all_root_cnodes[0].children[0].value
        self.assertEqual(unit.position, [0.5, 0.6])
        self.assertEqual(unit.charge, {"e": 1})
        self.assertEqual(unit.identifier, (0, 0))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[0].children[0].children), 0)

        unit = all_root_cnodes[0].children[1].value
        self.assertEqual(unit.position, [0.3, 0.4])
        self.assertEqual(unit.charge, {"e": -1})
        self.assertEqual(unit.identifier, (0, 1))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[0].children[1].children), 0)

        unit = all_root_cnodes[1].value
        self.assertEqual(unit.position, [0.9, 0.775])
        self.assertIsNone(unit.charge)
        self.assertEqual(unit.identifier, (1,))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[1].children), 2)

        unit = all_root_cnodes[1].children[0].value
        self.assertEqual(unit.position, [0.9, 0.8])
        self.assertEqual(unit.charge, {"e": 1})
        self.assertEqual(unit.identifier, (1, 0))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[1].children[0].children), 0)

        unit = all_root_cnodes[1].children[1].value
        self.assertEqual(unit.position, [0.9, 0.75])
        self.assertEqual(unit.charge, {"e": -1})
        self.assertEqual(unit.identifier, (1, 1))
        self.assertIsNone(unit.velocity)
        self.assertIsNone(unit.time_stamp)
        self.assertEqual(len(all_root_cnodes[1].children[1].children), 0)

    def test_extract_active_global_state_leaf_unit_active(self):
        self._state_handler.initialize(self._root_nodes)
        branch = Node(Unit(identifier=(0,), position=[0.2, 0.3], charge=None,
                           velocity=[0.2, 0], time_stamp=Time(0.0, 0.1)), weight=1)
        branch.add_child(Node(Unit(identifier=(0, 0), position=[0.4, 0.6], charge={"e": -1},
                                   velocity=[0.6, -0.1], time_stamp=Time(0.0, 0.2)), weight=0.5))
        self._state_handler.insert_into_global_state([branch])

        branches = self._state_handler.extract_active_global_state()
        self.assertEqual(len(branches), 1)
        info = branches[0].value
        self.assertEqual(info.position, [0.2, 0.3])
        self.assertEqual(info.charge, None)
        self.assertEqual(info.identifier, (0,))
        self.assertEqual(info.velocity, [0.2, 0])
        self.assertEqual(info.time_stamp, Time(0.0, 0.1))
        self.assertEqual(len(branches[0].children), 1)
        info = branches[0].children[0].value
        self.assertEqual(info.position, [0.4, 0.6])
        self.assertEqual(info.charge, {"e": 1})
        self.assertEqual(info.identifier, (0, 0))
        self.assertEqual(info.velocity, [0.6, -0.1])
        self.assertEqual(info.time_stamp, Time(0.0, 0.2))
        self.assertEqual(len(branches[0].children[0].children), 0)

    def test_extract_active_global_state_root_node_active(self):
        self._state_handler.initialize(self._root_nodes)
        branch = Node(Unit(identifier=(1,), position=[0.2, 0.3], charge=None,
                           velocity=[0.1, 0.2], time_stamp=Time(0.0, 0.0)), weight=1)
        branch.add_child(Node(Unit(identifier=(1, 0), position=[0.4, 0.6], charge={"e": -1},
                                   velocity=[0.6, -0.1], time_stamp=Time(-1.0, 0.8)), weight=0.5))
        branch.add_child(Node(Unit(identifier=(1, 1), position=[0.4, 0.6], charge={"e": -1},
                                   velocity=[0.6, -0.1], time_stamp=Time(-1.0, 0.8)), weight=0.5))
        self._state_handler.insert_into_global_state([branch])

        branches = self._state_handler.extract_active_global_state()
        self.assertEqual(len(branches), 1)
        info = branches[0].value
        self.assertEqual(info.position, [0.2, 0.3])
        self.assertEqual(info.charge, None)
        self.assertEqual(info.identifier, (1,))
        self.assertEqual(info.velocity, [0.1, 0.2])
        self.assertEqual(info.time_stamp, Time(0.0, 0.0))
        self.assertEqual(len(branches[0].children), 2)
        info = branches[0].children[0].value
        self.assertEqual(info.position, [0.4, 0.6])
        self.assertEqual(info.charge, {"e": 1})
        self.assertEqual(info.identifier, (1, 0))
        self.assertEqual(info.velocity, [0.6, -0.1])
        self.assertEqual(info.time_stamp, Time(-1.0, 0.8))
        self.assertEqual(len(branches[0].children[0].children), 0)
        info = branches[0].children[1].value
        self.assertEqual(info.position, [0.4, 0.6])
        self.assertEqual(info.charge, {"e": -1})
        self.assertEqual(info.identifier, (1, 1))
        self.assertEqual(info.velocity, [0.6, -0.1])
        self.assertEqual(info.time_stamp, Time(-1.0, 0.8))
        self.assertEqual(len(branches[0].children[1].children), 0)

    def test_extract_global_state_not_copied_velocity(self):
        self._state_handler.initialize(self._root_nodes)
        branch = Node(Unit(identifier=(0,), position=[0.2, 0.3], charge=None,
                           velocity=[0.5, 0], time_stamp=Time(0.0, 0.3)), weight=1)
        branch.add_child(Node(Unit(identifier=(0, 1), position=[0.4, 0.6],
                                   charge={"e": -1}, velocity=[1, 0], time_stamp=Time(0.0, 0.3)), weight=0.5))
        self._state_handler.insert_into_global_state([branch])
        all_root_cnodes = self._state_handler.extract_global_state()
        velocity = all_root_cnodes[0].value.velocity
        velocity[0] = -1
        all_root_cnodes = self._state_handler.extract_global_state()
        self.assertEqual(all_root_cnodes[0].value.velocity, velocity)

    def test_extract_global_state_not_copied_time_stamp(self):
        self._state_handler.initialize(self._root_nodes)
        branch = Node(Unit(identifier=(0,), position=[0.2, 0.3], charge=None,
                           velocity=[0.5, 0], time_stamp=Time(0.0, 0.3)), weight=1)
        branch.add_child(Node(Unit(identifier=(0, 1), position=[0.4, 0.6],
                                   charge={"e": -1}, velocity=[1, 0], time_stamp=Time(0.0, 0.3)), weight=0.5))
        self._state_handler.insert_into_global_state([branch])
        all_root_cnodes = self._state_handler.extract_global_state()
        time_stamp = all_root_cnodes[0].value.time_stamp
        time_stamp.update(Time(1.0, 0.2))
        all_root_cnodes = self._state_handler.extract_global_state()
        self.assertEqual(all_root_cnodes[0].value.time_stamp, time_stamp)

    def test_extract_from_global_state_copied_velocity(self):
        self._state_handler.initialize(self._root_nodes)
        branch = Node(Unit(identifier=(0,), position=[0.2, 0.3], charge=None,
                           velocity=[0.5, 0], time_stamp=Time(0.0, 0.3)), weight=1)
        branch.add_child(Node(Unit(identifier=(0, 1), position=[0.4, 0.6],
                                   charge={"e": -1}, velocity=[1, 0], time_stamp=Time(0.0, 0.3)), weight=0.5))
        self._state_handler.insert_into_global_state([branch])
        root_cnode = self._state_handler.extract_from_global_state((0, 1))
        velocity = root_cnode.value.velocity
        velocity[0] = -1
        all_root_cnodes = self._state_handler.extract_global_state()
        self.assertEqual(all_root_cnodes[0].value.velocity, [0.5, 0])

    def test_extract_from_global_state_copied_time_stamp(self):
        self._state_handler.initialize(self._root_nodes)
        branch = Node(Unit(identifier=(0,), position=[0.2, 0.3], charge=None,
                           velocity=[0.5, 0], time_stamp=Time(0.0, 0.3)), weight=1)
        branch.add_child(Node(Unit(identifier=(0, 1), position=[0.4, 0.6],
                                   charge={"e": -1}, velocity=[1, 0], time_stamp=Time(0.0, 0.3)), weight=0.5))
        self._state_handler.insert_into_global_state([branch])
        root_cnode = self._state_handler.extract_from_global_state((0, 1))
        time_stamp = root_cnode.value.time_stamp
        time_stamp.update(Time(1.0, 0.2))
        all_root_cnodes = self._state_handler.extract_global_state()
        self.assertEqual(all_root_cnodes[0].value.time_stamp, Time(0.0, 0.3))

    def test_insert_into_global_state_velocity_not_none_but_time_stamp_none_raises_error(self):
        self._state_handler.initialize(self._root_nodes)
        branch = Node(Unit(identifier=(0,), position=[0.2, 0.3], charge=None,
                           velocity=[0.5, 0], time_stamp=Time(0.0, 0.3)), weight=1)
        branch.add_child(Node(Unit(identifier=(0, 1), position=[0.4, 0.6], charge={"e": -1},
                                   velocity=[1, 0], time_stamp=Time(0.0, 0.3)), weight=0.5))
        self._state_handler.insert_into_global_state([branch])
        branch = Node(Unit(identifier=(0,), position=[0.1, 0.2], charge=None,
                           velocity=[0, 1], time_stamp=None), weight=1)
        branch.add_child(Node(Unit(identifier=(0, 1), position=[0.3, 0.4],
                                   charge={"e": -1}, velocity=[0, 1], time_stamp=None), weight=0.5))
        branch.add_child(Node(Unit(identifier=(0, 0), position=[0.5, 0.6],
                                   charge={"e": 1}, velocity=[0, 1], time_stamp=None), weight=0.5))
        with self.assertRaises(AssertionError):
            self._state_handler.insert_into_global_state([branch])

    def test_insert_into_global_state_velocity_none_but_time_stamp_not_none_raises_error(self):
        self._state_handler.initialize(self._root_nodes)
        branch = Node(Unit(identifier=(0,), position=[0.2, 0.3], charge=None,
                           velocity=[0.5, 0], time_stamp=Time(0.0, 0.3)), weight=1)
        branch.add_child(Node(Unit(identifier=(0, 1), position=[0.4, 0.6], charge={"e": -1},
                                   velocity=[1, 0], time_stamp=Time(0.0, 0.3)), weight=0.5))
        self._state_handler.insert_into_global_state([branch])
        branch = Node(Unit(identifier=(0,), position=[0.1, 0.2], charge=None,
                           velocity=None, time_stamp=Time(0.0, 0.5)), weight=1)
        branch.add_child(Node(Unit(identifier=(0, 1), position=[0.3, 0.4],
                                   charge={"e": -1}, velocity=None, time_stamp=Time(0.0, 0.5)), weight=0.5))
        branch.add_child(Node(Unit(identifier=(0, 0), position=[0.5, 0.6],
                                   charge={"e": 1}, velocity=None, time_stamp=Time(0.0, 0.5)), weight=0.5))
        with self.assertRaises(AssertionError):
            self._state_handler.insert_into_global_state([branch])


if __name__ == '__main__':
    main()
