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
import jellyfysh.base.node as node
import jellyfysh.base.unit as unit
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


# Inherit explicitly from TestCase class for Test functionality in PyCharm.
class TestNode(ExpandedTestCase, TestCase):
    def test_add_child(self):
        # Build up tree and check if children and parents are set correctly
        root_node = node.Node("RootNode")
        self.assertIsNone(root_node.parent)
        self.assertEqual(root_node.children, [])
        self.assertEqual(root_node.value, "RootNode")

        node_one = node.Node("One")
        self.assertEqual(node_one.value, "One")
        root_node.add_child(node_one)
        self.assertIsNone(root_node.parent)
        self.assertEqual(root_node.children, [node_one])
        self.assertIs(node_one.parent, root_node)
        self.assertEqual(node_one.children, [])

        node_two = node.Node("Two")
        self.assertEqual(node_two.value, "Two")
        node_one.add_child(node_two)
        self.assertIsNone(root_node.parent)
        self.assertEqual(root_node.children, [node_one])
        self.assertIs(node_one.parent, root_node)
        self.assertEqual(node_one.children, [node_two])
        self.assertIs(node_two.parent, node_one)
        self.assertEqual(node_two.children, [])

        node_three = node.Node("Three")
        self.assertEqual(node_three.value, "Three")
        root_node.add_child(node_three)
        self.assertIsNone(root_node.parent)
        self.assertEqual(root_node.children, [node_one, node_three])
        self.assertIs(node_one.parent, root_node)
        self.assertEqual(node_one.children, [node_two])
        self.assertIs(node_two.parent, node_one)
        self.assertEqual(node_two.children, [])
        self.assertIs(node_three.parent, root_node)
        self.assertEqual(node_three.children, [])

    def test_weight_not_initial_set(self):
        root_node = node.Node("RootNode")
        node_one = node.Node("One")
        node_two = node.Node("Two")
        node_three = node.Node("Three")
        node_four = node.Node("Four")
        node_five = node.Node("Five")

        root_node.children = [node_one, node_two, node_three]
        node_one.parent = root_node
        node_two.parent = root_node
        node_three.parent = root_node
        node_one.children = [node_four, node_five]
        node_four.parent = node_one
        node_five.parent = node_one

        self.assertEqual(root_node.weight, 1)
        self.assertEqual(node_one.weight, 1 / 3)
        self.assertEqual(node_two.weight, 1 / 3)
        self.assertEqual(node_three.weight, 1 / 3)
        self.assertEqual(node_four.weight, 0.5)
        self.assertEqual(node_five.weight, 0.5)

    def test_weight_initial_set(self):
        root_node = node.Node("RootNode", weight=0.01)
        node_one = node.Node("One", weight=0.1)
        node_two = node.Node("Two", weight=0.2)
        node_three = node.Node("Three", weight=0.3)
        node_four = node.Node("Four", weight=0.4)
        node_five = node.Node("Five", weight=0.5)

        root_node.children = [node_one, node_two, node_three]
        node_one.parent = root_node
        node_two.parent = root_node
        node_three.parent = root_node
        node_one.children = [node_four, node_five]
        node_four.parent = node_one
        node_five.parent = node_one

        self.assertEqual(root_node.weight, 0.01)
        self.assertEqual(node_one.weight, 0.1)
        self.assertEqual(node_two.weight, 0.2)
        self.assertEqual(node_three.weight, 0.3)
        self.assertEqual(node_four.weight, 0.4)
        self.assertEqual(node_five.weight, 0.5)

    def test_yield_leaf_nodes(self):
        root_node = node.Node("RootNode", weight=0.01)
        node_one = node.Node("One", weight=0.1)
        node_two = node.Node("Two", weight=0.2)
        node_three = node.Node("Three", weight=0.3)
        node_four = node.Node("Four", weight=0.4)
        node_five = node.Node("Five", weight=0.5)

        root_node.children = [node_one, node_two, node_three]
        node_one.parent = root_node
        node_two.parent = root_node
        node_three.parent = root_node
        node_one.children = [node_four, node_five]
        node_four.parent = node_one
        node_five.parent = node_one

        leaf_nodes_of_root_node = list(node.yield_leaf_nodes(root_node))
        self.assertEqual(len(leaf_nodes_of_root_node), 4)
        self.assertEqual(leaf_nodes_of_root_node[0].value, "Four")
        self.assertEqual(leaf_nodes_of_root_node[1].value, "Five")
        self.assertEqual(leaf_nodes_of_root_node[2].value, "Two")
        self.assertEqual(leaf_nodes_of_root_node[3].value, "Three")

    def test_yield_nodes_on_level_below(self):
        root_node = node.Node("RootNode", weight=0.01)
        node_one = node.Node("One", weight=0.1)
        node_two = node.Node("Two", weight=0.2)
        node_three = node.Node("Three", weight=0.3)
        node_four = node.Node("Four", weight=0.4)
        node_five = node.Node("Five", weight=0.5)

        root_node.children = [node_one, node_two, node_three]
        node_one.parent = root_node
        node_two.parent = root_node
        node_three.parent = root_node
        node_one.children = [node_four, node_five]
        node_four.parent = node_one
        node_five.parent = node_one

        level_zero_below_nodes = list(node.yield_nodes_on_level_below(root_node, 0))
        self.assertEqual(len(level_zero_below_nodes), 1)
        self.assertEqual(level_zero_below_nodes[0].value, "RootNode")

        level_one_below_nodes = list(node.yield_nodes_on_level_below(root_node, 1))
        self.assertEqual(len(level_one_below_nodes), 3)
        self.assertEqual(level_one_below_nodes[0].value, "One")
        self.assertEqual(level_one_below_nodes[1].value, "Two")
        self.assertEqual(level_one_below_nodes[2].value, "Three")

        level_two_below_nodes = list(node.yield_nodes_on_level_below(root_node, 2))
        self.assertEqual(len(level_two_below_nodes), 2)
        self.assertEqual(level_two_below_nodes[0].value, "Four")
        self.assertEqual(level_two_below_nodes[1].value, "Five")

        level_three_below_nodes = list(node.yield_nodes_on_level_below(root_node, 3))
        self.assertEqual(len(level_three_below_nodes), 0)

    def test_yield_nodes_on_level_below_raises_error_if_level_negative(self):
        root_node = node.Node("RootNode", weight=0.01)
        node_one = node.Node("One", weight=0.1)
        node_two = node.Node("Two", weight=0.2)
        node_three = node.Node("Three", weight=0.3)
        node_four = node.Node("Four", weight=0.4)
        node_five = node.Node("Five", weight=0.5)

        root_node.children = [node_one, node_two, node_three]
        node_one.parent = root_node
        node_two.parent = root_node
        node_three.parent = root_node
        node_one.children = [node_four, node_five]
        node_four.parent = node_one
        node_five.parent = node_one

        with self.assertRaises(AssertionError):
            list(node.yield_nodes_on_level_below(root_node, -1))

    def test_yield_closest_leaf_unit_positions(self):
        # Test with root node without children
        hypercubic_setting.HypercubicSetting(system_length=10.0)
        root_unit = unit.Unit(identifier=(0,), position=[2.1, 8.3, 0.4])
        root_cnode = node.Node(root_unit, weight=1)
        generated_values = list(node.yield_closest_leaf_unit_positions(root_cnode))
        self.assertEqual(len(generated_values), 1)
        self.assertTrue(all(len(generated_value) == 2 for generated_value in generated_values))
        self.assertIs(generated_values[0][0], root_unit)
        self.assertEqual(generated_values[0][1], [2.1, 8.3, 0.4])
        # Value of the node should be unchanged
        self.assertEqual(root_cnode.value.identifier, (0,))
        self.assertEqual(root_cnode.value.position, [2.1, 8.3, 0.4])
        self.assertEqual(len(root_cnode.children), 0)

        # Test with root node with three children where the children positions are already correct
        child_unit_one = unit.Unit(identifier=(0, 0), position=[0.1, 8.9, 0.3])
        child_unit_two = unit.Unit(identifier=(0, 1), position=[2.3, 8.2, 0.6])
        child_unit_three = unit.Unit(identifier=(0, 2), position=[3.9, 7.8, 0.3])
        root_cnode.add_child(node.Node(child_unit_one, weight=1.0 / 3.0))
        root_cnode.add_child(node.Node(child_unit_two, weight=1.0 / 3.0))
        root_cnode.add_child(node.Node(child_unit_three, weight=1.0 / 3.0))
        generated_values = list(node.yield_closest_leaf_unit_positions(root_cnode))
        self.assertEqual(len(generated_values), 3)
        self.assertTrue(all(len(generated_value) == 2 for generated_value in generated_values))
        self.assertIs(generated_values[0][0], child_unit_one)
        self.assertAlmostEqualSequence(generated_values[0][1], [0.1, 8.9, 0.3], places=12)
        self.assertIs(generated_values[1][0], child_unit_two)
        self.assertAlmostEqualSequence(generated_values[1][1], [2.3, 8.2, 0.6], places=12)
        self.assertIs(generated_values[2][0], child_unit_three)
        self.assertAlmostEqualSequence(generated_values[2][1], [3.9, 7.8, 0.3], places=12)
        # Value of the node should be unchanged
        self.assertEqual(root_cnode.value.identifier, (0,))
        self.assertEqual(root_cnode.value.position, [2.1, 8.3, 0.4])
        self.assertEqual(len(root_cnode.children), 3)
        self.assertEqual(root_cnode.children[0].value.identifier, (0, 0))
        self.assertAlmostEqualSequence(root_cnode.children[0].value.position, [0.1, 8.9, 0.3], places=12)
        self.assertEqual(len(root_cnode.children[0].children), 0)
        self.assertEqual(root_cnode.children[1].value.identifier, (0, 1))
        self.assertAlmostEqualSequence(root_cnode.children[1].value.position, [2.3, 8.2, 0.6], places=12)
        self.assertEqual(len(root_cnode.children[1].children), 0)
        self.assertEqual(root_cnode.children[2].value.identifier, (0, 2))
        self.assertAlmostEqualSequence(root_cnode.children[2].value.position, [3.9, 7.8, 0.3], places=12)
        self.assertEqual(len(root_cnode.children[2].children), 0)

        # Test with root node with three children where the children positions must be corrected
        root_cnode.children[0].value.position = [2.4, 8.9, 2.2]
        root_cnode.children[1].value.position = [9.5, 0.1, 9.1]
        root_cnode.children[2].value.position = [4.4, 5.9, 9.9]
        generated_values = list(node.yield_closest_leaf_unit_positions(root_cnode))
        self.assertEqual(len(generated_values), 3)
        self.assertTrue(all(len(generated_value) == 2 for generated_value in generated_values))
        self.assertIs(generated_values[0][0], child_unit_one)
        self.assertAlmostEqualSequence(generated_values[0][1], [2.4, 8.9, 2.2], places=12)
        self.assertIs(generated_values[1][0], child_unit_two)
        self.assertAlmostEqualSequence(generated_values[1][1], [-0.5, 10.1, -0.9], places=12)
        self.assertIs(generated_values[2][0], child_unit_three)
        self.assertAlmostEqualSequence(generated_values[2][1], [4.4, 5.9, -0.1], places=12)
        # Value of the node should be unchanged
        self.assertEqual(root_cnode.value.identifier, (0,))
        self.assertEqual(root_cnode.value.position, [2.1, 8.3, 0.4])
        self.assertEqual(len(root_cnode.children), 3)
        self.assertEqual(root_cnode.children[0].value.identifier, (0, 0))
        self.assertAlmostEqualSequence(root_cnode.children[0].value.position, [2.4, 8.9, 2.2], places=12)
        self.assertEqual(len(root_cnode.children[0].children), 0)
        self.assertEqual(root_cnode.children[1].value.identifier, (0, 1))
        self.assertAlmostEqualSequence(root_cnode.children[1].value.position, [9.5, 0.1, 9.1], places=12)
        self.assertEqual(len(root_cnode.children[1].children), 0)
        self.assertEqual(root_cnode.children[2].value.identifier, (0, 2))
        self.assertAlmostEqualSequence(root_cnode.children[2].value.position, [4.4, 5.9, 9.9], places=12)
        self.assertEqual(len(root_cnode.children[2].children), 0)

        setting.reset()


if __name__ == '__main__':
    main()
