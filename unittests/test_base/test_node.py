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
import base.node as node


class TestNode(TestCase):
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


if __name__ == '__main__':
    main()
