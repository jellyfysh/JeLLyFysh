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
"""Module for the Node class."""
from typing import Any, Iterable


class Node(object):
    """
    Class for a general node within a tree structure.

    Each node within the tree can have one parent node and several child nodes. If the parent node is not None, this
    node appears in the parent node's children sequence. Similarly each child node has the parent set to this node.
    The weight of a node is defined as one over the number of children in the parent node. However, in the application
    branches (meaning only a part of the tree) are constructed where not all children appear. Then the real weight of
    the node within the original tree can be set on initialization.
    Also each node can have a value associated with it.
    A tree structure with nodes is currently used in the TreeStateHandler. There the values are particles (see
    base.particle) or units (see base.unit).

    Attributes
    ----------
    value: Any
        The value associated with the Node.
    parent: Node or None
        The parent node.
    children: Sequence[Node]
        The sequence of child nodes.
    """

    def __init__(self, value: Any = None, weight: float = None):
        """
        The constructor of the Node class.

        Normally the weight returned by the weight property calculates one over the number of children of the parent
        node. However, if a branch gets constructed where some children are missing, this formula yields a weight which
        does not equal the weight of the node in the original tree. For this case, the weight should be set on
        initialization which replaces the usage of the formula for the weight.

        Parameters
        ----------
        value : Any, optional
            The value to be associated with the node.
        weight : float or None, optional
            The weight of the node. If none, the mentioned formula is used.
        """
        self.value = value  #: a Particle or Unit object to store real information
        self.parent = None  #: parent of this node
        self.children = []  #: list of children
        self._weight = weight
        if self._weight is not None:
            self._get_weight = self._get_weight_set
        else:
            self._get_weight = self._get_weight_not_set

    def add_child(self, node: "Node") -> None:
        """
        Add a child node to this node.

        Append the child to the children sequence and set the parent of the child node to be this node.

        Parameters
        ----------
        node : base.node.Node
            The child node.
        """
        self.children.append(node)
        node.parent = self

    @property
    def weight(self) -> float:
        """
        Return the weight of this node.

        If the weight has been set on initialization, this weight will be returned. Otherwise one over the number of
        children of the parent node will be returned. If there is no parent node, the weight is one.

        Returns
        -------
        float
            The weight of the node within the parents children.
        """
        return self._get_weight()

    def _get_weight_set(self) -> float:
        return self._weight

    def _get_weight_not_set(self) -> float:
        self._weight = 1 / len(self.parent.children) if self.parent is not None else 1
        self._get_weight = self._get_weight_set
        return self._weight


def yield_leaf_nodes(node: Node) -> Iterable[Node]:
    """
    Generate all leaf nodes below the given node.

    Parameters
    ----------
    node : Node
        The node.

    Yields
    ------
    Node
        The leaf nodes.
    """
    if not node.children:
        yield node
    else:
        for child in node.children:
            yield from yield_leaf_nodes(child)


def yield_nodes_on_level_below(node: Node, level: int) -> Iterable[Node]:
    """
    Generate all nodes which are the given level below the node.

    Parameters
    ----------
    node : Node
        The node.
    level : int
        The level.

    Yields
    -------
    Node
        The nodes which are the given level below the node.

    Raises
    ------
    AssertionError
        If level is negative.
    """
    assert level >= 0
    if level == 0:
        yield node
    else:
        for child in node.children:
            yield from yield_nodes_on_level_below(child, level - 1)
