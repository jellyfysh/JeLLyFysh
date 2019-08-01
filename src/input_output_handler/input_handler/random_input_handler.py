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
"""Module for the RandomInputHandler class."""
import logging
from typing import List
from base.logging import log_init_arguments
from base.node import Node
import setting
from .input_handler import InputHandler
from .random_node_creator import RandomNodeCreator


class RandomInputHandler(InputHandler):
    """
    Input handler which creates a random initial global physical state for the tree state handler.

    This class is designed to work together with the TreeStateHandler. Here, the global physical state is given
    by a sequence of trees. Each tree is specified by a root node, which themselves are connected to children nodes.
    Each node contains a Particle object, which stores the position and the charge.
    This input handler allows for a variable number of root nodes with the same number of children. Each tree is of
    height at most two. If the height is two, the root nodes correspond to composite objects and the child nodes to
    point masses. If the height is one, the root nodes correspond to point masses.
    To create the random initial global physical state, this class relies on a random node creator, which creates a
    single random root node.
    """

    def __init__(self, random_node_creator: RandomNodeCreator, number_of_root_nodes: int) -> None:
        """
        The constructor of the RandomInputHandler class.

        The constructor sets the number of root nodes, the number of nodes per root node and the number of node levels
        in the setting package.

        Parameters
        ----------
        random_node_creator : input_output_handler.input_handler.random_node_creator.RandomNodeCreator
            The random node creator which creates a single random root node.
        number_of_root_nodes : int
            The number of root nodes to create.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           random_node_creator=random_node_creator.__class__.__name__,
                           number_of_root_nodes=number_of_root_nodes)
        super().__init__()
        setting.set_number_of_root_nodes(number_of_root_nodes)
        setting.set_number_of_nodes_per_root_node(random_node_creator.number_of_nodes_per_root_node)
        setting.set_number_of_node_levels(random_node_creator.number_of_node_levels)
        self._random_node_creator = random_node_creator

    def read(self) -> List[Node]:
        """
        Return the initial global physical state.

        This method creates the number of root nodes which was specified on initialization and fills them using the
        random node creator.

        Returns
        -------
        List[base.node.Node]
            The initial global physical state.
        """
        all_nodes = [Node() for _ in range(setting.number_of_root_nodes)]
        for node in all_nodes:
            self._random_node_creator.fill_root_node(node)
        return all_nodes
