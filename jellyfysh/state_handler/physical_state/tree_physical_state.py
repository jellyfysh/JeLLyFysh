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
"""Module for the TreePhysicalState class."""
import logging
from typing import Iterable, Sequence, Tuple
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from .physical_state import PhysicalState


class TreePhysicalState(PhysicalState):
    """
    The tree lifting state implements a lifting state for the tree state handler.

    This class is designed to work together with the TreeStateHandler. This global physical state defines the
    identifiers to be tuples of integers, where the tuples can have different lengths.
    This class stores all positions of point masses and composite point objects in a tree, where each node contains a
    base.particle.Particle object. This object contains the time-sliced position and a charge (for the leaf nodes).
    """

    def __init__(self) -> None:
        """
        The constructor of the TreePhysicalState class.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__)
        super().__init__()
        self._root_nodes = None

    def initialize(self, global_physical_state: Sequence[Node]) -> None:
        """
        Initialize the state handler with the given full global physical state.

        Extends the initialize method of the Initializer class. This method must be extended and used once in the
        beginning of the run to initialize the physical state. Only after a call of this method, other public methods of
        this class can be called without raising an error.
        The global physical state constructed by an input handler should be given as a sequence of root nodes. Within
        each tree, each node should contain particle objects.

        Parameters
        ----------
        global_physical_state : Sequence[base.node.Node]
           The full global physical state.
        """
        super().initialize(global_physical_state)
        self._root_nodes = global_physical_state

    def get(self, identifier: Tuple[int, ...]) -> Node:
        """
        Return the node for the given global state identifier.

        Parameters
        ----------
        identifier : Tuple[int, ...]
            The global state identifier.

        Returns
        -------
        base.node.Node
            The node containing a particle object with the position and the charge corresponding to the identifier.
        """
        node = self._root_nodes[identifier[0]]
        for identifier_level in range(1, len(identifier)):
            node = node.children[identifier[identifier_level]]
        return node

    def set(self, identifier: Tuple[int, ...], position: Sequence[float]) -> None:
        """
        Store the given position for the global state identifier.

        Parameters
        ----------
        identifier : Tuple[int, ...]
            The global state identifier.
        position : Sequence[float]
            The position.
        """
        node = self._root_nodes[identifier[0]]
        for identifier_level in range(1, len(identifier)):
            node = node.children[identifier[identifier_level]]
        node.value.position = position

    def yield_identifiers(self) -> Iterable[Tuple[int, ...]]:
        """
        Generate all global state identifiers of all root nodes.

        This method is used in the extract_global_state method in the TreeStateHandler to extract the full global state.
        The identifiers generated here will be put into the extract_from_global_state method, which constructs a branch
        for each of these. Therefore we only generate the root node identifiers here, since the branches of the root
        nodes contain all children.

        Yields
        ------
        Tuple[int, ...]
            The global state identifiers of all root nodes.
        """
        for root_node_index, _ in enumerate(self._root_nodes):
            # noinspection PyRedundantParentheses
            yield (root_node_index,)
