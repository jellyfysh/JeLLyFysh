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
"""Module for the TreeStateHandler class."""
from copy import copy
import logging
from typing import Callable, List, Sequence, Tuple
from base.logging import log_init_arguments
from base.node import Node
from base.unit import Unit
from .lifting_state.tree_lifting_state import TreeLiftingState
from .physical_state.tree_physical_state import TreePhysicalState
from .state_handler import StateHandler


StateId = Tuple[int, ...]


class TreeStateHandler(StateHandler):
    """
    The tree state handler uses a tree structure to implement the abstract methods of a state handler.

    It separates the global state into the global physical state and the global lifting state. The global physical
    state stores all positions of point masses and composite point objects in a tree, where each node contains a
    base.particle.Particle object (see TreePhysicalState). This object contains the time-sliced position and a charge
    (for the leaf nodes).
    Root nodes can be iterated and they have a unique numbering. The children of each root node are also numbered.
    This implies unique identifiers of nodes and their particles as tuples of integers with different lengths (see
    StateId). The first entry gives the root node, followed by the entries on lower levels.
    The global lifting state maps such identifiers onto a velocity and the time-stamp of the last time-slicing (see
    TreeLiftingState).

    When the tree state handler communicates with other parts of the application via its method, it combines the
    global physical and the global lifting state in base.unit.Unit objects.
    As a design principle, the event handlers keep the time-slicing of composite point objects and its point masses
    consistent. Therefore the parts of the tree structure needs to be mirrored as well. For a given identifier, the
    tree state handler constructs branches, that is the information of a node with its ancestors and descendants,
    where each node contains a unit. Such nodes are called cnodes in variables and docstrings to distinguish them from
    nodes which contain particles.
    """

    def __init__(self, physical_state: TreePhysicalState, lifting_state: TreeLiftingState) -> None:
        """
        The constructor of the TreeStateHandler class.

        Parameters
        ----------
        physical_state : state_handler.physical_state.tree_physical_state.TreePhysicalState
            The class to store the physical state.
        lifting_state : state_handler.lifting_state.tree_lifting_state.TreeLiftingState
            The class to store the lifting state.
        """
        self._logger = logging.getLogger(__name__)
        self._logger_enabled_for_debug = self._logger.isEnabledFor(logging.DEBUG)
        log_init_arguments(self._logger.debug, self.__class__.__name__,
                           physical_state=physical_state.__class__.__name__,
                           lifting_state=lifting_state.__class__.__name__)
        super().__init__(physical_state, lifting_state)

    def initialize(self, global_physical_state: Sequence[Node]) -> None:
        """
        Initialize the state handler with the given full global physical state.

        Extends the initialize method of the Initializer class. This method must be extended and used once in the
        beginning of the run to initialize the state handler. Only after a call of this method, other public methods of
        this class can be called without raising an error.
        The global state constructed by an input handler should be given as a sequence of root nodes. Within each tree,
        each node should contain particle objects. Via this method only the global physical state can be initialized.
        The global lifting state is initialized via the out-state of the StartOfRunEventHandler, which commits the first
        out-state via insert_into_global_state.

        Parameters
        ----------
        global_physical_state : Sequence[base.node.Node]
           The full global physical state.
        """
        super().initialize(global_physical_state)
        self._physical_state.initialize(global_physical_state)

    def extract_from_global_state(self, identifier: StateId) -> Node:
        """
        Extract a part of the global state based on a global state identifier.

        For the given identifier, this method constructs a branch of cnodes. When constructing the units, the positions
        and velocities are copied, so that event handlers may modify the branch. This method then returns the root cnode
        of the branch.
        The identifier is returned by the activator and the extracted part of the global state will be sent to the
        event handlers by the mediator.

        Parameters
        ----------
        identifier : StateId
            The global state identifier.

        Returns
        -------
        base.node.Node
            The root cnode of the branch corresponding the the global state identifier.
        """
        sliced_identifier = (identifier[0],)
        old_node = self._physical_state.get(sliced_identifier)
        velocity, time_stamp = self._lifting_state.get(sliced_identifier)
        unit = Unit(sliced_identifier, copy(old_node.value.position), old_node.value.charge,
                    copy(velocity), time_stamp)
        parent_cnode = Node(unit, old_node.weight)
        old_cnode = parent_cnode
        for identifier_level in range(1, len(identifier)):
            sliced_identifier = identifier[:identifier_level + 1]
            next_node = old_node.children[identifier[identifier_level]]
            velocity, time_stamp = self._lifting_state.get(sliced_identifier)
            unit = Unit(sliced_identifier, copy(next_node.value.position), next_node.value.charge,
                        copy(velocity), time_stamp)
            next_cnode = Node(unit, next_node.weight)
            old_cnode.add_child(next_cnode)
            old_cnode = next_cnode
            old_node = next_node
        for index, child in enumerate(old_node.children):
            next_cnode = self._construct_cnode_with_all_children_cnodes(child, identifier + (index,), copy)
            old_cnode.add_child(next_cnode)
        return parent_cnode

    def _construct_cnode_with_all_children_cnodes(
            self, starting_node: Node, starting_identifier: StateId,
            copy_method: Callable[[Sequence[float]], Sequence[float]] = lambda vector: vector) -> Node:
        """
        Construct the cnode for the given node and add all children after converting them to cnodes.

        The copy method will be applied to the position and the velocity. Per default this method does not copy them.
        """
        velocity, time_stamp = self._lifting_state.get(starting_identifier)
        unit = Unit(starting_identifier, copy_method(starting_node.value.position), starting_node.value.charge,
                    copy_method(velocity), time_stamp)
        cnode = Node(unit, starting_node.weight)
        for index, child in enumerate(starting_node.children):
            next_cnode = self._construct_cnode_with_all_children_cnodes(child, starting_identifier + (index,),
                                                                        copy_method)
            cnode.add_child(next_cnode)
        return cnode

    def insert_into_global_state(self, extracted_global_state: Sequence[Node]) -> None:
        """
        Insert an extracted part of the global state into the global state.

        The extracted global state is a sequence of root cnodes of branches. For each cnode in all branches, this method
        just updates the global physical and lifting state based on the information stored in the unit.
        The extracted global states are the out-states of the event handlers which changed internally the extracted
        global state they received by the method extract_from_global_state via the mediator.

        Parameters
        ----------
        extracted_global_state : Sequence[base.node.Node]
            The part of the global state which should be inserted into the global state.
        """
        for cnode in extracted_global_state:
            identifier = cnode.value.identifier
            self._physical_state.set(identifier, cnode.value.position)
            self._lifting_state.set(identifier, cnode.value.velocity, cnode.value.time_stamp)
            self.insert_into_global_state(cnode.children)

    def extract_active_global_state(self) -> List[Node]:
        """
        Extract the active part of the global state.

        The extracted active part of the global state is constructed as a sequence of root cnodes of branches, where
        each cnode contains an active unit. For this, this method relies on the global lifting state which provides
        a method which generates all independent lifted identifiers. When constructing the units, the positions
        and velocities are copied.
        The output of this method is passed on to the activator, which uses this information to determine the event
        handlers and their in-state identifiers to run next by the mediator. Also this method can be useful when an
        event handler wants to time-slice the full active global state.

        Returns
        -------
        List[base.node.Node]
            The active global state.
        """
        # TODO Maybe add a copy argument here at some point, so that the active part is not always copied
        if self._logger_enabled_for_debug:
            self._logger.debug("Independent active global state identifiers: {0}"
                               .format([identifier
                                        for identifier in self._lifting_state.yield_independent_lifted_identifiers()]))
        return [self.extract_from_global_state(identifier)
                for identifier in self._lifting_state.yield_independent_lifted_identifiers()]

    def extract_global_state(self) -> List[Node]:
        """
        Extract the full global state.

        The full global state is given by constructing a branch of cnodes for all root nodes. For this, this method
        relies on the global physical state which provides a method which generates all identifiers of the root nodes.
        When constructing the units, the positions and velocities are not copied.
        The output of this method is given to the output handlers.

        Returns
        -------
        List[base.node.Node]
            The full global state.
        """
        return [self._construct_cnode_with_all_children_cnodes(self._physical_state.get(root_node_identifier),
                                                               root_node_identifier)
                for root_node_identifier in self._physical_state.yield_identifiers()]

    def update_logging(self):
        """
        Update the logging of this class.

        This method is called in resume.py which might be run with a different logging level compared to the run which
        created the dump. This method then ensures that this class logs on the correct level.
        """
        self._logger = logging.getLogger(__name__)
        self._logger_enabled_for_debug = self._logger.isEnabledFor(logging.DEBUG)
