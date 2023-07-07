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
"""Module for the TreeStateHandler class."""
from copy import copy
import logging
from math import factorial, pi, sqrt
from random import normalvariate
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node, yield_leaf_nodes
from jellyfysh.base.particle import NewtonianParticle
from jellyfysh.base.time import Time
from jellyfysh.base.unit import Unit
import jellyfysh.setting as setting
from .state_handler import StateHandler


StateId = Tuple[int, ...]


class NewtonianTreeStateHandler(StateHandler):
    """
    The tree state handler uses a tree structure to implement the abstract methods of a state handler.

    It separates the global state into the global physical state and the global lifting state. The global physical
    state stores all positions of point masses and composite point objects in a tree, where each node contains a
    base.particle.Particle object (see TreePhysicalState). This object contains the time-sliced position, and possibly
    a charge (only for the leaf nodes).

    Root nodes can be iterated and they have a unique numbering. The children of each root node are also numbered.
    This implies unique identifiers of nodes and their particles as tuples of integers with different lengths (see
    StateId). The first entry gives the root node, followed by the entries on lower levels.

    The global lifting state maps such identifiers onto a velocity and the time stamp of the last time slicing (see
    TreeLiftingState). In order to avoid loss of precision during long runs of JF, time stamps (and candidate event
    times) are not stored as simple floats but as the quotient and remainder of an integer division of the time stamp
    with 1 (see base.time.Time class for more information).

    When the tree state handler communicates with other parts of the application via its method, it combines the
    global physical and the global lifting state in base.unit.Unit objects. Here, positions, velocities, and time stamps
    are copied so that other parts of JF can modify them without modifying the global state.

    As a design principle, the event handlers keep the time-slicing of composite point objects and its point masses
    consistent. Therefore the parts of the tree structure needs to be mirrored as well. For a given identifier, the
    tree state handler constructs branches, that is the information of a node with its ancestors and descendants,
    where each node contains a unit. Such nodes are called cnodes in variables and docstrings to distinguish them from
    nodes which contain particles.
    """

    def __init__(self) -> None:
        """
        The constructor of the TreeStateHandler class.
        """
        self._logger = logging.getLogger(__name__)
        self._logger_enabled_for_debug = self._logger.isEnabledFor(logging.DEBUG)
        log_init_arguments(self._logger.debug, self.__class__.__name__)
        super().__init__(None, None)
        self._root_nodes = None
        self._lifting_dictionary = {}
        self._lifted_identifiers = {identifier_length: set()
                                    for identifier_length in range(1, setting.number_of_node_levels + 1)}
        if setting.number_of_node_levels == 1:
            self._yield_independent_lifted_identifiers = self._yield_independent_lifted_identifiers_simple

    def _convert_subtree_to_newtonian(self, node: Node) -> Node:
        newtonian_node = Node(
            NewtonianParticle(node.value.position, None, node.value.charge))
        for child in node.children:
            newtonian_node.add_child(self._convert_subtree_to_newtonian(child))
        return newtonian_node

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
        self._root_nodes = [self._convert_subtree_to_newtonian(node) for node in global_physical_state]
        self.sample_maxwell_boltzmann_velocities()

    def sample_maxwell_boltzmann_velocities(self):
        """
        Sample a velocity from the Maxwell-Boltzmann distribution for every leaf unit.

        This method sets the center-of-mass velocity to zero, and the kinetic energy to one.
        """
        total_mass_leaf_nodes = 0
        center_of_mass_velocity = [0.0 for _ in range(setting.dimension)]
        for root_node in self._root_nodes:
            for leaf_node in yield_leaf_nodes(root_node):
                assert "mass" in leaf_node.value.charge
                assert leaf_node.value.charge["mass"] > 0.0
                total_mass_leaf_nodes += leaf_node.value.charge["mass"]
                leaf_node.value.velocity = [
                    normalvariate(0.0, sqrt(1.0 / (setting.beta * leaf_node.value.charge["mass"])))
                    for _ in range(setting.dimension)]
                for d in range(setting.dimension):
                    center_of_mass_velocity[d] += leaf_node.value.charge["mass"] * leaf_node.value.velocity[d]
        center_of_mass_velocity = [v / total_mass_leaf_nodes for v in center_of_mass_velocity]
        twice_kinetic_energy = 0.0
        mean_velocity_squared = 0.0
        for root_node in self._root_nodes:
            for leaf_node in yield_leaf_nodes(root_node):
                for d in range(setting.dimension):
                    leaf_node.value.velocity[d] -= center_of_mass_velocity[d]
                    mean_velocity_squared += leaf_node.value.velocity[d] * leaf_node.value.velocity[d]
                    twice_kinetic_energy += (leaf_node.value.charge["mass"] * leaf_node.value.velocity[d]
                                             * leaf_node.value.velocity[d])
        # See https://manual.gromacs.org/current/reference-manual/algorithms/molecular-dynamics.html, Eq. (23).
        # The factor shouldn't really matter, but we still use it.
        # degrees_of_freedom = (setting.dimension * setting.number_of_root_nodes * setting.number_of_nodes_per_root_node
        #                      - setting.dimension)
        # correction_factor = sqrt(degrees_of_freedom / (setting.beta * twice_kinetic_energy))
        correction_factor = sqrt(2.0 / twice_kinetic_energy)
        mean_velocity_squared /= (setting.number_of_root_nodes * setting.number_of_nodes_per_root_node)
        root_mean_velocity_squared = sqrt(mean_velocity_squared)
        # Correct the mean velocity to one.
        # See Maxwell-Boltzmann distribution in n-dimensional space on
        # https://en.wikipedia.org/wiki/Maxwell–Boltzmann_distribution, and use Legendre duplication formula as well as
        # the factorial expression for integer arguments for the Gamma function (see
        # https://en.wikipedia.org/wiki/Gamma_function).
        if setting.dimension % 2 == 0:
            additional_factor = (sqrt(setting.dimension / 2.0) * (factorial(setting.dimension // 2 - 1)) ** 2
                                 / (2 ** (1 - setting.dimension) * sqrt(pi) * factorial(setting.dimension - 1)))
        else:
            additional_factor = (sqrt(setting.dimension / 2.0) / (factorial((setting.dimension - 1) // 2) ** 2)
                                 * 2 ** (1 - setting.dimension) * sqrt(pi) * factorial(setting.dimension - 1))
        for root_node in self._root_nodes:
            for leaf_node in yield_leaf_nodes(root_node):
                for d in range(setting.dimension):
                    # leaf_node.value.velocity[d] *= correction_factor
                    leaf_node.value.velocity[d] *= additional_factor / root_mean_velocity_squared

        mean = 0.0
        for root_node in self._root_nodes:
            for leaf_node in yield_leaf_nodes(root_node):
                for v in leaf_node.value.velocity:
                    mean += v * v
        print(f"Root mean square velocity: {sqrt(mean / (setting.number_of_root_nodes * setting.number_of_nodes_per_root_node))}")

        # kin = 0.0
        # for root_node in self._root_nodes:
        #     for leaf_node in yield_leaf_nodes(root_node):
        #         for d in range(setting.dimension):
        #             kin += (0.5 * leaf_node.value.charge["mass"] * leaf_node.value.velocity[d]
        #                     * leaf_node.value.velocity[d])
        # print(f"Kinetic Energy: {kin}")

        if setting.number_of_node_levels > 1:
            assert setting.number_of_node_levels == 2
            for lifted_root_identifier in self._lifted_identifiers[1]:
                assert len(lifted_root_identifier) == 1
                self._root_nodes[lifted_root_identifier[0]].value.velocity = [0.0 for _ in range(setting.dimension)]
            for lifted_leaf_identifier in self._lifted_identifiers[2]:
                assert len(lifted_leaf_identifier) == 2
                assert (lifted_leaf_identifier[0],) in self._lifted_identifiers[1]
                root_node = self._root_nodes[lifted_leaf_identifier[0]]
                leaf_node = root_node.children[lifted_leaf_identifier[1]]
                for d in range(setting.dimension):
                    root_node.value.velocity[d] += leaf_node.value.velocity[d] * leaf_node.weight
        # TODO: At the moment, composite objects do not consider the weighted average of the positions/velocities of
        # their children.

    def _get_physical(self, identifier: StateId) -> Node:
        node = self._root_nodes[identifier[0]]
        for identifier_level in range(1, len(identifier)):
            node = node.children[identifier[identifier_level]]
        return node

    def _set_physical(self, identifier: StateId, position: Sequence[float], velocity: Sequence[float]) -> None:
        node = self._root_nodes[identifier[0]]
        for identifier_level in range(1, len(identifier)):
            node = node.children[identifier[identifier_level]]
        node.value.position = position
        node.value.velocity = velocity

    def _delete_lifting(self, identifier: StateId) -> None:
        if identifier in self._lifting_dictionary.keys():
            del self._lifting_dictionary[identifier]
            self._lifted_identifiers[len(identifier)].remove(identifier)

    def _get_lifting(self, identifier: StateId) -> Optional[Time]:
        return self._lifting_dictionary.get(identifier, None)

    def _set_lifting(self, identifier: StateId, time_stamp: Optional[Time]):
        if time_stamp is not None:
            self._lifting_dictionary[identifier] = time_stamp
            self._lifted_identifiers[len(identifier)].add(identifier)
        else:
            self._delete_lifting(identifier)

    def _yield_independent_lifted_identifiers(self) -> Iterable[StateId]:
        for lifted_root_identifier in self._lifted_identifiers[1]:
            lifted_identifiers = []
            for leaf_unit_identifier in range(setting.number_of_nodes_per_root_node):
                identifier = lifted_root_identifier + (leaf_unit_identifier,)
                if identifier in self._lifted_identifiers[2]:
                    lifted_identifiers.append(identifier)
            if len(lifted_identifiers) == setting.number_of_nodes_per_root_node:
                yield lifted_root_identifier
            else:
                yield from lifted_identifiers

    def _yield_independent_lifted_identifiers_simple(self) -> Iterable[Tuple[int, ...]]:
        yield from self._lifting_dictionary.keys()

    def extract_from_global_state(self, identifier: StateId) -> Node:
        """
        Extract a part of the global state based on a global state identifier.

        For the given identifier, this method constructs a branch of cnodes. When constructing the units, the positions,
        velocities, and time stamps are copied, so that event handlers may modify the branch. This method then returns
        the root cnode of the branch.

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
        old_node = self._get_physical(sliced_identifier)
        time_stamp = self._get_lifting(sliced_identifier)
        unit = Unit(sliced_identifier, copy(old_node.value.position), old_node.value.charge,
                    copy(old_node.value.velocity), copy(time_stamp))
        parent_cnode = Node(unit, old_node.weight)
        old_cnode = parent_cnode
        for identifier_level in range(1, len(identifier)):
            sliced_identifier = identifier[:identifier_level + 1]
            next_node = old_node.children[identifier[identifier_level]]
            time_stamp = self._get_lifting(sliced_identifier)
            unit = Unit(sliced_identifier, copy(next_node.value.position), next_node.value.charge,
                        copy(next_node.value.velocity), copy(time_stamp))
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
            copy_method: Callable[[Any], Any] = lambda object_to_copy: object_to_copy) -> Node:
        """
        Construct the cnode for the given node and add all children after converting them to cnodes.

        The copy method will be applied to the position, the velocity, and the time stamp. Per default this method does
        not copy them.
        """
        time_stamp = self._get_lifting(starting_identifier)
        unit = Unit(starting_identifier, copy_method(starting_node.value.position), starting_node.value.charge,
                    copy_method(starting_node.value.velocity), copy_method(time_stamp))
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
            self._set_physical(identifier, cnode.value.position, cnode.value.velocity)
            self._set_lifting(identifier, cnode.value.time_stamp)
            self.insert_into_global_state(cnode.children)

    def extract_active_global_state(self) -> List[Node]:
        """
        Extract the active part of the global state.

        The extracted active part of the global state is constructed as a sequence of root cnodes of branches, where
        each cnode contains an active unit. For this, this method relies on the global lifting state which provides
        a method which generates all independent lifted identifiers. When constructing the units, the positions,
        velocities, and time stamps are copied.

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
                                        for identifier in self._yield_independent_lifted_identifiers()]))
        return [self.extract_from_global_state(identifier)
                for identifier in self._yield_independent_lifted_identifiers()]

    def extract_global_state(self) -> List[Node]:
        """
        Extract the full global state.

        The full global state is given by constructing a branch of cnodes for all root nodes. For this, this method
        relies on the global physical state which provides a method which generates all identifiers of the root nodes.
        When constructing the units, the positions, velocities, and time stamps are not copied.

        The output of this method is given to the output handlers.

        Returns
        -------
        List[base.node.Node]
            The full global state.
        """
        #kin = 0.0
        #for root_node in self._root_nodes:
        #    for leaf_node in yield_leaf_nodes(root_node):
        #        for d in range(setting.dimension):
        #            kin += (0.5 * leaf_node.value.charge["mass"] * leaf_node.value.velocity[d]
        #                                     * leaf_node.value.velocity[d])
        #print(f"Kinetic Energy: {kin}")
        return [self._construct_cnode_with_all_children_cnodes(self._get_physical((root_node_index,)),
                                                               (root_node_index,))
                for root_node_index, _ in enumerate(self._root_nodes)]

    def update_logging(self):
        """
        Update the logging of this class.

        This method is called in resume.py which might be run with a different logging level compared to the run which
        created the dump. This method then ensures that this class logs on the correct level.
        """
        self._logger = logging.getLogger(__name__)
        self._logger_enabled_for_debug = self._logger.isEnabledFor(logging.DEBUG)
