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
"""Module for abstract EndOfChainEventHandler class."""
from abc import ABCMeta, abstractmethod
from typing import Any, List, Sequence, Tuple
from base.node import Node, yield_leaf_nodes
import setting
from state_handler.tree_state_handler import StateId
from .abstracts import LeavesEventHandler
from event_handler.helper_functions import analyse_velocity


class EndOfChainEventHandler(LeavesEventHandler, metaclass=ABCMeta):
    """
    The base class for all end of chain event handlers.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    An end of chain event handler changes the lifting state, meaning that the active unit can change and that the
    direction of motion can change.
    To provide this, this class depends on three methods to be implemented by the inheriting class:
    1. A method to sample a new chain length. This time is added to the previously returned candidate event time
    to compute the next candidate event time.
    2. A method to determine the identifiers of units which should be independently active after an event of this class.
    3. A method to determine the new direction of motion based on the old direction of motion.
    The instance of this event handler should always be active and its events should not be trashed by other events.
    Only after an event of this event handler, a new event should be computed, since otherwise step 1. yields a bug.
    On a request of the candidate event time, this event handler returns the new event time together with the
    identifiers of the units which should be activated. The latter are transformed into branches and transmitted,
    together with the branches of the currently independent active units, in the out-state request. There, the
    velocities are exchanged and also the direction of motion is changed.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        The constructor of the EndOfChainEventHandler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)
        self._event_time = 0.0

    def send_event_time(self) -> Tuple[float, List[Sequence[StateId]]]:
        """
        Return the candidate event time together with the identifiers of the units to be activated.

        Besides the candidate event time, this method returns a sequence of identifiers of units which should be made
        independently active in an event of this event handler.

        Returns
        -------
        (float, List[Sequence[state_handler.tree_state_handler.StateId]])
            The candidate event time, A list containing the sequence of unit identifiers.
        """
        self._event_time += self._get_new_chain_length()
        return self._event_time, [self._get_new_active_identifiers()]

    def send_out_state(self, cnodes_with_active_units: Sequence[Node],
                       cnodes_with_new_active_units: Sequence[Node]) -> Sequence[Node]:
        """
        Return the out-state.

        In the out-state, the previously independent active units have a zero velocity and the new independent active
        units have the same speed, but the direction of motion has changed.

        Parameters
        ----------
        cnodes_with_active_units : Sequence[base.node.Node]
            The root cnode of the branch of the independent active unit.
        cnodes_with_new_active_units : Sequence[base.node.Node]
            The root cnode of the branch of the unit which should get active.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.

        Raises
        ------
        AssertionError
            If not all leaf units in cnodes_with_active_units have the same velocity.
        AssertionError
            If the leaf units in cnodes_with_new_active_units have a non-zero velocity and time stamp but they do not
            appear in cnodes_with_active_units.
        """
        self._store_in_state(cnodes_with_active_units)
        self._time_slice_all_units_in_state()
        self._construct_leaf_cnodes()

        for index in range(1, len(self._leaf_units)):
            assert self._leaf_units[index].velocity == self._leaf_units[0].velocity
        old_direction_of_motion, speed = analyse_velocity(self._leaf_units[0].velocity)
        new_direction_of_motion = self._get_new_direction_of_motion(old_direction_of_motion)

        old_active_identifiers = set()
        velocity_changes = {}
        for index, unit in enumerate(self._leaf_units):
            velocity_changes[unit.identifier] = ([0.0 for _ in range(setting.dimension)], index)
            unit.velocity[old_direction_of_motion] = 0.0
            velocity_changes[unit.identifier][0][old_direction_of_motion] -= speed
            old_active_identifiers.add(unit.identifier)

        new_leaf_cnodes = []
        new_leaf_units = []
        for cnode in cnodes_with_new_active_units:
            for leaf_cnode in yield_leaf_nodes(cnode):
                new_leaf_cnodes.append(leaf_cnode)
                new_leaf_units.append(leaf_cnode.value)

        for index, unit in enumerate(new_leaf_units):
            if unit.identifier in old_active_identifiers:
                velocity_changes[unit.identifier][0][new_direction_of_motion] += speed
                self._leaf_units[velocity_changes[unit.identifier][1]].velocity[new_direction_of_motion] = speed
            else:
                assert unit.velocity is None
                assert unit.time_stamp is None
                cnode = new_leaf_cnodes[index]
                self._leaf_units.append(unit)
                self._leaf_cnodes.append(cnode)
                while cnode.parent:
                    cnode = cnode.parent
                self._state.append(cnode)
                velocity_changes[unit.identifier] = ([0.0 for _ in range(setting.dimension)],
                                                     len(self._leaf_units) - 1)
                velocity_changes[unit.identifier][0][new_direction_of_motion] += speed
                unit.velocity = [0.0 for _ in range(setting.dimension)]
                unit.velocity[new_direction_of_motion] = speed
                unit.time_stamp = self._event_time

        for old_leaf_unit in self._leaf_units:
            # noinspection PyTypeChecker
            if all(velocity_component < 1e-6 for velocity_component in old_leaf_unit.velocity):
                old_leaf_unit.velocity = None
                old_leaf_unit.time_stamp = None

        for velocity_change, index in velocity_changes.values():
            self._register_velocity_change_leaf_cnode(self._leaf_cnodes[index], velocity_change)
        self._commit_non_leaf_velocity_changes()
        return self._state

    @abstractmethod
    def _get_new_direction_of_motion(self, old_direction_of_motion: int) -> int:
        """
        Return the new direction of motion.

        Parameters
        ----------
        old_direction_of_motion : int
            The old direction of motion.

        Returns
        -------
        int
            The new direction of motion.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_new_chain_length(self) -> float:
        """
        Return the new chain length.

        Returns
        -------
        float
            The new chain length.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_new_active_identifiers(self) -> Sequence[StateId]:
        """
        Return the unit identifiers in the tree state handler which should become active.

        Returns
        -------
        Sequence[state_handler.tree_state_handler.StateId]
            The unit identifiers.
        """
        raise NotImplementedError
