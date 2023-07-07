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
"""Module for abstract EndOfChainEventHandler class."""
from abc import ABCMeta, abstractmethod
from copy import copy
from typing import Any, List, Sequence, Tuple
from jellyfysh.base.node import Node, yield_leaf_nodes
from jellyfysh.base.time import Time
import jellyfysh.setting as setting
from jellyfysh.state_handler.tree_state_handler import StateId
from .abstracts import LeavesEventHandler


class EndOfChainEventHandler(LeavesEventHandler, metaclass=ABCMeta):
    """
    The base class for all end-of-chain event handlers.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    An end-of-chain event handler changes the lifting state, meaning that the independent active units can change and
    that their velocities can change.

    This class assumes that all independent active units have the same velocity and the same time stamp. For the
    implementation of an end-of-chain event, this class depends on three methods to be implemented by the inheriting
    class:
    1. A method to sample a new chain time. This time is added to the time stamp of the active units at the moment
    of the candidate-event-time request of this event handler. This makes it possible to deactivate this event
    handler, and also to trash and recreate events from it.
    2. A method to determine the identifiers of units which should be independent active after an event of this class
    based on the active global state at the candidate-event-time request.
    3. A method to determine the new velocity based on the old velocity.

    On a request of the candidate event time, this event handler returns the new event time together with the
    identifiers of the units which should become independent active. The latter are transformed into branches and
    transmitted, together with the branches of the currently independent active units, in the out-state request. There,
    the velocities are changed and transferred to the new independent and induced active units.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information). The chain length that is added to the old candidate event
    time, however, can stay a simple float because it is always of the same order of magnitude during a run of JF.
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
        self._event_time = Time(0.0, 0.0)

    def send_event_time(self, root_cnodes_of_independent_active_units: Sequence[Node]) \
            -> Tuple[Time, List[Sequence[StateId]]]:
        """
        Return the candidate event time together with the identifiers of the units that should become independent
        active.

        The in-state should contain the branches of all independent active units (i.e., the active global state).

        Parameters
        ----------
        root_cnodes_of_independent_active_units : Sequence[base.node.Node]
            The active global state.

        Returns
        -------
        (base.time.Time, List[Sequence[state_handler.tree_state_handler.StateId]])
            The candidate event time, a list containing the sequence of new independent active unit identifiers.

        Raises
        ------
        AssertionError
            If the time stamps of the root units in the in-state are not equal.
        """
        current_time_stamp = root_cnodes_of_independent_active_units[0].value.time_stamp
        assert all(node.value.time_stamp == current_time_stamp for node in root_cnodes_of_independent_active_units)
        self._event_time = current_time_stamp + self._get_new_chain_time(current_time_stamp)
        return self._event_time, [self._get_new_active_identifiers(root_cnodes_of_independent_active_units)]

    def send_out_state(self, cnodes_with_active_units: Sequence[Node],
                       cnodes_with_new_active_units: Sequence[Node]) -> Sequence[Node]:
        """
        Return the out-state.

        In the out-state, the previously independent active units have a zero velocity and the new independent active
        units have the updated velocity (that is determined by the _get_new_velocity method).

        Parameters
        ----------
        cnodes_with_active_units : Sequence[base.node.Node]
            The root cnodes of the branches of the independent active units.
        cnodes_with_new_active_units : Sequence[base.node.Node]
            The root cnodes of the branches of the units which should get independent active.

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

        assert all(self._leaf_units[index].velocity == self._leaf_units[0].velocity
                   for index in range(1, len(self._leaf_units)))
        old_velocity = self._leaf_units[0].velocity
        new_velocity = self._get_new_velocity(old_velocity)

        old_active_identifiers = set()
        velocity_changes = {}
        for index, unit in enumerate(self._leaf_units):
            unit.velocity = [0.0 for _ in range(setting.dimension)]
            velocity_changes[unit.identifier] = ([-velocity_component for velocity_component in old_velocity], index)
            old_active_identifiers.add(unit.identifier)

        new_leaf_cnodes = []
        new_leaf_units = []
        for cnode in cnodes_with_new_active_units:
            for leaf_cnode in yield_leaf_nodes(cnode):
                new_leaf_cnodes.append(leaf_cnode)
                new_leaf_units.append(leaf_cnode.value)

        for index, unit in enumerate(new_leaf_units):
            if unit.identifier in old_active_identifiers:
                for velocity_index in range(setting.dimension):
                    velocity_changes[unit.identifier][0][velocity_index] += new_velocity[velocity_index]
                self._leaf_units[velocity_changes[unit.identifier][1]].velocity = copy(new_velocity)
            else:
                assert unit.velocity is None
                assert unit.time_stamp is None
                cnode = new_leaf_cnodes[index]
                self._leaf_units.append(unit)
                self._leaf_cnodes.append(cnode)
                while cnode.parent:
                    cnode = cnode.parent
                if cnode not in self._state:
                    self._state.append(cnode)
                velocity_changes[unit.identifier] = (copy(new_velocity), len(self._leaf_units) - 1)
                unit.velocity = copy(new_velocity)
                unit.time_stamp = copy(self._event_time)

        for leaf_unit in self._leaf_units:
            # noinspection PyTypeChecker
            if all(abs(velocity_component) < 1.0e-13 for velocity_component in leaf_unit.velocity):
                leaf_unit.velocity = None
                leaf_unit.time_stamp = None

        for velocity_change, index in velocity_changes.values():
            self._register_velocity_change_leaf_cnode(self._leaf_cnodes[index], velocity_change)
        self._commit_non_leaf_velocity_changes()
        return self._state

    @abstractmethod
    def _get_new_velocity(self, old_velocity: Sequence[float]) -> Sequence[float]:
        """
        Return the new velocity based on the old velocity.

        Parameters
        ----------
        old_velocity : Sequence[float]
            The old velocity.

        Returns
        -------
        Sequence[float]
            The new velocity.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_new_chain_time(self, current_time_stamp: Time) -> float:
        """
        Return the new chain time that will be added to the given current time stamp to determine the next candidate
        event time.

        Parameters
        ----------
        current_time_stamp : base.time.Time
            The current time stamp of the active global state at the moment of the candidate-event-time request.

        Returns
        -------
        float
            The new chain time.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_new_active_identifiers(self, root_cnodes_of_independent_active_units: Sequence[Node]) -> Sequence[StateId]:
        """
        Return the unit identifiers in the tree state handler which should become independent active based on the
        active global state at the moment of the candidate-event-time request.

        Parameters
        ----------
        root_cnodes_of_independent_active_units : Sequence[base.node.Node]
            The active global state.

        Returns
        -------
        Sequence[state_handler.tree_state_handler.StateId]
            The unit identifiers.
        """
        raise NotImplementedError


class NewtonianEndOfChainEventHandler(EndOfChainEventHandler, metaclass=ABCMeta):
    """
    Abstract end-of-chain event handler for the NewtonianTreeStateHandler.

    The mediator defines a specific mediating method for this class, where the velocities of all leaf units are sampled
    again from the Maxwell-Boltzmann distribution.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        The constructor of the NewtonianEndOfChainEventHandler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)
