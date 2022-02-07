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
"""Module for the SingleIndependentActivePeriodicDirectionEndOfChainEventHandler class."""
import logging
from random import randint
from typing import List, Sequence, Tuple
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
import jellyfysh.setting as setting
from jellyfysh.state_handler.tree_state_handler import StateId
from .abstracts import EndOfChainEventHandler


class SingleIndependentActivePeriodicDirectionEndOfChainEventHandler(EndOfChainEventHandler):
    """
    Event handler which ends an event chain in fixed time intervals.

    This event handler aligns the velocity of a single independent active (root or leaf) unit periodically with the
    cartesian axes in the positive direction (e.g., in three dimensions x -> y -> z -> x -> ...), while keeping the
    entry in the velocity along the relevant axis constant. The send_out_state method therefore only succeeds if the
    current velocity of the independent active unit is also aligned to one of the cartesian axes (see _get_new_velocity
    method that raises an error otherwise). After an end-of-chain event of this event handler, a random unit is
    independent active. This unit is a leaf (root) unit if the independent active unit at the moment of the computation
    of the candidate event time of this event handler was also a leaf (root) unit. This implies that the end-of-chain
    event should be trashed and recomputed if the simulation switches from an independent active leaf to root unit.

    The instance of this event handler should always be active because it computes the new candidate event time based
    on the time of the previously committed event (i.e., based on the time of the last call of the send_out_state
    method).

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    This class implements an EndOfChainEventHandler. For more details on the implementation of this event handler, see
    .abstracts.end_of_chain_event_handler.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information). The chain time that is added to the old candidate event
    time, however, can stay a simple float because it is always of the same order of magnitude during a run of JF.
    """

    def __init__(self, chain_time: float) -> None:
        """
        The constructor of the SingleIndependentActivePeriodicDirectionEndOfChainEventHandler class.

        Parameters
        ----------
        chain_time : float
            The time interval after which a new end-of-chain event occurs.

        Raises
        ------
        base.exceptions.ConfigurationError:
            If the chain time is not greater than zero.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, chain_time=chain_time)
        super().__init__()
        if not chain_time > 0.0:
            raise ConfigurationError("The chain_time in the event handler {0} must be > 0.0."
                                     .format(self.__class__.__name__))
        self._chain_time = chain_time
        self._last_committed_event_time = Time(0.0, 0.0)

    def send_event_time(self, root_cnodes_of_independent_active_units: Sequence[Node]) \
            -> Tuple[Time, List[Sequence[StateId]]]:
        """
        Return the candidate event time together with the identifiers of the units that should become independent
        active.

        The in-state should contain the branches of all independent active units (i.e., the active global state). This
        event handler only allows for a single independent active unit.

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
            If there is more than one root cnode (i.e., more than one independent active unit).
            If the given root cnode is not active.
        """
        assert len(root_cnodes_of_independent_active_units) == 1
        assert root_cnodes_of_independent_active_units[0].value.time_stamp is not None
        return super().send_event_time(root_cnodes_of_independent_active_units)

    def send_out_state(self, cnodes_with_active_units: Sequence[Node],
                       cnodes_with_new_active_units: Sequence[Node]) -> Sequence[Node]:
        """
        Return the out-state.

        In the out-state, the previously independent active unit has a zero velocity and the new independent active
        unit has the updated velocity (that is determined by the _get_new_velocity method).

        This class stores the time stamp of the (currently) independent active unit. This is used in the
        _get_new_chain_time method to achieve an end-of-chain event in fixed intervals of the chain time even when
        end-of-chain events are trashed.

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
            If there is more than one currently independent active unit or more than one new independent active unit.
            If the supposedly currently independent active unit is not active.
        """
        assert len(cnodes_with_active_units) == len(cnodes_with_new_active_units) == 1
        assert cnodes_with_active_units[0].value.time_stamp is not None
        self._last_committed_event_time = cnodes_with_active_units[0].value.time_stamp
        return super().send_out_state(cnodes_with_active_units, cnodes_with_new_active_units)

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

        Raises
        ------
        AssertionError
            If the old velocity is not parallel to one of the cartesian axes.
        """
        direction_of_motions = [index for index, component in enumerate(old_velocity) if component != 0.0]
        assert len(direction_of_motions) == 1
        new_velocity = [0.0 for _ in range(setting.dimension)]
        new_velocity[(direction_of_motions[0] + 1) % setting.dimension] = old_velocity[direction_of_motions[0]]
        return new_velocity

    def _get_new_chain_time(self, current_time_stamp: Time) -> float:
        """
        Return the new chain time that will be added to the given current time stamp to determine the next candidate
        event time.

        Parameters
        ----------
        current_time_stamp : base.time.Time
            The current time stamp of the active global state at the moment candidate-event-time request.

        Returns
        -------
        float
            The new chain time.

        Raises
        ------
        AssertionError
            If the current time stamp of the active global state is smaller than the last committed event time.
            If the current time stamp of the active global state is greater than the last committed event time plus the
            chain time (which might be because the event handler was deactivated).
        """
        assert 0.0 <= (current_time_stamp - self._last_committed_event_time) <= self._chain_time
        return (self._last_committed_event_time - current_time_stamp) + self._chain_time

    def _get_new_active_identifiers(self, root_cnodes_of_independent_active_units: Sequence[Node]) -> Sequence[StateId]:
        """
        Return the unit identifiers in the tree state handler which should become independent active based on the
        active global state at the moment of the candidate-event-time request.

        If the given independent active unit is a leaf (root) unit, this method returns a random identifier of a leaf
        (root) unit.

        Parameters
        ----------
        root_cnodes_of_independent_active_units : Sequence[base.node.Node]
            The active global state.

        Returns
        -------
        Sequence[state_handler.tree_state_handler.StateId]
            [The global-state identifier of a random leaf or root unit.]

        Raises
        ------
        AssertionError
            If there is more than one independent active unit.
            If there are no composite objects in the simulation but the given root cnode of the independent active unit
            still has child cnodes.
        """
        assert len(root_cnodes_of_independent_active_units) == 1
        if setting.number_of_node_levels == 1:
            assert len(root_cnodes_of_independent_active_units[0].children) == 0
            return [(randint(0, setting.number_of_root_nodes - 1),)]
        else:
            if len(root_cnodes_of_independent_active_units[0].children) == setting.number_of_nodes_per_root_node:
                # Root unit was independent active.
                return [(randint(0, setting.number_of_root_nodes - 1),)]
            else:
                # Leaf unit was independent active.
                return [(randint(0, setting.number_of_root_nodes - 1),
                         randint(0, setting.number_of_nodes_per_root_node - 1))]
