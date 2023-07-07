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
"""Module for InitialChainStartOfRunEventHandler class."""
import logging
import random
from typing import List, Sequence, Tuple
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
import jellyfysh.setting as setting
from jellyfysh.state_handler.tree_state_handler import StateId
from .abstracts import LeavesEventHandler, StartOfRunEventHandler


class SingleIndependentActiveRandomNewtonianLeafUnitStartOfRunEventHandler(StartOfRunEventHandler, LeavesEventHandler):
    """
    Event handler which starts a run and sets the initial lifting.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    This class implements a StartOfRunEventHandler. At the beginning of a run, the activator specifically returns an
    instance of this class to be run by the mediator.

    The out-state of this event handler consists of the branch of a single independent active unit. The velocity is
    aligned in positive direction along an axis.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information).
    """

    def __init__(self) -> None:
        """
        The constructor of the InitialChainStartOfRunEventHandler class.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the speed is not larger than zero.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__)
        super().__init__()
        if setting.number_of_node_levels == 1:
            self._initial_active_identifier = (random.randint(0, setting.number_of_root_nodes - 1),)
        else:
            self._initial_active_identifier = (random.randint(0, setting.number_of_root_nodes - 1),
                                               random.randint(0, setting.number_of_nodes_per_root_node - 1))
        self._event_time = Time(0.0, 0.0)

    def send_event_time(self) -> Tuple[Time, List[StateId]]:
        """
        Return the candidate event time together with the initially active identifier.

        Returns
        -------
        (base.time.Time, List[state_handler.tree_state_handler.StateId])
            The candidate event time, the list containing the initially active unit identifier.
        """
        return self._event_time, [self._initial_active_identifier]

    def send_out_state(self, cnode_with_initially_active_unit: Node) -> Sequence[Node]:
        """
        Return the out-state.

        This method receives the branch of the initially active unit.

        Parameters
        ----------
        cnode_with_initially_active_unit : base.node.Node
            The branch of the initially active unit.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.
        """
        self._store_in_state([cnode_with_initially_active_unit])
        self._construct_leaf_cnodes()

        for leaf_cnode in self._leaf_cnodes:
            unit = leaf_cnode.value
            assert unit.time_stamp is None
            unit.time_stamp = Time(0.0, 0.0)
            self._register_velocity_change_leaf_cnode(leaf_cnode, leaf_cnode.value.velocity)
        self._commit_non_leaf_velocity_changes()
        return self._state
