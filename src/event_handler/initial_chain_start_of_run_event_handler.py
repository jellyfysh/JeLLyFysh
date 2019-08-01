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
"""Module for InitialChainStartOfRunEventHandler class."""
import logging
from typing import List, Sequence, Tuple
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.node import Node
import setting
from state_handler.tree_state_handler import StateId
from .abstracts import LeavesEventHandler, StartOfRunEventHandler


class InitialChainStartOfRunEventHandler(StartOfRunEventHandler, LeavesEventHandler):
    """
    Event handler which starts a run and sets the initial lifting.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    This class implements a StartOfRunEventHandler. At the beginning of a run, the activator specifically returns an
    instance of this class to be run by the mediator.
    The out-state of this event handler consists of the branch of a single independent active unit. The velocity is
    aligned in positive direction along an axis.
    """

    def __init__(self, initial_direction_of_motion: int, speed: float,
                 initial_active_identifier: Sequence[int]) -> None:
        """
        The constructor of the InitialChainStartOfRunEventHandler class.

        Parameters
        ----------
        initial_direction_of_motion : int
            The initial direction of motion.
        speed : float
            The initial absolute value of the velocity.
        initial_active_identifier : List[int]
            The global state identifier of the initially active leaf unit.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the initial direction of motion exceeds the dimension.
        base.exceptions.ConfigurationError
            If the speed is not larger than zero.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           initial_direction_of_motion=initial_direction_of_motion, speed=speed,
                           initial_active_identifier=initial_active_identifier)
        super().__init__()
        if initial_direction_of_motion >= setting.dimension:
            raise ConfigurationError("The index of the initial direction of motion {0} "
                                     "has to be smaller than the dimension {1} "
                                     "in the event handler {2}."
                                     .format(initial_direction_of_motion, setting.dimension, self.__class__.__name__))
        if speed <= 0.0:
            raise ConfigurationError("The speed in the event handler {0} should be larger than 0.0."
                                     .format(self.__class__.__name__))
        self._initial_velocity = [0.0] * setting.dimension
        self._initial_velocity[initial_direction_of_motion] = speed
        self._initial_active_identifier = tuple(initial_active_identifier)
        # TODO extend to to multiple initial active units

    def send_event_time(self) -> Tuple[float, List[StateId]]:
        """
        Return the candidate event time together with the initially active identifier.

        Returns
        -------
        (float, List[state_handler.tree_state_handler.StateId])
            The candidate event time, the list containing the initially active unit identifier.
        """
        self._event_time = 0.0
        return 0.0, [self._initial_active_identifier]

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
            assert unit.velocity is None
            assert unit.time_stamp is None
            unit.velocity = self._initial_velocity.copy()
            unit.time_stamp = 0.0
            self._register_velocity_change_leaf_cnode(leaf_cnode, self._initial_velocity)

        self._commit_non_leaf_velocity_changes()
        return self._state
