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
"""Module for the SameActivePeriodicDirectionEndOfChainEventHandler class."""
import logging
from typing import Sequence
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.node import Node
import setting
from .abstracts import EndOfChainEventHandler


class SameActivePeriodicDirectionEndOfChainEventHandler(EndOfChainEventHandler):
    """
    Event handler which ends an event chain in fixed intervals.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    This class implements an EndOfChainEventHandler. For more details on the implementation of this event handler, see
    .abstracts.end_of_chain_event_handler. This end of chain event handler changes the direction of motion periodically
    (x -> y -> z -> x ...) and keeps the same independent active unit active.
    The instance of this event handler should always be active and its events should not be trashed by other events.
    Only after an event of this event handler, a new event should be computed.
    """
    def __init__(self, chain_length: float) -> None:
        """
        The constructor of the SameActivePeriodicDirectionEndOfChainEventHandler class.

        Parameters
        ----------
        chain_length : float
            The time interval of the ending of event chains.

        Raises
        ------
        base.exceptions.ConfigurationError:
            If the chain length is not greater than zero.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, chain_length=chain_length)
        super().__init__()
        if not chain_length > 0.0:
            raise ConfigurationError("The chain_length in the event handler {0} must be > 0.0."
                                     .format(self.__class__.__name__))
        self._chain_length = chain_length

    def send_out_state(self, cnodes_with_active_units: Sequence[Node],
                       cnodes_with_new_active_units: Sequence[Node]) -> Sequence[Node]:
        """
        Return the out-state.

        In the out-state, the previously independent active units have a zero velocity and the new independent active
        units have the same speed, but the direction of motion has changed.
        This class uses the trick, that the _get_new_active_identifiers method returns just an empty list, which means
        that cnodes_with_new_active units is empty. For the send_out_state method of the base class to work, this class
        uses cnodes_with_active units instead.

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
            If cnodes_with_new_active_units is not an empty sequence.
        """
        assert len(cnodes_with_new_active_units) == 0
        return super().send_out_state(cnodes_with_active_units, cnodes_with_active_units)

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
        return (old_direction_of_motion + 1) % setting.dimension

    def _get_new_chain_length(self) -> float:
        """
        Return the new chain length.

        Returns
        -------
        float
            The new chain length.
        """
        return self._chain_length

    def _get_new_active_identifiers(self) -> []:
        """
        Return the unit identifiers in the tree state handler which should become active.

        This method returns an empty list, since the independent active units should be kept active.

        Returns
        -------
        []
            The empty list of unit identifiers.
        """
        return []
