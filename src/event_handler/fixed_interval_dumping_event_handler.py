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
"""Module for the FixedIntervalDumpingEventHandler class."""
import logging
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from .abstracts import DumpingEventHandler


class FixedIntervalDumpingEventHandler(DumpingEventHandler):
    """
    Event handler which triggers a dump of the run in fixed intervals.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    This class implements a DumpingEventHandler. It is connected to an output handler and the mediator defines
    a specific mediating method for this class. There the mediator object itself is handed to the output handler
    so that the whole run can be dumped.
    The instance of this event handler should always be active and its events should not be trashed by other events.
    Only after an event of this event handler, a new event should be computed.
    """

    def __init__(self, dumping_interval: float, output_handler: str) -> None:
        """
        The constructor of the FixedIntervalDumpingEventHandler class.

        Parameters
        ----------
        dumping_interval : float
            The time interval of the dumping.
        output_handler :
            The name of the output handler.

        Raises
        ------
        base.exceptions.ConfigurationError:
            If the dumping interval is not greater than zero.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           dumping_interval=dumping_interval, output_handler=output_handler)
        super().__init__(output_handler=output_handler)
        if not dumping_interval > 0.0:
            raise ConfigurationError("The dumping_interval in the event handler {0} has to be > 0.0."
                                     .format(self.__class__.__name__))
        self._dumping_interval = dumping_interval
        self._event_time = 0.0

    def send_event_time(self) -> float:
        """
        Return the candidate event time.

        Returns
        -------
        float
            The candidate event time.
        """
        self._event_time += self._dumping_interval
        return self._event_time

    def send_out_state(self) -> []:
        """
        Return the out-state.

        The out-state of this event handler is empty.

        Returns
        -------
        []
            The empty out-state.
        """
        return []
