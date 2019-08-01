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
"""Module for the HeapScheduler class."""
import heapq
import logging
from typing import Any
from base.exceptions import SchedulerError
from base.logging import log_init_arguments
from .event_identifier import EventIdentifier
from .scheduler import Scheduler


class HeapScheduler(Scheduler):
    """
    The heap scheduler uses a heap queue to implement the abstract methods of a scheduler.

    In JF, the scheduler keeps track of candidate events and their associated event handler references. It can select
    among the candidate events the one with the smallest candidate event time. Moreover, it can delete events.
    Generally, an event stored in the scheduler consists of a candidate event time and an arbitrary associated object.
    When asked for the succeeding event, the scheduler should return the object which is associated to the shortest
    time. Events can be trashed based on the associated objects.
    Although the associated object is in JF always an event handler reference (which is also hinted by the argument
    names), they will have the type Any in this class.
    """

    def __init__(self) -> None:
        """
        The constructor of the HeapScheduler class.
        """
        self._logger = logging.getLogger(__name__)
        self._logger_enabled_for_debug = self._logger.isEnabledFor(logging.DEBUG)
        log_init_arguments(self._logger.debug, self.__class__.__name__)
        super().__init__()
        self._times = []
        self._event_identifier_creator = EventIdentifier()
        self._identifier_event_handler_dictionary = {}

    def push_event(self, time: float, event_handler: Any) -> None:
        """
        Push an event into the scheduler.

        This method does not push the tuple (time, event_handler) itself into the heap because the heap compares the
        pushed objects. The tuple comparison leads to a failing comparison between the associated objects if the times
        are equal.
        To solve this, we create integers via the _EventIdentifier class, which are guaranteed to be different.

        Parameters
        ----------
        time : float
            The event time.
        event_handler : Any
            The associated object.
        """
        identifier = self._event_identifier_creator.identifier()
        self._identifier_event_handler_dictionary[identifier] = event_handler
        heapq.heappush(self._times, (time, identifier))

    def get_succeeding_event(self) -> Any:
        """
        Get the object currently stored in the scheduler which was pushed with the smallest event time.

        Returns
        -------
        Any
            The object associated to the smallest stored event time.
        """
        if self._logger_enabled_for_debug:
            self._logger.debug("Smallest event time in the scheduler: {0}".format(self._times[0][0]))
        return self._identifier_event_handler_dictionary[self._times[0][1]]

    def trash_event(self, event_handler: Any) -> None:
        """
        Delete an event in the scheduler based on the associated object.

        Parameters
        ----------
        event_handler : Any
            The associated object.

        Raises
        ------
        base.exceptions.SchedulerError
            If the associated object is not present in the scheduler.
        """
        # TODO If want to delete efficiently, we must establish a map from elements to heap indices
        for index, (_, identifier) in enumerate(self._times):
            if self._identifier_event_handler_dictionary[identifier] is event_handler:
                del self._times[index]
                del self._identifier_event_handler_dictionary[identifier]
                self._event_identifier_creator.delete_identifier(identifier)
                heapq.heapify(self._times)
                return
        raise SchedulerError("Event handler {0} not present in scheduler.".format(event_handler.__class__.__name__))

    def update_logging(self) -> None:
        """
        Update the logging of this class.

        This method is called in resume.py which might be run with a different logging level compared to the run which
        created the dump. This method then ensures that this class logs on the correct level.
        """
        self._logger = logging.getLogger(__name__)
        self._logger_enabled_for_debug = self._logger.isEnabledFor(logging.DEBUG)
