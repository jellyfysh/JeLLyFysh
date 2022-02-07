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
"""Module for the ListScheduler class."""
import logging
from typing import Any
from jellyfysh.base.exceptions import SchedulerError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.time import Time
from .scheduler import Scheduler


class _Element(object):
    """
    Tuple-like class that stores a candidate event time and a corresponding associated object.

    Although the associated object is in JF always an event handler reference (which is also hinted by the argument
    names), they will have the type Any in this class.

    This class overwrites the equality comparison with an event handler instance. An _Element instance compares equal if
    the stored event handler instance is the compared other event handler instance. This class can then be used to
    remove an _Element instance from a list of _Element instances via the list's remove method, given just the event
    handler instance.
    """

    def __init__(self, time: Time, event_handler: Any) -> None:
        """
        The constructor of the _Element class.

        Parameters
        ----------
        time : base.time.Time
            The candidate event time.
        event_handler : Any
            The associated object.
        """
        self.time = time
        self.event_handler = event_handler

    def __eq__(self, event_handler: Any) -> bool:
        """
        Return whether the other associated object is the same object as the stored associated object.

        Parameters
        ----------
        event_handler : Any
            The other associated object.

        Returns
        -------
        bool
            Whether the other associated object is the same as the stored associated object.
        """
        return self.event_handler is event_handler


class ListScheduler(Scheduler):
    """
    The list scheduler uses a a simple list to implement the abstract methods of a scheduler.

    In JF, the scheduler keeps track of candidate event times and their associated event handler references. It can
    select among the candidate events the one with the smallest candidate event time. Moreover, it can delete events.

    Generally, an event stored in the scheduler consists of a candidate event time and an arbitrary associated object.
    Here, the candidate event time is an instance of the base.time.Time class to avoid loss of precision during long
    runs of JF (see documentation of the Time class for details).

    When asked for the succeeding event, the scheduler should return the object which is associated to the shortest
    time. Events can also be trashed based on the associated objects.

    Although the associated object is in JF always an event handler reference (which is also hinted by the argument
    names), they will have the type Any in this class.

    This scheduler uses a simple python list that stores instances of the _Element class. Each _Element instance stores
    a pair consisting of the candidate event time and the corresponding event handler. Pushing an event into the
    scheduler amounts to appending a new _Element instance to the internal list. Python's builtin min method can then be
    used to get the succeeding event with the smallest candidate event time. Likewise, the remove method of the internal
    list trashes an event.
    """

    def __init__(self, warn_on_equal_event_times: bool = False) -> None:
        """
        The constructor of the ListScheduler class.

        Parameters
        ----------
        warn_on_equal_event_times : bool, optional
            Whether this scheduler should log a warning when succeeding event times are equal.
        """
        self._logger = logging.getLogger(__name__)
        self._logger_enabled_for_debug = self._logger.isEnabledFor(logging.DEBUG)
        log_init_arguments(self._logger.debug, self.__class__.__name__)
        super().__init__()
        self._times = []
        self._last_returned_event = (Time(-float("inf"), -float("inf")), None)
        self._warn_on_equal_event_times = warn_on_equal_event_times

    def push_event(self, time: Time, event_handler: Any) -> None:
        """
        Push an event into the scheduler.

        Parameters
        ----------
        time : base.time.Time
            The candidate event time.
        event_handler : Any
            The associated object.
        """
        self._times.append(_Element(time, event_handler))

    def get_succeeding_event(self) -> Any:
        """
        Get the valid object currently stored in the scheduler which was pushed with the smallest candidate event time.

        Note that the _event_time_increasing method is called via an assert so that it can be skipped using the -O
        option of the interpreter. This method raises a SchedulerError if the event time is not increasing.

        Returns
        -------
        Any
            The object associated to the smallest stored event time.

        Raises
        ------
        base.exceptions.SchedulerError
            If the newest smallest event time is greater than the last returned event time.
        base.exceptions.SchedulerError
            If the scheduler does not contain any event.
        """
        try:
            smallest_element = min(self._times, key=lambda element: element.time)
        except ValueError:
            raise SchedulerError("The succeeding event was requested from the class {0}. However, the scheduler "
                                 "does not contain any events.".format(self.__class__.__name__))
        if self._logger_enabled_for_debug:
            self._logger.debug("Smallest event time in the scheduler: {0}".format(smallest_element.time))
        assert self._event_time_increasing(smallest_element.time, smallest_element.event_handler.__class__.__name__)
        return smallest_element.event_handler

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
        try:
            self._times.remove(event_handler)
        except ValueError:
            raise SchedulerError("Event handler {0} not present in scheduler.".format(event_handler.__class__.__name__))

    def update_logging(self) -> None:
        """
        Update the logging of this class.

        This method is called in resume.py which might be run with a different logging level compared to the run which
        created the dump. This method then ensures that this class logs on the correct level.
        """
        self._logger = logging.getLogger(__name__)
        self._logger_enabled_for_debug = self._logger.isEnabledFor(logging.DEBUG)

    def _event_time_increasing(self, event_time: Time, event_handler_class_name: str) -> bool:
        """
        Check whether the newest smallest event time is greater than the last returned event time.

        Optionally logs a warning if the event times are equal, and raises an exception if the newest smallest event
        time is smaller than the last returned event time.
        """
        if self._warn_on_equal_event_times and event_time == self._last_returned_event[0]:
            self._logger.warning("The last returned event time {0} calculated by the event handler {1} is equal to the "
                                 "new smallest event time {2} calculated by the event handler {3}."
                                 .format(*self._last_returned_event, event_time, event_handler_class_name))
        if event_time < self._last_returned_event[0]:
            raise SchedulerError("The last returned event time {0} calculated by the event handler {1} is greater than "
                                 "the new smallest event time {2} calculated by the event handler {3}."
                                 .format(*self._last_returned_event, event_time, event_handler_class_name))
        self._last_returned_event = (event_time, event_handler_class_name)
        return True
