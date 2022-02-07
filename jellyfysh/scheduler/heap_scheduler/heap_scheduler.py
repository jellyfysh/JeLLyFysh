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
"""Module for the HeapScheduler class."""
import logging
from typing import Any, Mapping, MutableMapping
from sys import implementation
from jellyfysh.base.exceptions import SchedulerError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.time import Time, inf
from jellyfysh.scheduler import Scheduler
# noinspection PyUnresolvedReferences
from ._heap import ffi, lib
if implementation.name == "pypy":
    from __pypy__ import add_memory_pressure
else:
    # noinspection PyMissingOrEmptyDocstring
    def add_memory_pressure(_: int) -> None:
        return

# Directly import C functions used in performance relevant parts of the code.
_lib_event_valid_callback = lib.event_valid_callback
_lib_insert = lib.insert
_lib_root = lib.root
_lib_delete_events = lib.delete_events
_new_handle = ffi.new_handle
_from_handle = ffi.from_handle


class HeapScheduler(Scheduler):
    """
    The heap scheduler uses a binary min heap to implement the abstract methods of a scheduler.

    In JF, the scheduler keeps track of candidate event times and their associated event handler references. It can
    select among the candidate events the one with the smallest candidate event time. Moreover, it can delete events.

    Generally, an event stored in the scheduler consists of a candidate event time and an arbitrary associated object.
    Here, the candidate event time is an instance of the base.time.Time class to avoid loss of precision during long
    runs of JF. In this class, the candidate event time is stored as the quotient and remainder of an integer division
    of the float time with 1 (see documentation of the Time class for details).

    When asked for the succeeding event, the scheduler should return the object which is associated to the shortest
    time. Events can be trashed based on the associated objects.

    Although the associated object is in JF always an event handler reference (which is also hinted by the argument
    names), they will have the type Any in this class.

    The binary min heap is implemented in C in the files heap.c and heap.h. The cffi package is used to call the C code.
    The executable module heap_build.py can be used to compile the C code and to create the necessary files.

    The heap uses lazy deletion. This is done by introducing a valid counter for each event handler reference.
    This counter is initially zero and stored in this Python class. When a new event should be inserted into the heap,
    the current value of the counter is stored in the heap as well. Trashing an event based on an event handler
    reference just increases the valid counter. On a request of the root entry of the heap (i.e., the event with the
    smallest candidate event time), the heap relies on the 'delete_callback' method, which allows the C code to
    compare the current value of the counter with the counter of the current root entry. If the current value is larger
    (i.e., the event of the event handler was trashed in between), the entry is neglected. This is repeated until a
    still valid entry is found and returned.

    When a new event is inserted into the heap by using the 'insert' function, this C function receives the quotient and
    remainder of the candidate event time that is stored in an Time instance as two separate arguments.
    """

    def __init__(self, warn_on_equal_event_times: bool = False) -> None:
        """
        The constructor of the HeapScheduler class.

        Parameters
        ----------
        warn_on_equal_event_times : bool, optional
            Whether this scheduler should log a warning when succeeding event times are equal.

        Raises
        ------
        MemoryError
            If the C code fails to allocate memory.
        """
        self._logger = logging.getLogger(__name__)
        self._logger_enabled_for_debug = self._logger.isEnabledFor(logging.DEBUG)
        log_init_arguments(self._logger.debug, self.__class__.__name__)
        super().__init__()
        c_heap = lib.construct_heap()
        if c_heap == ffi.NULL:
            raise MemoryError("Could not allocate memory for the class {0}.".format(self.__class__.__name__))
        self._heap = ffi.gc(c_heap, lib.destroy_heap, size=lib.estimated_size(c_heap))
        self._minimal_valid_counter = {}
        self._last_returned_event = (Time(-float("inf"), -float("inf")), None)
        self._allocated_memory_bytes = 0
        self._scheduler_handle = _new_handle(self)
        self._event_handler_handles = {}
        self._warn_on_equal_event_times = warn_on_equal_event_times

    def push_event(self, time: Time, event_handler: Any) -> None:
        """
        Push an event into the binary min heap.

        Note that the 'insert' function implemented in C returns the number of bytes that are currently allocated for
        all heap entries (a failed re-allocation is signaled by a return value of -1). If the PyPy interpreter is used,
        and if the number of allocated bytes increased, one should add memory pressure by using the
        'add_memory_pressure' function. If the CPython interpreter is used, this function does nothing (see
        https://cffi.readthedocs.io/en/latest/using.html#memory-pressure-pypy). For the heap structure itself, this is
        solved by using the 'size' keyword argument in the ffi.gc call (see '__init__' method of this class).

        Also, the counter used for the lazy deletion is unbounded in Python. This might lead to the fact, that it cannot
        be passed to the C function because C uses a bounded integer. If this is detected, all events from the
        associated event handler are deleted in the heap (which necessarily have lower counter values) and the counter
        is reset to 0.

        Parameters
        ----------
        time : base.time.Time
            The event time.
        event_handler : Any
            The event handler.

        Raises
        ------
        MemoryError
            If the C code fails to re-allocate memory for more heap entries.
        """
        if time < inf:
            if event_handler not in self._event_handler_handles.keys():
                self._event_handler_handles[event_handler] = _new_handle(event_handler)
            try:
                new_size = _lib_insert(self._heap, time.quotient, time.remainder,
                                       self._event_handler_handles[event_handler],
                                       self._minimal_valid_counter.setdefault(event_handler, 0))
            except OverflowError:
                # Counter for event handler is too large to pass it to the C function.
                _lib_delete_events(self._heap, self._event_handler_handles[event_handler])
                self._minimal_valid_counter[event_handler] = 0
                new_size = _lib_insert(self._heap, time.quotient, time.remainder,
                                       self._event_handler_handles[event_handler], 0)
            if new_size == self._allocated_memory_bytes:
                return
            elif new_size > self._allocated_memory_bytes:
                add_memory_pressure(new_size - self._allocated_memory_bytes)
                self._allocated_memory_bytes = new_size
            elif new_size < self._allocated_memory_bytes:
                raise MemoryError("Could not reallocate memory for the class {0}.".format(self.__class__.__name__))

    def get_succeeding_event(self) -> Any:
        """
        Get the valid event handler reference currently stored in the scheduler which was pushed with the smallest
        candidate event time.

        Note that the _event_time_increasing method is called via an assert so that it can be skipped using the -O
        option of the interpreter. This method raises a SchedulerError if the event time is not increasing.

        The 'root' function implemented in C returns a root element, where the time is -inf, the event handler points to
        ffi.NULL, and the counter is undefined if the heap is empty after the lazy deletion was carried out. If this is
        detected, a SchedulerError is raised.

        Returns
        -------
        Any
            The event handler associated to the smallest stored event time.

        Raises
        ------
        base.exceptions.SchedulerError
            If the newest smallest event time is greater than the last returned event time.
        base.exceptions.SchedulerError
            If the scheduler does not contain any event.
        """
        # TODO: Add automatic garbage collection if number of elements in the heap becomes too large?
        top = _lib_root(self._heap, self._scheduler_handle, _lib_event_valid_callback)
        try:
            event_handler = _from_handle(top.event_handler)
        except RuntimeError as error:
            if (top.time_quotient == -float("inf") and top.time_remainder == -float("inf")
                    and top.event_handler == ffi.NULL):
                raise SchedulerError("The succeeding event was requested from the class {0}. However, the scheduler "
                                     "does not contain any events.".format(self.__class__.__name__))
            raise error
        if self._logger_enabled_for_debug:
            self._logger.debug("Smallest event time in the scheduler: {0}"
                               .format(str(top.time_quotient + top.time_remainder)))
        assert self._event_time_increasing(Time(top.time_quotient, top.time_remainder),
                                           event_handler.__class__.__name__)
        return event_handler

    def trash_event(self, event_handler: Any) -> None:
        """
        Delete an event in the scheduler based on the associated event handler.

        Note that the scheduler does not check whether an event of the associated event handler is present in it.

        Parameters
        ----------
        event_handler : Any
            The event handler.
        """
        self._minimal_valid_counter[event_handler] = self._minimal_valid_counter.get(event_handler, 0) + 1

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

    def event_valid_callback(self, event_handler_handle: ffi.CData, counter: int) -> bool:
        """
        Determine whether the event of the given event handler and counter value is still valid.

        This is effectively the callback function used in the 'root' function implemented in C (see
        get_succeeding_event method).

        Parameters
        ----------
        event_handler_handle : ffi.CData
            The cffi handle (created via ffi.new_handle) of the event handler.
        counter : int
            The counter of the event.

        Returns
        -------
        bool
            Whether the event is still valid.
        """
        return self._minimal_valid_counter[_from_handle(event_handler_handle)] > counter

    def __copy__(self):
        """No shallow copies can be created of this class."""
        raise NotImplementedError

    # noinspection PyDefaultArgument
    def __deepcopy__(self, _={}):
        """No deep copies can be created of this class."""
        raise NotImplementedError

    def __getstate__(self) -> Mapping[str, Any]:
        """
        Return a state of this class that can be pickled.

        This method removes _heap, _scheduler_handle, and _event_handler_handles from the self.__dict__ dictionary so
        that it can be pickled. Moreover, all heap entries are retrieved and pickled as well, so that they can be
        re-inserted into the heap in the __setstate__ method.

        Returns
        -------
        Mapping[str, Any]
            The state that can be pickled.
        """
        state = self.__dict__.copy()
        heap_entries = []
        index = 0
        while True:
            entry = lib.entry(self._heap, index)
            if entry.event_handler == ffi.NULL:
                break
            index += 1
            heap_entries.append((entry.time_quotient, entry.time_remainder, _from_handle(entry.event_handler),
                                 entry.counter))
        del state["_heap"]
        del state["_scheduler_handle"]
        del state["_event_handler_handles"]
        state["heap_entries"] = heap_entries
        return state

    def __setstate__(self, state: MutableMapping[str, Any]) -> None:
        """
        Use the state dictionary to initialize this class.

        This method creates the _heap, _scheduler_handle, and _event_handler_handles attributes that were deleted in the
        __getstate__ method. Moreover, all heap entries that were retrieved in the __getstate__ method are inserted into
        the heap.

        Parameters
        ----------
        state : MutableMapping[str, Any]
            The state.

        Raises
        ------
        MemoryError
            If the C code fails to allocate memory.
        """
        heap_entries = state["heap_entries"]
        del state["heap_entries"]
        self.__dict__.update(state)
        c_heap = lib.construct_heap()
        if c_heap == ffi.NULL:
            raise MemoryError("Could not allocate memory for the class {0}.".format(self.__class__.__name__))
        self._heap = ffi.gc(c_heap, lib.destroy_heap, size=lib.estimated_size(c_heap))
        new_size = 0
        self._event_handler_handles = {}
        for time_quotient, time_remainder, event_handler, counter in heap_entries:
            if event_handler not in self._event_handler_handles.keys():
                self._event_handler_handles[event_handler] = _new_handle(event_handler)
            new_size = _lib_insert(self._heap, time_quotient, time_remainder,
                                   self._event_handler_handles[event_handler], counter)
            if new_size < 0:
                raise MemoryError("Could not reallocate memory for the class {0}.".format(self.__class__.__name__))
        self._allocated_memory_bytes = new_size
        add_memory_pressure(self._allocated_memory_bytes)
        self._scheduler_handle = _new_handle(self)


@ffi.def_extern()
def event_valid_callback(scheduler_handle: ffi.CData, event_handler_handle: ffi.CData, counter: int) -> bool:
    """
    Callback function that is passed to the 'root' function implemented in C, which determines whether the event with
    the given event handler and counter value is still valid.

    This function just passes the call on to 'event_valid_callback' method of the object behind the handle.

    Parameters
    ----------
    scheduler_handle : ffi.CData
        The cffi handle (created via ffi.new_handle) of the object whose 'delete_identifier' method will be used.
    event_handler_handle : ffi.CData
        The cffi handle (created via ffi.new_handle) of the event handler responsible for the event.
    counter : int
        The counter of the event.

    Returns
    -------
    bool
        Whether the event is still valid.
    """
    return _from_handle(scheduler_handle).event_valid_callback(event_handler_handle, counter)
