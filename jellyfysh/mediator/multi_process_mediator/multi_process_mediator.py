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
"""Module for the MultiProcessMediator class."""
import collections
from enum import Enum
import logging
import multiprocessing
import multiprocessing.connection as connection
import os
import types
from jellyfysh.activator import Activator
from jellyfysh.base.exceptions import ConfigurationError, MediatorError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.input_output_handler import InputOutputHandler
from jellyfysh.mediator.mediator import Mediator
from jellyfysh.scheduler import Scheduler
from jellyfysh.state_handler import StateHandler
from .or_event import create_or_event


def _communicate_via_pipe_without_arguments(func):
    def _pipe_wrapper(pipe):
        pipe.send(func())
    return _pipe_wrapper


def _communicate_via_pipe_with_arguments(func, unpack_arguments):
    if unpack_arguments:
        def _pipe_wrapper(pipe):
            arguments = pipe.recv()
            pipe.send(func(*arguments))
    else:
        def _pipe_wrapper(pipe):
            arguments = pipe.recv()
            pipe.send(func(arguments))
    return _pipe_wrapper


def run_in_process(self, pipe: multiprocessing.Pipe, start_event: multiprocessing.Event,
                   continue_event: multiprocessing.Event, start_or_continue_event: multiprocessing.Event,
                   semaphore: multiprocessing.Semaphore) -> None:
    """
    Define the iteration loop for an event handler which communicates via pipes with the mediator and waits for
    multiprocessing events to be set by the mediator.

    This method will first decorate the send_event_time and send_out_state methods of the event handler so that the
    arguments are received via a pipe and objects are returned via the same pipe.
    Then this methods enters an infinite iteration loop. There it first waits for the or event. If the start event
    was the relevant one, the event is cleared, the send_event_time is called and the result is put into the pipe. The
    arguments of the method are received via the same pipe. During the calculation of the candidate event time the
    semaphore is acquired to block resources. If the continue event was the relevant event, an exception is raised.
    After the send_event_time method was called, this method again waits for a start or an continue event. On a start
    event, the iteration loops resumes to the start. On a continue event, the event is cleared and the send_out_state
    method is called. The result is again put into the pipe and the arguments are received via the pipe.

    Parameters
    ----------
    self : event_handler.EventHandler
        The event handler instance.
    pipe : multiprocessing.Pipe
        The pipe.
    start_event : multiprocessing.Event
        The start event.
    continue_event : multiprocessing.Event
        The continue event.
    start_or_continue_event : multiprocessing.Event
        The or event of the start and the continue event.
    semaphore : multiprocessing.Semaphore
        The semaphore.

    Raises
    ------
    base.exceptions.MediatorError
        If the continue event was set at the beginning of the iteration loop.
    base.exceptions.MediatorError
        If the or event failed.
    """
    if self.number_send_event_time_arguments > 1:
        raise MediatorError("Method send_event_time only allows for 0 or 1 arguments.")
    self.send_event_time = (_communicate_via_pipe_without_arguments(self.send_event_time)
                            if self.number_send_event_time_arguments == 0
                            else _communicate_via_pipe_with_arguments(self.send_event_time, False))
    self.send_out_state = (_communicate_via_pipe_without_arguments(self.send_out_state)
                           if self.number_send_out_state_arguments == 0
                           else _communicate_via_pipe_with_arguments(self.send_out_state, True))
    while True:
        start_or_continue_event.wait()
        assert sum(1 if event.is_set() else 0 for event in [start_event, continue_event]) == 1
        if start_event.is_set():
            start_event.clear()
            semaphore.acquire()
            self.send_event_time(pipe)
            semaphore.release()
        elif continue_event.is_set():
            raise MediatorError("Continue event is not allowed in idle state!")
        else:
            raise MediatorError("Or event fired although no event is set!")

        start_or_continue_event.wait()
        assert sum(1 if event.is_set() else 0 for event in [start_event, continue_event]) == 1
        if start_event.is_set():
            continue
        elif continue_event.is_set():
            continue_event.clear()
            self.send_out_state(pipe)
        else:
            raise MediatorError("Or event fired although no event is set!")


class EventHandlerState(Enum):
    """The stages the mediator assigns to the event handlers."""
    idle = 0
    event_time_started = 1
    suspended = 2
    out_state_started = 3


class MultiProcessMediator(Mediator):
    """
    This class implements a mediator where the events are computed in separate processes compared to the mediator.

    This mediator adds the run_in_process method to each created instance of an event handler. This method runs as an
    autonomous iteration loop in a separate process and reacts to multiprocessing events set by the mediator.
    In addition, the send_event_time and send_out_state methods of the event handlers get decorated, so that the
    mediator communicates with them via pipes. The same pipe is used to receive the candidate event time and the
    out-state. The mediator assigns four different stages to the event handlers which determine which multiprocessing
    events can be set to start the send_event_time or the send_out_state method and to determine the nature of the
    objects in the pipe.
    If enough processors are present, the event handlers may compute out-states in advance. This is only relevant for
    event handlers which do not have any arguments in their send_out_state methods.
    After these modifications, this mediator follows the same nine general steps as the single-process mediator:
    1. Extract the active global state from the state handler.
    2. Based on this, obtain from the activator the event handlers, whose send_event_time method should be called, and
    their global in-state identifiers needed to construct the in-state arguments.
    3. Extract the in-states from the state handler using the global in-state identifiers.
    4. Request the candidate event times from all the event handlers returned by the activator and push them into the
    scheduler.
    5. Obtain the event handler which created the earliest candidate event time from the scheduler.
    6. Receive the out-state of the event handler with the earliest candidate event time. If the event handler defines
    an argument construction method in the Mediator base class, the objects returned together with the candidate event
    time will be handed to this argument construction method in order to construct the arguments of the send_out_state
    method of the event handler.
    7. Commit the out-state to the global state using the state handler.
    8. Based on the event handler, which committed the event to the global state, receive the event handlers from the
    activator, whose events are trashed in the scheduler.
    (9. Optionally, if the event handler which committed the event to the global state defines a mediating method in the
    Mediator base class, this mediating method is run.)
    (For more details, see [Hoellmer2020] in References.bib.)
    """

    def __init__(self, input_output_handler: InputOutputHandler, state_handler: StateHandler, scheduler: Scheduler,
                 activator: Activator, number_cores: int = os.cpu_count()) -> None:
        """
        The constructor of the MultiProcessMediator class.

        Parameters
        ----------
        input_output_handler : input_output_handler.InputOutputHandler
            The input-output handler.
        state_handler : state_handler.StateHandler
            The state handler.
        scheduler : scheduler.Scheduler
            The scheduler.
        activator : activator.Activator
            The activator.
        number_cores : int, optional
            The number of cores to use.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the number of cores is not larger than one.
        """
        self._logger = logging.getLogger(__name__)
        self._logger_enabled_for_debug = self._logger.isEnabledFor(logging.DEBUG)
        log_init_arguments(self._logger.debug, self.__class__.__name__,
                           input_output_handler=input_output_handler.__class__.__name__,
                           state_handler=state_handler.__class__.__name__,
                           scheduler=scheduler.__class__.__name__,
                           activator=activator.__class__.__name__)
        if not number_cores > 1:
            raise ConfigurationError("The multi processing mediator should only be used "
                                     "when more than one processor is available.")
        state_handler.initialize(input_output_handler.read())
        super().__init__(input_output_handler, state_handler, scheduler, activator)
        self._out_states = {}
        self._number_cores = number_cores
        self._pipes = {}
        self._event_handlers = {}
        self._os_processes = []
        self._start_events = {}
        self._send_out_state_events = {}
        self._event_handlers_state = {}
        for event_handler in self._event_handlers_list:
            event_handler.run_in_process = types.MethodType(run_in_process, event_handler)
        self._start_processes()

    def _start_processes(self) -> None:
        """Start the processes of the event handlers and set up the pipes."""
        semaphore = multiprocessing.BoundedSemaphore(value=self._number_cores - 1)
        for event_handler in self._event_handlers_list:
            pipe, process_pipe = multiprocessing.Pipe()
            self._pipes[event_handler] = pipe
            self._event_handlers[pipe] = event_handler
            self._send_out_state_events[pipe] = multiprocessing.Event()
            self._start_events[pipe] = multiprocessing.Event()
            self._os_processes.append(multiprocessing.Process(
                target=event_handler.run_in_process,
                args=(process_pipe, self._start_events[pipe], self._send_out_state_events[pipe],
                      create_or_event(self._start_events[pipe], self._send_out_state_events[pipe]), semaphore)))
            self._os_processes[-1].start()
            self._event_handlers_state[pipe] = EventHandlerState.idle
            process_pipe.close()

    def run(self) -> None:
        """
        Loop over the legs of the continuous-time evolution of the event-chain Monte Carlo algorithm.

        This method is called by run.py and resume.py. The loop should only be interrupted when a
        base.exceptions.EndOfRun exception is raised, which is caught in the scripts.
        """
        while True:
            # Extract active global state
            active_global_state = self._state_handler.extract_active_global_state()

            # Fetch event handlers to activate
            event_handlers_in_state_dictionary = self._activator.get_event_handlers_to_run(
                active_global_state, self._event_handler_with_shortest_event_time)
            if self._logger_enabled_for_debug:
                self._logger.debug(
                    "Event handlers that will be run with their in-state identifiers: {0}"
                    .format({event_handler.__class__.__name__: in_state_identifier
                             for event_handler, in_state_identifier in event_handlers_in_state_dictionary.items()}))

            # Fetch in-states
            for event_handler, in_state_identifiers in event_handlers_in_state_dictionary.items():
                if in_state_identifiers is not None:
                    event_handlers_in_state_dictionary[event_handler] = [
                        self._state_handler.extract_from_global_state(identifier)
                        for identifier in in_state_identifiers]

            # Send in-states
            pipes = []
            for event_handler, in_state in event_handlers_in_state_dictionary.items():
                pipe = self._pipes[event_handler]
                if self._event_handlers_state[pipe] == EventHandlerState.idle:
                    self._start_events[pipe].set()
                else:
                    raise MediatorError("Event Process not ready!")
                if event_handler.number_send_event_time_arguments:
                    assert in_state is not None
                    pipe.send(in_state)
                self._event_handlers_state[pipe] = EventHandlerState.event_time_started
                pipes.append(pipe)

            # Receive event times, and let them continue on out-state if there are idle os-processes.
            pipes_time_received = collections.deque()
            event_times_received = 0
            while event_times_received < len(event_handlers_in_state_dictionary):
                for pipe in connection.wait(pipes):
                    if self._event_handlers_state[pipe] == EventHandlerState.event_time_started:
                        self._event_handlers_state[pipe] = EventHandlerState.suspended
                        returned = pipe.recv()
                        try:
                            event_time, self._out_state_arguments[pipe] = returned
                        except TypeError:
                            event_time, self._out_state_arguments[pipe] = returned, []
                        # If no arguments for out-state, one can calculate it in advance if there are resources
                        if not self._event_handlers[pipe].number_send_out_state_arguments:
                            pipes_time_received.append(pipe)
                        event_times_received += 1
                        if (0 < len(event_handlers_in_state_dictionary) - event_times_received < self._number_cores - 1
                                and len(pipes_time_received)):
                            next_pipe = pipes_time_received.popleft()
                            self._send_out_state_events[next_pipe].set()
                            if self._event_handlers[next_pipe].number_send_out_state_arguments:
                                next_pipe.send(
                                    self._out_state_arguments_methods[self._event_handlers[next_pipe]](
                                        *self._out_state_arguments[next_pipe]))
                            self._event_handlers_state[next_pipe] = EventHandlerState.out_state_started
                        self._scheduler.push_event(event_time, self._event_handlers[pipe])
                        if self._logger_enabled_for_debug:
                            self._logger.debug("Pushed candidate event time to the scheduler: {0} ({1})"
                                               .format(event_time, self._event_handlers[pipe].__class__.__name__))
                    elif self._event_handlers_state[pipe] == EventHandlerState.out_state_started:
                        self._event_handlers_state[pipe] = EventHandlerState.idle
                        if len(pipes_time_received):
                            next_pipe = pipes_time_received.popleft()
                            self._send_out_state_events[next_pipe].set()
                            if self._event_handlers[next_pipe].number_send_out_state_arguments:
                                next_pipe.send(
                                    self._out_state_arguments_methods[self._event_handlers[next_pipe]](
                                        *self._out_state_arguments[next_pipe]))
                            self._event_handlers_state[next_pipe] = EventHandlerState.out_state_started
                        self._out_states[self._event_handlers[pipe]] = pipe.recv()
                    else:
                        raise MediatorError("Event process with pipe {0} to mediator is already finished"
                                            " and shouldn't receive anything anymore!".format(pipe))

            # Request shortest time
            # Pop the earliest event handler from scheduler, and let it calculate out-state
            self._event_handler_with_shortest_event_time = self._scheduler.get_succeeding_event()
            if self._logger_enabled_for_debug:
                self._logger.debug("Event handler which created the event with the shortest event time: {0}"
                                   .format(self._event_handler_with_shortest_event_time.__class__.__name__))
            pipe_with_shortest_event_time = self._pipes[self._event_handler_with_shortest_event_time]
            if self._event_handlers_state[pipe_with_shortest_event_time] == EventHandlerState.suspended:
                self._send_out_state_events[pipe_with_shortest_event_time].set()
                self._event_handlers_state[pipe_with_shortest_event_time] = EventHandlerState.out_state_started
                if self._event_handler_with_shortest_event_time.number_send_out_state_arguments:
                    pipe_with_shortest_event_time.send(
                        self._out_state_arguments_methods[self._event_handler_with_shortest_event_time](
                            *self._out_state_arguments[pipe_with_shortest_event_time]))
            if self._event_handlers_state[pipe_with_shortest_event_time] == EventHandlerState.out_state_started:
                out_state = pipe_with_shortest_event_time.recv()
                self._event_handlers_state[pipe_with_shortest_event_time] = EventHandlerState.idle
                self._out_states[self._event_handler_with_shortest_event_time] = out_state

            out_state = self._out_states[self._event_handler_with_shortest_event_time]
            pipe_with_shortest_event_time = self._pipes[self._event_handler_with_shortest_event_time]
            assert self._event_handlers_state[pipe_with_shortest_event_time] == EventHandlerState.idle

            # Commit out-state
            self._state_handler.insert_into_global_state(out_state)

            # Trash
            for event_handler in self._activator.get_trashable_events(
                    self._event_handler_with_shortest_event_time):
                if self._logger_enabled_for_debug:
                    self._logger.debug("Event handler trashed in the scheduler: {0}"
                                       .format(event_handler.__class__.__name__))
                self._scheduler.trash_event(event_handler)
                if event_handler in self._out_states:
                    del self._out_states[event_handler]
                pipe = self._pipes[event_handler]
                if self._event_handlers_state[pipe] == EventHandlerState.suspended:
                    self._event_handlers_state[pipe] = EventHandlerState.idle
                elif self._event_handlers_state[pipe] == EventHandlerState.out_state_started:
                    pipe.recv()
                    self._event_handlers_state[pipe] = EventHandlerState.idle

            # Other optional operations, mainly output
            self._mediating_methods.get(self._event_handler_with_shortest_event_time, lambda: None)()

    def post_run(self) -> None:
        """
        Call the post_run method of the base class and terminate all processes.

        After a base.exceptions.EndOfRun exception was raised in the run method of this class, this method is called by
        run.py and resume.py.
        """
        super().post_run()
        for process in self._os_processes:
            if process.is_alive():
                process.terminate()
                process.join()

    def update_logging(self) -> None:
        """
        Update the logging of this class and do the same for the scheduler and the state handler.

        This method is called in resume.py which might be run with a different logging level compared to the run which
        created the dump. This method then ensures that this class logs on the correct level.
        """
        self._logger = logging.getLogger(__name__)
        self._logger_enabled_for_debug = self._logger.isEnabledFor(logging.DEBUG)
        self._scheduler.update_logging()
