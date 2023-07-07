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
"""Module for the SingleProcessMediator class."""
from copy import copy
import logging
from jellyfysh.activator import Activator
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.event_handler import EventHandlerWithUnconfirmedEvents
from jellyfysh.input_output_handler import InputOutputHandler
from jellyfysh.mediator.mediator import Mediator
from jellyfysh.state_handler import StateHandler
from jellyfysh.scheduler import Scheduler


class SingleProcessMediator(Mediator):
    """
    This class implements a mediator where the whole application runs in a single process.

    All other used classes of JF then only provide public methods which are used by this mediator.
    On one leg of the continuous time-evolution, the mediator goes through nine steps:
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
                 activator: Activator) -> None:
        """
        The constructor of the SingleProcessMediator class.

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
        """
        self._logger = logging.getLogger(__name__)
        self._logger_enabled_for_debug = self._logger.isEnabledFor(logging.DEBUG)
        log_init_arguments(self._logger.debug, self.__class__.__name__,
                           input_output_handler=input_output_handler.__class__.__name__,
                           state_handler=state_handler.__class__.__name__,
                           scheduler=scheduler.__class__.__name__,
                           activator=activator.__class__.__name__)
        # Obtain nodes from input handler (via the input-output handler) and hands them over to the state handler
        state_handler.initialize(input_output_handler.read())
        super().__init__(input_output_handler, state_handler, scheduler, activator)
        self._last_velocity = None
        self._last_time_stamp = None
        self._total_time = 0.0
        self._total_distance = 0.0
        self._number_events = {event_handler: 0 for event_handler in self._event_handlers_list}

    def run(self) -> None:
        """
        Loop over the legs of the continuous-time evolution of the event-chain Monte Carlo algorithm.

        This method is called by run.py and resume.py. The loop should only be interrupted when a
        base.exceptions.EndOfRun exception is raised, which is caught in the scripts.
        """
        while True:
            # Extract active global state
            active_global_state = self._state_handler.extract_active_global_state()

            if len(active_global_state) > 0:
                assert len(active_global_state) == 1
                assert len(active_global_state[0].children) == 1
                current_velocity = active_global_state[0].children[0].value.velocity
                current_time_stamp = active_global_state[0].children[0].value.time_stamp
                if self._last_velocity is None:
                    print(f"Initial time stamp: {current_time_stamp}")
                    self._last_velocity = copy(current_velocity)
                    self._last_time_stamp = copy(current_time_stamp)
                else:
                    time_difference = current_time_stamp - self._last_time_stamp
                    self._total_distance += (sum(v * v for v in self._last_velocity) ** 0.5) * time_difference
                    self._total_time += time_difference
                    self._last_velocity = copy(current_velocity)
                    self._last_time_stamp = copy(current_time_stamp)
            # Fetch event handlers to activate
            event_handlers_in_state_dictionary = self._activator.get_event_handlers_to_run(
                active_global_state, self._event_handler_with_shortest_event_time)
            if self._logger_enabled_for_debug:
                self._logger.debug(
                    "Event handlers that will be run with their in-state identifiers: {0}"
                    .format([(event_handler.__class__.__name__, in_state_identifier)
                             for event_handler, in_state_identifier in event_handlers_in_state_dictionary.items()]))

            # Fetch in-states
            for event_handler, in_state_identifiers in event_handlers_in_state_dictionary.items():
                if in_state_identifiers is not None:
                    event_handlers_in_state_dictionary[event_handler] = [
                        self._state_handler.extract_from_global_state(identifier)
                        for identifier in in_state_identifiers]

            for event_handler, in_state in event_handlers_in_state_dictionary.items():
                # Request candidate event times
                if event_handler.number_send_event_time_arguments:
                    assert in_state is not None
                    returned = event_handler.send_event_time(in_state)
                else:
                    assert in_state is None
                    returned = event_handler.send_event_time()

                try:
                    event_time, self._out_state_arguments[event_handler] = returned
                except TypeError:
                    event_time, self._out_state_arguments[event_handler] = returned, []

                self._scheduler.push_event(event_time, event_handler)
                if self._logger_enabled_for_debug:
                    self._logger.debug("Pushed candidate event time to the scheduler: {0} ({1})"
                                       .format(event_time, event_handler.__class__.__name__))

            while True:
                # Request shortest time
                self._event_handler_with_shortest_event_time = self._scheduler.get_succeeding_event()
                if self._logger_enabled_for_debug:
                    self._logger.debug("Event handler which created the event with the shortest event time: {0}"
                                       .format(self._event_handler_with_shortest_event_time.__class__.__name__))

                # Request out-state
                if self._event_handler_with_shortest_event_time.number_send_out_state_arguments:
                    out_state = self._event_handler_with_shortest_event_time.send_out_state(
                        *self._out_state_arguments_methods[self._event_handler_with_shortest_event_time](
                            *self._out_state_arguments[self._event_handler_with_shortest_event_time]))
                else:
                    out_state = self._event_handler_with_shortest_event_time.send_out_state()

                # Resend event time if event is unconfirmed (i.e., out_state is None)
                if out_state is None:
                    assert isinstance(self._event_handler_with_shortest_event_time, EventHandlerWithUnconfirmedEvents)
                    returned = self._event_handler_with_shortest_event_time.resend_event_time()
                    try:
                        event_time, self._out_state_arguments[self._event_handler_with_shortest_event_time] = returned
                    except TypeError:
                        event_time, self._out_state_arguments[self._event_handler_with_shortest_event_time] = (returned,
                                                                                                               [])
                    self._scheduler.trash_event(self._event_handler_with_shortest_event_time)
                    self._scheduler.push_event(event_time, self._event_handler_with_shortest_event_time)
                else:
                    break
            self._number_events[self._event_handler_with_shortest_event_time] += 1
            # Commit out-state
            self._state_handler.insert_into_global_state(out_state)

            # Trash
            for event_handler in self._activator.get_trashable_events(
                    self._event_handler_with_shortest_event_time):
                if self._logger_enabled_for_debug:
                    self._logger.debug("Event handler trashed in the scheduler: {0}"
                                       .format(event_handler.__class__.__name__))
                self._scheduler.trash_event(event_handler)

            # Other optional operations, mainly output
            self._mediating_methods.get(self._event_handler_with_shortest_event_time, lambda: None)()

    def update_logging(self) -> None:
        """
        Update the logging of this class and do the same for the scheduler and the state handler.

        This method is called in resume.py which might be run with a different logging level compared to the run which
        created the dump. This method then ensures that this class logs on the correct level.
        """
        self._logger = logging.getLogger(__name__)
        self._logger_enabled_for_debug = self._logger.isEnabledFor(logging.DEBUG)
        self._scheduler.update_logging()
        self._state_handler.update_logging()
