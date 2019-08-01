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
"""Module for the TagActivator class."""
import copy
import logging
from typing import Any, Dict, List, Sequence, Union
from base.exceptions import ConfigurationError, TagActivatorError
from base.logging import log_init_arguments
from event_handler import EventHandler
from event_handler.abstracts import StartOfRunEventHandler
from .activator import Activator
from .internal_state import InternalState
from .tagger import Tagger


class TagActivator(Activator):
    """
    The tag activator uses taggers to implement the abstract methods of an activator.

    Each tagger provides a tag, a list of event handlers, a method to generate the global state identifiers of the
    in-states needed in the send_event_time method of its event handlers, and lists of tags, whose taggers will be asked
    for event handlers to run (creates list) or trash (trashes list) after an event of an event handler out of the
    tagger has been committed to the global state. The identifiers can be of Any type. The precise format is
    specified by the used state handler.
    The create and trash action can be overruled with the activated and deactivated taggers. If a tagger is deactivated,
    it will generate no in-state identifiers.
    For each tagger, this class stores internally which event handlers are currently 'running' (meaning that the
    candidate event of them is stored in the scheduler) and 'not running'.
    When first asked for new event handlers to run, this activator will specifically return the StartOfRunEventHandler
    together with its in-state identifiers.
    """

    def __init__(self, taggers: Sequence[Tagger], internal_states: Sequence[InternalState] = ()) -> None:
        """
        The constructor of the TagActivator class.

        Parameters
        ----------
        taggers : Sequence[activator.tagger.Tagger]
            Sequence of taggers involved.
        internal_states : Sequence[activator.internal_state.InternalState]
            Sequence of internal states involved.

        Raises
        ------
        base.exceptions.ConfigurationError
            If not exactly one StartOfRunEventHandler has been provided in all the taggers.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           taggers=[tagger.__class__.__name__ for tagger in taggers],
                           internal_states=[internal_state.__class__.__name__ for internal_state in internal_states])
        self._taggers = taggers
        super().__init__(sum((tagger.event_handlers for tagger in taggers), []), internal_states)

        self._event_handler_tagger_dictionary = {event_handler: tagger
                                                 for tagger in self._taggers for event_handler in tagger.event_handlers}
        self._running_event_handlers = {tagger: [] for tagger in self._taggers}
        self._not_running_event_handlers = {tagger: copy.copy(tagger.event_handlers) for tagger in self._taggers}

        # Search the one and only instance of StartOfRunEventHandler
        self._start_of_run_event_handler = None
        for event_handler in self._event_handlers:
            if isinstance(event_handler, StartOfRunEventHandler):
                if self._start_of_run_event_handler is None:
                    self._start_of_run_event_handler = event_handler
                else:
                    raise ConfigurationError("Please provide only one StartOfRunEventHandler!")
        if self._start_of_run_event_handler is None:
            raise ConfigurationError("An StartOfRunEventHandler is required to run the program!")

        self._create_taggers = self._build_tagger_dictionary("creates")
        self._trash_taggers = self._build_tagger_dictionary("trashes")
        self._activate_taggers = self._build_tagger_dictionary("activates")
        self._deactivate_taggers = self._build_tagger_dictionary("deactivates")

    def _build_tagger_dictionary(self, list_attribute: str) -> Dict[Tagger, List[Tagger]]:
        tag_to_tagger_dictionary = {tagger.tag: tagger for tagger in self._taggers}
        tagger_dictionary = {tagger: [] for tagger in self._taggers}
        for tagger in self._taggers:
            for tag in getattr(tagger, list_attribute):
                if tag not in tag_to_tagger_dictionary.keys():
                    raise ConfigurationError("Given tag '{0}' in the list attribute '{1}'"
                                             " of the tagger '{2}' does not exist as a tagger!"
                                             .format(tag, list_attribute, tagger.tag))
                tagger_dictionary[tagger].append(tag_to_tagger_dictionary[tag])
        return tagger_dictionary

    def initialize(self, extracted_global_state: Any) -> None:
        """
        Initialize all the taggers based on the extracted global state from the state handler and all internal states.

        Extends the initialize method of the abstract Activator class. This method is called once in the beginning of
        the run by the mediator. Only after a call of this method, other public methods of this class can be called
        without raising an error.
        The internal states get initialized in the super().initialize(extracted_global_state) call.
        The precise format of the extracted_active_global_state is specified by the used state handler. Since it is
        just passed through to the taggers and the internal states, only these need to be implemented for different
        versions of state handlers.

        Parameters
        ----------
        extracted_global_state : Any
            The full extracted global state from the state handler.
        """
        super().initialize(extracted_global_state)
        for tagger in self._taggers:
            tagger.initialize(extracted_global_state, self._internal_states)

    def get_event_handlers_to_run(self, extracted_active_global_state: Any, preceding_event_handler: EventHandler) \
            -> Dict[EventHandler, Union[Sequence[Any], None]]:
        """
        Return the event handlers to run together with the sequence of in-state identifiers based on the extracted
        active global state and the preceding event handler which committed an event to the global state last.

        The sequence of the in-state identifiers can be replaced by None, if the event handler needs no in-state to
        calculate its candidate event time.
        This method can be called only once at the beginning, when preceding_event_handler is None. Later calls will be
        redirected to _get_event_handlers_to_run_update.
        For this call, the tagger containing the StartOfRunEventHandler will be asked to activate and deactivate other
        taggers. Then, the StartOfRunEventHandler is returned together with its in-state identifiers.
        Only 'not running' event handlers can be returned here. Returned event handlers are then stored internally as
        'running'.
        Overwrites the get_event_handlers_to_run method of the abstract Activator class.
        The precise format of the extracted_active_global_state is specified by the used state handler. Since it is
        just passed through to the taggers and the internal states, only these need to be implemented for different
        versions of state handlers.

        Parameters
        ----------
        extracted_active_global_state : Any
            The extracted active global state information.
        preceding_event_handler : event_handler.EventHandler
            The event handler which committed the last event to the global state.

        Returns
        -------
        Dict[event_handler.event_handler.EventHandler: Sequence[Any] or None]
            The map from the event handlers to run onto their in-state identifiers.

        Raises
        ------
        AssertionError
            If the preceding_event_handler is not None or the length of the active_units sequence is unequal zero.
        base.exceptions.TagActivatorError
            If the list of not running event handlers of a tagger is empty but an event handler of this tagger should
            be run.
        """
        assert preceding_event_handler is None
        event_handlers_identifiers_dictionary = {}
        start_of_run_tagger = self._event_handler_tagger_dictionary[self._start_of_run_event_handler]
        for tagger in self._activate_taggers[start_of_run_tagger]:
            tagger.activate()
        for tagger in self._deactivate_taggers[start_of_run_tagger]:
            tagger.deactivate()

        for identifiers_send_event_time in start_of_run_tagger.yield_identifiers_send_event_time(
                extracted_active_global_state):
            try:
                event_handler = self._not_running_event_handlers[start_of_run_tagger].pop()
            except IndexError:
                raise TagActivatorError("Not-running event handler list of the tagger {0} is empty. "
                                        "Increase the number of event handlers of this tagger."
                                        .format(start_of_run_tagger.__class__.__name__))

            self._running_event_handlers[start_of_run_tagger].append(event_handler)
            event_handlers_identifiers_dictionary[event_handler] = identifiers_send_event_time
        # noinspection PyAttributeOutsideInit
        self.get_event_handlers_to_run = self._get_event_handlers_to_run_update
        return event_handlers_identifiers_dictionary

    def _get_event_handlers_to_run_update(
            self, extracted_active_global_state: Any, preceding_event_handler: EventHandler) \
            -> Dict[EventHandler, Union[Sequence[Any], None]]:
        """
        Return the event handlers to run together with the in-state identifiers based on the extracted active global
        state and the preceding event handler which committed an event to the global state last.

        The sequence of the in-state identifiers can be replaced by None, if the event handler needs no in-state to
        calculate its candidate event time.
        This method replaces the get_event_handlers_to_run_method after the first call, where preceding_event_handler
        is None.
        The method implements four steps:
        1. Identify the tagger of the preceding event handler (preceding_event_tagger).
        2. Activate and deactivate taggers based on the preceding_event_tagger.
        3. Update the internal state based on the active global state information.
        4. Ask each tagger in the creates list of the preceding_event_tagger for the event handlers and their in-states
        identifiers.
        Only 'not running' event handlers can be returned here. Returned event handlers are then stored internally as
        'running'.
        The precise format of the extracted_active_global_state is specified by the used state handler. Since it is
        just passed through to the taggers and the internal states, only these need to be implemented for different
        versions of state handlers.

        Parameters
        ----------
        extracted_active_global_state : Any
            The extracted active global state information.
        preceding_event_handler : event_handler.EventHandler
            The event handler which committed the preceding event to the global state.

        Returns
        -------
        Dict[event_handler.event_handler.EventHandler: Sequence[Any] or None]
            The map from the event handlers to run onto their in-state identifiers.

        Raises
        ------
        base.exceptions.TagActivatorError
            If the list of not running event handlers of a tagger is empty but an event handler of this tagger should
            be run.
        """
        preceding_event_tagger = self._event_handler_tagger_dictionary[preceding_event_handler]
        for tagger in self._activate_taggers[preceding_event_tagger]:
            tagger.activate()
        for tagger in self._deactivate_taggers[preceding_event_tagger]:
            tagger.deactivate()
        event_handlers_identifiers_dictionary = {}

        for internal_state in self._internal_states:
            internal_state.update(extracted_active_global_state)

        for tagger in self._create_taggers[preceding_event_tagger]:
            for identifiers_send_event_time in tagger.yield_identifiers_send_event_time(extracted_active_global_state):
                try:
                    event_handler = self._not_running_event_handlers[tagger].pop()
                except IndexError:
                    raise TagActivatorError("Not-running event handler list of the tagger {0} is empty. "
                                            "Increase the number of event handlers of this tagger."
                                            .format(tagger.__class__.__name__))
                self._running_event_handlers[tagger].append(event_handler)
                event_handlers_identifiers_dictionary[event_handler] = identifiers_send_event_time
        return event_handlers_identifiers_dictionary

    def get_trashable_events(self, preceding_event_handler: EventHandler) -> List[EventHandler]:
        """
        Return the event handlers whose events should be trashed in the scheduler.

        The method asks each tagger in the trashes list of the preceding_event_tagger for the event handlers to trash.
        The preceding_event_handler should also be returned in this method. Returned event handlers are stored
        internally as 'not running' so that they can be returned by get_new_event_handlers_to_run again.
        Overwrites the get_trashable_events method of the abstract Activator class.

        Parameters
        ----------
        preceding_event_handler : event_handler.EventHandler
            The event handler which committed the preceding event to the global state.

        Returns
        -------
        List[event_handler.EventHandler]
            The sequence of event handlers whose events should be trashed in the scheduler.
        """
        trashable_events = []
        for tagger in self._trash_taggers[self._event_handler_tagger_dictionary[preceding_event_handler]]:
            trashable_events += self._running_event_handlers[tagger]
            self._not_running_event_handlers[tagger] += self._running_event_handlers[tagger]
            self._running_event_handlers[tagger] = []
        assert preceding_event_handler in trashable_events
        return trashable_events

    def get_info_internal_state(self, event_handler_asking: EventHandler, identifier_in_internal_state: Any) -> Any:
        """
        Return the global state identifier associated to the identifier of the internal state.

        This method provides the interface to get information of the internal state. In the mediator, it is called after
        an event handler has sent its candidate event time. To calculate the out-state, the event handler needs
        information of the internal state based on an identifier (for example the cell veto event handler samples a
        target cell together with the candidate event time, and in order to compute the out-state it needs the
        occupancy of this target cell).
        Overwrites the get_info_internal_state method of the abstract Activator class.
        The precise format of the internal state identifier and the returned global state identifier is specified by the
        internal state itself and the used state handler.

        Parameters
        ----------
        event_handler_asking : event_handler.EventHandler
            The event handler that requests internal state information.
        identifier_in_internal_state : Any
            The internal state identifier.

        Returns
        -------
        Any
            The global state identifier associated with the internal state identifier.
        """
        return self._event_handler_tagger_dictionary[event_handler_asking].internal_state[identifier_in_internal_state]
