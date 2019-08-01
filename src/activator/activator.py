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
"""Module for the abstract Activator class."""
from abc import ABCMeta, abstractmethod
from typing import Any, Mapping, Sequence, Union
from base.initializer import Initializer
from event_handler import EventHandler
from .internal_state import InternalState


class Activator(Initializer, metaclass=ABCMeta):
    """
    Abstract class for a general activator to be used in the mediator.

    In JF, the activator decides which event handlers are run (by the mediator) after an event has been committed to
    the global state. The global state identifiers of the in-states needed by these event handlers to compute their next
    candidate event time is also provided by the activator. The identifiers can be of Any type. The precise format is
    specified by the used state handler.
    The activator also decides which events to trash in the scheduler after an event has been committed.
    Finally, it maintains internal states (for example cell-occupancy systems). These internal state should be
    connected to the event handlers whose in-states are based on these internal states.
    """

    def __init__(self, event_handlers: Sequence[EventHandler], internal_states: Sequence[InternalState]) -> None:
        """
        The constructor of the abstract Activator class.

        Parameters
        ----------
        event_handlers : Sequence[event_handler.EventHandler]
            The sequence of all event handlers.
        internal_states : Sequence[activator.internal_state.InternalState]
            The sequence of all internal states.
        """
        super().__init__()
        self._event_handlers = event_handlers
        self._internal_states = internal_states

    def initialize(self, extracted_global_state: Any) -> None:
        """
        Initialize all the internal states based on the full extracted global state.

        Extends the initialize method of the Initializer class. This method is called once in the beginning of the run
        by the mediator. Only after a call of this method, other public methods of this class can be
        called without raising an error.
        The precise format of the argument is specified by the used state handler.

        Parameters
        ----------
        extracted_global_state : Any
            The full extracted global state from the state handler.
        """
        super().initialize()
        for internal_state in self._internal_states:
            internal_state.initialize(extracted_global_state)

    @property
    def event_handlers(self) -> Sequence[EventHandler]:
        """
        Return the sequence of all event handlers.

        Returns
        -------
        Sequence[event_handler.EventHandler]
            Sequence of all created event handlers.
        """
        return self._event_handlers

    @abstractmethod
    def get_event_handlers_to_run(self, extracted_active_global_state: Any, preceding_event_handler: EventHandler) \
            -> Mapping[EventHandler, Union[Sequence[Any], None]]:
        """
        Return the event handlers to run together with the sequence of in-state identifiers based on the extracted
        active global state and the preceding event handler which committed an event to the global state last.

        The in-state identifiers of the global state are transformed into global state information in the mediator using
        the state handler. The precise format of the extracted active global state is specified by the used state
        handler.
        The sequence of the in-state identifiers can be replaced by None, if the event handler needs no in-state to
        calculate its candidate event time.
        For the very first call of this method, preceding_event_handler is None.
        If there are internal states stored in the activator, this method should also keep the internal state consistent
        with the global state.

        Parameters
        ----------
        extracted_active_global_state : Any
            The extracted active global state information.
        preceding_event_handler : event_handler.EventHandler
            The event handler which committed the preceding event to the global state.

        Returns
        -------
        Mapping[event_handler.event_handler.EventHandler: Sequence[Any] or None]
            The map from the event handlers to run onto their in-state identifiers.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trashable_events(self, preceding_event_handler: EventHandler) -> Sequence[EventHandler]:
        """
        Return the event handlers whose events should be trashed in the scheduler.

        The preceding_event_handler should also be returned in this method.

        Parameters
        ----------
        preceding_event_handler : event_handler.EventHandler
            The event handler which committed the preceding event to the global state.

        Returns
        -------
        Sequence[event_handler.EventHandler]
            The sequence of event handlers whose events should be trashed in the scheduler.
        """
        raise NotImplementedError

    @abstractmethod
    def get_info_internal_state(self, event_handler_asking: EventHandler, identifier_in_internal_state: Any) -> Any:
        """
        Return the global state identifier associated to the identifier of the internal state.

        This method provides the interface to get information of the internal state. In the mediator, it is called after
        an event handler has sent its candidate event time. To calculate the out-state, the event handler needs
        information of the internal state based on an identifier (for example the cell veto event handler samples a
        target cell together with the candidate event time, and in order to compute the out-state it needs the
        occupancy of this target cell).

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
        raise NotImplementedError
