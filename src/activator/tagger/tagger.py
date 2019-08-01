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
"""Module for the abstract Tagger class."""
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Iterable, List, Sequence, Tuple
from activator.internal_state import InternalState
from base.factory import get_alias
from base.initializer import Initializer
from base.node import Node
from base.strings import to_snake_case
from event_handler import EventHandler
from state_handler.tree_state_handler import StateId


class Tagger(Initializer, metaclass=ABCMeta):
    """
    This class is a tagger which gets used in the TagActivator.

    A tagger provides a tag, a list of event handlers, a method to generate the in-state identifiers for the
    send_event_time method of its event handlers, and lists of tags, whose taggers will be asked for event handlers
    to run (creates list) or trash (trashes list) after an event of an event handler out of the tagger has been
    committed to the global state.
    A tagger can also activate and deactivate other taggers. If a tagger is deactivated, it will generate no in-state
    identifiers.
    The number of event handlers inside a tagger should meet the maximum number of events with the given tag
    simultaneously in the scheduler.

    This class is designed to work together with the TreeStateHandler. The in-state identifiers are then a sequence of
    tuples of integers, where the tuples can have different lengths. Each tuple in the sequence specifies a particle in
    the global state (see StateId in state_handler.tree_state_handler.py).
    """

    def __init__(self, create: Sequence[str], trash: Sequence[str], event_handler: EventHandler,
                 number_event_handlers: int, tag: str = None, activate: Sequence[str] = (),
                 deactivate: Sequence[str] = ()) -> None:
        """
        The constructor of the abstract tagger class.

        Parameters
        ----------
        create : Sequence[str]
            Sequence of tags to create after an event handler of this tagger has committed an event to the global state.
        trash : Sequence[str]
            Sequence of tags to trash after an event handler of this tagger has committed an event to the global state.
        event_handler : event_handler.EventHandler
            A single event handler instance.
        number_event_handlers : int
            Number of event handlers to prepare. The tagger will deepcopy the given event handler instance to create
            this number of event handlers.
        tag : str or None, optional
            Tag used in all four lists (also of other taggers). If None, the class name (or the alias set in the
            factory) will be used as the tag.
        activate : Sequence[str], optional
            Sequence of tags to activate after an event handler of this tagger has committed an event to the global
            state.
        deactivate : Sequence[str], optional
            Sequence of tags to deactivate after an event handler of this tagger has committed an event to the global
            state.
        """
        self._activates = activate
        self._deactivates = deactivate
        # Do this before calling super().__init__(), because Initializer class will overwrite public methods
        self._deactivated_yield_identifiers_send_event_time = lambda separated_active_units: iter(())
        self._activated_yield_identifiers_send_event_time = self.yield_identifiers_send_event_time
        super().__init__()
        # If no tag is given, we want to use the alias of the .ini file which is included in the __class__.__name__
        # property in the factory. get_alias() extracts this alias
        self._tag = tag if tag is not None else to_snake_case(get_alias(self.__class__.__name__))
        self._creates = create
        self._trashes = trash
        self._event_handlers = [event_handler] + [deepcopy(event_handler) for _ in range(1, number_event_handlers)]

    @property
    def tag(self) -> str:
        """
        Return the tag.

        Returns
        -------
        str
            The tag.
        """
        return self._tag

    def initialize(self, extracted_global_state: Sequence[Node], internal_states: Sequence[InternalState]) -> None:
        """
        Initialize the taggers based on the full extracted global state and all initialized internal states.

        Extends the initialize method of the Initializer class. Use this method once in the beginning of the run to
        initialize the tagger. Only after a call of this method, other public methods of this class can be called
        without raising an error.
        Per default, this method does nothing except calling the initialize method of the Initializer. Some taggers
        might need to gain access to their internal state (if they need one), or their event handlers should be
        initialized (for example with a cell system). Such taggers must extend this method.
        The full extracted global state is given as a sequence of cnodes of all root nodes stored in the global state.

        Parameters
        ----------
        extracted_global_state : Sequence[base.node.Node]
            The full extracted global state from the state handler.
        internal_states : Sequence[activator.internal_state.InternalState]
            Sequence of all initialized internal states in the activator.
        """
        super().initialize()

    @property
    def creates(self) -> Sequence[str]:
        """
        Return the tags of taggers which should be created after an event handler of this tagger has committed an event
        to the global state.

        Returns
        -------
        Sequence[str]
            The tags to create.
        """
        return self._creates

    @property
    def trashes(self) -> Sequence[str]:
        """
        Return the tags of taggers which should be trashed after an event handler of this tagger has committed an event
        to the global state.

        Returns
        -------
        Sequence[str]
            The tags to trash.
        """
        return self._trashes

    @property
    def activates(self) -> Sequence[str]:
        """
        Return the tags of taggers which should be activated after an event handler of this tagger has committed an
        event to the global state.

        Returns
        -------
        Sequence[str]
            The tags to activate.
        """
        return self._activates

    @property
    def deactivates(self) -> Sequence[str]:
        """
        Return the tags of taggers which should be deactivated after an event handler of this tagger has committed an
        event to the global state.

        Returns
        -------
        Sequence[str]
            The tags to deactivate.
        """
        return self._deactivates

    @property
    def event_handlers(self) -> List[EventHandler]:
        """
        Return all event handlers of this tagger.

        Returns
        -------
        List[event_handler.EventHandler]
            List of all event handlers.
        """
        return self._event_handlers

    @abstractmethod
    def yield_identifiers_send_event_time(
            self, extracted_active_global_state: Sequence[Node]) -> Iterable[Tuple[StateId, ...]]:
        """
        Generate in-state identifiers for the send_event_time method of this taggers event handlers.

        The in-state identifiers will be transformed into real in-states using the state handler via the mediator.
        The active global state is given by a sequence of root cnodes where each cnode branch only contains active
        units. The in-state identifiers should be generated as a tuple of global state identifiers. If None is
        generated, no in-state will be constructed in the state handler.

        Parameters
        ----------
        extracted_active_global_state : Sequence[base.node.Node]
            The active global state information.

        Yields
        ------
        Tuple[activator.tag_activator.StateId, ...] or None
            The global state in-state identifiers.
        """
        raise NotImplementedError

    def deactivate(self) -> None:
        """
        Deactivate this tagger.

        This tagger is deactivated by replacing the yield_identifiers_send_event_time method by an emtpy iterator.
        """
        # noinspection PyAttributeOutsideInit
        self.yield_identifiers_send_event_time = self._deactivated_yield_identifiers_send_event_time

    def activate(self) -> None:
        """
        Activate this tagger.

        This tagger is activated by enabling access to the implemented yield_identifiers_send_event_time method.
        """
        # noinspection PyAttributeOutsideInit
        self.yield_identifiers_send_event_time = self._activated_yield_identifiers_send_event_time
