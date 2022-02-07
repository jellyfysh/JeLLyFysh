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
"""Module for the abstract Tagger class."""
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Iterable, List, Sequence, Tuple
from jellyfysh.activator.internal_state import InternalState
from jellyfysh.base.factory import get_alias
from jellyfysh.base.initializer import Initializer
from jellyfysh.base.node import Node
from jellyfysh.base.strings import to_snake_case
from jellyfysh.event_handler import EventHandler
from jellyfysh.state_handler.tree_state_handler import StateId


class Tagger(Initializer, metaclass=ABCMeta):
    """
    This class is a tagger which gets used in the TagActivator.

    A tagger provides a tag, a list of event handlers, and a method that generates the in-state identifiers for the
    send_event_time method of its event handlers. It further provides lists of tags, whose taggers will be asked for
    event handlers to run (creates property) or trash (trashes property) after an event of an event handler out of this
    tagger has been committed to the global state. Likewise, a tagger can also activate and deactivate other taggers via
    the activates and deactivates properties. If a tagger is deactivated, it will generate no in-state identifiers, and
    its event handlers will not compute any events.

    On initialization, a tagger receives a single event handler instance that should be able to compute the events for
    the in-states that this tagger produces. Since there can be more than one event with the same tag simultaneously
    in the scheduler, the event handler that was received on initialization is deepcopied in the initialize method of
    this class. The number of created event handlers is set on initialization. The number of event handlers inside a
    tagger should meet the maximum number of events with the given tag simultaneously in the scheduler.

    Note that the initialize method, which deepcopies the event handler that is stored in the
    self._event_handler_to_copy attribute, is called by the tag activator after the initialize_with_internal_states
    method. Taggers might initialize their event handler in the self._event_handler_to_copy attribute with information
    about the internal state (e.g., with a cell system). By this, the (possibly numerically complex) initialization
    of an event handler has to be only run once, and the already initialized event handler is deepcopied to create
    all event handlers that are available to this tagger.

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
        # Do this before calling super().__init__(), because Initializer class will overwrite public methods.
        self._deactivated_yield_identifiers_send_event_time = lambda separated_active_units: iter(())
        self._activated_yield_identifiers_send_event_time = self.yield_identifiers_send_event_time
        super().__init__()
        # If no tag is given, we want to use the alias of the .ini file which is included in the __class__.__name__.
        # property in the factory. get_alias() extracts this alias
        self._tag = tag if tag is not None else to_snake_case(get_alias(self.__class__.__name__))
        self._creates = create
        self._trashes = trash
        self._number_event_handlers = number_event_handlers
        self._event_handler_to_copy = event_handler
        self._event_handlers = None

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

    def initialize_with_internal_states(self, internal_states: Sequence[InternalState]) -> None:
        """
        Initialize the tagger based on the initialized internal states.

        Adds a second initialize method relevant to the base Initializer class. This method is called once in the
        beginning of the run by the tag activator. However, this method does not call the initialize method of the
        Initializer class. Therefore, other public methods of this class can still not be called without raising an
        error after this method has been used. To finalize the initialization of this class, use the initialize method
        (which should be called after this method).

        Per default, this method does nothing. Some taggers might need to gain access to their internal state (if
        they need one), or their event handler in the self._event_handler_to_copy attribute should be initialized (for
        example, with a cell system). Such taggers must extend this method. The subsequent call of the initialize method
        by the tag activator will then deepcopy the initialized event handler to create all event handlers for this
        tagger.

        Parameters
        ----------
        internal_states : Sequence[activator.internal_state.InternalState]
            Sequence of all initialized internal states in the activator.
        """
        pass

    def initialize(self) -> None:
        """
        Initialize the tagger by deepcopying the single instance of the event handler which was received on
        initialization of this class.

        Extends the initialize method of the Initializer class. Use this method once in the beginning of the run to
        initialize the tagger. Only after a call of this method, other public methods of this class can be called
        without raising an error.

        This method creates all event handlers of this tagger (accessible via the get_event_handlers method) by
        deepcopying the self._event_handler_to_copy attribute. The number of event handlers that are created was set
        on initialization of this class.

        Note that this method is called after the initialize_with_internal_states method. There, the event handler
        that will be copied might have been initialized with information of the internal state. By this, the
        initialization of the event handler is only done once.
        """
        super().initialize()
        self._event_handlers = [self._event_handler_to_copy] + [deepcopy(self._event_handler_to_copy)
                                                                for _ in range(1, self._number_event_handlers)]

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

    def get_event_handlers(self) -> List[EventHandler]:
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
        Generate in-state identifiers for the send_event_time method of this tagger's event handlers.

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
