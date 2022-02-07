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
"""Module for the abstract TaggerWithInternalState class."""
from abc import ABCMeta
from typing import Sequence
from jellyfysh.activator.internal_state import InternalState
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.factory import get_alias
from jellyfysh.event_handler import EventHandler
from jellyfysh.base.strings import to_snake_case
from .tagger import Tagger


class TaggerWithInternalState(Tagger, metaclass=ABCMeta):
    """
    Abstract class for Taggers which use an internal state.

    Some taggers need access to an internal state in order to generate in-state identifiers for their event handlers.
    This class then already extends the initialize method to extract the correct internal state from all initialized
    internal states.

    This class is designed to work together with the TreeStateHandler. The in-state identifiers are then a sequence of
    tuples of integers, where the tuples can have different lengths. Each tuple in the sequence specifies a particle in
    the global state (see StateId in state_handler.tree_state_handler.py).
    """

    def __init__(self, create: Sequence[str], trash: Sequence[str], event_handler: EventHandler,
                 number_event_handlers: int, internal_state_label: str, tag: str = None) -> None:
        """
        The constructor of the abstract TaggerWithInternalState class.

        Note that the activate and deactivate sequences are always empty for a tagger of this kind.

        The label of the internal state should be the class name of the internal state or the alias introduced in the
        ini file set by the factory.

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
        internal_state_label : str
            The label of the internal state this tagger wants to use.
        tag : str or None, optional
            Tag used in all four lists (also of other taggers). If None, the class name (or the alias set in the
            factory) will be used as the tag.

        """
        super().__init__(create, trash, event_handler, number_event_handlers=number_event_handlers, tag=tag)
        self._internal_state_label = internal_state_label
        self._internal_state = None

    def initialize_with_internal_states(self, internal_states: Sequence[InternalState]) -> None:
        """
        Initialize the tagger based on the initialized internal states.

        Adds a second initialize method relevant to the base Initializer class. This method is called once in the
        beginning of the run by the tag activator. However, this method does not call the initialize method of the
        Initializer class. Therefore, other public methods of this class can still not be called without raising an
        error after this method has been used. To finalize the initialization of this class, use the initialize method
        (which should be called after this method).

        This method extracts the internal state corresponding to the internal state label out of the sequence of
        internal states. If a tagger needs to further initialize its event handler in the self._event_handler_to_copy
        attribute (for example, with a cell system), it should extend this method.

        The subsequent call of the initialize method will then deepcopy the initialized event handler to create all
        event handlers for this tagger.

        Parameters
        ----------
        internal_states : Sequence[activator.internal_state.InternalState]
            Sequence of all initialized internal states in the activator.
        """
        super().initialize_with_internal_states(internal_states)
        relevant_internal_state = [state for state in internal_states
                                   if to_snake_case(get_alias(state.__class__.__name__)) == self._internal_state_label]
        if len(relevant_internal_state) == 0:
            raise ConfigurationError("The given internal state label '{0}' does not exist!"
                                     .format(self._internal_state_label))
        if len(relevant_internal_state) > 1:
            raise ConfigurationError("The given internal state label '{0}' exists more than once!"
                                     .format(self._internal_state_label))
        self._internal_state = relevant_internal_state[0]

    @property
    def internal_state(self) -> InternalState:
        """
        Return the internal state that this tagger uses.

        Returns
        -------
        activator.internal_state.InternalState
            The internal state.
        """
        return self._internal_state
