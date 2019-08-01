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
"""Module for the ActiveRootUnitInStateTagger class."""
import logging
from typing import Iterable, Sequence, Tuple
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.node import Node
from event_handler import EventHandler
import setting
from state_handler.tree_state_handler import StateId
from .tagger import Tagger


class ActiveRootUnitInStateTagger(Tagger):
    """
    Tagger for an event handler which needs the active units corresponding to the root nodes as the in-state.

    An example is the RootLeafUnitActiveSwitcher.
    This class is designed to work together with the TreeStateHandler. The in-state identifiers are then a sequence of
    tuples of integers, where the tuples can have different lengths. Each tuple in the sequence specifies a particle in
    the global state (see StateId in state_handler.tree_state_handler.py).
    """

    def __init__(self, create: Sequence[str], trash: Sequence[str], event_handler: EventHandler,
                 number_event_handlers: int = 1, tag: str = None, activate: Sequence[str] = (),
                 deactivate: Sequence[str] = ()) -> None:
        """
        The constructor of the ActiveRootUnitInStateTagger class.

        Parameters
        ----------
        create : Sequence[str]
            Sequence of tags to create after an event handler of this tagger has committed an event to the global state.
        trash : Sequence[str]
            Sequence of tags to trash after an event handler of this tagger has committed an event to the global state.
        event_handler : event_handler.EventHandler
            A single event handler instance.
        number_event_handlers : int, optional
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

        Raises
        ------
        base.exceptions.ConfigurationError
            If no composite point objects are involved in the run (setting.number_of_node_levels not > 1).
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           event_handler=event_handler.__class__.__name__, number_event_handlers=number_event_handlers,
                           create=create, trash=trash, activate=activate, deactivate=deactivate, tag=tag)
        super().__init__(create, trash, event_handler, number_event_handlers=number_event_handlers, tag=tag,
                         activate=activate, deactivate=deactivate)
        if not setting.number_of_node_levels > 1:
            raise ConfigurationError("Tagger {0} can only be used when composite point objects are involved in the"
                                     "simulation (setting.number_of_node_levels > 1)."
                                     .format(self.__class__.__name__))

    def yield_identifiers_send_event_time(
            self, extracted_active_global_state: Sequence[Node]) -> Iterable[Tuple[StateId]]:
        """
        Generate in-state identifiers for the send_event_time method of this taggers event handlers.

        The in-state identifiers will be transformed into real in-states using the state handler via the mediator.
        The active global state is given by a sequence of root cnodes where each cnode branch only contains active
        units. The in-state identifiers are generated as a tuple of global state identifiers. If None is generated, no
        in-state will be constructed in the state handler.

        The generated in-state is just the identifier of the active unit whose identifier has length 1.

        Parameters
        ----------
        extracted_active_global_state : Sequence[base.node.Node]
            The active global state information.

        Yields
        ------
        Tuple[activator.tag_activator.StateId]
            The global state in-state identifiers.
        """
        for root_cnode in extracted_active_global_state:
            yield (root_cnode.value.identifier,)
