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
"""Module for the ActiveGlobalStateInStateTagger class."""
import logging
from typing import Iterable, Sequence, Tuple
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.event_handler import EventHandler
import jellyfysh.setting as setting
from jellyfysh.state_handler.tree_state_handler import StateId
from .tagger import Tagger


class ActiveGlobalStateInStateTagger(Tagger):
    """
    Tagger for event handlers which require the full active global state as the in-state.

    An example for such an event handler is the abstract EndOfChainEventHandler.

    This class is designed to work together with the TreeStateHandler. The in-state identifiers are then a sequence of
    tuples of integers, where the tuples can have different lengths. Each tuple in the sequence specifies a particle in
    the global state (see StateId in state_handler.tree_state_handler.py).
    """

    def __init__(self, create: Sequence[str], trash: Sequence[str], event_handler: EventHandler,
                 tag: str = None, activate: Sequence[str] = (), deactivate: Sequence[str] = ()) -> None:
        """
        The constructor of the ActiveGlobalStateInStateTagger class.

        There is only one event handler instance in this tagger.

        Parameters
        ----------
        create : Sequence[str]
            Sequence of tags to create after an event handler of this tagger has committed an event to the global state.
        trash : Sequence[str]
            Sequence of tags to trash after an event handler of this tagger has committed an event to the global state.
        event_handler : event_handler.EventHandler
            A single event handler instance.
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
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           event_handler=event_handler.__class__.__name__, number_event_handlers=1,
                           create=create, trash=trash, activate=activate, deactivate=deactivate, tag=tag)
        super().__init__(create, trash, event_handler, number_event_handlers=1, tag=tag,
                         activate=activate, deactivate=deactivate)

    def initialize(self) -> None:
        """
        Initialize the tagger by replacing the yield_identifiers_send_event_time method if no composite objects are
        present in the simulation.

        If there are no composite objects, all root cnodes in the extracted active global state are independently
        active. This is not true with composite objects where some units are only induced active.

        Extends the initialize method of the Tagger class.
        """
        super().initialize()
        if setting.number_of_node_levels == 1:
            # noinspection PyAttributeOutsideInit
            self.yield_identifiers_send_event_time = self._yield_identifiers_send_event_time_no_composite_objects

    def yield_identifiers_send_event_time(
            self, extracted_active_global_state: Sequence[Node]) -> Iterable[Tuple[StateId]]:
        """
        Generate in-state identifiers for the send_event_time method of this tagger's event handlers.

        The in-state identifiers will be transformed into real in-states using the state handler via the mediator.
        The active global state is given by a sequence of root cnodes where each cnode branch only contains active
        units. The in-state identifiers are generated as a tuple of global state identifiers.

        This method is only used if composite objects are present in the simulation. In this case, this method has to
        distinguish between independent and induced active units. If no composite objects are used in the simulation,
        this method is replaced by the _yield_identifiers_send_event_time_no_composite_objects method in the initialize
        method.

        The generated in-state contains the branches of the independent active (root or leaf) units.

        Parameters
        ----------
        extracted_active_global_state : Sequence[base.node.Node]
            The active global state information.

        Yields
        ------
        Tuple[activator.tag_activator.StateId]
            The global state in-state identifiers of the independent active units.
        """
        independent_identifiers = []
        for root_cnode in extracted_active_global_state:
            if len(root_cnode.children) == setting.number_of_nodes_per_root_node:
                independent_identifiers.append(root_cnode.value.identifier)
            else:
                for child in root_cnode.children:
                    independent_identifiers.append(child.value.identifier)
        yield tuple(independent_identifiers)

    def _yield_identifiers_send_event_time_no_composite_objects(
            self, extracted_active_global_state: Sequence[Node]) -> Iterable[Tuple[StateId]]:
        """
        Generate in-state identifiers for the send_event_time method of this tagger's event handlers.

        The in-state identifiers will be transformed into real in-states using the state handler via the mediator.
        The active global state is given by a sequence of root cnodes where each cnode branch only contains active
        units. The in-state identifiers are generated as a tuple of global state identifiers.

        This method is used in place of the yield_identifiers_send_event_time method if no composite objects are present
        in the simulation. In this case, all units in the active global state are independent active.

        The generated in-state contains the branches of the independent active units.

        Parameters
        ----------
        extracted_active_global_state : Sequence[base.node.Node]
            The active global state information.

        Yields
        ------
        Tuple[activator.tag_activator.StateId]
            The global state in-state identifiers of the independent active units.
        """
        yield tuple(node.value.identifier for node in extracted_active_global_state)
