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
"""Module for the CellVetoTagger class."""
import logging
from typing import Iterable, Sequence, Tuple
from activator.internal_state import CellOccupancy, InternalState
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.node import Node
from event_handler.abstracts import CellVetoEventHandler
from event_handler import EventHandler
from state_handler.tree_state_handler import StateId
from .abstracts import TaggerWithInternalState


class CellVetoTagger(TaggerWithInternalState):
    """
    Tagger which generates in-states for a CellVetoEventHandler.

    This class does not deal with exceptional target particles, i.e. particles in excluded cells and surplus particles.
    These should be taken care of with the ExcludedCellsTagger and the SurplusCellsTagger.
    This tagger only works for a single active unit on the cell level of the cell-occupancy system.
    This class is designed to work together with the TreeStateHandler. The in-state identifiers are then a sequence of
    tuples of integers, where the tuples can have different lengths. Each tuple in the sequence specifies a particle in
    the global state (see StateId in state_handler.tree_state_handler.py).
    """

    def __init__(self, create: Sequence[str], trash: Sequence[str], event_handler: EventHandler,
                 internal_state_label: str, tag: str = None) -> None:
        """
        The constructor of the CellVetoTagger class.

        This class uses an internal state and therefore inherits from the TaggerWithInternalState class.
        The internal_state_label should refer to a cell-occupancy system.
        Note that the activate and deactivate sequences are always empty for a tagger of this kind.
        Also there is only one event handler instance in this tagger.
        Finally the event handler should be an instance of a CellVetoEventHandler.

        Parameters
        ----------
        create : Sequence[str]
            Sequence of tags to create after an event handler of this tagger has committed an event to the global state.
        trash : Sequence[str]
            Sequence of tags to trash after an event handler of this tagger has committed an event to the global state.
        event_handler : event_handler.cell_boundary_event_handler.CellBoundaryEventHandler
            A single event handler instance.
        internal_state_label : str
            The label of the internal state this tagger wants to use.
        tag : str or None, optional
            Tag used in all four lists (also of other taggers). If None, the class name (or the alias set in the
            factory) will be used as the tag.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the event handler is not an instance of a CellVetoEventHandler.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           event_handler=event_handler.__class__.__name__, internal_state_label=internal_state_label,
                           create=create, trash=trash, tag=tag)
        if not isinstance(event_handler, CellVetoEventHandler):
            raise ConfigurationError("The class {0} can only be used with a class inheriting from the class"
                                     " CellVetoEventHandler!".format(self.__class__.__name__))
        super().__init__(create, trash, event_handler, number_event_handlers=1,
                         internal_state_label=internal_state_label, tag=tag)

    def initialize(self, extracted_global_state: Sequence[Node], internal_states: Sequence[InternalState]) -> None:
        """
        Initialize the taggers based on the full extracted global state and all initialized internal states.

        Extends the initialize method of the TaggerWithInternalState class. Use this method once in the beginning of the
        run to initialize the tagger. Only after a call of this method, other public methods of this class can be called
        without raising an error.
        This method checks if the internal state of this label is a cell-occupancy system.
        Also, the event handler gets knowledge about the cell system, the cell level of the cell-occupancy system
        and the full extracted global state.
        The full extracted global state is given as a sequence of cnodes of all root nodes stored in the global state.

        Parameters
        ----------
        extracted_global_state : Sequence[base.node.Node]
            The full extracted global state from the state handler.
        internal_states : Sequence[activator.internal_state.InternalState]
            Sequence of all initialized internal states in the activator.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the internal state is not an instance of CellOccupancy.
        """
        super().initialize(extracted_global_state, internal_states)
        if not isinstance(self._internal_state, CellOccupancy):
            raise ConfigurationError("Class CellVetoTagger can only be used with an instance of CellOccupancy "
                                     "as the internal state!")
        self._event_handlers[0].initialize(self._internal_state.cells, self._internal_state.cell_level,
                                           extracted_global_state)

    def yield_identifiers_send_event_time(
            self, extracted_active_global_state: Sequence[Node]) -> Iterable[Tuple[StateId]]:
        """
        Generate in-state identifiers for the send_event_time method of this taggers event handlers.

        The in-state identifiers will be transformed into real in-states using the state handler via the mediator.
        The active global state is given by a sequence of root cnodes where each cnode branch only contains active
        units. The in-state identifiers are generated as a tuple of global state identifiers (which are themselves
        tuples). If None is generated, no in-state will be constructed in the state handler.

        The generated in-state is just the active identifier in the cell-occupancy system.

        Parameters
        ----------
        extracted_active_global_state : Sequence[base.node.Node]
            The active global state information.

        Yields
        ------
        Tuple[activator.tag_activator.StateId]
            The global state in-state identifiers.
        """
        for _, active_identifier in self._internal_state.yield_active_cells():
            yield (active_identifier,)
