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
"""Module for the CellBoundingPotentialTagger class."""
import logging
from typing import Iterable, Sequence, Tuple
from jellyfysh.activator.internal_state import CellOccupancy, InternalState
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.event_handler.abstracts import CellBoundingPotentialEventHandler
from jellyfysh.event_handler import EventHandler
from jellyfysh.state_handler.tree_state_handler import StateId
from .abstracts import TaggerWithInternalState


class CellBoundingPotentialTagger(TaggerWithInternalState):
    """
    Tagger which generates in-states for instances of the CellBoundingPotentialEventHandler class.

    For each active cell in the cell-occupancy system, this tagger generates a single in-state for all other non-nearby
    target cells that are not empty. Each in-state contains the active unit's identifier and all occupants in the
    respective target cell. Units in nearby cells, or surplus units in the cell-occupancy system can be taken care of by
    using, e.g., the ExcludedCellsTagger and the SurplusCellsTagger.

    This class is designed to work together with the TreeStateHandler. The in-state identifiers are then a sequence of
    tuples of integers, where the tuples can have different lengths. Each tuple in the sequence specifies a particle in
    the global state (see StateId in state_handler.tree_state_handler.py).
    """

    def __init__(self, create: Sequence[str], trash: Sequence[str], event_handler: EventHandler,
                 number_event_handlers: int, internal_state_label: str, tag: str = None) -> None:
        """
        The constructor of the CellBoundingPotentialTagger class.

        This class uses an internal state and therefore inherits from the TaggerWithInternalState class. The
        internal_state_label should refer to a cell-occupancy system.

        Note that the activate and deactivate sequences are always empty for a tagger of this kind.

        The event handler should be an instance of both the CellBoundingPotentialEventHandler and EventHandler classes.

        Parameters
        ----------
        create : Sequence[str]
            Sequence of tags to create after an event handler of this tagger has committed an event to the global state.
        trash : Sequence[str]
            Sequence of tags to trash after an event handler of this tagger has committed an event to the global state.
        event_handler : event_handler.cell_boundary_event_handler.CellBoundaryEventHandler
            A single event handler instance.
        number_event_handlers : int
            Number of event handlers to prepare. The tagger will deepcopy the given event handler instance to create
            this number of event handlers.
        internal_state_label : str
            The label of the internal state this tagger wants to use.
        tag : str or None, optional
            Tag used in all four lists (also of other taggers). If None, the class name (or the alias set in the
            factory) will be used as the tag.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the event handler is not an instance of both a CellBoundingPotentialEventHandler and an EventHandler.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           event_handler=event_handler.__class__.__name__, number_event_handlers=number_event_handlers,
                           internal_state_label=internal_state_label, create=create, trash=trash, tag=tag)
        if not (isinstance(event_handler, CellBoundingPotentialEventHandler)
                and isinstance(event_handler, EventHandler)):
            raise ConfigurationError("The class {0} can only be used with an instance of a class that inherits from "
                                     "both the EventHandler and CellBoundingPotentialEventHandler classes."
                                     .format(self.__class__.__name__))
        super().__init__(create, trash, event_handler, internal_state_label=internal_state_label,
                         number_event_handlers=number_event_handlers, tag=tag)

    def initialize_with_internal_states(self, internal_states: Sequence[InternalState]) -> None:
        """
        Initialize the tagger based on the initialized internal states.

        Extends the initialize_with_internal_states method of the base TaggerWithInternal class. This method is a second
        initialize method relevant to the base Initializer class. It is called once in the beginning of the run by the
        tag activator. However, this method does not call the initialize method of the Initializer class. Therefore,
        other public methods of this class can still not be called without raising an error after this method has been
        used. To finalize the initialization of this class, use the initialize method (which should be called after this
        method).

        This method checks if the internal state of this tagger is a cell-occupancy system. Also, the cell bounding
        potential event handler in the self._event_handler_to_copy attribute gets knowledge about the cell system, and
        is initialized itself. This event handler is deepcopied in the subsequent call of the initialize method to
        create the desired number of (initialized) event handlers for this tagger (see Tagger base class).

        Parameters
        ----------
        internal_states : Sequence[activator.internal_state.InternalState]
            Sequence of all initialized internal states in the activator.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the internal state is not an instance of CellOccupancy.
        """
        super().initialize_with_internal_states(internal_states)
        if not isinstance(self._internal_state, CellOccupancy):
            raise ConfigurationError("The tagger {0} can only be used with an instance of the CellOccupancy class as "
                                     "the internal state.".format(self.__class__.__name__))
        # noinspection PyUnresolvedReferences
        self._event_handler_to_copy.initialize(self._internal_state.cells)

    def yield_identifiers_send_event_time(
            self, extracted_active_global_state: Sequence[Node]) -> Iterable[Tuple[StateId, StateId]]:
        """
        Generate in-state identifiers for the send_event_time method of this tagger's event handlers.

        The in-state identifiers will be transformed into real in-states using the state handler via the mediator.
        The active global state is given by a sequence of root cnodes where each cnode branch only contains active
        units. The in-state identifiers are generated as a tuple of global state identifiers.

        For each active unit in the cell-occupancy system and each non-nearby target cell of the active cell that is
        not empty, the generated in-states consist of a tuple that contains the active unit's identifier and all
        occupants in the target cell.

        Parameters
        ----------
        extracted_active_global_state : Sequence[base.node.Node]
            The active global state information.

        Yields
        ------
        Tuple[activator.tag_activator.StateId, activator.tag_activator.StateId]
            The global state in-state identifiers.
        """
        for active_cell, active_identifier in self._internal_state.yield_active_cells():
            yield from ((active_identifier,)
                        + tuple(occupant_identifier for occupant_identifier in self._internal_state[cell])
                        for cell in self._internal_state.cells.yield_cells()
                        if self._internal_state[cell]
                        and cell not in self._internal_state.cells.nearby_cells(active_cell))
