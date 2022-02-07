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
"""Module for the CellVetoTagger class."""
import logging
from typing import Iterable, Sequence, Tuple
from jellyfysh.activator.internal_state import CellOccupancy, InternalState
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.event_handler.abstracts import CellVetoEventHandler
from jellyfysh.event_handler import EventHandler
from jellyfysh.state_handler.tree_state_handler import StateId
from .abstracts import TaggerWithInternalState


class CellVetoTagger(TaggerWithInternalState):
    """
    Tagger which generates in-states for instances of the CellVetoEventHandler class.

    For each active cell in the cell-occupancy system, this tagger generates a single in-state that only consists of the
    respective active unit's identifier. The cell-veto algorithm that is implemented in the CellVetoEventHandler does
    not consider possible units in nearby cells, or surplus units in the cell-occupancy system. These should be taken
    care of by using, e.g., the ExcludedCellsTagger and the SurplusCellsTagger.

    This class does not deal with exceptional target particles, i.e. particles in excluded cells and surplus particles.
    These should be taken care of with the ExcludedCellsTagger and the SurplusCellsTagger.
    This tagger only works for a single active unit on the cell level of the cell-occupancy system.

    This class is designed to work together with the TreeStateHandler. The in-state identifiers are then a sequence of
    tuples of integers, where the tuples can have different lengths. Each tuple in the sequence specifies a particle in
    the global state (see StateId in state_handler.tree_state_handler.py).
    """

    def __init__(self, create: Sequence[str], trash: Sequence[str], event_handler: EventHandler,
                 internal_state_label: str, number_event_handlers: int = 1, tag: str = None) -> None:
        """
        The constructor of the CellVetoTagger class.

        This class uses an internal state and therefore inherits from the TaggerWithInternalState class. The
        internal_state_label should refer to a cell-occupancy system.

        Note that the activate and deactivate sequences are always empty for a tagger of this kind.

        The event handler should be an instance of a CellVetoEventHandler. The number of event handlers should be
        set to the number of active units that are relevant to the cell-occupancy system.

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
        number_event_handlers : int, optional
            Number of event handlers to prepare. The tagger will deepcopy the given event handler instance to create
            this number of event handlers.
        tag : str or None, optional
            Tag used in all four lists (also of other taggers). If None, the class name (or the alias set in the
            factory) will be used as the tag.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the event handler is not an instance of a CellVetoEventHandler.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           event_handler=event_handler.__class__.__name__, number_event_handlers=number_event_handlers,
                           internal_state_label=internal_state_label, create=create, trash=trash, tag=tag)
        if not isinstance(event_handler, CellVetoEventHandler):
            raise ConfigurationError("The class {0} can only be used with an instance of the class "
                                     "CellVetoEventHandler as the event handler".format(self.__class__.__name__))
        super().__init__(create, trash, event_handler, number_event_handlers=number_event_handlers,
                         internal_state_label=internal_state_label, tag=tag)

    def initialize_with_internal_states(self, internal_states: Sequence[InternalState]) -> None:
        """
        Initialize the tagger based on the initialized internal states.

        Extends the initialize_with_internal_states method of the base TaggerWithInternal class. This method is a second
        initialize method relevant to the base Initializer class. It is called once in the beginning of the run by the
        tag activator. However, this method does not call the initialize method of the Initializer class. Therefore,
        other public methods of this class can still not be called without raising an error after this method has been
        used. To finalize the initialization of this class, use the initialize method (which should be called after this
        method).

        This method checks if the internal state of this tagger is a cell-occupancy system. Also, the cell veto
        event handler in the self._event_handler_to_copy attribute gets knowledge about the cell system, and the cell
        level of the cell-occupancy system. By this, the event handler is initialized. This event handler is deepcopied
        in the subsequent call of the initialize method to create the desired number of (initialized) event handlers for
        this tagger (see Tagger base class).

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
        self._event_handler_to_copy.initialize(self._internal_state.cells, self._internal_state.cell_level)

    def yield_identifiers_send_event_time(
            self, extracted_active_global_state: Sequence[Node]) -> Iterable[Tuple[StateId]]:
        """
        Generate in-state identifiers for the send_event_time method of this tagger's event handlers.

        The in-state identifiers will be transformed into real in-states using the state handler via the mediator.
        The active global state is given by a sequence of root cnodes where each cnode branch only contains active
        units. The in-state identifiers are generated as a tuple of global state identifiers.

        The generated in-states for the CellVetoEventHandlers are just the active unit identifiers of the cell-occupancy
        system.

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
            # noinspection PyRedundantParentheses
            yield (active_identifier,)
