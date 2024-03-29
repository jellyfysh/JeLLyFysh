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
"""Module for the NoInStateTagger class."""
import logging
from typing import Iterable, Sequence
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.event_handler import EventHandler
from .tagger import Tagger


class NoInStateTagger(Tagger):
    """
    Tagger for an event handler which needs no in-state.

    An example for such an event handler is the FixedIntervalSamplingEventHandler.

    This class is designed to work together with the TreeStateHandler. The in-state identifiers are then a sequence of
    tuples of integers, where the tuples can have different lengths. Each tuple in the sequence specifies a particle in
    the global state (see StateId in state_handler.tree_state_handler.py).
    """

    def __init__(self, create: Sequence[str], trash: Sequence[str], event_handler: EventHandler, tag: str = None,
                 activate: Sequence[str] = (), deactivate: Sequence[str] = ()) -> None:
        """
        The constructor of the NoInStateTagger class.

        There is only one event handler instance in this tagger.

        Parameters
        ----------
        create : Sequence[str]
            Sequence of tags to create after an event handler of this tagger has committed an event to the global state.
        trash : Sequence[str]
            Sequence of tags to trash after an event handler of this tagger has committed an event to the global state.
        event_handler : event_handler.cell_boundary_event_handler.CellBoundaryEventHandler
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
                           event_handler=event_handler.__class__.__name__, create=create, trash=trash,
                           activate=activate, deactivate=deactivate, tag=tag)
        super().__init__(create, trash, event_handler, number_event_handlers=1, tag=tag, activate=activate,
                         deactivate=deactivate)

    def yield_identifiers_send_event_time(self, extracted_active_global_state: Sequence[Node]) -> Iterable[None]:
        """
        Generate in-state identifiers for the send_event_time method of this tagger's event handler.

        The in-state identifiers will be transformed into real in-states using the state handler via the mediator.
        The active global state is given by a sequence of root cnodes where each cnode branch only contains active
        units.

        The generated in-state is just None which means that no in-state is constructed in the state handler.

        Parameters
        ----------
        extracted_active_global_state : Sequence[base.node.Node]
            The active global state information.

        Yields
        ------
        None
            The global state in-state identifiers.
        """
        yield None
