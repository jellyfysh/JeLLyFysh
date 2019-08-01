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
"""Module for the FactorTypeMapInStateTagger class."""
import logging
from typing import Iterable, Sequence, Tuple
from base.logging import log_init_arguments
from base.strings import to_camel_case
from base.node import Node, yield_leaf_nodes
from event_handler import EventHandler
from state_handler.tree_state_handler import StateId
from .factor_type_maps import FactorTypeMaps
from .tagger import Tagger


class FactorTypeMapInStateTagger(Tagger):
    """
    Tagger which generates in-states based on factor type maps parsed out of a file.

    A factor type map is explained briefly in the FactorTypeMaps class.
    This class is designed to work together with the TreeStateHandler. The in-state identifiers are then a sequence of
    tuples of integers, where the tuples can have different lengths. Each tuple in the sequence specifies a particle in
    the global state (see StateId in state_handler.tree_state_handler.py).
    """

    def __init__(self, create: Sequence[str], trash: Sequence[str], event_handler: EventHandler,
                 number_event_handlers: int, factor_type_maps: FactorTypeMaps, tag: str = None,
                 factor_type_maps_label: str = None) -> None:
        """
        The constructor of the FactorTypeMapInState class.

        Note that the activate and deactivate sequences are always empty for a tagger of this kind.

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
        factor_type_maps: activator.tagger.factor_type_maps.FactorTypeMaps
            The factor type maps instance which parses the factor type maps out of a file.
        tag : str or None, optional
            Tag used in all four lists (also of other taggers). If None, the class name (or the alias set in the
            factory) will be used as the tag.
        factor_type_maps_label: str or None, optional
            The label of the factor used in the file for the factor type maps. If None, the class name (or the alias
            set in the factory will be used as this label.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           event_handler=event_handler.__class__.__name__, number_event_handlers=number_event_handlers,
                           create=create, trash=trash, tag=tag, factor_type_maps=factor_type_maps.__class__.__name__,
                           factor_type_maps_label=factor_type_maps_label)
        super().__init__(create, trash, event_handler, number_event_handlers=number_event_handlers, tag=tag)
        # TODO Infer number event handlers from the factor type map
        self._factor_type_map = (factor_type_maps[to_camel_case(factor_type_maps_label)]
                                 if factor_type_maps_label is not None
                                 else factor_type_maps[to_camel_case(self._tag)])

    def yield_identifiers_send_event_time(
            self, extracted_active_global_state: Sequence[Node]) -> Iterable[Tuple[StateId, ...]]:
        """
        Generate in-state identifiers for the send_event_time method of this taggers event handlers.

        The in-state identifiers will be transformed into real in-states using the state handler via the mediator.
        The active global state is given by a sequence of root cnodes where each cnode branch only contains active
        units. The in-state identifiers are generated as a tuple of global state identifiers. If None is generated, no
        in-state will be constructed in the state handler.

        The generated in-states are given by the factor type map.

        Parameters
        ----------
        extracted_active_global_state : Sequence[base.node.Node]
            The active global state information.

        Yields
        ------
        Tuple[activator.tag_activator.StateId, ...]
            The global state in-state identifiers.
        """
        # TODO Change yield_factor_identifier of factor type map to not call set here?
        yield from set(factor for root_cnode in extracted_active_global_state
                       for leaf_cnode in yield_leaf_nodes(root_cnode)
                       for factor in self._factor_type_map.yield_factor_identifier(leaf_cnode.value.identifier))
