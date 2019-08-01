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
"""Module for the SingleActiveCellOccupancy class."""
import logging
from typing import Iterable, Sequence, Tuple, Union
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.node import Node, yield_nodes_on_level_below
import setting
from state_handler.tree_state_handler import StateId
from .cell_occupancy import CellOccupancy
from .cell_occupancy.cells import Cells


class SingleActiveCellOccupancy(CellOccupancy):
    """
    This class builds a cell-occupancy system with a maximum of one single active unit.

    This cell-occupancy system can include a charge, meaning that only global state identifiers of unit with this
    charge are stored.
    This class is designed to work together with the TreeStateHandler. An in-state identifier is then a tuple of
    integers, where the tuple can have different lengths (see StateId in state_handler.tree_state_handler.py).
    """

    def __init__(self, cells: Cells, cell_level: int, charge: str = None) -> None:
        """
        The constructor of the SingleActiveCellOccupancy class.

        Parameters
        ----------
        cells : activator.internal_state.cell_occupancy.cells.Cells
            The underlying cell system.
        cell_level : int
            The length of the global state identifiers which should be stored in this internal state.
        charge : str or None, optional
            The charge of the unit, which must be unequal zero, in order for the corresponding identifier to be stored.
            If None, all global state identifiers with the correct length are stored.

        Raises
        ------
        base.exceptions.ConfigurationError
            If cell_level corresponds to composite point objects which cannot have a charge but the charge is set.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, cells=cells.__class__.__name__,
                           cell_level=cell_level, charge=charge)
        super().__init__(cells, cell_level)
        if cell_level < setting.number_of_node_levels and charge is not None:
            raise ConfigurationError("Chosen cell level stores composite point objects which cannot have a charge!")
        self._surplus = {}
        self._occupant = [None for _ in self._cells.yield_cells()]
        self._active_unit_identifier = None
        self._is_relevant_unit = (lambda unit: unit.charge[charge] != 0) if charge is not None else lambda unit: True
        self._active_cell = None

    def initialize(self, extracted_global_state: Sequence[Node]) -> None:
        """
        Initialize the internal state based on the full extracted global state from the state handler.

        Extends the initialize method of the InternalState class. Use this method once in the beginning of the run to
        initialize the internal state. Only after a call of this method, other public methods of this class can be
        called without raising an error.
        The full extracted global state is given as a sequence of cnodes of all root nodes stored in the global state.

        Parameters
        ----------
        extracted_global_state : Sequence[base.node.Node]
            The full extracted global state from the state handler.
        """
        super().initialize(extracted_global_state)
        for root_cnode in extracted_global_state:
            for relevant_cnode in yield_nodes_on_level_below(root_cnode, self._cell_level - 1):
                unit = relevant_cnode.value
                if self._is_relevant_unit(unit):
                    cell = self._cells.position_to_cell(unit.position)
                    if self._occupant[cell] is None:
                        self._occupant[cell] = unit.identifier
                    else:
                        self._surplus.setdefault(cell, []).append(unit.identifier)

    def __getitem__(self, internal_state_identifier: int) -> Union[StateId, None]:
        """
        Return the stored global state identifier based on a cell identifier of the underlying cell system.

        If there is no stored global state identifier in the given cell, this method returns None.
        Surplus identifiers and identifiers of active units are not returned in this method.
        Overwrites the __getitem__ method of the abstract InternalState class.

        Parameters
        ----------
        internal_state_identifier : int
            The cell identifier.

        Returns
        -------
        state_handler.tree_state_handler.StateId or None
           The global state identifier whose unit is located in the cell specified by the cell identifier.
        """
        # noinspection PyTypeChecker
        return self._occupant[internal_state_identifier]

    def update(self, extracted_active_global_state: Sequence[Node]) -> None:
        """
        Update the internal state based on the extracted active global state.
        
        Use this method to keep the internal state consistent with the global state. The active global state information
        is given by a sequence of root cnodes where each cnode branch only contains active units.
        The method extracts the active unit on the cell level of this class. This class assumes, that there is only
        one active point mass, therefore on each level there is only one active unit.
        If the active unit identifier has changed, this method updates the cell-occupancy system. If the active unit
        has not changed, only the active cell is determined again.

        Parameters
        ----------
        extracted_active_global_state : Sequence[base.node.Node]
            The active global state information.

        Raises
        ------
        AssertionError
            If separated_active_units does not include exactly one active unit on the cell_level of this class.
        """
        active_units_on_cell_level = [cnode.value for root_cnode in extracted_active_global_state
                                      for cnode in yield_nodes_on_level_below(root_cnode, self.cell_level - 1)]
        assert len(active_units_on_cell_level) == 1
        active_unit = active_units_on_cell_level[0]
        if active_unit.identifier != self._active_unit_identifier:
            if self._active_unit_identifier is not None:
                if self._occupant[self._active_cell] is None:
                    self._occupant[self._active_cell] = self._active_unit_identifier
                else:
                    self._surplus.setdefault(self._active_cell, []).append(self._active_unit_identifier)

            if self._is_relevant_unit(active_unit):
                self._active_cell = self._cells.position_to_cell(active_unit.position)
                self._active_unit_identifier = active_unit.identifier
                if self._occupant[self._active_cell] == active_unit.identifier:
                    self._occupant[self._active_cell] = self._surplus.get(self._active_cell, [None]).pop()
                    if not self._surplus.get(self._active_cell, True):
                        del self._surplus[self._active_cell]
                else:
                    self._surplus[self._active_cell].remove(active_unit.identifier)
                    if not self._surplus[self._active_cell]:
                        del self._surplus[self._active_cell]
            else:
                self._active_unit_identifier = None
                self._active_cell = None
        else:
            self._active_cell = self._cells.position_to_cell(active_unit.position)

    def yield_surplus(self) -> Iterable[StateId]:
        """
        Generate the surplus identifiers.

        Overwrites the yield_surplus method of the abstract CellOccupancy class.

        Yields
        ------
        state_handler.tree_state_handler.StateId
            Surplus identifier.
        """
        for value in self._surplus.values():
            yield from value

    def yield_active_cells(self) -> Iterable[Tuple[int, StateId]]:
        """
        Generate the cell identifiers and the global state identifiers of the active units.

        Overwrites the yield_active_cells method of the abstract CellOccupancy class.

        Yields
        ------
        (int, StateId)
            The cell identifier, the global state identifier of the active unit.
        """
        if self._active_cell is not None:
            yield self._active_cell, self._active_unit_identifier
