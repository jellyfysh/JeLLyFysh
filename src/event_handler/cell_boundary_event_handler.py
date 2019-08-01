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
"""Module for the CellBoundaryEventHandler class."""
import logging
from typing import Sequence
from activator.internal_state.cell_occupancy.cells import Cells
from base.initializer import Initializer
from base.logging import log_init_arguments
from base.node import Node
import setting
from .abstracts import BasicEventHandler


class CellBoundaryEventHandler(BasicEventHandler, Initializer):
    """
    Event handler which triggers an event, when a unit crosses a cell-boundary.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    This event handler can only treat an in-state with a single independent active unit and the direction of motion
    should be along a positive direction of one axis.
    """

    def __init__(self) -> None:
        """The constructor of the CellBoundaryEventHandler class."""
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__)
        super().__init__()
        self._cells = None
        self._cell_level = None
        self._relevant_unit = None
        self._boundary = None
        self._direction = None
        self._cell_level = None

    def initialize(self, cells: Cells, cell_level: int) -> None:
        """
        Initialize the cell boundary event handler.

        Here, the event handler gets access to the relevant cell system and the cell level of the corresponding
        cell-occupancy system. For the tree state handler, the cell level is the length of the global state
        identifier of the unit.
        Extends the initialize method of the abstract Initializer class. This method is called once in the beginning of
        the run by the activator. Only after a call of this method, other public methods of this class can be called
        without raising an error.

        Parameters
        ----------
        cells : activator.internal_state.cell_occupancy.cells.Cells
            The cell system.
        cell_level : int
            The cell level.
        """
        super().initialize()
        self._cells = cells
        self._cell_level = cell_level

    def send_event_time(self, in_states: Sequence[Node]) -> float:
        """
        Return the candidate event time.

        The in-state should be just the branch of a single independent active unit.

        Parameters
        ----------
        in_states : Sequence[base.node.Node]
            The in-state.

        Returns
        -------
        float
            The candidate event time.

        Raises
        ------
        AssertionError
            If the in-state contains more than one branch.
        AssertionError
            If the unit on the cell level is not active.
        AssertionError
            If the velocity of the active independent unit is not aligned in positive direction with an axis.
        """
        assert len(in_states) == 1
        cnode = in_states[0]
        while len(cnode.value.identifier) < self._cell_level:
            assert len(cnode.children) == 1
            cnode = cnode.children[0]

        self._store_in_state(in_states)
        self._relevant_unit = cnode.value

        cell = self._cells.position_to_cell(self._relevant_unit.position)

        assert self._relevant_unit.velocity is not None
        assert sum(1 if velocity_component != 0.0 else 0
                   for velocity_component in self._relevant_unit.velocity) == 1

        for direction, velocity_component in enumerate(self._relevant_unit.velocity):
            if velocity_component != 0.0:

                assert velocity_component > 0.0

                next_cell = self._cells.successor(cell, direction)
                self._boundary = self._cells.cell_min(next_cell)[direction]
                self._direction = direction

                separation = self._boundary - self._relevant_unit.position[direction]
                if separation < 0.0:
                    separation = setting.periodic_boundaries.next_image(separation, direction)
                self._event_time = (separation / self._relevant_unit.velocity[direction]
                                    + self._relevant_unit.time_stamp)

                return self._event_time

    def send_out_state(self) -> Sequence[Node]:
        """
        Return the out-state.

        This method just places the relevant active unit on the next cell boundary and keeps all units in the in-state
        consistent.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.
        """
        self._time_slice_all_units_in_state()
        # Keep this to avoid floating point errors
        self._relevant_unit.position[self._direction] = self._boundary
        return self._state
