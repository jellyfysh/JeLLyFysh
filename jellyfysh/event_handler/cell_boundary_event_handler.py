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
"""Module for the CellBoundaryEventHandler class."""
from math import inf
import logging
from typing import Sequence
from jellyfysh.activator.internal_state.cell_occupancy.cells import PeriodicCells
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.initializer import Initializer
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
import jellyfysh.setting as setting
from .abstracts import BasicEventHandler


class CellBoundaryEventHandler(BasicEventHandler, Initializer):
    """
    Event handler which triggers an event when a relevant active unit crosses a cell boundary in a cell-occupancy
    system.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    This event handler can only treat an in-state with a single independent active unit on the cell level of the
    cell-occupancy system. For the tree state handler, the cell level is the length of the global state identifier
    of the active unit.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information).
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

    def initialize(self, cells: PeriodicCells, cell_level: int) -> None:
        """
        Initialize the cell boundary event handler.

        Here, the event handler gets access to the relevant periodic cell system and the cell level of the corresponding
        cell-occupancy system.

        This cell boundary event handler requires a periodic cell system (and not a simple cell system) so that each
        cell always has a neighbor cell.

        Extends the initialize method of the abstract Initializer class. This method is called once in the beginning of
        the run by the activator. Only after a call of this method, other public methods of this class can be called
        without raising an error.

        Parameters
        ----------
        cells : activator.internal_state.cell_occupancy.cells.PeriodicCells
            The periodic cell system.
        cell_level : int
            The cell level.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the cell system is not an instance of the PeriodicCells class.
        """
        super().initialize()
        if not isinstance(cells, PeriodicCells):
            raise ConfigurationError("The event handler {0} can only be initialized with an instance of the "
                                     "PeriodicCells class.".format(self.__class__.__name__))
        self._cells = cells
        self._cell_level = cell_level

    def send_event_time(self, in_states: Sequence[Node]) -> Time:
        """
        Return the candidate event time.

        The in-state should be just the branch of a single independent active unit.

        Parameters
        ----------
        in_states : Sequence[base.node.Node]
            The in-state.

        Returns
        -------
        base.time.Time
            The candidate event time.

        Raises
        ------
        AssertionError
            If the in-state contains more than one branch.
            If the unit on the cell level is not active.
            If there is more than one active unit above or on the cell level.
            If no velocity component of the relevant active unit is unequal zero.
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

        current_smallest_time_to_boundary = inf
        for direction, velocity_component in enumerate(self._relevant_unit.velocity):
            if velocity_component != 0.0:
                if velocity_component > 0.0:
                    neighbor_boundary = self._cells.neighbor_cell(cell, direction, True).cell_min[direction]
                    separation = neighbor_boundary - self._relevant_unit.position[direction]
                    if separation < 0.0:
                        separation = setting.periodic_boundaries.next_image(separation, direction)
                    time_to_boundary = separation / velocity_component
                else:
                    neighbor_boundary = self._cells.neighbor_cell(cell, direction, False).cell_max[direction]
                    separation = self._relevant_unit.position[direction] - neighbor_boundary
                    if separation < 0.0:
                        separation = setting.periodic_boundaries.next_image(separation, direction)
                    time_to_boundary = separation / abs(velocity_component)
                if time_to_boundary < current_smallest_time_to_boundary:
                    current_smallest_time_to_boundary = time_to_boundary
                    self._boundary = neighbor_boundary
                    self._direction = direction
        assert current_smallest_time_to_boundary < inf
        self._event_time = self._relevant_unit.time_stamp + current_smallest_time_to_boundary
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
