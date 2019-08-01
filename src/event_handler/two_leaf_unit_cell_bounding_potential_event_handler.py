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
"""Module for the TwoLeafUnitCellBoundingPotentialEventHandler class."""
import logging
import math
import random
from typing import Sequence
from activator.internal_state.cell_occupancy.cells import PeriodicCells
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.node import Node
from potential import Potential
from potential.cell_bounding_potential import CellBoundingPotential
import setting
from .abstracts import CellBoundingPotentialEventHandler, EventHandlerWithBoundingPotential


class TwoLeafUnitCellBoundingPotentialEventHandler(EventHandlerWithBoundingPotential,
                                                   CellBoundingPotentialEventHandler):
    """
    Event handler which uses a cell bounding potential for an interaction between two leaf units.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    Only one of the two leaf units should be active with an direction of motion in positive direction along an axis.
    This class uses an potential and a cell bounding potential which expect a single separation. For the potential,
    the separation should be between two leaf units, for the cell bounding potential it is the cell separation.
    Also, this event handler can pass a charge to the potential. The separation vector and the cell separation are
    corrected for periodic boundaries.
    """

    def __init__(self, potential: Potential, bounding_potential: CellBoundingPotential, charge: str = None) -> None:
        """
        The constructor of the TwoLeafUnitCellBoundingPotentialEventHandler class.

        Parameters
        ----------
        potential : potential.Potential
            The potential between the leaf units.
        bounding_potential : potential.cell_bounding_potential.CellBoundingPotential
            The invertible cell bounding potential between the leaf units.
        charge : str or None, optional
            The relevant charge for this event handler.

        Raises
        ------
        base.exceptions.ConfigurationError:
            If the cell bounding potential does not expect exactly one separation.
        base.exceptions.ConfigurationError:
            If the charge is not None but the potential or the cell bounding potential expects more than two charges.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential.__class__.__name__,
                           bounding_potential=bounding_potential.__class__.__name__, charge=charge)
        super().__init__(potential=potential, bounding_potential=bounding_potential)
        self._charge = charge

        if self._bounding_potential.number_separation_arguments != 1:
            raise ConfigurationError("The event handler {0} expects a cell bounding potential "
                                     "which handles exactly one separation!".format(self.__class__.__name__))

        if charge is None:
            self._potential_charges = (lambda unit_one, unit_two:
                                       tuple(1.0 for _ in range(self._potential.number_charge_arguments)))
            self._bounding_potential_charges = (lambda unit_one, unit_two:
                                                tuple(1.0 for _ in
                                                      range(self._bounding_potential.number_charge_arguments)))
        else:
            if self._potential.number_charge_arguments == 2 and self._bounding_potential.number_charge_arguments == 2:
                self._potential_charges = lambda unit_one, unit_two: (unit_one.charge[charge], unit_two.charge[charge])
                self._bounding_potential_charges = self._potential_charges
            else:
                raise ConfigurationError("The event handler {0} was initialized with a charge which is not None,"
                                         " but its potential {1} and/or its bounding potential {2}"
                                         "expects not exactly 2 charges."
                                         .format(self.__class__.__name__, self._potential.__class__.__name__,
                                                 self._bounding_potential.__class__.__name__))

    # noinspection PyMethodOverriding
    def initialize(self, cells: PeriodicCells, extracted_global_state: Sequence[Node]) -> None:
        """
        Initialize the cell bounding potential event handler base class.

        This is done by handing over the cells, the extracted global state and the charge to the base class. This event
        handler requires the cells to be an instance of PeriodicCells.
        Extends the initialize method of the abstract Initializer class. This method is called once in the beginning of
        the run by the activator. Only after a call of this method, other public methods of this class can be called
        without raising an error.

        Parameters
        ----------
        cells : activator.internal_state.cell_occupancy.cells.Cells
            The cell system.
        extracted_global_state : Sequence[base.node.Node]
            The extracted global state of the tree state handler.

        Raises
        ------
        AssertionError
            If the cell system is not an instance of PeriodicCells.
        """
        if not isinstance(cells, PeriodicCells):
            raise ConfigurationError("The event handler {0} needs an instance of PeriodicCells!"
                                     .format(self.__class__.__name__))
        super().initialize(cells, extracted_global_state, self._charge)

    def send_event_time(self, in_state: Sequence[Node]) -> float:
        """
        Return the candidate event time.

        The in-state should consist of all branches of the two leaf units which take part in the interaction treated in
        this event handler. The candidate event time is calculated using the cell bounding potential. For this,
        the cell separation between the leaf units is calculated. The cell separation is calculated under consideration
        of periodic boundaries.

        Parameters
        ----------
        in_state : Sequence[base.node.Node]
            The in-state.

        Returns
        -------
        float
            The candidate event time.

        Raises
        ------
        AssertionError
            If the in-state contains not exactly two leaf units.
        AssertionError
            If the cell of the target leaf unit is an excluded cell of the active leaf unit's cell.
        """
        self._store_in_state(in_state)
        self._construct_leaf_cnodes()
        self._extract_active_leaf_unit()
        assert len(self._leaf_cnodes) == 2
        self._active_cell = self._cells.position_to_cell(self._leaf_units[self._active_leaf_unit_index].position)
        non_active_cell = self._cells.position_to_cell(self._leaf_units[self._active_leaf_unit_index ^ 1].position)
        assert non_active_cell not in self._cells.excluded_cells(self._active_cell)

        self._relative_cell = self._cells.relative_cell(non_active_cell, self._active_cell)

        displacement = self._bounding_potential.displacement(
            self._direction_of_motion, self._relative_cell,
            *self._bounding_potential_charges(self._leaf_units[0], self._leaf_units[1]),
            random.expovariate(setting.beta))
        self._event_time = self._active_leaf_unit.time_stamp + displacement / self._speed
        self._time_slice_all_units_in_state()
        return self._event_time

    # noinspection PyTypeChecker
    def send_out_state(self) -> Sequence[Node]:
        """
        Return the out-state.

        First, this method confirms the event. If it is confirmed, the velocities of the two leaf units are exchanged
        and the branches are kept consistent.
        If the active leaf unit changed the cell in the send_event_time method, this method returns a None out-state.
        This is never relevant, since a cell boundary event should have been triggered before. If this is not the case,
        the None will yield an exception in the state handler.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.
        """
        if (not all(not math.isnan(position) for position in self._leaf_units[self._active_leaf_unit_index].position)
                or not (self._cells.position_to_cell(self._leaf_units[self._active_leaf_unit_index].position)
                        == self._active_cell)):
            # Active unit changed cell and bounded event rate is effectively wrong.
            # Will never be relevant, since cell boundary event comes in first -> return None
            return None
        self._bounding_event_rate = self._bounding_potential.derivative(
            self._direction_of_motion, self._relative_cell,
            *self._bounding_potential_charges(self._leaf_units[0], self._leaf_units[1]))
        self._calculate_out_state_of_two_leaf_unit_bounding_potential(
            setting.periodic_boundaries.separation_vector(self._leaf_units[self._active_leaf_unit_index].position,
                                                          self._leaf_units[self._active_leaf_unit_index ^ 1].position),
            self._potential_charges(self._leaf_units[0], self._leaf_units[1]))
        return self._state
