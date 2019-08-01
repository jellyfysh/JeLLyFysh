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
"""Module for the TwoCompositeObjectCellBoundingPotentialEventHandler class."""
import logging
import math
import random
from typing import Sequence
from activator.internal_state.cell_occupancy.cells import PeriodicCells
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.node import Node
from lifting import Lifting
from potential import Potential
from potential.cell_bounding_potential import CellBoundingPotential
import setting
from .abstracts import CellBoundingPotentialEventHandler, TwoCompositeObjectBoundingPotentialEventHandler
from .helper_functions import bounding_potential_warning


class TwoCompositeObjectCellBoundingPotentialEventHandler(TwoCompositeObjectBoundingPotentialEventHandler,
                                                          CellBoundingPotentialEventHandler):
    """
    Event handler which treats a two-composite-object interaction using a cell bounding potential.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    To calculate the potential between the two composite objects, this class sums the potential between all pairs of
    leaf units located in different composite objects.
    Only one of the leaf units should be active with an direction of motion in positive direction along an axis.
    This class uses an potential and a cell bounding potential which expect a single separation. For the potential,
    the separation should be between two leaf units, for the cell bounding potential it is the cell separation between
    between the two composite objects.
    Also, this event handler can pass a charge to the potential. The cell bounding potential however, since it treats
    composite objects, gets all charges as one. The charges must have been considered there during the estimation
    of the bounds of the derivatives for all cell separations. The separation vector and the cell separation are
    corrected for periodic boundaries.
    Since the interaction involves more than two leaf units, a lifting scheme is required.
    """
    def __init__(self, potential: Potential, bounding_potential: CellBoundingPotential,
                 lifting: Lifting, charge: str = None) -> None:
        """
        The constructor of the TwoCompositeObjectCellBoundingPotentialEventHandler class.

        Parameters
        ----------
        potential : potential.Potential
            The potential between the leaf units.
        bounding_potential : potential.cell_bounding_potential.CellBoundingPotential
            The invertible cell bounding potential between the composite objects.
        lifting : lifting.Lifting
            The lifting scheme.
        charge : str or None, optional
            The relevant charge for this event handler.

        Raises
        ------
        base.exceptions.ConfigurationError:
            If the cell bounding potential does not expect exactly one separation.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential.__class__.__name__,
                           bounding_potential=bounding_potential.__class__.__name__,
                           lifting=lifting.__class__.__name__, charge=charge)
        super().__init__(potential=potential, bounding_potential=bounding_potential, lifting=lifting,
                         charge=charge)

        if self._bounding_potential.number_separation_arguments != 1:
            raise ConfigurationError("The event handler {0} expects a cell bounding potential "
                                     "which handles exactly one separation!".format(self.__class__.__name__))

        # The Cell Bounding Potential considers composite point objects without charges! We set them to one here
        self._bounding_potential_charges = (lambda unit_one, unit_two:
                                            tuple(1.0 for _ in
                                                  range(self._bounding_potential.number_charge_arguments)))

        self._root_units = None
        self._active_root_unit_index = None
        self._active_root_unit = None
        self._charge = charge

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

        The in-state should consist of all branches of the composite objects which take part in the interaction treated
        in this event handler. The candidate event time is calculated using the cell bounding potential. For this,
        the cell separation between composite objects is calculated. The cell separation is calculated under
        consideration of periodic boundaries.

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
            If the in-state contains not exactly two branches given as root cnodes.
        AssertionError
            If both root cnodes are active.
        AssertionError
            If the cell of the target composite object is an excluded cell of the active composite object's cell.
        """
        assert len(in_state) == 2
        assert len(in_state[0].value.identifier) == len(in_state[1].value.identifier) == 1
        self._store_in_state(in_state)
        self._construct_leaf_cnodes()
        self._extract_active_leaf_unit()

        self._root_units = [in_state[0].value, in_state[1].value]
        active_root_units = [(index, unit) for index, unit in enumerate(self._root_units)
                             if unit.velocity is not None]
        assert len(active_root_units) == 1
        self._active_root_unit_index = active_root_units[0][0]
        self._active_root_unit = self._root_units[self._active_root_unit_index]

        self._active_cell = self._cells.position_to_cell(
            self._root_units[self._active_root_unit_index].position)
        non_active_cell = self._cells.position_to_cell(
            self._root_units[self._active_root_unit_index ^ 1].position)
        self._relative_cell = self._cells.relative_cell(non_active_cell, self._active_cell)
        assert self._relative_cell not in self._cells.excluded_cells(0)

        # This event handler assumes composite point objects without a charge
        displacement = self._bounding_potential.displacement(
            self._direction_of_motion, self._relative_cell, *self._bounding_potential_charges(None, None),
            random.expovariate(setting.beta))
        self._event_time = self._active_leaf_unit.time_stamp + displacement / self._speed
        self._time_slice_all_units_in_state()
        return self._event_time

    def send_out_state(self):
        """
        Return the out-state.

        First, this method confirms the event. If it is confirmed, the lifting scheme determines the new active leaf
        unit, which is imprinted in the out-state consisting of both branches of the two composite objects.
        If the active composite object changed the cell in the send_event_time method, this method returns a None
        out-state. This is never relevant, since a cell boundary event should have been triggered before. If this is not
        the case, the None out-state will yield an exception in the state handler.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.

        Raises
        ------
        AssertionError
            If the bounding event rate is smaller than zero.
        AssertionError
            If the lifting scheme failed.
        """
        new_root_unit_position = (
            self._root_units[self._active_root_unit_index].position.copy())
        new_root_unit_position[self._direction_of_motion] += (
            self._root_units[self._active_root_unit_index].velocity[self._direction_of_motion]
            * (self._event_time - self._root_units[self._active_root_unit_index].time_stamp))
        new_root_unit_position[self._direction_of_motion] = (
            setting.periodic_boundaries.correct_position_entry(
                new_root_unit_position[self._direction_of_motion], self._direction_of_motion))
        if (not all(not math.isnan(position) for position in self._leaf_units[self._active_leaf_unit_index].position)
                or not (self._cells.position_to_cell(new_root_unit_position) == self._active_cell)):
            # Active unit changed cell and bounded event rate is effectively wrong.
            # Will never be relevant, since cell boundary event comes in first -> return None
            return None
        self._construct_leaf_units_of_composite_objects()
        bounding_event_rate = self._bounding_potential.derivative(self._direction_of_motion, self._relative_cell,
                                                                  *self._bounding_potential_charges(None, None))
        factor_derivative = 0.0
        target_composite_object_factor_derivatives = [0.0] * len(self._target_leaf_units)
        for index, leaf_unit in enumerate(self._target_leaf_units):
            pairwise_derivative = self._potential.derivative(
                self._direction_of_motion,
                setting.periodic_boundaries.separation_vector(self._active_leaf_unit.position, leaf_unit.position),
                *self._potential_charges(self._active_leaf_unit, leaf_unit))
            factor_derivative += pairwise_derivative
            target_composite_object_factor_derivatives[index] -= pairwise_derivative
        event_rate = max(0.0, factor_derivative)
        assert bounding_event_rate >= 0.0
        bounding_potential_warning(self.__class__.__name__, bounding_event_rate, event_rate)
        if event_rate <= random.uniform(0.0, bounding_event_rate):
            return self._state

        self._fill_lifting(self._local_leaf_units, self._target_leaf_units,
                           factor_derivative, target_composite_object_factor_derivatives)

        next_active_identifier = self._lifting.get_active_identifier()
        next_active_cnode = [cnode for cnode in self._leaf_cnodes if cnode.value.identifier == next_active_identifier]
        assert len(next_active_cnode) == 1
        self._exchange_velocity(self._leaf_cnodes[self._active_leaf_unit_index], next_active_cnode[0])
        return self._state
