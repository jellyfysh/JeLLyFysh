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
"""Module for the TwoCompositeObjectCellBoundingPotentialEventHandler class."""
import logging
import math
import random
from typing import Sequence
from jellyfysh.activator.internal_state.cell_occupancy.cells import PeriodicCells
from jellyfysh.base.exceptions import ConfigurationError, bounding_potential_warning
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
from jellyfysh.lifting import Lifting
from jellyfysh.potential import Potential
from jellyfysh.potential.cell_bounding_potential import CellBoundingPotential
import jellyfysh.setting as setting
from .abstracts import CellBoundingPotentialEventHandler, TwoCompositeObjectBoundingPotentialEventHandler


class TwoCompositeObjectCellBoundingPotentialEventHandler(TwoCompositeObjectBoundingPotentialEventHandler,
                                                          CellBoundingPotentialEventHandler):
    """
    Event handler which treats a two-composite-object interaction using a cell bounding potential.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    To calculate the potential between the two composite objects, this class sums the potential between all pairs of
    leaf units located in different composite objects.

    Only one of the leaf units should be active. This class uses an potential and a cell bounding potential which expect
    a single separation. For the potential, the separation should be between two leaf units, for the cell bounding
    potential it is the cell separation between the cells of the two composite objects. The separation vector and the
    cell separation are corrected for periodic boundaries.

    If the potentials can consider charges, this event handler can pass the charges to the potentials. For the
    potential, this event handler uses the charges of the two relevant point masses. For the cell bounding potential,
    this event handler uses the charge of the active point mass, and the charges of all point masses in the target
    composite point object. The name of the used charge is set on initialization.

    Since the cell bounding potential requires a potential change argument in its displacement method, this event
    handler always samples a potential change.

    Since the interaction involves more than two leaf units, a lifting scheme is required.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information). The time displacements returned by the cell bounding
    potential's displacement method, however, are still simple floats because they are always of the same order of
    magnitude during a run of JF.
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
        base.exceptions.ConfigurationError:
            If the charge is not None and the cell bounding potential does not expect exactly two charge arguments.
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

        # self._potential_charges is set in TwoCompositeObjectBoundingPotentialEventHandler base class
        if charge is None:
            self._bounding_potential_charges = (lambda unit_one, target_units:
                                                tuple(1.0 for _ in
                                                      range(self._bounding_potential.number_charge_arguments)))
        else:
            if self._bounding_potential.number_charge_arguments == 2:
                self._bounding_potential_charges = (lambda unit_one, target_units:
                                                    (unit_one.charge[charge],
                                                     tuple(unit.charge[charge] for unit in target_units)))
            else:
                raise ConfigurationError("The event handler {0} was initialized with a charge which is not None,"
                                         " but its bounding potential {2} expects not exactly 2 charges."
                                         .format(self.__class__.__name__, self._potential.__class__.__name__,
                                                 self._bounding_potential.__class__.__name__))

        self._root_units = None
        self._active_root_unit_index = None
        self._active_root_unit = None
        self._charge = charge

    # noinspection PyMethodOverriding
    def initialize(self, cells: PeriodicCells) -> None:
        """
        Initialize the cell bounding potential event handler base class.

        This is done by handing the cells to the base class, and by telling it whether the cell bounding potential needs
        to determine a lower bound on the derivatives for all not excluded cell separations, or not. The lower bounds
        are required if this event handler considers a charge.

        Since in this version of JeLLyFysh, cell separations are only defined for periodic cell systems (i.e., with
        taking periodic boundary conditions into account), a cell bounding potential can only be initialized with an
        instance of the 'PeriodicCells' class. The same restriction thus holds for this event handler (see 'initialize'
        method of base 'CellBoundingPotentialEventHandler' class).

        Extends the initialize method of the abstract Initializer class. This method is called once in the beginning of
        the run by the activator. Only after a call of this method, other public methods of this class can be called
        without raising an error.

        Parameters
        ----------
        cells : activator.internal_state.cell_occupancy.cells.Cells
            The cell system.
        """
        super().initialize(cells, self._charge is not None)

    def send_event_time(self, in_state: Sequence[Node]) -> Time:
        """
        Return the candidate event time.

        The in-state should consist of all branches of the composite objects which take part in the interaction treated
        in this event handler. The candidate event time is calculated using the cell bounding potential. For this,
        the cell separation between the composite objects is calculated. The cell separation is calculated under
        consideration of periodic boundaries.

        Parameters
        ----------
        in_state : Sequence[base.node.Node]
            The in-state.

        Returns
        -------
        base.time.Time
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
        self._construct_leaf_units_of_composite_objects()

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
        assert self._relative_cell not in self._cells.nearby_cells(self._cells.zero_cell)

        time_displacement = self._bounding_potential.displacement(
            self._active_leaf_unit.velocity, self._relative_cell, *self._bounding_potential_charges(
                self._active_leaf_unit, self._target_leaf_units),
            random.expovariate(setting.beta))
        self._event_time = self._active_leaf_unit.time_stamp + time_displacement
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
        if (not all(not math.isnan(position) for position in self._leaf_units[self._active_leaf_unit_index].position)
                or not (self._cells.position_to_cell(self._root_units[self._active_root_unit_index].position)
                        == self._active_cell)):
            # Active unit changed cell and bounded event rate is effectively wrong.
            # Will never be relevant, since cell boundary event comes in first -> return None
            return None

        bounding_event_rate = self._bounding_potential.derivative(
            self._active_leaf_unit.velocity, self._relative_cell,
            *self._bounding_potential_charges(self._active_leaf_unit, self._target_leaf_units))
        factor_derivative = 0.0
        target_composite_object_factor_derivatives = [0.0] * len(self._target_leaf_units)
        for index, leaf_unit in enumerate(self._target_leaf_units):
            pairwise_derivative = self._potential.derivative(
                self._active_leaf_unit.velocity,
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
