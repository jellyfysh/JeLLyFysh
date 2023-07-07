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
"""Module for the CompositeObjectCellVetoEventHandler class"""
import logging
import random
from typing import List, Sequence, Tuple, Union
from jellyfysh.activator.internal_state.cell_occupancy.cells import PeriodicCells
from jellyfysh.base.exceptions import bounding_potential_warning
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node, yield_leaf_nodes
from jellyfysh.base.time import Time
from jellyfysh.base import vectors
from jellyfysh.estimator import Estimator
from jellyfysh.lifting import Lifting
from jellyfysh.potential import Potential
import jellyfysh.setting as setting
from .abstracts import CellVetoEventHandler, TwoCompositeObjectBoundingPotentialEventHandler
from .fibonacci_sphere import FibonacciSphere


class CompositeObjectCellVetoEventHandler(CellVetoEventHandler, TwoCompositeObjectBoundingPotentialEventHandler):
    """
    Event handler which treats a two-composite-object interaction using the cell-veto algorithm.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    To calculate the potential between the two composite objects, this class sums the potential between all pairs of
    leaf units located in the different composite objects.

    Only a single leaf unit should be active with an direction of motion in positive direction along an axis.
    The base class sets up and uses Walker's algorithm to sample a target cell and a candidate event time. The target
    cell is sampled under consideration of periodic boundary conditions.

    This class is used, when the relevant cell-occupancy system stores composite objects, each with more than one point
    mass. Then, the out-state can be calculated using a lifting scheme after the confirmation of the event.

    This event handler can consider the charge of the active leaf unit by using the charge correction factor of the
    estimator that is used to estimate upper and lower bounds on the derivative for any non-nearby cell separation. The
    name of the used charge is set on initialization.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information).
    """

    def __init__(self, estimator: Estimator, lifting: Lifting, fibonacci_sphere: FibonacciSphere = FibonacciSphere(),
                 potential: Potential = None, charge: str = None, derivative_bounds_input_filename: str = None,
                 derivative_bounds_correction_factor: float = 1.0,
                 derivative_bounds_output_filename: str = None) -> None:
        """
        The constructor of the CompositeObjectCellVetoEventHandler class.

        If the potential is None, the potential of the estimator is used.

        Parameters
        ----------
        estimator : estimator.Estimator
            The estimator used to determine bounds for the derivatives.
        lifting : lifting.Lifting
            The lifting scheme.
        potential : potential.Potential or None, optional
            The potential between two leaf units.
        charge : str or None, optional
            The relevant charge for this event handler.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           estimator=estimator.__class__.__name__, lifting=lifting.__class__.__name__,
                           fibonacci_sphere=fibonacci_sphere.__class__.__name__,
                           potential=None if potential is None else potential.__class__.__name__, charge=charge,
                           derivative_bounds_input_filename=derivative_bounds_input_filename,
                           derivative_bounds_correction_factor=derivative_bounds_correction_factor,
                           derivative_bounds_output_filename=derivative_bounds_output_filename)
        super().__init__(estimator=estimator, lifting=lifting, fibonacci_sphere=fibonacci_sphere,
                         potential=estimator.potential if potential is None else potential, charge=charge,
                         derivative_bounds_input_filename=derivative_bounds_input_filename,
                         derivative_bounds_correction_factor=derivative_bounds_correction_factor,
                         derivative_bounds_output_filename=derivative_bounds_output_filename)
        self._charge = charge

    # noinspection PyMethodOverriding
    def initialize(self, cells: PeriodicCells, cell_level: int) -> None:
        """
        Initialize this event handler.

        Extends the initialize method of the abstract CellVetoEventHandler class. This method is called once in the
        beginning of the run by the activator. Only after a call of this method, other public methods of this class can
        be called without raising an error.

        This method passes through all relevant arguments to the initialize method of the base class. There the Walker
        class is initialized properly.

        Parameters
        ----------
        cells : activator.internal_state.cell_occupancy.cells.PeriodicCells
            The periodic cell system.
        cell_level : int
            The cell level of the cell-occupancy system this event handler corresponds to. For the tree state handler
            this number equals the length of the stored global state identifiers.
        """
        super().initialize(cells, cell_level, self._charge)

    def send_event_time(self, in_state: Sequence[Node]) -> Tuple[Time, List[int]]:
        """
        Return the candidate event time together with the sampled target cell using the send_event_time method of the
        abstract CellVetoEventHandler base class.

        This method additionally stores the sorted leaf units in the composite point object that contains the active
        leaf unit. This is used in the send_out_state method.

        Parameters
        ----------
        in_state : Sequence[base.node.Node]
            The branch of the independent active unit.

        Returns
        -------
        (base.time.Time, [int])
            The candidate event time, the sampled target cell.
        """
        return_value = super().send_event_time(in_state)
        self._local_leaf_units = sorted(self._leaf_units, key=lambda unit: unit.identifier)
        return return_value

    def send_out_state(self, target_cnode: Union[Node, None]) -> Union[Sequence[Node], None]:
        """
        Return the out-state.

        This method receives the branch of the composite object in the sampled target cell. If it is None, the
        time-sliced active composite object branch which was transmitted in the send_event_time method is returned.
        Otherwise, first the event is confirmed. If it is confirmed, the lifting scheme determines the new active leaf
        unit, which is imprinted in the out-state consisting of both branches of the two composite objects.

        Parameters
        ----------
        target_cnode : Node or None
            The root cnode of the composite object in the sampled target cell.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.
        """
        if target_cnode is not None:
            assert target_cnode.children
            target_leaf_cnodes = []
            target_leaf_units = []
            for leaf_cnode in yield_leaf_nodes(target_cnode):
                target_leaf_cnodes.append(leaf_cnode)
                target_leaf_units.append(leaf_cnode.value)
            target_leaf_units = sorted(target_leaf_units, key=lambda unit: unit.identifier)
            factor_derivative = 0.0
            factor_gradient = [0.0 for _ in range(setting.dimension)]
            target_composite_object_factor_gradients = [[0.0 for _ in range(setting.dimension)]
                                                        for _ in range(len(target_leaf_units))]
            for index, target_leaf_unit in enumerate(target_leaf_units):
                pairwise_gradient = self._potential.gradient(
                    setting.periodic_boundaries.separation_vector(
                        self._active_leaf_unit.position, target_leaf_unit.position),
                    *self._potential_charges(self._active_leaf_unit, target_leaf_unit))
                for i, g in enumerate(pairwise_gradient):
                    factor_gradient[i] += g
                    target_composite_object_factor_gradients[index][i] -= g
                factor_derivative += vectors.dot(self._active_leaf_unit.velocity, pairwise_gradient)
            event_rate = max(0.0, factor_derivative)
            bounding_potential_warning(self.__class__.__name__, self._bounding_event_rate, event_rate)
            if random.uniform(0.0, self._bounding_event_rate) < event_rate:
                self._state.append(target_cnode)
                for target_leaf_cnode in target_leaf_cnodes:
                    self._leaf_cnodes.append(target_leaf_cnode)
                    self._leaf_units.append(target_leaf_cnode.value)
                new_local_velocities, new_target_velocities = self._fill_lifting(
                    self._local_leaf_units, target_leaf_units, factor_gradient,
                    target_composite_object_factor_gradients)
                new_active_identifier, change_velocities = self._lifting.get_active_identifier()
                assert not (not change_velocities and new_active_identifier == self._active_leaf_unit.identifier)
                time_stamp = self._active_leaf_unit.time_stamp
                self._active_leaf_unit.time_stamp = None
                self._register_velocity_change_leaf_cnode(self._leaf_cnodes[self._active_leaf_unit_index],
                                                          [-c for c in self._active_leaf_unit.velocity])
                new_active_cnode = [cnode for cnode in self._leaf_cnodes
                                    if cnode.value.identifier == new_active_identifier]
                assert len(new_active_cnode) == 1
                new_active_cnode[0].value.time_stamp = time_stamp
                if change_velocities:
                    for index_1, local_unit in enumerate(self._local_leaf_units):
                        local_unit.velocity = new_local_velocities[index_1]
                    for index_2, target_unit in enumerate(target_leaf_units):
                        target_unit.velocity = new_target_velocities[index_2]
                self._register_velocity_change_leaf_cnode(new_active_cnode[0],
                                                          new_active_cnode[0].value.velocity)
                self._commit_non_leaf_velocity_changes()
                return self._state
        return None
