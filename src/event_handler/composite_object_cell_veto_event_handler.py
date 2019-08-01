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
"""Module for the CompositeObjectCellVetoEventHandler class"""
import logging
import random
from typing import Sequence, Union
from activator.internal_state.cell_occupancy.cells import PeriodicCells
from base.logging import log_init_arguments
from base.node import Node, yield_leaf_nodes
from estimator import Estimator
from lifting import Lifting
import setting
from .abstracts import CellVetoEventHandler, TwoCompositeObjectBoundingPotentialEventHandler
from .helper_functions import bounding_potential_warning


class CompositeObjectCellVetoEventHandler(CellVetoEventHandler, TwoCompositeObjectBoundingPotentialEventHandler):
    """
    Event handler which treats a two-composite-object interaction using the cell-veto algorithm.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    To calculate the potential between the two composite objects, this class sums the potential between all pairs of
    leaf units located in the different composite objects.
    The base class sets up and uses Walker's algorithm to sample a target cell and a candidate event time. The target
    cell is sampled under consideration of periodic boundary conditions.
    This class is used, when the relevant cell-occupancy system stores composite objects, each with more than one point
    mass. It assumes that a single leaf unit is active. Then the out-state can be calculated using a lifting scheme
    after the confirmation of the event.
    This event handler can include a charge, which is used in the estimator when it estimates bounds for the derivatives
    for different cell separations.
    """

    def __init__(self, estimator: Estimator, lifting: Lifting, charge: str = None) -> None:
        """
        The constructor of the CompositeObjectCellVetoEventHandler class.

        Parameters
        ----------
        estimator : estimator.Estimator
            The estimator used to determine bounds for the derivatives.
        lifting : lifting.Lifting
            The lifting scheme.
        charge : str or None, optional
            The relevant charge for this event handler.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           estimator=estimator.__class__.__name__, lifting=lifting.__class__.__name__,
                           charge=charge)
        super().__init__(estimator=estimator, lifting=lifting, charge=charge)
        self._charge = charge

    # noinspection PyMethodOverriding
    def initialize(self, cells: PeriodicCells, cell_level: int, extracted_global_state: Sequence[Node]) -> None:
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
        extracted_global_state : Sequence[base.node.Node]
            The extracted global state of the tree state handler.
        """
        super().initialize(cells, cell_level, extracted_global_state, self._charge)

    def send_out_state(self, target_cnode: Union[Node, None]) -> Sequence[Node]:
        """
        Return the out-state.

        This method receives the branch of the composite object in the sampled target cell.
        If it is None, the time-sliced active composite object branch which was transmitted in the send_event_time
        method is returned. Otherwise, first the event is confirmed. If it is confirmed, the lifting scheme determines
        the new active leaf unit, which is imprinted in the out-state consisting of both branches of the two composite
        objects.

        Parameters
        ----------
        target_cnode : Node or None
            The root cnode of the composite object in the sampled target cell.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.
        """
        if target_cnode is None:
            return self._state
        else:
            assert target_cnode.children
            self._state.append(target_cnode)
            for leaf_cnode in yield_leaf_nodes(target_cnode):
                self._leaf_cnodes.append(leaf_cnode)
                self._leaf_units.append(leaf_cnode.value)
            self._construct_leaf_units_of_composite_objects()

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
            assert self._bounding_event_rate >= 0.0
            bounding_potential_warning(self.__class__.__name__, self._bounding_event_rate, event_rate)
            if event_rate <= random.uniform(0.0, self._bounding_event_rate):
                return self._state

            self._fill_lifting(self._local_leaf_units, self._target_leaf_units,
                               factor_derivative, target_composite_object_factor_derivatives)

            next_active_identifier = self._lifting.get_active_identifier()
            next_active_cnode = [cnode for cnode in self._leaf_cnodes
                                 if cnode.value.identifier == next_active_identifier]
            assert len(next_active_cnode) == 1
            self._exchange_velocity(self._leaf_cnodes[self._active_leaf_unit_index], next_active_cnode[0])
            return self._state
