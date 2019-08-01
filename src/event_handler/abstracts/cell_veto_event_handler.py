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
"""Module for the abstract CellVetoEventHandler class."""
from abc import ABCMeta
from copy import copy
import random
from typing import Any, List, Sequence, Tuple, Union
from activator.internal_state.cell_occupancy.cells import PeriodicCells
from base.exceptions import ConfigurationError
from base.initializer import Initializer
import base.node as node
from estimator import Estimator
from event_handler.walker import Walker, WalkerItem
import setting
from .event_handler_with_bounding_potential import EventHandlerWithBoundingPotential


class CellVetoEventHandler(EventHandlerWithBoundingPotential, Initializer, metaclass=ABCMeta):
    """
    The base class for all event handlers implementing the cell-veto algorithm.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    It sets up and uses Walker's algorithm to sample a target cell and a candidate event time. The calculation of the
    out-state which might include a lifting is left open. The target cell is sampled under consideration of periodic
    boundary conditions.
    The mediating method of this class gets the sampled target cell, retrieves the stored global state identifier in the
    cell-occupancy system which is stored in the activator and converts it into a branch using the tree state handler.
    This branch then accompanies the out-state request.
    This class uses an estimator, to determine upper bounds for the derivatives for all not excluded cell separations.
    The event rates are put into a Walker scheme (see event_handler.Walker) which stores the total cell event rate and
    allows to sample a target cell.
    """

    def __init__(self, estimator: Estimator, **kwargs: Any) -> None:
        """
        The constructor of the CellVetoEventHandler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        estimator : estimator.Estimator
            The estimator used to determine bounds for the derivatives.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(potential=estimator.potential, **kwargs)
        self._estimator = estimator
        self._cells = None
        self._walker = None
        self._derivative_upper_bounds = None
        self._cell_level = None

    def initialize(self, cells: PeriodicCells, cell_level: int, extracted_global_state: Sequence[node.Node],
                   charge: Union[str, None]) -> None:
        """
        Initialize the cell veto event handler.

        This is done by storing the cells and also by extracting the maximum charge from the extracted global state.
        This charge is used to initialize the estimator. If the charge is None, the estimator is initialized with a
        maximum charge one.
        Then the estimator is used to set up the Walker class.
        Extends the initialize method of the abstract Initializer class. This method is called once in the beginning of
        the run by the activator. Only after a call of this method, other public methods of this class can be called
        without raising an error.

        Parameters
        ----------
        cells : activator.internal_state.cell_occupancy.cells.PeriodicCells
            The periodic cell system.
        cell_level : int
            The cell level of the cell-occupancy system this event handler corresponds to. For the tree state handler
            this number equals the length of the stored global state identifiers.
        extracted_global_state : Sequence[base.node.Node]
            The extracted global state of the tree state handler.
        charge : str or None
            The relevant charge for this event handler.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the cells are not an instance of PeriodicCells.
        """
        if not isinstance(cells, PeriodicCells):
            raise ConfigurationError("The event handler {0} needs an instance of PeriodicCells!"
                                     .format(self.__class__.__name__))
        Initializer.initialize(self)
        max_charge = (max(abs(leaf_node.value.charge[charge])
                          for root_cnode in extracted_global_state
                          for leaf_node in node.yield_leaf_nodes(root_cnode))
                      if charge is not None else 1.0)
        self._estimator.initialize(max_charge)
        self._cells = cells
        self._cell_level = cell_level
        self._derivative_upper_bounds = []
        print("Initializing the cell-veto event handler {0}.".format(self.__class__.__name__))
        for cell in self._cells.yield_cells():
            self._derivative_upper_bounds.append([None for _ in range(setting.dimension)])

            if cell not in self._cells.excluded_cells(0):
                cell_lower_corner = copy(self._cells.cell_min(cell))
                cell_upper_corner = copy(self._cells.cell_max(cell))
                for d in range(setting.dimension):
                    while cell_upper_corner[d] < cell_lower_corner[d]:
                        cell_upper_corner[d] = setting.periodic_boundaries.next_image(cell_upper_corner[d], d)

                ref_cell_lower_corner = copy(self._cells.cell_min(0))
                ref_cell_upper_corner = copy(self._cells.cell_max(0))
                for d in range(setting.dimension):
                    while ref_cell_upper_corner[d] < ref_cell_lower_corner[d]:
                        ref_cell_upper_corner[d] = setting.periodic_boundaries.next_image(ref_cell_upper_corner[d], d)

                lower_corner = [cell_lower_corner[d] - ref_cell_upper_corner[d] for d in range(setting.dimension)]
                upper_corner = [cell_upper_corner[d] - ref_cell_lower_corner[d] for d in range(setting.dimension)]

                for direction in range(setting.dimension):
                    upper_bound, lower_bound = self._estimator.derivative_bound(
                        lower_corner, upper_corner, direction, calculate_lower_bound=True)
                    self._derivative_upper_bounds[cell][direction] = max(-lower_bound, upper_bound)

        walker_item_lists = [[] for _ in range(setting.dimension)]
        for cell in self._cells.yield_cells():
            if cell not in self._cells.excluded_cells(0):
                for direction in range(setting.dimension):
                    # noinspection PyTypeChecker
                    walker_item_lists[direction].append(WalkerItem(cell,
                                                                   self._derivative_upper_bounds[cell][direction]))
        self._walker = [Walker(item_list) for item_list in walker_item_lists]

        print("Finished initialization of the cell-veto event handler {0}.".format(self.__class__.__name__))

    def send_event_time(self, in_state: Sequence[node.Node]) -> Tuple[float, List[int]]:
        """
        Return the candidate event time together with the sampled target cell.

        This is done using the total event rate stored in the Walker class. The in-state consists of a single root
        cnode, which contains a single independent active unit on the cell level.
        This method returns the sampled target cell besides the candidate event time, so that the corresponding branch
        can be received in the send_out_state method.
        Also, the event rate bound between the cell of the active unit on cell level and the target cell is recorded in
        the _bounding_event_rate attribute.

        Parameters
        ----------
        in_state : Sequence[base.node.Node]
            The branch of the independent active unit.

        Returns
        -------
        (float, [int])
            The candidate event time, the sampled target cell.

        Raises
        ------
        AssertionError
            If the in-state consists of more than one root cnode.
        AssertionError
            If the bounding event rate is smaller than zero.
        """
        assert len(in_state) == 1
        self._store_in_state(in_state)
        self._construct_leaf_cnodes()
        self._extract_active_leaf_unit()

        relevant_cnode = self._leaf_cnodes[0]
        while len(relevant_cnode.value.identifier) > self._cell_level:
            relevant_cnode = relevant_cnode.parent

        active_cell = self._cells.position_to_cell(relevant_cnode.value.position)
        total_rate = self._walker[self._direction_of_motion].total_rate
        relative_cell = self._walker[self._direction_of_motion].sample_cell()
        self._bounding_event_rate = self._derivative_upper_bounds[relative_cell][self._direction_of_motion]
        assert self._bounding_event_rate > 0.0
        target_cell = self._cells.translate(active_cell, relative_cell)
        # TODO add a seeding option at each place a random number is used so that we can insert random numbers
        displacement = random.expovariate(setting.beta) / total_rate
        self._event_time = self._active_leaf_unit.time_stamp + displacement / self._speed
        self._time_slice_all_units_in_state()

        return self._event_time, [target_cell]
