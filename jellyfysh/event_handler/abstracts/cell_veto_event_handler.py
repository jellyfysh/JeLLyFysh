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
"""Module for the abstract CellVetoEventHandler class."""
from abc import ABCMeta
import random
from typing import Any, List, Sequence, Tuple, Union
from jellyfysh.activator.internal_state.cell_occupancy.cells import PeriodicCells
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.initializer import Initializer
import jellyfysh.base.node as node
from jellyfysh.base.time import Time
from jellyfysh.estimator import Estimator
from jellyfysh.event_handler.walker import Walker, WalkerItem
import jellyfysh.setting as setting
from .event_handler_with_bounding_potential import EventHandlerWithBoundingPotential


class CellVetoEventHandler(EventHandlerWithBoundingPotential, Initializer, metaclass=ABCMeta):
    """
    The base class for all event handlers implementing the cell-veto algorithm for a single active leaf unit with an
    direction of motion in positive direction along an axis.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    Although only a single leaf unit should be active, the cell system may track composite point objects to treat
    interactions between composite point objects.

    It sets up and uses Walker's algorithm to sample a target cell and a candidate event time. The calculation of the
    out-state which might include a lifting is left open. The target cell is sampled under consideration of periodic
    boundary conditions.

    The mediating method of this class gets the sampled target cell, retrieves the stored global state identifier in the
    cell-occupancy system which is stored in the activator and converts it into a branch using the tree state handler.
    This branch then accompanies the out-state request.

    This class uses an estimator to determine upper and lower bounds for the derivatives for all non-nearby cell
    separations. The event rates are put into a Walker scheme (see event_handler.Walker) which stores the total cell
    event rate and allows to sample a target cell.

    This event handler can consider the charge of the active leaf unit by using the charge correction factor of the
    estimator that is used to estimate upper and lower bounds on the derivative for any non-nearby cell separation. The
    name of the used charge is set in the initialize method.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information).
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
        super().__init__(**kwargs)
        self._estimator = estimator
        self._cells = None
        self._upper_bound_walker = None
        self._lower_bound_walker = None
        self._derivative_bounds = None
        self._cell_level = None
        self._charge_of_unit = None

    def initialize(self, cells: PeriodicCells, cell_level: int, charge: Union[str, None]) -> None:
        """
        Initialize the cell veto event handler.

        This is done by storing the cells, and the cell level of the relevant cell occupancy system for this event
        handler. Further, the charge of this event handler is an argument of this method. (Note that the charge is not
        an argument of the __init__ method to solve problems with multiple inheritance. This occurs, for example, in
        the CompositeObjectCellVetoEventHandler.)

        Then the estimator is used to set up two Walker classes, one for the upper and one for the lower bounds.
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
        self._cells = cells
        self._cell_level = cell_level
        self._derivative_bounds = {}
        if charge is None:
            self._charge_of_unit = lambda unit: 1.0
        else:
            self._charge_of_unit = lambda unit: unit.charge[charge]
        print("Initializing the cell-veto event handler {0}.".format(self.__class__.__name__))
        for cell in self._cells.yield_cells():
            if cell not in cells.nearby_cells(cells.zero_cell):
                cell_separation = cells.relative_cell(cell, cells.zero_cell)
                self._derivative_bounds[cell_separation] = []
                lower_corner = [cell.cell_min[direction] - cells.zero_cell.cell_max[direction]
                                for direction in range(setting.dimension)]
                upper_corner = [cell.cell_max[direction] - cells.zero_cell.cell_min[direction]
                                for direction in range(setting.dimension)]

                for direction in range(setting.dimension):
                    upper_bound, lower_bound = self._estimator.derivative_bound(
                        lower_corner, upper_corner, direction, calculate_lower_bound=True)
                    self._derivative_bounds[cell_separation].append((upper_bound, -lower_bound))

        upper_bound_walker_items = [[] for _ in range(setting.dimension)]
        lower_bound_walker_items = [[] for _ in range(setting.dimension)]
        for cell in self._cells.yield_cells():
            if cell not in self._cells.nearby_cells(cells.zero_cell):
                cell_separation = cells.relative_cell(cell, cells.zero_cell)
                for direction in range(setting.dimension):
                    upper_bound_walker_items[direction].append(
                        WalkerItem(cell, max(self._derivative_bounds[cell_separation][direction][0], 0.0)))
                    lower_bound_walker_items[direction].append(
                        WalkerItem(cell, max(self._derivative_bounds[cell_separation][direction][1], 0.0)))
        self._upper_bound_walker = [Walker(item_list) for item_list in upper_bound_walker_items]
        self._lower_bound_walker = [Walker(item_list) for item_list in lower_bound_walker_items]
        print("Finished initialization of the cell-veto event handler {0}.".format(self.__class__.__name__))

    def send_event_time(self, in_state: Sequence[node.Node]) -> Tuple[Time, List[int]]:
        """
        Return the candidate event time together with the sampled target cell.

        This is done using the total event rate stored in the relevant Walker class. The in-state consists of a single
        root cnode, which contains a single independent active unit on the cell level, and a single active leaf unit.
        The relevant Walker class depends on the charge correction factor that is returned by the estimator based
        on the charge of the active leaf unit. If it is positive, the upper bound Walker class is used, otherwise,
        the lower bound Walker class.

        This method is only implemented for velocities of the active leaf unit that are aligned in positive direction
        with one of the cartesian axes.

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
        (base.time.Time, [int])
            The candidate event time, the sampled target cell.

        Raises
        ------
        AssertionError
            If the in-state consists of more than one root cnode.
        AssertionError
            If the bounding event rate is smaller than zero.
        AssertionError
            If the velocity is not in the positive direction parallel to one of the cartesian axes.
        """
        assert len(in_state) == 1
        self._store_in_state(in_state)
        self._construct_leaf_cnodes()
        self._extract_active_leaf_unit()
        direction_of_motions = [index for index, component in enumerate(self._active_leaf_unit.velocity)
                                if component != 0.0]
        assert len(direction_of_motions) == 1
        direction_of_motion = direction_of_motions[0]
        speed = self._active_leaf_unit.velocity[direction_of_motion]
        assert speed > 0.0

        relevant_cnode = self._leaf_cnodes[0]
        while len(relevant_cnode.value.identifier) > self._cell_level:
            relevant_cnode = relevant_cnode.parent

        active_cell = self._cells.position_to_cell(relevant_cnode.value.position)

        charge_factor = self._estimator.charge_correction_factor(self._charge_of_unit(self._active_leaf_unit))
        if charge_factor > 0.0:
            walker = self._upper_bound_walker[direction_of_motion]
            bounding_event_rate_index = 0
        else:
            charge_factor *= -1.0
            walker = self._lower_bound_walker[direction_of_motion]
            bounding_event_rate_index = 1

        total_rate = walker.total_rate * charge_factor
        relative_cell = walker.sample_cell()
        self._bounding_event_rate = (
                self._derivative_bounds[relative_cell][direction_of_motion][bounding_event_rate_index]
                * charge_factor)
        assert self._bounding_event_rate > 0.0
        target_cell = self._cells.translate(active_cell, relative_cell)
        # TODO add a seeding option at each place a random number is used so that we can insert random numbers
        time_displacement = random.expovariate(setting.beta) / (total_rate * speed)
        self._event_time = self._active_leaf_unit.time_stamp + time_displacement
        self._time_slice_all_units_in_state()

        return self._event_time, [target_cell]
