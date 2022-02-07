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
"""Module for the LeafUnitCellVetoEventHandler class."""
import logging
from typing import Sequence, Union
from jellyfysh.activator.internal_state.cell_occupancy.cells import PeriodicCells
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node, yield_leaf_nodes
from jellyfysh.estimator import Estimator
from jellyfysh.potential import Potential
import jellyfysh.setting as setting
from .abstracts import CellVetoEventHandler


class LeafUnitCellVetoEventHandler(CellVetoEventHandler):
    """
    Event handler which treats a two-leaf-unit interaction using the cell-veto algorithm.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    Only a single leaf unit should be active with an direction of motion in positive direction along an axis.
    The base class sets up and uses Walker's algorithm to sample a target cell and a candidate event time. The target
    cell is sampled under consideration of periodic boundary conditions.

    This class is used, when the relevant cell-occupancy system stores point masses. The out-state can be calculated by
    exchanging the velocities of the two leaf units after the confirmation of the event.

    This event handler can consider the charge of the active leaf unit by using the charge correction factor of the
    estimator that is used to estimate upper and lower bounds on the derivative for any non-nearby cell separation. The
    name of the used charge is set on initialization.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information).
    """

    def __init__(self, estimator: Estimator, potential: Potential = None, charge: str = None) -> None:
        """
        The constructor of the LeafUnitCellVetoEventHandler class.

        If the potential is None, the potential of the estimator is used.

        Parameters
        ----------
        estimator : estimator.Estimator
            The estimator used to determine bounds for the derivatives.
        potential : potential.Potential or None, optional
            The potential between two leaf units.
        charge : str or None, optional
            The relevant charge for this event handler.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the charge is not None but the potential expects more than two charges.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           estimator=estimator.__class__.__name__,
                           potential=None if potential is None else potential.__class__.__name__, charge=charge)
        super().__init__(estimator=estimator, potential=estimator.potential if potential is None else potential)
        self._charge = charge
        if charge is None:
            self._charges = lambda unit_one, unit_two: tuple(1.0 for _ in
                                                             range(self._potential.number_charge_arguments))
        else:
            if self._potential.number_charge_arguments == 2:
                self._charges = lambda unit_one, unit_two: (unit_one.charge[charge], unit_two.charge[charge])
            else:
                raise ConfigurationError("The event handler {0} was initialized with a charge which is not None,"
                                         " but its potential {1} expects not exactly 2 charges."
                                         .format(self.__class__.__name__, self._potential.__class__.__name__))

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

    def send_out_state(self, target_unit_root_cnode: Union[Node, None]) -> Sequence[Node]:
        """
        Return the out-state.

        This method receives the branch of the leaf unit in the sampled target cell. If it is None, the time-sliced
        active leaf unit branch which was transmitted in the send_event_time method is returned. Otherwise, first the
        event is confirmed. If it is confirmed, the velocities are exchanged and the branches are kept consistent in the
        out-state.

        Parameters
        ----------
        target_unit_root_cnode : Node or None
            The branch of the leaf unit in the sampled target cell.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.
        """
        if target_unit_root_cnode is None:
            return self._state
        else:
            self._state.append(target_unit_root_cnode)
            for leaf_cnode in yield_leaf_nodes(target_unit_root_cnode):
                self._leaf_cnodes.append(leaf_cnode)
                self._leaf_units.append(leaf_cnode.value)
            assert len(self._leaf_units) == 2
            self._calculate_out_state_of_two_leaf_unit_bounding_potential(
                setting.periodic_boundaries.separation_vector(
                    self._leaf_units[self._active_leaf_unit_index].position,
                    self._leaf_units[self._active_leaf_unit_index ^ 1].position),
                self._charges(self._leaf_units[0], self._leaf_units[1]),)
            return self._state
