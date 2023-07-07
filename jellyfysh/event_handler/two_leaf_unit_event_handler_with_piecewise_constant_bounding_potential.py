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
"""Module for the TwoLeafUnitEventHandlerWithPiecewiseConstantBoundingPotential class."""
import logging
import random
from typing import List, Sequence, Tuple
from jellyfysh.base.exceptions import ConfigurationError, bounding_potential_warning
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
from jellyfysh.base.unit import Unit
from jellyfysh.potential import Potential
import jellyfysh.setting as setting
from .abstracts import EventHandlerWithPiecewiseConstantBoundingPotential


class TwoLeafUnitEventHandlerWithPiecewiseConstantBoundingPotential(EventHandlerWithPiecewiseConstantBoundingPotential):
    """
    Event handler which uses a dynamically created piecewise constant bounding potential for an interaction between two
    leaf units.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    The bounding potential is implemented in the base class EventHandlerWithPiecewiseConstantBoundingPotential. See the
    documentation of this class for more details.

    Only one of the two leaf units should be active. This class uses an potential which expects a single separation
    between two leaf units. The separation vector between the interacting leaf units is corrected for periodic
    boundaries.

    If the potentials can consider charges, this event handler can pass the charges of the two leaf units to the
    potential. The name of the used charge is set on initialization.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information). The maximum time displacement, however, can stay a simple
    float because it is always of the same order of magnitude during a run of JF. The same is true for the time
    displacement returned by the dynamically constructed bounding potential.
    """

    def __init__(self, potential: Potential, offset: float, max_displacement: float, charge: str = None) -> None:
        """
        The constructor of the TwoLeafUnitEventHandlerWithPiecewiseConstantBoundingPotential class.

        Parameters
        ----------
        potential : potential.Potential
            The potential between the leaf units.
        offset : float
            The offset used to create piecewise constant bounding potential.
        max_displacement :
            The maximum time displacement used to create piecewise constant bounding potential.
        charge : str or None, optional
            The relevant charge for this event handler.

        Raises
        ------
        base.exceptions.ConfigurationError:
            If the potential does not expect exactly one separation.
        base.exceptions.ConfigurationError:
            If the charge is not None but the potential expects more than two charges.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential.__class__.__name__, offset=offset, max_displacement=max_displacement,
                           charge=charge)
        super().__init__(potential=potential, offset=offset, max_displacement=max_displacement)
        if self._potential.number_separation_arguments != 1:
            raise ConfigurationError("The event handler {0} expects a potential "
                                     "which handles exactly one separation!".format(self.__class__.__name__))

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

    def send_event_time(self, in_state: Sequence[Node]) -> Time:
        """
        Return the candidate event time.

        The in-state should consist of all branches of the two leaf units which take part in the interaction treated in
        this event handler.

        This method uses the dynamically created piecewise constant bounding potential to compute the candidate event
        time. For this, it extracts all leaf units from the in-state. Then it uses the
        _displacement_from_piecewise_constant_bounding_potential method that is implemented in the
        EventHandlerWithPiecewiseConstantBoundingPotential base class.

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
            If the in-state contains not exactly two leaf units.
        """
        self._store_in_state(in_state)
        self._construct_leaf_cnodes()
        self._extract_active_leaf_unit()
        assert len(self._leaf_cnodes) == 2
        time_displacement = self._displacement_from_piecewise_constant_bounding_potential(
            random.expovariate(setting.beta))
        self._event_time = self._active_leaf_unit.time_stamp + time_displacement
        self._time_slice_all_units_in_state()
        return self._event_time

    def send_out_state(self) -> Sequence[Node]:
        """
        Return the out-state.

        First, this method confirms the event. If it is confirmed, the velocities of the two leaf units are exchanged
        and the branches are kept consistent.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.
        """
        charges = self._get_charges(self._leaf_units)
        separation = self._get_separations([unit.position for unit in self._leaf_units])[0]
        self._bounding_event_rate = self._event_rate_from_piecewise_constant_bounding_potential()
        if self._bounding_event_rate is not None:
            real_derivative = self._potential.derivative(self._active_leaf_unit.velocity, separation, *charges)
            if real_derivative > 0:
                bounding_potential_warning(self.__class__.__name__, self._bounding_event_rate, real_derivative)
                if random.uniform(0, self._bounding_event_rate) < real_derivative:
                    self._exchange_velocity(self._leaf_cnodes[self._active_leaf_unit_index],
                                            self._leaf_cnodes[self._active_leaf_unit_index ^ 1],
                                            self._potential.gradient(separation, *charges))
        return self._state

    def _get_separations(self, positions: Sequence[Sequence[float]]) -> List[Sequence[float]]:
        """
        Return the sequence of separations relevant for the potential based on the positions of the leaf units.

        Note that the positions in the arguments are in the same order as self._leaf_units.

        This method returns the shortest separation vector between the two leaf units.

        Parameters
        ----------
        positions : Sequence[Sequence[float]]
            The sequence of positions of the leaf units.

        Returns
        -------
        List[Sequence[float]]
            The list of separations.
        """
        return [setting.periodic_boundaries.separation_vector(positions[self._active_leaf_unit_index],
                                                              positions[self._active_leaf_unit_index ^ 1])]

    def _get_charges(self, units: Sequence[Unit]) -> Tuple[float, ...]:
        """
        Return the sequence of charges relevant for the potential.

        Parameters
        ----------
        units : Sequence[base.unit.Unit]
            The sequence of leaf units.

        Returns
        -------
        Tuple[float, ...]
            The tuple of charges.
        """
        return self._charges(units[0], units[1])
