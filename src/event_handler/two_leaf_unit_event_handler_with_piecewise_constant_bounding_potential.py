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
"""Module for the TwoLeafUnitEventHandlerWithPiecewiseConstantBoundingPotential class."""
import logging
import random
from typing import List, Sequence, Tuple
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.node import Node
from base.unit import Unit
from potential import Potential
import setting
from .abstracts import EventHandlerWithPiecewiseConstantBoundingPotential
from .helper_functions import bounding_potential_warning


class TwoLeafUnitEventHandlerWithPiecewiseConstantBoundingPotential(EventHandlerWithPiecewiseConstantBoundingPotential):
    """
    Event handler which uses a dynamically created piecewise constant bounding potential for an interaction between two
    leaf units.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    The bounding potential is implemented in the base class EventHandlerWithPiecewiseConstantBoundingPotential. See the
    documentation of this class for more details.
    This class uses an potential which expects a single separation between two leaf units. Also, this event handler
    can pass a charge to the potential. The separation vector between the interacting leaf units is corrected for
    periodic boundaries.
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
            The maximum displacement used to create piecewise constant bounding potential.
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

    def send_event_time(self, in_state: Sequence[Node]) -> float:
        """
        Return the candidate event time.

        The in-state should consist of all branches of the two leaf units which take part in the interaction treated in
        this event handler.
        This method uses the dynamically created piecewise constant bounding potential to compute the candidate event
        time. For this, it extracts all leaf units from the in-state. Then it constructs the separations which are
        passed to the potential using the _get_separations method.

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
        """
        self._store_in_state(in_state)
        self._construct_leaf_cnodes()
        self._extract_active_leaf_unit()
        assert len(self._leaf_cnodes) == 2
        displacement = self._displacement_from_piecewise_constant_bounding_potential(random.expovariate(setting.beta))
        self._event_time = self._active_leaf_unit.time_stamp + displacement / self._speed
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
        separation = self._get_separations(self._leaf_units)[0]
        self._bounding_event_rate = self._event_rate_from_piecewise_constant_bounding_potential()
        if self._bounding_event_rate is not None:
            real_derivative = self._potential.derivative(self._direction_of_motion, separation, *charges)
            if real_derivative > 0:
                bounding_potential_warning(self.__class__.__name__, self._bounding_event_rate, real_derivative)
                if random.uniform(0, self._bounding_event_rate) < real_derivative:
                    self._exchange_velocity(self._leaf_cnodes[self._active_leaf_unit_index],
                                            self._leaf_cnodes[self._active_leaf_unit_index ^ 1])
        return self._state

    def _get_separations(self, units: Sequence[Unit]) -> List[Sequence[float]]:
        """
        Return the sequence of separations relevant for the potential.

        This method returns the shortest separation vector between the two leaf units.

        Parameters
        ----------
        units : Sequence[base.unit.Unit]
            The sequence of leaf units.

        Returns
        -------
        List[Sequence[float]]
            The list of separations.
        """
        return [setting.periodic_boundaries.separation_vector(units[self._active_leaf_unit_index].position,
                                                              units[self._active_leaf_unit_index ^ 1].position)]

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
