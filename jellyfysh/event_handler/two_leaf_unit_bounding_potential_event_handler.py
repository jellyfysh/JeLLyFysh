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
"""Module for the TwoLeafUnitBoundingPotentialEventHandler class."""
import logging
import random
from typing import Sequence
from jellyfysh.base.exceptions import ConfigurationError, bounding_potential_warning
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
import jellyfysh.base.vectors as vectors
from jellyfysh.potential import Potential, InvertiblePotential
import jellyfysh.setting as setting
from .abstracts import EventHandlerWithBoundingPotential


class TwoLeafUnitBoundingPotentialEventHandler(EventHandlerWithBoundingPotential):
    """
    Event handler which uses a bounding potential (or the sum of several bounding potentials) for an interaction between
    two leaf units.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    Only one of the two leaf units should be active. This class uses an potential and one or several bounding
    potentials which expect a single separation between two leaf units. The separation vector between the interacting
    leaf units is corrected for periodic boundaries.

    If the (bounding) potentials can consider charges, this event handler can pass the charges of the two leaf units to
    the potentials. The name of the used charge is set on initialization.

    This event handler always samples a potential change that is passed to the displacement methods of the bounding
    potentials. If several bounding potentials are used, the minimum time displacement determines the next proposed
    event. For the confirmation of this event, the sum of the event rates of all bounding potentials is used.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information). The time displacements returned by the bounding potential's
    displacement method, however, are still simple floats because they are always of the same order of magnitude during
    a run of JF.
    """

    def __init__(self, potential: Potential, bounding_potential: Sequence[InvertiblePotential],
                 charge: str = None) -> None:
        """
        The constructor of the TwoLeafUnitBoundingPotentialEventHandler class.

        Parameters
        ----------
        potential : potential.Potential
            The potential between the leaf units.
        bounding_potential : Sequence[potential.InvertiblePotential]
            The invertible bounding potentials between the leaf units.
        charge : str or None, optional
            The relevant charge for this event handler.

        Raises
        ------
        base.exceptions.ConfigurationError:
            If the bounding potentials do not expect exactly one separation.
        base.exceptions.ConfigurationError:
            If the charge is not None but the potential or one of the bounding potentials does not expect exactly two
            charges.
        base.exceptions.ConfigurationError:
            If the displacement method of one the bounding potentials does not require a potential change.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential.__class__.__name__,
                           bounding_potential=[bounding_potential.__class__.__name__
                                               for bounding_potential in bounding_potential],
                           charge=charge)
        super().__init__(potential=potential)
        self._bounding_potentials = bounding_potential
        for bounding_potential in self._bounding_potentials:
            if bounding_potential.number_separation_arguments != 1:
                raise ConfigurationError("The event handler {0} expects (possibly several) bounding potentials "
                                         "that handle exactly one separation! The bounding potential {1} expects "
                                         "{2} separations.".format(self.__class__.__name__,
                                                                   bounding_potential.__class__.__name__,
                                                                   bounding_potential.number_separation_arguments))

        if charge is None:
            self._potential_charges = (
                lambda unit_one, unit_two: tuple(1.0 for _ in range(self._potential.number_charge_arguments)))
            self._bounding_potentials_charges = [
                lambda unit_one, unit_two: tuple(1.0 for _ in range(bounding_potential.number_charge_arguments))
                for bounding_potential in self._bounding_potentials]
        else:
            if (self._potential.number_charge_arguments == 2
                    and all(bounding_potential.number_charge_arguments == 2
                            for bounding_potential in self._bounding_potentials)):
                self._potential_charges = lambda unit_one, unit_two: (unit_one.charge[charge], unit_two.charge[charge])
                self._bounding_potentials_charges = [self._potential_charges for _ in self._bounding_potentials]
            else:
                raise ConfigurationError("The event handler {0} was initialized with a charge which is not None,"
                                         " but its potential {1} and/or one of its bounding potentials {2}"
                                         " expects not exactly 2 charges."
                                         .format(self.__class__.__name__, self._potential.__class__.__name__,
                                                 [bounding_potential.__class__.__name__
                                                  for bounding_potential in self._bounding_potentials]))
        for bounding_potential in self._bounding_potentials:
            if not bounding_potential.potential_change_required:
                raise ConfigurationError("The event handler {0} expects (possibly several) bounding potentials that "
                                         "require a potential change in their displacement methods. The bounding "
                                         "potential {1} does not expect a potential change."
                                         .format(self.__class__.__name__, bounding_potential.__class__.__name__))

    def send_event_time(self, in_state: Sequence[Node]) -> Time:
        """
        Return the candidate event time.

        The in-state should consist of all branches of the two leaf units which take part in the interaction treated in
        this event handler. The candidate event time is calculated from the minimum time displacement of the bounding
        potentials.

        Note that the separation vector between the two leaf units is only computed once and then passed to all bounding
        potentials. If several bounding potentials are used, these should therefore not modify the separation argument
        in their displacement method.

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
        separation = setting.periodic_boundaries.separation_vector(
            self._leaf_units[self._active_leaf_unit_index].position,
            self._leaf_units[self._active_leaf_unit_index ^ 1].position)
        time_displacement = min(bounding_potential.displacement(
            self._active_leaf_unit.velocity, separation, *charges(self._leaf_units[0], self._leaf_units[1]),
            random.expovariate(setting.beta)) for bounding_potential, charges in zip(self._bounding_potentials,
                                                                                     self._bounding_potentials_charges))
        self._event_time = self._active_leaf_unit.time_stamp + time_displacement
        self._time_slice_all_units_in_state()
        return self._event_time

    def send_out_state(self) -> Sequence[Node]:
        """
        Return the out-state.

        First, this method confirms the event. For this it uses the real event rate at the time of the event, and
        compares it to the sum of the event rates of all bounding potentials. If the event is confirmed, the velocities
        of the two leaf units are exchanged and the branches are kept consistent.

        Note that the separation vector between the two leaf units is only computed once and then passed to all bounding
        potentials. If several bounding potentials are used, these should therefore not modify the separation argument
        in their derivative method.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.
        """
        separation = setting.periodic_boundaries.separation_vector(
            self._leaf_units[self._active_leaf_unit_index].position,
            self._leaf_units[self._active_leaf_unit_index ^ 1].position)
        bounding_event_rate = sum(
            max(0.0, vectors.dot(self._active_leaf_unit.velocity,
                                 bounding_potential.gradient(separation,
                                                             *charges(self._leaf_units[0], self._leaf_units[1]))))
            for bounding_potential, charges in zip(self._bounding_potentials, self._bounding_potentials_charges))
        potential_charges = self._potential_charges(self._leaf_units[0], self._leaf_units[1])
        real_gradient = self._potential.gradient(separation, *potential_charges)
        real_derivative = vectors.dot(self._active_leaf_unit.velocity, real_gradient)
        if real_derivative > 0:
            bounding_potential_warning(self.__class__.__name__, bounding_event_rate, real_derivative)
            if random.uniform(0, bounding_event_rate) < real_derivative:
                self._exchange_velocity(self._leaf_cnodes[self._active_leaf_unit_index],
                                        self._leaf_cnodes[self._active_leaf_unit_index ^ 1],
                                        real_gradient)
        return self._state
