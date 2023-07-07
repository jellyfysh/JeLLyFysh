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
"""Module for the TwoCompositeObjectSummedBoundingPotentialEventHandler class."""
import logging
import random
from typing import Sequence, Union
from jellyfysh.base.exceptions import ConfigurationError, bounding_potential_warning
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
from jellyfysh.base import vectors
from jellyfysh.lifting import Lifting
from jellyfysh.potential import Potential
import jellyfysh.setting as setting
from .abstracts import TwoCompositeObjectBoundingPotentialEventHandler


class TwoCompositeObjectEventHandlerWithPiecewiseConstantBoundingPotential(
    TwoCompositeObjectBoundingPotentialEventHandler):
    """
    Event handler which treats a two-composite-object interaction using a bounding potential.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    To calculate the (bounding) potential between the two composite objects, this class sums the (bounding) potential
    between all pairs of leaf units located in different composite objects.

    Only one of the leaf units should be active. This class uses an potential and a bounding potential which expect a
    single separation between two leaf units. The separation vectors between the leaf units are corrected for periodic
    boundaries.

    If the potentials can consider charges, this event handler can pass the charges of the leaf units to the potentials.
    The name of the used charge is set on initialization.

    This event handler always samples a potential change that is passed to the displacement method of the bounding
    potential.

    Since the interaction involves more than two leaf units, a lifting scheme is required.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information). The time displacements returned by the bounding potential's
    displacement method, however, are still simple floats because they are always of the same order of magnitude during
    a run of JF.
    """

    def __init__(self, potential: Potential, lifting: Lifting, offset: float, max_displacement: float,
                 charge: str = None) -> None:
        """
        The constructor of the TwoCompositeObjectSummedBoundingPotentialEventHandler class.

        Parameters
        ----------
        potential : potential.Potential
            The potential between the leaf units.
        lifting : lifting.Lifting
            The lifting scheme.
        charge : str or None, optional
            The relevant charge for this event handler.

        Raises
        ------
        base.exceptions.ConfigurationError:
            If the bounding potential does not expect exactly one separation.
        base.exceptions.ConfigurationError:
            If the charge is not None but the potential or the bounding potential expects more than two charges.
        base.exceptions.ConfigurationError:
            If the displacement method of the bounding potential does not require a potential change.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential.__class__.__name__, lifting=lifting.__class__.__name__, offset=offset,
                           max_displacement=max_displacement, charge=charge)
        super().__init__(charge=charge, potential=potential, lifting=lifting)
        self._offset = offset
        if not max_displacement > 0.0:
            raise ConfigurationError("Please use a value for max_displacement > 0.0 in the class {0}."
                                     .format(self.__class__.__name__))
        self._max_displacement = max_displacement
        self._bounding_event_rate = None
        self._number_events = 0
        self._number_accepted = 0

    def send_event_time(self, in_state: Sequence[Node]) -> Time:
        """
        Return the candidate event time.

        This method stores the transmitted in-state internally and then uses the resend_event_time to determine the
        candidate event time. The in-state should consist of all branches of the composite objects which take part in
        the interaction treated in this event handler.

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
            If the in-state contains not more than one branch.
        """
        assert len(in_state) > 1
        self._store_in_state(in_state)
        self._construct_leaf_cnodes()
        self._extract_active_leaf_unit()
        self._construct_leaf_units_of_composite_objects()
        return self.resend_event_time()

    def resend_event_time(self) -> Time:
        """
        Return the candidate event time based on the internally stored in-state.

        Returns
        -------
        base.time.Time
            The candidate event time.

        """
        # TODO: If this turns out to be a bottleneck, one can probably use that the derivatives at the initial position
        # were already computed (in the last call of resend_event_time or in the call of send_out_state).
        derivative_one = 0.0
        derivative_two = 0.0
        speed = vectors.norm(self._active_leaf_unit.velocity)
        max_time_displacement = self._max_displacement / speed
        proposed_position = [setting.periodic_boundaries.correct_position_entry(
            self._active_leaf_unit.position[d] + self._active_leaf_unit.velocity[d] * max_time_displacement, d)
            for d in range(setting.dimension)]
        for index, target_leaf_unit in enumerate(self._target_leaf_units):
            charges = self._potential_charges(self._active_leaf_unit, target_leaf_unit)
            derivative_one += vectors.dot(self._active_leaf_unit.velocity,
                                          self._potential.gradient(
                                              setting.periodic_boundaries.separation_vector(
                                                  self._active_leaf_unit.position, target_leaf_unit.position),
                                              *charges))
            derivative_two += vectors.dot(self._active_leaf_unit.velocity,
                                          self._potential.gradient(
                                              setting.periodic_boundaries.separation_vector(
                                                  proposed_position, target_leaf_unit.position), *charges))
        constant_derivative = max(derivative_one, derivative_two) + self._offset * speed
        potential_change = random.expovariate(setting.beta)
        if constant_derivative <= 0.0:
            self._bounding_event_rate = None
            time_displacement = max_time_displacement
        elif potential_change / constant_derivative < max_time_displacement:
            self._bounding_event_rate = constant_derivative
            time_displacement = potential_change / constant_derivative
        else:
            self._bounding_event_rate = None
            time_displacement = max_time_displacement
        self._event_time = self._active_leaf_unit.time_stamp + time_displacement
        self._time_slice_all_units_in_state()
        return self._event_time

    def send_out_state(self) -> Union[Sequence[Node], None]:
        """
        Return the out-state.

        First, this method confirms the event. Here, the overall bounding event rate is the sum of the absolute values
        of the bounding event rates for each pair of leaf units within the two different composite objects. If the event
        is confirmed, the lifting scheme determines the new active leaf unit, which is imprinted in the out-state
        consisting of both branches of the two composite objects.

        Returns
        -------
        Sequence[base.node.Node] or None
            The out-state.

        Raises
        ------
        AssertionError
            If the lifting scheme failed.
        """
        self._number_events += 1
        if self._bounding_event_rate is not None:
            factor_derivative = 0.0
            factor_gradient = [0.0 for _ in range(setting.dimension)]
            target_composite_object_factor_gradients = [[0.0 for _ in range(setting.dimension)]
                                                        for _ in range(len(self._target_leaf_units))]
            for index, target_leaf_unit in enumerate(self._target_leaf_units):
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
                self._number_accepted += 1
                new_local_velocities, new_target_velocities = self._fill_lifting(
                    self._local_leaf_units, self._target_leaf_units, factor_gradient,
                    target_composite_object_factor_gradients)
                new_active_identifier, change_velocities = self._lifting.get_active_identifier()
                assert not (not change_velocities and new_active_identifier == self._active_leaf_unit.identifier)
                time_stamp = self._active_leaf_unit.time_stamp
                self._active_leaf_unit.time_stamp = None
                self._register_velocity_change_leaf_cnode(self._leaf_cnodes[self._active_leaf_unit_index],
                                                          [-c for c in self._active_leaf_unit.velocity])
                new_active_cnode = [cnode for cnode in self._leaf_cnodes if cnode.value.identifier == new_active_identifier]
                assert len(new_active_cnode) == 1
                new_active_cnode[0].value.time_stamp = time_stamp
                if change_velocities:
                    for index_1, local_unit in enumerate(self._local_leaf_units):
                        local_unit.velocity = new_local_velocities[index_1]
                    for index_2, target_unit in enumerate(self._target_leaf_units):
                        target_unit.velocity = new_target_velocities[index_2]
                self._register_velocity_change_leaf_cnode(new_active_cnode[0],
                                                          new_active_cnode[0].value.velocity)
                self._commit_non_leaf_velocity_changes()
                return self._state
        return None

    def info(self):
        print(f"Acceptance rate {self.__class__.__name__}: "
              f"{self._number_accepted / self._number_events if self._number_events != 0 else 0.0}")
