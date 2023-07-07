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
"""Module for the FixedSeparationsEventHandlerWithPiecewiseConstantBoundingPotential class."""
import logging
import random
from typing import List, Sequence, Tuple, Union
from jellyfysh.base.exceptions import ConfigurationError, bounding_potential_warning
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
from jellyfysh.base.unit import Unit
import jellyfysh.base.vectors as vectors
from jellyfysh.lifting import Lifting
from jellyfysh.potential import Potential
import jellyfysh.setting as setting
from .abstracts import EventHandlerWithPiecewiseConstantBoundingPotential


class FixedSeparationsEventHandlerWithPiecewiseConstantBoundingPotential(
    EventHandlerWithPiecewiseConstantBoundingPotential):
    """
    Event handler which uses a dynamically created piecewise constant bounding potential for an interaction between more
    than two leaf units.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    The bounding potential is implemented in the base class EventHandlerWithPiecewiseConstantBoundingPotential. See the
    documentation of this class for more details.

    Only one of the leaf units should be active. This class generalizes the creation of the separation arguments for its
    potential between the leaf units. On instantiation, it receives a scheme which generates the separations which are
    passed to the potential's derivative method. The separation vectors between the leaf units are corrected for
    periodic boundaries.

    Each charge argument the potential expects is set to one.

    Also, this class assumes that the interaction is between more than two leaf units, which yields a lifting scheme
    necessary.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information). The maximum time displacement, however, can stay a simple
    float because it is always of the same order of magnitude during a run of JF. The same is true for the time
    displacement returned by the dynamically constructed bounding potential.
    """

    def __init__(self, potential: Potential, lifting: Lifting, offset: float, max_displacement: float,
                 separations: Sequence[int]) -> None:
        """
        The constructor of the FixedSeparationsEventHandlerWithPiecewiseConstantBoundingPotential class.

        Parameters
        ----------
        potential : potential.Potential
            The potential between the leaf units.
        lifting : lifting.Lifting
            The lifting scheme.
        offset : float
            The offset used to create piecewise constant bounding potential.
        max_displacement : float
            The maximum time displacement used to create piecewise constant bounding potential.
        separations : Sequence[int]
            A sequence of integers in the format [i1, j1, i2, j2...in, jn]. The separations passed to the potential
            will be [r_j1 - r_i1, r_j2 - r_i2, ..., r_jn - r_in].

        Raises
        ------
        base.exceptions.ConfigurationError:
            If the separations sequence is not divisible by two.
        base.exceptions.ConfigurationError:
            If the number of separations which can be constructed from the separations sequence does not equal the
            number of separation arguments of the potential.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential.__class__.__name__,
                           lifting=lifting.__class__.__name__, offset=offset, max_displacement=max_displacement,
                           separations=separations)
        super().__init__(potential=potential, offset=offset, max_displacement=max_displacement)
        self._lifting = lifting
        self._separations = separations
        if len(self._separations) % 2 != 0:
            raise ConfigurationError("The given array of indices {0} which should be used to calculate the separations"
                                     " handed over to the potential is not divisible by two!".format(separations))
        if self._potential.number_separation_arguments != len(self._separations) // 2:
            raise ConfigurationError("The event handler {0} expects a potential "
                                     "which handles exactly the number of separations specified"
                                     " by the list of identifiers {1} used to calculate these separations"
                                     " (length of the list divided by 2)!"
                                     .format(self.__class__.__name__, self._separations))
        # The charges the potential expects will always be set to 1.0
        self._number_charges = self._potential.number_charge_arguments
        self._number_events = 0
        self._number_accepted = 0

    def send_event_time(self, in_state: Sequence[Node]) -> Time:
        """
        Return the candidate event time.

        This method stores the transmitted in-state internally and then uses the resend_event_time to determine the
        candidate event time. The in-state should consist of all branches of the leaf units which take part in the
        interaction treated in this event handler.

        Parameters
        ----------
        in_state : Sequence[base.node.Node]
            The in-state.

        Returns
        -------
        base.time.Time
            The candidate event time.
        """
        self._store_in_state(in_state)
        self._construct_leaf_cnodes()
        self._extract_active_leaf_unit()
        return self.resend_event_time()

    def resend_event_time(self) -> Time:
        """
        Return the candidate event time based on the internally stored in-state.

        This method uses the dynamically created piecewise constant bounding potential to compute the candidate event
        time. For this, it extracts all leaf units from the in-state. Then it uses the
        _displacement_from_piecewise_constant_bounding_potential method that is implemented in the
        EventHandlerWithPiecewiseConstantBoundingPotential base class.

        Returns
        -------
        base.time.Time
            The candidate event time.
        """
        time_displacement = self._displacement_from_piecewise_constant_bounding_potential(
            random.expovariate(setting.beta))
        self._event_time = self._active_leaf_unit.time_stamp + time_displacement
        self._time_slice_all_units_in_state()
        return self._event_time

    # noinspection PyUnresolvedReferences
    def send_out_state(self) -> Union[Sequence[Node], None]:
        """
        Return the out-state.

        First, this method confirms the event. If it is confirmed, the lifting scheme determines the new active leaf
        unit, which is imprinted in the out-state consisting of all branches of the leaf units taking part in the
        interaction.

        Returns
        -------
        Sequence[base.node.Node] or None
            The out-state.

        Raises
        ------
        AssertionError
            If the bounding event rate is smaller than zero.
        AssertionError
            If the lifting scheme failed.
        """
        self._number_events += 1
        bounding_event_rate = self._event_rate_from_piecewise_constant_bounding_potential()
        if bounding_event_rate is not None:
            assert bounding_event_rate >= 0.0
            separations = self._get_separations([unit.position for unit in self._leaf_units])
            potential_gradients = self._potential.gradient(*separations, *self._get_charges())
            active_unit_gradient = potential_gradients[self._active_leaf_unit_index]
            gradient_dot_velocity = vectors.dot(active_unit_gradient, self._active_leaf_unit.velocity)
            if gradient_dot_velocity > 0.0:
                bounding_potential_warning(self.__class__.__name__, bounding_event_rate, gradient_dot_velocity)
                if random.uniform(0.0, bounding_event_rate) < gradient_dot_velocity:
                    self._number_accepted += 1
                    self._lifting.reset()
                    sum_gradient_squared = 0.0
                    sum_velocity_dot_gradient = 0.0
                    for index, leaf_unit in enumerate(self._leaf_units):
                        sum_gradient_squared += vectors.norm_sq(potential_gradients[index])
                        sum_velocity_dot_gradient += vectors.dot(leaf_unit.velocity, potential_gradients[index])
                    prefactor = -2.0 * sum_velocity_dot_gradient / sum_gradient_squared
                    new_velocities = []
                    for index, leaf_unit in enumerate(self._leaf_units):
                        velocity_change = [prefactor * g for g in potential_gradients[index]]
                        new_velocities.append([v + c for v, c in zip(leaf_unit.velocity, velocity_change)])
                        self._lifting.insert(vectors.dot(leaf_unit.velocity, potential_gradients[index]),
                                             (index, False), leaf_unit is self._active_leaf_unit)
                        self._lifting.insert(vectors.dot(new_velocities[-1], potential_gradients[index]),
                                             (index, True), False)
                    new_active_index, change_velocities = self._lifting.get_active_identifier()
                    assert not (not change_velocities and new_active_index == self._active_leaf_unit_index)
                    time_stamp = self._active_leaf_unit.time_stamp
                    self._active_leaf_unit.time_stamp = None
                    self._register_velocity_change_leaf_cnode(self._leaf_cnodes[self._active_leaf_unit_index],
                                                              [-c for c in self._active_leaf_unit.velocity])
                    self._leaf_units[new_active_index].time_stamp = time_stamp
                    if change_velocities:
                        for index, leaf_unit in enumerate(self._leaf_units):
                            leaf_unit.velocity = new_velocities[index]
                    self._register_velocity_change_leaf_cnode(self._leaf_cnodes[new_active_index],
                                                              self._leaf_cnodes[new_active_index].value.velocity)
                    self._commit_non_leaf_velocity_changes()
                    return self._state
        return None

    def _get_separations(self, positions: Sequence[Sequence[float]]) -> List[Sequence[float]]:
        """
        Return the sequence of separations relevant for the potential based on the positions of the leaf units.

        Note that the positions in the arguments are in the same order as self._leaf_units.

        This method uses the separations sequence from the initialization to extract the separations from the units.
        For each pair of units, the shortest separation vector is calculated.

        Parameters
        ----------
        positions : Sequence[Sequence[float]]
            The sequence of positions of the leaf units.

        Returns
        -------
        List[Sequence[float]]
            The list of separations.
        """
        assert len(positions) > max(self._separations)
        separations_iterable = iter(self._separations)
        # Iterate over self._separations in chunks of size two by using zip trick
        return [setting.periodic_boundaries.separation_vector(positions[index_one], positions[index_two])
                for index_one, index_two in zip(separations_iterable, separations_iterable)]

    def _get_charges(self, units: Sequence[Unit] = ()) -> Tuple[float, ...]:
        """
        Return the sequence of charges relevant for the potential.

        This method just returns one for each charge the potential expects.

        Parameters
        ----------
        units : Sequence[base.unit.Unit]
            The sequence of leaf units.

        Returns
        -------
        Tuple[float, ...]
            The tuple of charges.
        """
        return tuple(1.0 for _ in range(self._number_charges))

    def info(self):
        print(f"Acceptance rate {self.__class__.__name__}: "
              f"{self._number_accepted / self._number_events if self._number_events != 0 else 0.0}")

