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
"""Module for the FixedSeparationsEventHandlerWithPiecewiseConstantBoundingPotential class."""
import logging
import random
from typing import List, Sequence, Tuple
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.node import Node
from base.unit import Unit
from lifting import Lifting
from potential import Potential
import setting
from .abstracts import EventHandlerWithPiecewiseConstantBoundingPotential
from .helper_functions import bounding_potential_warning


class FixedSeparationsEventHandlerWithPiecewiseConstantBoundingPotential(
      EventHandlerWithPiecewiseConstantBoundingPotential):
    """
    Event handler which uses a dynamically created piecewise constant bounding potential for an interaction between more
    than two leaf units.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    The bounding potential is implemented in the base class EventHandlerWithPiecewiseConstantBoundingPotential. See the
    documentation of this class for more details.
    This class generalizes the creation of the separation arguments for its potential between the leaf units. On
    instantiation it receives a scheme which generates the separations which are passed to the potential. Each charge
    argument the potential expects is set to one. Also, this class assumes that there is a single active leaf unit and
    that the interaction is between more than three leaf units, which yields a lifting scheme necessary. The separation
    vectors between the interacting leaf units are corrected for periodic boundaries.
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
            The maximum displacement used to create piecewise constant bounding potential.
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

    def send_event_time(self, in_state: Sequence[Node]) -> float:
        """
        Return the candidate event time.

        The in-state should consist of all branches of the leaf units which take part in the interaction treated in this
        event handler.
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
        """
        self._store_in_state(in_state)
        self._construct_leaf_cnodes()
        self._extract_active_leaf_unit()
        displacement = self._displacement_from_piecewise_constant_bounding_potential(random.expovariate(setting.beta))
        self._event_time = self._active_leaf_unit.time_stamp + displacement / self._speed
        self._time_slice_all_units_in_state()
        return self._event_time

    def send_out_state(self) -> Sequence[Node]:
        """
        Return the out-state.

        First, this method confirms the event. If it is confirmed, the lifting scheme determines the new active leaf
        unit, which is imprinted in the out-state consisting of all branches of the leaf units taking part in the
        interaction.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.

        Raises
        ------
        AssertionError
            If the bounding event rate is smaller than zero.
        AssertionError
            If the lifting scheme failed.
        """
        separations = self._get_separations(self._leaf_units)
        bounding_event_rate = self._event_rate_from_piecewise_constant_bounding_potential()
        if bounding_event_rate is not None:
            assert bounding_event_rate >= 0.0
            potential_derivatives = self._potential.derivative(
                self._direction_of_motion, *separations, *self._get_charges())
            active_unit_derivative = potential_derivatives[self._active_leaf_unit_index]
            if active_unit_derivative > 0:
                bounding_potential_warning(self.__class__.__name__, bounding_event_rate,
                                           potential_derivatives[self._active_leaf_unit_index])
                if random.uniform(0.0, bounding_event_rate) < active_unit_derivative:
                    self._lifting.reset()
                    for index, leaf_unit in enumerate(self._leaf_units):
                        self._lifting.insert(potential_derivatives[index], leaf_unit.identifier,
                                             index == self._active_leaf_unit_index)
                    new_active_identifier = self._lifting.get_active_identifier()
                    new_active_indices = [index for index, leaf_unit in enumerate(self._leaf_units)
                                          if leaf_unit.identifier == new_active_identifier]
                    assert len(new_active_indices) == 1
                    self._exchange_velocity(self._leaf_cnodes[self._active_leaf_unit_index],
                                            self._leaf_cnodes[new_active_indices[0]])

        return self._state

    def _get_separations(self, units: Sequence[Unit]) -> List[Sequence[float]]:
        """
        Return the sequence of separations relevant for the potential.

        This method uses the separations sequence from the initialization to extract the separations from the units.
        For each pair of units, the shortest separation vectors is calculated.

        Parameters
        ----------
        units : Sequence[base.unit.Unit]
            The sequence of leaf units.

        Returns
        -------
        List[Sequence[float]]
            The list of separations.
        """
        assert len(units) > max(self._separations)
        separations_iterable = iter(self._separations)
        # Iterate over self._separations in chunks of size two by using zip trick
        return [setting.periodic_boundaries.separation_vector(units[index_one].position, units[index_two].position)
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
