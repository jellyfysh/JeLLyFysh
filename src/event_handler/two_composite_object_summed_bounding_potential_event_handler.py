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
"""Module for the TwoCompositeObjectSummedBoundingPotentialEventHandler class."""
import logging
import random
from typing import Sequence
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.node import Node
from lifting import Lifting
from potential import Potential, InvertiblePotential
import setting
from .abstracts import TwoCompositeObjectBoundingPotentialEventHandler
from .helper_functions import bounding_potential_warning


class TwoCompositeObjectSummedBoundingPotentialEventHandler(TwoCompositeObjectBoundingPotentialEventHandler):
    """
    Event handler which treats a two-composite-object interaction using a bounding potential.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    To calculate the (bounding) potential between the two composite objects, this class sums the (bounding) potential
    between all pairs of leaf units located in different composite objects.
    Only one of the leaf units should be active with an direction of motion in positive direction along an axis.
    This class uses an potential and a bounding potential which expect a single separation between two leaf units.
    Also, this event handler can pass a charge to the (bounding) potential. The separation vectors are corrected for
    periodic boundaries.
    Since the interaction involves more than two leaf units, a lifting scheme is required.
    """

    def __init__(self, potential: Potential, bounding_potential: InvertiblePotential,
                 lifting: Lifting, charge: str = None) -> None:
        """
        The constructor of the TwoCompositeObjectSummedBoundingPotentialEventHandler class.

        Parameters
        ----------
        potential : potential.Potential
            The potential between the leaf units.
        bounding_potential : potential.InvertiblePotential
            The invertible bounding potential between the leaf units.
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
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential.__class__.__name__,
                           bounding_potential=bounding_potential.__class__.__name__,
                           lifting=lifting.__class__.__name__, charge=charge)
        super().__init__(charge=charge, potential=potential, lifting=lifting)
        self._bounding_potential = bounding_potential
        if self._bounding_potential.number_separation_arguments != 1:
            raise ConfigurationError("The event handler {0} expects a potential and a bounding potential "
                                     "which handles exactly one separation!".format(self.__class__.__name__))

        if charge is None:
            self._bounding_potential_charges = (lambda unit_one, unit_two:
                                                tuple(1.0 for _ in
                                                      range(self._bounding_potential.number_charge_arguments)))
        else:
            if self._bounding_potential.number_charge_arguments == 2:
                self._bounding_potential_charges = lambda unit_one, unit_two: (unit_one.charge[charge],
                                                                               unit_two.charge[charge])

            else:
                raise ConfigurationError("The event handler {0} was initialized with a charge which is not None,"
                                         " but its bounding potential {1}"
                                         "expects not exactly 2 charges."
                                         .format(self.__class__.__name__, self._bounding_potential.__class__.__name__))

    def send_event_time(self, in_state: Sequence[Node]) -> float:
        """
        Return the candidate event time.

        The in-state should consist of all branches of the composite objects which take part in the interaction treated
        in this event handler. The candidate event time is calculated using the bounding potential. For this, the
        minimum displacement of each pair interaction of the leaf units in different composite objects is used.

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
            If the in-state contains not more than one branch.
        """
        assert len(in_state) > 1
        self._store_in_state(in_state)
        self._construct_leaf_cnodes()
        self._extract_active_leaf_unit()
        self._construct_leaf_units_of_composite_objects()
        displacement = min(self._bounding_potential.displacement(
            self._direction_of_motion,
            setting.periodic_boundaries.separation_vector(self._active_leaf_unit.position, target_unit.position),
            *self._bounding_potential_charges(self._active_leaf_unit, target_unit),
            random.expovariate(setting.beta)) for target_unit in self._target_leaf_units)
        self._event_time = self._active_leaf_unit.time_stamp + displacement / self._speed
        self._time_slice_all_units_in_state()
        return self._event_time

    def send_out_state(self) -> Sequence[Node]:
        """
        Return the out-state.

        First, this method confirms the event. Here, the overall bounding event rate is the sum of the absolute values
        of the bounding event rates for each pair of leaf units within the two different composite objects. If the event
        is confirmed, the lifting scheme determines the new active leaf unit, which is imprinted in the out-state
        consisting of both branches of the two composite objects.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.

        Raises
        ------
        AssertionError
            If the lifting scheme failed.
        """
        bounding_event_rate = 0.0
        factor_derivative = 0.0
        target_composite_object_factor_derivatives = [0.0] * len(self._target_leaf_units)
        for index, target_leaf_unit in enumerate(self._target_leaf_units):
            separation = setting.periodic_boundaries.separation_vector(self._active_leaf_unit.position,
                                                                       target_leaf_unit.position)
            bounding_event_rate += max(
                0.0, self._bounding_potential.derivative(
                    self._direction_of_motion, separation,
                    *self._bounding_potential_charges(self._active_leaf_unit, target_leaf_unit)))
            pairwise_derivative = self._potential.derivative(
                self._direction_of_motion, separation, *self._potential_charges(self._active_leaf_unit,
                                                                                target_leaf_unit))
            factor_derivative += pairwise_derivative
            target_composite_object_factor_derivatives[index] -= pairwise_derivative
        event_rate = max(0.0, factor_derivative)
        bounding_potential_warning(self.__class__.__name__, bounding_event_rate, event_rate)
        if event_rate <= random.uniform(0.0, bounding_event_rate):
            return self._state

        self._fill_lifting(self._local_leaf_units, self._target_leaf_units, event_rate,
                           target_composite_object_factor_derivatives)

        next_active_identifier = self._lifting.get_active_identifier()
        next_active_cnode = [cnode for cnode in self._leaf_cnodes if cnode.value.identifier == next_active_identifier]
        assert len(next_active_cnode) == 1
        self._exchange_velocity(self._leaf_cnodes[self._active_leaf_unit_index], next_active_cnode[0])
        return self._state
