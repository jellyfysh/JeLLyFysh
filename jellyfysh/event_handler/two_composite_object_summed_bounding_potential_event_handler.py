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
from typing import Sequence
from jellyfysh.base.exceptions import ConfigurationError, bounding_potential_warning
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
from jellyfysh.base import vectors
from jellyfysh.lifting import Lifting
from jellyfysh.potential import Potential, InvertiblePotential
import jellyfysh.setting as setting
from .abstracts import TwoCompositeObjectBoundingPotentialEventHandler


class TwoCompositeObjectSummedBoundingPotentialEventHandler(TwoCompositeObjectBoundingPotentialEventHandler):
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

    def __init__(self, potential: Potential, bounding_potential: Sequence[InvertiblePotential],
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
        base.exceptions.ConfigurationError:
            If the displacement method of the bounding potential does not require a potential change.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential.__class__.__name__,
                           bounding_potential=bounding_potential.__class__.__name__,
                           lifting=lifting.__class__.__name__, charge=charge)
        super().__init__(charge=charge, potential=potential, lifting=lifting)
        self._bounding_potentials = bounding_potential
        for bounding_potential in self._bounding_potentials:
            if bounding_potential.number_separation_arguments != 1:
                raise ConfigurationError("The event handler {0} expects (possibly several) bounding potentials "
                                         "that handle exactly one separation! The bounding potential {1} expects "
                                         "{2} separations.".format(self.__class__.__name__,
                                                                   bounding_potential.__class__.__name__,
                                                                   bounding_potential.number_separation_arguments))
        if charge is None:
            self._bounding_potentials_charges = [
                lambda unit_one, unit_two: tuple(1.0 for _ in range(bounding_potential.number_charge_arguments))
                for bounding_potential in self._bounding_potentials]
        else:
            if all(bounding_potential.number_charge_arguments == 2 for bounding_potential in self._bounding_potentials):
                self._bounding_potentials_charges = [lambda unit_one, unit_two: (unit_one.charge[charge],
                                                                                 unit_two.charge[charge])
                                                     for _ in self._bounding_potentials]
            else:
                raise ConfigurationError("The event handler {0} was initialized with a charge which is not None,"
                                         " but one of its bounding potentials {1}"
                                         " expects not exactly 2 charges."
                                         .format(self.__class__.__name__,
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

        The in-state should consist of all branches of the composite objects which take part in the interaction treated
        in this event handler. The candidate event time is calculated using the bounding potential. For this, the
        minimum displacement of each pair interaction of the leaf units in different composite objects is used.

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
        time_displacement = min(bounding_potential.displacement(
            self._active_leaf_unit.velocity,
            setting.periodic_boundaries.separation_vector(self._active_leaf_unit.position, target_unit.position),
            *charges(self._active_leaf_unit, target_unit), random.expovariate(setting.beta))
                                for target_unit in self._target_leaf_units
                                for bounding_potential, charges in zip(self._bounding_potentials,
                                                                       self._bounding_potentials_charges))
        self._event_time = self._active_leaf_unit.time_stamp + time_displacement
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
        factor_gradient = [0.0 for _ in range(setting.dimension)]
        target_composite_object_factor_gradients = [[0.0 for _ in range(setting.dimension)]
                                                    for _ in range(len(self._target_leaf_units))]
        for index, target_leaf_unit in enumerate(self._target_leaf_units):
            separation = setting.periodic_boundaries.separation_vector(self._active_leaf_unit.position,
                                                                       target_leaf_unit.position)
            bounding_event_rate += sum(
                max(0.0, bounding_potential.derivative(
                    self._active_leaf_unit.velocity, separation, *charges(self._active_leaf_unit, target_leaf_unit)))
                for bounding_potential, charges in zip(self._bounding_potentials, self._bounding_potentials_charges))
            pairwise_gradient = self._potential.gradient(
                separation, *self._potential_charges(self._active_leaf_unit, target_leaf_unit))
            pairwise_derivative = vectors.dot(self._active_leaf_unit.velocity, pairwise_gradient)
            for i, g in enumerate(pairwise_gradient):
                factor_gradient[i] += g
                target_composite_object_factor_gradients[index][i] -= g
            factor_derivative += pairwise_derivative
        event_rate = max(0.0, factor_derivative)
        bounding_potential_warning(self.__class__.__name__, bounding_event_rate, event_rate)
        if event_rate <= random.uniform(0.0, bounding_event_rate):
            return self._state
        old_velocity_norm = vectors.norm(self._active_leaf_unit.velocity)
        factor_gradient_norm_squared = vectors.norm_sq(factor_gradient)
        new_velocity = [-v + 2.0 * g * factor_derivative / factor_gradient_norm_squared
                        for v, g in zip(self._active_leaf_unit.velocity, factor_gradient)]
        new_velocity = vectors.normalize(new_velocity, new_norm=old_velocity_norm)

        self._fill_lifting(self._local_leaf_units, self._target_leaf_units, factor_gradient,
                           target_composite_object_factor_gradients, new_velocity)
        next_active_identifier = self._lifting.get_active_identifier()
        next_active_cnode = [cnode for cnode in self._leaf_cnodes if cnode.value.identifier == next_active_identifier]
        assert len(next_active_cnode) == 1
        self._register_velocity_change_leaf_cnode(self._leaf_cnodes[self._active_leaf_unit_index],
                                                  [-component for component in
                                                   self._active_leaf_unit.velocity])
        self._register_velocity_change_leaf_cnode(next_active_cnode[0], new_velocity)
        next_active_cnode[0].value.velocity = new_velocity
        next_active_cnode[0].value.time_stamp = self._active_leaf_unit.time_stamp
        self._active_leaf_unit.velocity = None
        self._active_leaf_unit.time_stamp = None
        self._commit_non_leaf_velocity_changes()

        return self._state
