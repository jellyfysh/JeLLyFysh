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
"""Module for the RootUnitActiveTwoCompositeObjectSummedBoundingPotentialEventHandler class."""
import logging
import random
from typing import List, Sequence, Tuple
from jellyfysh.base.exceptions import ConfigurationError, bounding_potential_warning
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
from jellyfysh.potential import Potential, InvertiblePotential
import jellyfysh.setting as setting
from jellyfysh.state_handler.tree_state_handler import StateId
from .abstracts import CompositeObjectsLifting


class RootUnitActiveTwoCompositeObjectSummedBoundingPotentialEventHandler(CompositeObjectsLifting):
    """
    Event handler which treats a summed interaction between leaf units in two different composite objects using a
    bounding potential for the case of an active composite object.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    To calculate the (bounding) potential between the leaf units in two composite objects, this class sums the
    (bounding) potential between all pairs of leaf units located in different composite objects.

    A complete composite object should be active. The leaf units taking part in the interaction should be located in two
    different composite objects. This class uses an potential and a bounding potential which expect a single separation
    between two leaf units. The separation vector between the leaf units is corrected for periodic boundaries.

    If the (bounding) potential can consider charges, this event handler can pass the charges of the two respective leaf
    units to the potential. The name of the used charge is set on initialization.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information). The time displacements returned by the bounding potential's
    displacement method, however, are still simple floats because they are always of the same order of magnitude during
    a run of JF.
    """

    def __init__(self, potential: Potential, bounding_potential: InvertiblePotential, charge: str = None) -> None:
        """
        The constructor of the RootUnitActiveTwoCompositeObjectSummedBoundingPotentialEventHandler class.

        Parameters
        ----------
        potential : potential.Potential
            The potential between the leaf units.
        bounding_potential : potential.InvertiblePotential
            The invertible bounding potential between the leaf units.
        charge : str or None, optional
            The relevant charge for this event handler.

        Raises
        ------
        base.exceptions.ConfigurationError:
            If the bounding potential does not expect exactly one separation.
        base.exceptions.ConfigurationError:
            If the potential does not expect exactly one separation.
        base.exceptions.ConfigurationError:
            If the charge is not None but the potential or the bounding potential expects more than two charges.
        base.exceptions.ConfigurationError
            If the number of nodes per root node is one and therefore no composite objects are present in the run.
        base.exceptions.ConfigurationError:
            If the displacement method of the bounding potential does not require a potential change.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential.__class__.__name__,
                           bounding_potential=bounding_potential.__class__.__name__, charge=charge)
        super().__init__()
        self._potential = potential
        self._bounding_potential = bounding_potential
        if self._bounding_potential.number_separation_arguments != 1:
            raise ConfigurationError("The event handler {0} expects a potential and a bounding potential "
                                     "which handles exactly one separation!".format(self.__class__.__name__))
        if not setting.number_of_nodes_per_root_node > 1:
            raise ConfigurationError("The class {0} can only be "
                                     "used when composite point objects are present!".format(self.__class__.__name__))
        if self._potential.number_separation_arguments != 1:
            raise ConfigurationError("The event handler {0} expects a potential "
                                     "which handles exactly one separation!".format(self.__class__.__name__))

        if charge is None:
            self._potential_charges = (lambda unit_one, unit_two:
                                       tuple(1.0 for _ in range(self._potential.number_charge_arguments)))
            self._bounding_potential_charges = (lambda unit_one, unit_two:
                                                tuple(1.0 for _ in
                                                      range(self._bounding_potential.number_charge_arguments)))
        else:
            if self._potential.number_charge_arguments == 2 and self._bounding_potential.number_charge_arguments == 2:
                self._potential_charges = lambda unit_one, unit_two: (unit_one.charge[charge], unit_two.charge[charge])
                self._bounding_potential_charges = self._potential_charges
            else:
                raise ConfigurationError("The event handler {0} was initialized with a charge which is not None,"
                                         " but its potential {1} and/or its bounding potential {2}"
                                         "expects not exactly 2 charges."
                                         .format(self.__class__.__name__, self._potential.__class__.__name__,
                                                 self._bounding_potential.__class__.__name__))
        if not self._bounding_potential.potential_change_required:
            raise ConfigurationError("The event handler {0} expects a bounding potential that requires a potential "
                                     "change in its displacement method.".format(self.__class__.__name__))

    def send_event_time(self, in_state: Sequence[Node]) -> Tuple[Time, List[StateId]]:
        """
        Return the candidate event time.

        The in-state should consist of all branches of the leaf units located in two different composite objects which
        take part in the interaction treated in this event handler. The candidate event time is calculated using the
        bounding potential. For this, the minimum displacement of each pair interaction of the leaf units in different
        composite objects is used.

        The send_out_state method of this class needs the full composite object branches the leaf units belong to as
        an argument in order to pass the velocity from the active composite object to the other. Therefore, this method
        also returns the identifiers of the involved composite objects.

        Parameters
        ----------
        in_state : Sequence[base.node.Node]
            The in-state.

        Returns
        -------
        (base.time.Time, List[state_handler.tree_state_handler.StateId])
            The candidate event time, the composite object identifiers the leaf units belong to.

        Raises
        ------
        AssertionError
            If the in-state contains not more than one branch.
        AssertionError
            If not all leaf units of the active composite object have the same velocity.
        """
        assert len(in_state) > 1
        self._store_in_state(in_state)
        self._construct_leaf_cnodes()
        self._construct_leaf_units_of_composite_objects()
        assert all(unit.velocity == self._local_leaf_units[0].velocity for unit in self._local_leaf_units)
        time_displacement = min(self._bounding_potential.displacement(
            local_leaf_unit.velocity,
            setting.periodic_boundaries.separation_vector(local_leaf_unit.position, target_leaf_unit.position),
            *self._bounding_potential_charges(local_leaf_unit, target_leaf_unit),
            random.expovariate(setting.beta))
                           for local_leaf_unit in self._local_leaf_units
                           for target_leaf_unit in self._target_leaf_units)
        self._event_time = self._local_leaf_units[0].time_stamp + time_displacement
        self._time_slice_all_units_in_state()
        return self._event_time, [(self._local_leaf_units[0].identifier[0],),
                                  (self._target_leaf_units[0].identifier[0],)]

    def send_out_state(self, composite_object_root_cnodes: Sequence[Node]) -> Sequence[Node]:
        """
        Return the out-state.

        This method receives the branches of the two composite objects the interacting leaf units belong to. Only one of
        these should be active.

        First, this method confirms the event. Here, the overall bounding event rate is the sum of the absolute values
        of the bounding event rates for each pair of leaf units within the two different composite objects. If the event
        is confirmed, the out-state has exchanged velocities between the two composite objects and both branches are
        kept consistent.

        Parameters
        ----------
        composite_object_root_cnodes : Sequence[base.node.Node]
            The branches of the two composite objects.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.
        """
        bounding_event_rate = 0.0
        factor_derivative = 0.0
        for active_leaf_unit in self._local_leaf_units:
            for target_leaf_unit in self._target_leaf_units:
                separation = setting.periodic_boundaries.separation_vector(active_leaf_unit.position,
                                                                           target_leaf_unit.position)
                bounding_event_rate += max(0.0, self._bounding_potential.derivative(
                    active_leaf_unit.velocity, separation, *self._bounding_potential_charges(active_leaf_unit,
                                                                                             target_leaf_unit)))
                factor_derivative += self._potential.derivative(
                    active_leaf_unit.velocity, separation, *self._potential_charges(active_leaf_unit, target_leaf_unit))
        bounding_potential_warning(self.__class__.__name__, bounding_event_rate, factor_derivative)
        self._store_in_state(composite_object_root_cnodes)
        self._time_slice_all_units_in_state()
        if factor_derivative > 0:
            if random.uniform(0, bounding_event_rate) < factor_derivative:
                self._construct_leaf_cnodes()
                self._construct_leaf_units_of_composite_objects()
                self._pass_composite_object_velocity()
        return self._state
