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
"""Module for the TwoLeafUnitEventHandler class."""
import logging
import random
from typing import Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
from jellyfysh.potential import InvertiblePotential
import jellyfysh.setting as setting
from .abstracts import SingleActiveLeafUnitEventHandler


class TwoLeafUnitEventHandler(SingleActiveLeafUnitEventHandler):
    """
    Event handler which uses an invertible potential as the interaction between two leaf units.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    Only one of the two leaf units should be active. This class uses an potential which expects a single separation
    between two leaf units. The separation vector between the interacting leaf units is corrected for periodic
    boundaries.

    If the potential can consider charges, this event handler can pass the charges of the two leaf units to the
    potential. The name of the used charge is set on initialization.

    This event handler only samples a potential change that is passed to the displacement method of the potential, if
    the potential requires a potential change (some potentials as, e.g., the Hard-Sphere potential do not require a
    sampled potential change).

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information). The time displacements returned by the potential's
    displacement method, however, are still simple floats because they are always of the same order of magnitude during
    a run of JF.
    """

    def __init__(self, potential: InvertiblePotential, charge: str = None) -> None:
        """
        The constructor of the TwoLeafUnitEventHandler class.

        Parameters
        ----------
        potential : potential.InvertiblePotential
            The invertible potential.
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
                           potential=potential.__class__.__name__, charge=charge)
        super().__init__()
        self._potential = potential
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

        if self._potential.potential_change_required:
            self._potential_displacement = (
                lambda *args, **kwargs: self._potential.displacement(*args, **kwargs,
                                                                     potential_change=random.expovariate(setting.beta)))
        else:
            self._potential_displacement = self._potential.displacement

    def send_event_time(self, in_state: Sequence[Node]) -> Time:
        """
        Return the candidate event time.

        The in-state should consist of all branches of the two leaf units which take part in the interaction treated in
        this event handler.

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
            If the in-state does not contain exactly two leaf units.
        """
        self._store_in_state(in_state)
        self._construct_leaf_cnodes()
        self._extract_active_leaf_unit()
        assert len(self._leaf_cnodes) == 2
        time_displacement = self._potential_displacement(
            self._active_leaf_unit.velocity,
            setting.periodic_boundaries.separation_vector(self._leaf_units[self._active_leaf_unit_index].position,
                                                          self._leaf_units[self._active_leaf_unit_index ^ 1].position),
            *self._charges(self._leaf_units[0], self._leaf_units[1]))
        self._event_time = self._active_leaf_unit.time_stamp + time_displacement
        self._time_slice_all_units_in_state()
        return self._event_time

    def send_out_state(self) -> Sequence[Node]:
        """
        Return the out-state.

        In the out-state, the velocity of the independent active leaf unit is transferred to the non-active target leaf
        unit. Here, both branches are kept consistent.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.
        """
        self._exchange_velocity(self._leaf_cnodes[self._active_leaf_unit_index],
                                self._leaf_cnodes[self._active_leaf_unit_index ^ 1])
        return self._state
