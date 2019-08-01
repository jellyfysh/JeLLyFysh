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
"""Module for the TwoLeafUnitEventHandler class."""
import logging
import random
from typing import Sequence
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.node import Node
from potential import InvertiblePotential
import setting
from .abstracts import SingleActiveLeafUnitEventHandler


class TwoLeafUnitEventHandler(SingleActiveLeafUnitEventHandler):
    """
    Event handler which uses an invertible potential as the interaction between two leaf units.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    Only one of the two leaf units should be active with an direction of motion in positive direction along an axis.
    This class uses an potential which expects a single separation between two leaf units. Also, this event handler
    can pass a charge to the potential. The separation vector between the interacting leaf units is corrected for
    periodic boundaries.
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

    def send_event_time(self, in_state: Sequence[Node]) -> float:
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
        displacement = self._potential.displacement(
            self._direction_of_motion,
            setting.periodic_boundaries.separation_vector(self._leaf_units[self._active_leaf_unit_index].position,
                                                          self._leaf_units[self._active_leaf_unit_index ^ 1].position),
            *self._charges(self._leaf_units[0], self._leaf_units[1]), random.expovariate(setting.beta))
        self._event_time = self._active_leaf_unit.time_stamp + displacement / self._speed
        self._time_slice_all_units_in_state()
        return self._event_time

    def send_out_state(self) -> Sequence[Node]:
        """
        Return the out-state.

        The out-state has exchanged velocities between the two leaf units and both branches are kept consistent.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.
        """
        self._exchange_velocity(self._leaf_cnodes[self._active_leaf_unit_index],
                                self._leaf_cnodes[self._active_leaf_unit_index ^ 1])
        return self._state
