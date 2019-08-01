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
"""Module for the RootUnitActiveTwoLeafUnitEventHandler class."""
import logging
from typing import List, Sequence, Tuple
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.node import Node
from potential import InvertiblePotential
import setting
from state_handler.tree_state_handler import StateId
from .abstracts import CompositeObjectsLifting
from .two_leaf_unit_event_handler import TwoLeafUnitEventHandler


class RootUnitActiveTwoLeafUnitEventHandler(CompositeObjectsLifting, TwoLeafUnitEventHandler):
    """
    Event handler which uses an invertible potential as the interaction between two leaf units for the case of an active
    composite object.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    A complete composite object should be active with an direction of motion in positive direction along an axis.
    The two leaf units taking part in the interaction should be located in two different composite objects.
    This class uses an potential which expects a single separation between two leaf units. Also, this event handler
    can pass a charge to the potential. The separation vector between the interacting leaf units is corrected for
    periodic boundaries.
    """

    def __init__(self, potential: InvertiblePotential, charge: str = None):
        """
        The constructor of the RootUnitActiveTwoLeafUnitEventHandler class.

        Parameters
        ----------
        potential : potential.InvertiblePotential
            The invertible potential.
        charge : str or None, optional
            The relevant charge for this event handler.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the number of nodes per root node is one and therefore no composite objects are present in the run.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           potential=potential.__class__.__name__, charge=charge)
        super().__init__(potential=potential, charge=charge)
        if not setting.number_of_node_levels > 1:
            raise ConfigurationError("The event handler {0} should only be used when composite point objects are "
                                     "involved!".format(self.__class__.__name__))

    def send_event_time(self, in_state: Sequence[Node]) -> Tuple[float, List[StateId]]:
        """
        Return the candidate event time.

        The in-state should consist of the branches of the two leaf units which take part in the interaction treated in
        this event handler.
        The send_out_state method of this class needs the full composite object branches the two leaf units belong to as
        an argument in order to pass the velocity from the active composite object to the other. Therefore, this method
        also returns the identifiers of the involved composite objects.

        Parameters
        ----------
        in_state : Sequence[base.node.Node]
            The in-state.

        Returns
        -------
        (float, List[state_handler.tree_state_handler.StateId])
            The candidate event time, the composite object identifiers the two leaf units belong to.

        Raises
        ------
        AssertionError
            If the in-state contains not exactly two branches.
        """
        event_time = super().send_event_time(in_state)
        assert len(self._state) == 2
        return event_time, [cnode.value.identifier for cnode in self._state]

    # noinspection PyMethodOverriding
    def send_out_state(self, composite_objects_root_cnodes: Sequence[Node]) -> Sequence[Node]:
        """
        Return the out-state.

        This method receives the branches of the two composite objects the two interacting leaf units belong to.
        One of these is active. The out-state has exchanged velocities between the two composite objects and both
        branches are kept consistent.

        Parameters
        ----------
        composite_objects_root_cnodes : Sequence[base.node.Node]
            The branches of the two composite objects.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.
        """
        self._store_in_state(composite_objects_root_cnodes)
        self._time_slice_all_units_in_state()
        self._construct_leaf_cnodes()
        self._construct_leaf_units_of_composite_objects()
        self._pass_composite_object_velocity()
        return self._state
