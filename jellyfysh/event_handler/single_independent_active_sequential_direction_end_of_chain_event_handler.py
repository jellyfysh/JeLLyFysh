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
"""Module for the SingleIndependentActiveSequentialDirectionEndOfChainEventHandler class."""
import logging
import math
from typing import Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
import jellyfysh.setting as setting
from .single_independent_active_periodic_direction_end_of_chain_event_handler \
    import SingleIndependentActivePeriodicDirectionEndOfChainEventHandler


class SingleIndependentActiveSequentialDirectionEndOfChainEventHandler(
    SingleIndependentActivePeriodicDirectionEndOfChainEventHandler):
    """
    Event handler which ends an event chain in fixed time intervals.

    This event handler is implemented only for two dimensions. Here, it rotates the velocity vector of a single
    independent active (root or leaf) unit by an angle around the origin that is set on initialization. The absolute
    value of the velocity is kept constant. After an end-of-chain event of this event handler, a random unit is
    independent active. This unit is a leaf (root) unit if the independent active unit at the moment of the computation
    of the candidate event time of this event handler was also a leaf (root) unit. This implies that the end-of-chain
    event should be trashed and recomputed if the simulation switches from an independent active leaf to root unit.

    The instance of this event handler should always be active because it computes the new candidate event time based
    on the time of the previously committed event (i.e., based on the time of the last call of the send_out_state
    method).

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    This class implements an EndOfChainEventHandler. For more details on the implementation of this event handler, see
    .abstracts.end_of_chain_event_handler. This class is also very similar to the
    SingleIndependentActivePeriodicDirectionEndOfChainEventHandler that aligns the velocity of a single independent
    active unit periodically with the cartesian axes in the positive direction. We therefore use inheritance and only
    change the computation of the new velocity vector.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information). The chain time that is added to the old candidate event
    time, however, can stay a simple float because it is always of the same order of magnitude during a run of JF.
    """

    def __init__(self, chain_time: float, delta_phi_degree: float) -> None:
        """
        The constructor of the SingleIndependentActiveSequentialDirectionEndOfChainEventHandler class.

        The rotation angle in degrees by which the two-dimensional velocity vector is rotated on each end-of-chain event
        should be greater than 0.0 and smaller than 360.0. Moreover, 180.0 degrees is excluded to assure irreducibility
        of the Markov chain.

        Parameters
        ----------
        chain_time : float
            The time interval after which a new end-of-chain event occurs.
        delta_phi_degree : float
            The rotation angle in degrees by which the two-dimensional velocity vector is rotated on each end-of-chain
            event.

        Raises
        ------
        base.exceptions.ConfigurationError:
            If the rotation angle in degrees is not in the interval (0, 360.0) or equal to 180.0.
            If the dimension in the setting package is not set to two.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, chain_time=chain_time,
                           delta_phi_degree=delta_phi_degree)
        super().__init__(chain_time=chain_time)
        if not 0.0 < delta_phi_degree < 360.0 or delta_phi_degree == 180.0:
            raise ConfigurationError("The rotation angle of the velocity after an end-of-chain event in the event "
                                     "handler {0} has to be larger than 0.0 and smaller than 360.0, and is further not "
                                     "allowed to be exactly 180.0.".format(self.__class__.__name__))
        if not setting.dimension == 2:
            raise ConfigurationError("The event handler {0} can only be used in two dimensions."
                                     .format(self.__class__.__name__))
        delta_phi = delta_phi_degree * math.pi / 180.0
        self._cos_delta_phi = math.cos(delta_phi)
        self._sin_delta_phi = math.sin(delta_phi)

    def _get_new_velocity(self, old_velocity: Sequence[float]) -> Sequence[float]:
        """
        Return the new velocity based on the old velocity.

        Parameters
        ----------
        old_velocity : Sequence[float]
            The old velocity.

        Returns
        -------
        Sequence[float]
            The new velocity.

        Raises
        ------
        AssertionError
            If the old velocity does not have exactly two components.
        """
        assert len(old_velocity) == 2
        return [old_velocity[0] * self._cos_delta_phi - old_velocity[1] * self._sin_delta_phi,
                old_velocity[0] * self._sin_delta_phi + old_velocity[1] * self._cos_delta_phi]
