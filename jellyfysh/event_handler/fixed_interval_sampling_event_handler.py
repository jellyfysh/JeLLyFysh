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
"""Module for the FixedIntervalSamplingEventHandler class."""
import logging
from typing import Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
from .abstracts import SamplingEventHandler


class FixedIntervalSamplingEventHandler(SamplingEventHandler):
    """
    Event handler which triggers a sampling in fixed intervals.

    This class implements a SamplingEventHandler. It is connected to an output handler and the mediator defines
    a specific mediating method for this class. There the extracted full global state is handed to the output handler
    so that it can start its sampling.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    The instance of this event handler should always be active and its events should not be trashed by other events.
    Only after an event of this event handler, a new event should be computed.

    This event handler returns candidate event times that are separated by a given sampling time interval. Optionally,
    it can return 0.0 as the first candidate event time. Otherwise, the first candidate event time equals the sampling
    interval.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information). The sampling interval that is added to the old candidate
    event time, however, can stay a simple float because it is always of the same order of magnitude during a run of JF.
    """

    def __init__(self, sampling_interval: float, output_handler: str, first_event_time_zero: bool = False) -> None:
        """
        The constructor of the FixedIntervalSamplingEventHandler class.

        Parameters
        ----------
        sampling_interval : float
            The time interval of the sampling.
        output_handler : str
            The name of the output handler.
        first_event_time_zero : bool, optional
            If the first returned candidate event time is zero.

        Raises
        ------
        base.exceptions.ConfigurationError:
            If the sampling interval is not greater than zero.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           sampling_interval=sampling_interval, output_handler=output_handler)
        super().__init__(output_handler=output_handler)
        if not sampling_interval > 0.0:
            raise ConfigurationError("The sampling_interval in the event handler {0} has to be > 0.0."
                                     .format(self.__class__.__name__))
        self._sampling_interval = sampling_interval
        self._event_time = Time(0.0, 0.0) if not first_event_time_zero else Time.from_float(-sampling_interval)

    def send_event_time(self) -> Time:
        """
        Return the candidate event time.

        Returns
        -------
        base.time.Time
            The candidate event time.
        """
        self._event_time += self._sampling_interval
        return self._event_time

    def send_out_state(self, cnodes_with_active_units: Sequence[Node]) -> Sequence[Node]:
        """
        Return the out-state.

        This method receives the branches of all independent active units. These are time-sliced and returned.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.
        """
        self._store_in_state(cnodes_with_active_units)
        self._time_slice_all_units_in_state()
        return self._state
