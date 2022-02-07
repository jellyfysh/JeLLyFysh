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
"""Module for the FinalTimeEndOfRunEventHandler class."""
import logging
from typing import Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
from .abstracts import EndOfRunEventHandler


class FinalTimeEndOfRunEventHandler(EndOfRunEventHandler):
    """
    Event handler which ends a run after a fixed final time.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    This class implements an EndOfRunEventHandler. It is optionally connected to an output handler, which gets access to
    the full extracted global state at the end of the run. Also, the mediating method for this class raises an EndOfRun
    exception.

    The instance of this event handler should always be active and its events should not be trashed by other events.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information).
    """

    def __init__(self, end_of_run_time: float, output_handler: str = None) -> None:
        """
        The constructor of the FinalTimeEndOfRunEventHandler class.

        Parameters
        ----------
        end_of_run_time : float
            The event time at which the run is ended.
        output_handler : str or None, optional
            The name of the output handler.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the end of run time is not greater than or equal to zero (the latter case logs a warning).
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           end_of_run_time=end_of_run_time, output_handler=output_handler)
        super().__init__(output_handler=output_handler)
        if not end_of_run_time >= 0.0:
            raise ConfigurationError("The end_of_run_time in the event handler {0} has to be >= 0.0."
                                     .format(self.__class__.__name__))
        if end_of_run_time == 0.0:
            logging.getLogger(__name__).warning("The end_of_run_time in the event handler {0} is equal to 0.0. The "
                                                "simulation will stop immediately once the run is started."
                                                .format(self.__class__.__name__))
        self._event_time = Time.from_float(end_of_run_time)

    def send_event_time(self) -> Time:
        """
        Return the candidate event time.

        Returns
        -------
        base.time.Time
            The candidate event time.
        """
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
