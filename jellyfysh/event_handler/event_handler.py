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
"""Module for the abstract EventHandler class."""
from abc import ABCMeta, abstractmethod
import inspect
from typing import Any, Sequence, Tuple, Union
from jellyfysh.base.time import Time


class EventHandler(metaclass=ABCMeta):
    """
    Abstract class for a general event handler to be used in the mediator.

    Event handlers calculate events in JF. An event terminates each straight-line leg of the time evolution of a
    piecewise non-interacting system. It consists of the candidate event time and the out-state, that is the updated
    starting configuration for the ensuing leg of the non-interacting time evolution.

    Each event handler only requires the global state reduced to a single factor in order to determine candidate event
    times and out-states (called the in-state). Similarly, the out-state does not consist of the full global state but
    only of the changed parts.

    Event handlers that realize factors or pseudo-factors receive the in-state as an argument of the send_event_time
    method and store it internally, so that the send_out_state method does not need any arguments. In contrast, for
    event handlers that realize set of factors or pseudo-factors the element of the set that triggers the event is
    unknown at the event-time request. Then, the send_event_time method has the part of the in-state which is necessary
    to calculate the candidate event time as the argument. Also, it may return supplementary arguments, which are used
    by the mediator to construct the full in-state. The remaining part of the in-state is then the argument of the
    send_out_state method. This class records the number of arguments for the both methods, which is used in the
    mediator to distinguish these cases.

    The precise format of the in-state and the out-state is specified by the used state handler.

    Note that in order to avoid loss of precision during long runs of JF, candidate event times and time stamps of
    active units are not stored as simple floats but as the quotient and remainder of an integer division of the time
    with 1 (see base.time.Time class for more information).

    Attributes
    ----------
    number_send_event_time_arguments : int
        The number of arguments of the send_event_time method.
    number_send_out_state_arguments : int
        The number of arguments of the send_out_state method.
    """

    def __init__(self, **kwargs: Any):
        """
        The constructor of the EventHandler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        self.number_send_event_time_arguments = len(inspect.signature(self.send_event_time).parameters)
        self.number_send_out_state_arguments = len(inspect.signature(self.send_out_state).parameters)
        super().__init__(**kwargs)

    @abstractmethod
    def send_event_time(self, *in_state: Sequence[Any]) -> Union[Time, Tuple[Time, Sequence[Any]]]:
        """
        Return the candidate event time optionally together with arguments needed by the mediator to construct arguments
        of the send_out_state method.

        The precise format of the in-state depends on the event handler and on the used state handler.

        If this method returns objects which are needed by the mediator to construct the arguments of the send_out_state
        method, this method should return a tuple of the event time and a sequence of these arguments. This sequence is
        unpacked in the mediator when they are passed to the method which gets the arguments.

        Parameters
        ----------
        in_state : Sequence[Any]
            The in-state or part of the in-state needed to calculate the candidate event time.

        Returns
        -------
        base.time.Time or (base.time.Time, Sequence[Any])
            The candidate event time optionally together with a sequence of arguments needed to construct the arguments
            of the send_out_state method,
        """
        raise NotImplementedError

    @abstractmethod
    def send_out_state(self, *args: Any) -> Any:
        """
        Return the out-state.

        The precise format of the out-state depends on the used state handler.

        If the send_event_time method did not receive the complete in-state, the remaining part is the argument of
        this method. The precise format of this also depends on the used state handler.

        Parameters
        ----------
        args : Any
            The remaining part of the in-state needed to calculate the out-state (optional).

        Returns
        -------
        Any
            The out-state.
        """
        raise NotImplementedError


class EventHandlerWithUnconfirmedEvents(EventHandler, metaclass=ABCMeta):
    """
    Abstract class for an event handler that can create unconfirmed events.

    Events in JF can be not confirmed if a bounding potential was used. If such an unconfirmed event led to the smallest
    candidate event time, only the event handler that was responsible for the unconfirmed event needs to compute
    a new candidate event time. Other events do not have to be trashed and recomputed.
    An unconfirmed event has 'None' as its out-state. A new candidate event time can then be calculated by using the
    resend_event_time method which never receives any arguments. This is because the (possibly) necessary in-state was
    should already be stored internally in the call of the send_event_time method.
    """

    def __init__(self, **kwargs: Any):
        """
        The constructor of the EventHandlerWithUnconfirmedEvents class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)

    @abstractmethod
    def send_out_state(self, *args: Any) -> Union[Any, None]:
        """
        Return the out-state.

        The precise format of the out-state of a confirmed event depends on the used state handler.
        If the send_event_time method did not receive the complete in-state, the remaining part is the argument of
        this method. The precise format of this also depends on the used state handler.
        If the event is not confirmed, None is returned as the out-state.

        Parameters
        ----------
        args : Any
            The remaining part of the in-state needed to calculate the out-state (optional).

        Returns
        -------
        Any or None
            The out-state.
        """
        raise NotImplementedError

    @abstractmethod
    def resend_event_time(self) -> Union[Time, Tuple[Time, Sequence[Any]]]:
        """
        Return the candidate event time, optionally together with arguments needed by the mediator to construct
        arguments of the send_out_state method, based on the internally stored in-state.

        This method should use the in-state that was internally stored in the send_event_time method.
        The precise format of the in-state depends on the event handler and on the used state handler.
        If this method returns objects which are needed by the mediator to construct the arguments of the send_out_state
        method, this method should return a tuple of the event time and a sequence of these arguments. This sequence is
        unpacked in the mediator when they are passed to the method which gets the arguments.

        Returns
        -------
        base.time.Time or (base.time.Time, Sequence[Any])
            The candidate event time optionally together with a sequence of arguments needed to construct the arguments
            of the send_out_state method,
        """
        raise NotImplemented
