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
"""Module for the Time class."""
from math import isinf
from math import inf as float_inf


class Time(object):
    """
    Class that represents a time as the quotient and remainder of an integer division with 1.

    Instances of this class are used to store event times. These are part of events that are computed by the event
    handlers of JF. Event times are then compared in the scheduler. Moreover, active units contain a time stamp that is
    an instance of this class.

    In long simulations of JF, the candidate event times and time stamps of active units always increase. At the same
    time, event handlers usually first compute a time displacement and then add this time displacement to a time stamp
    of an active unit to determine the candidate event time. Here, the time displacements stay in the same order of
    magnitude during a simulation. If floats are used to store the involved time stamp and resulting candidate event
    time, events with very close time displacements may yield an equal candidate event time if the time stamp is already
    rather large. This leads to a decreasing precision during the simulation.

    This problem is solved by this class. By splitting the time stamp and candidate event times into a quotient and
    remainder of an integer division with 1, the precision of the remainder stays constant during a simulation. Time
    displacements that are computed in event handlers can still be represented as floats and then added to an instance
    of this class.
    """

    def __init__(self, quotient: float, remainder: float) -> None:
        """
        The constructor of the Time class.

        Parameters
        ----------
        quotient : float
            The quotient of the result of an integer division of the time with 1.
        remainder : float
            The remainder of the result of an integer division of the time with 1.
        """
        self._quotient = quotient
        self._remainder = remainder

    @property
    def quotient(self) -> float:
        """
        Return the quotient of an integer division of the time stored in this instance with 1.

        Returns
        -------
        float
            The quotient.
        """
        return self._quotient

    @property
    def remainder(self) -> float:
        """
        Return the remainder of an integer division of the time stored in this instance with 1.

        Returns
        -------
        float
            The remainder.
        """
        return self._remainder

    @staticmethod
    def from_float(time: float) -> 'Time':
        """
        Create a Time instance that stores the time in the given argument.

        Parameters
        ----------
        time : float
            The time as a float.

        Returns
        -------
        Time
            The time as a instance of this class.
        """
        return Time(*divmod(time, 1.0)) if not isinf(time) else Time(time, time)

    def update(self, other: 'Time') -> None:
        """
        Update the time stored in this instance to the time stored in the given Time instance.

        Parameters
        ----------
        other : Time
            The time as a time instance that should be stored in this instance.
        """
        self._quotient = other.quotient
        self._remainder = other.remainder

    def __add__(self, other: float) -> 'Time':
        """
        Add the given float time to the time of this instance and return a new Time instance with the result.

        Parameters
        ----------
        other : float
            The float time that should be added to the time of this instance.

        Returns
        -------
        Time
            The resulting time as a Time instance.
        """
        if not isinf(other):
            add_quotient, new_remainder = divmod(self._remainder + other, 1.0)
            return Time(self._quotient + add_quotient, new_remainder)
        else:
            return Time(other, other)

    def __sub__(self, other: 'Time') -> float:
        """
        Subtract the given time in a Time instance from the time of this instance and return the result as a float.

        Parameters
        ----------
        other : Time
            The Time instance whose time should be subtracted from the time of this instance.

        Returns
        -------
        float
            The resulting time as a float.
        """
        return self._quotient - other.quotient + self._remainder - other._remainder

    def __eq__(self, other: 'Time') -> bool:
        """
        Return whether the time stored in this instance and the time in the given Time instance are equal.

        Parameters
        ----------
        other : Time
            The other time to which the time of this instance is compared to.

        Returns
        -------
        bool
            True if the two times are equal and false otherwise.
        """
        return self._quotient == other.quotient and self._remainder == other.remainder

    def __lt__(self, other: 'Time') -> bool:
        """
        Return whether the time stored in this instance is smaller than the time in the given instance.

        Parameters
        ----------
        other: Time
            The other time to which the time of this instance is compared to.

        Returns
        -------
        bool
            True if the time in this instance is smaller than the other time and false otherwise.
        """
        return self._quotient < other.quotient or (self._quotient == other.quotient
                                                   and self._remainder < other.remainder)

    def __gt__(self, other: 'Time') -> bool:
        """
        Return whether the time stored in this instance is greater than the time in the given instance.

        Parameters
        ----------
        other : Time
            The other time to which the time of this instance is compared to.

        Returns
        -------
        bool
            True if the time in this instance is greater than the other time and false otherwise.
        """
        lt_result = self.__lt__(other)
        return not lt_result and self.__ne__(other)

    def __le__(self, other: 'Time') -> bool:
        """
        Return whether the time stored in this instance is smaller than or equal to the time in the given instance.

        Parameters
        ----------
        other : Time
            The other time to which the time of this instance is compared to.

        Returns
        -------
        bool
            True if the time in this instance is smaller than or equal to the other time and false otherwise.
        """
        lt_result = self.__lt__(other)
        return lt_result or self.__eq__(other)

    def __ge__(self, other: 'Time') -> bool:
        """
        Return whether the time stored in this instance is greater than or equal to the time in the given instance.

        Parameters
        ----------
        other : Time
            The other time to which the time of this instance is compared to.

        Returns
        -------
        bool
            True if the time in this instance is greater than or equal to the other time and false otherwise.
        """
        lt_result = self.__lt__(other)
        return not lt_result

    def __str__(self) -> str:
        """
        Return the informal string representation of the time stored in this Time instance.

        Returns
        -------
        str
            The string representation.
        """
        return str(self._quotient + self._remainder)

    def __repr__(self) -> str:
        """
        Return the official string representation of this Time instance.

        Returns
        -------
        str
            The string representation.
        """
        return f"Time({self._quotient}, {self._remainder})"


inf = Time(float_inf, float_inf)
"""Instance of the time class that stores an infinite time."""
