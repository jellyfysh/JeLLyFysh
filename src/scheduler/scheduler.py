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
"""Module for the abstract Scheduler class."""
from abc import ABCMeta, abstractmethod
from typing import Any


class Scheduler(metaclass=ABCMeta):
    """
    Abstract class for a general scheduler to be used in the mediator.

    In JF, the scheduler keeps track of candidate events and their associated event handler references. It can select
    among the candidate events the one with the smallest candidate event time. Moreover, it can delete events.
    Generally, an event stored in the scheduler consists of a candidate event time and an arbitrary associated object.
    When asked for the succeeding event, the scheduler should return the object which is associated to the shortest
    time. Events can be trashed based on the associated objects.
    Although the associated object is in JF always an event handler reference (which is also hinted by the argument
    names), they will have the type Any in this class.
    """

    def __init__(self):
        """
        The constructor of the abstract Scheduler class.
        """
        pass

    @abstractmethod
    def push_event(self, time: float, event_handler: Any) -> None:
        """
        Push an event into the scheduler.

        Parameters
        ----------
        time : float
            The event time.
        event_handler : Any
            The associated object.
        """
        raise NotImplementedError

    @abstractmethod
    def get_succeeding_event(self) -> Any:
        """
        Get the object currently stored in the scheduler which was pushed with the smallest event time.

        Returns
        -------
        Any
            The object associated to the smallest stored event time.
        """
        raise NotImplementedError

    @abstractmethod
    def trash_event(self, event_handler: Any) -> None:
        """
        Delete an event in the scheduler based on the associated object.

        Parameters
        ----------
        event_handler : Any
            The associated object.
        """
        raise NotImplementedError

    @abstractmethod
    def update_logging(self) -> None:
        """
        Update the logging of this class.

        This method is called in resume.py which might be run with a different logging level compared to the run which
        created the dump. This method then ensures that this class logs on the correct level.
        """
        raise NotImplementedError
