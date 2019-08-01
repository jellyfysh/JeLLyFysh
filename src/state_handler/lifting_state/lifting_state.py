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
"""Module for the abstract LiftingState class."""
from abc import ABCMeta, abstractmethod
from typing import Any, Iterable, Sequence, Tuple, Union


class LiftingState(metaclass=ABCMeta):
    """
    Abstract class for a global lifting state.

    The global lifting state maps global state identifiers onto a velocity and a time-stamp of the last time-slicing.
    It should also be able to generate all independent lifted identifiers.
    The precise format of the global state identifiers depends on the used state handler. Therefore we use the type
    Any for them throughout this class.
    """

    def __init__(self):
        """
        The constructor of the abstract LiftingState class.
        """
        pass

    @abstractmethod
    def set(self, identifier: Any, velocity: Union[Sequence[float]], time_stamp: Union[float]) -> None:
        """
        Store the given velocity and time stamp for the global state identifier.

        If the velocity and the time stamp are None, the global state identifier should be deleted from the global
        lifting state.

        Parameters
        ----------
        identifier : Any
            The global state identifier.
        velocity : Sequence[float] or None
            The velocity.
        time_stamp : float or None
            The time stamp.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, identifier: Any) -> Union[Tuple[Sequence[float], float], Tuple[None, None]]:
        """
        Return the velocity and the time stamp for the global state identifier.

        If the global state identifier is not stored within the global lifting state, this method returns (None, None).

        Parameters
        ----------
        identifier : Any
            The global state identifier.

        Returns
        -------
        (Sequence[float], float) or (None, None)
            The velocity, the time stamp.
        """
        raise NotImplementedError

    @abstractmethod
    def yield_independent_lifted_identifiers(self) -> Iterable[Any]:
        """
        Generate all independent lifted identifiers stored in the global lifting state.

        For the example of the TreeStateHandler, independent velocities means the following: If a point mass is active,
        this induces a velocity of the composite point object it belongs to (velocity multiplied by the weight of the
        node of the point mass). Similarly a nonzero velocity of a composite point object leads to the fact, that all
        point masses of this composite point object have the same velocity. For the first case, only the identifier of
        the point mass should be returned, for the latter case only the identifier of the composite point object.

        Yields
        ------
        Any
            The independently lifted global state identifiers.
        """
        raise NotImplementedError
