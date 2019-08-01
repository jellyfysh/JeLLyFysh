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
"""Module for the Unit class."""
from typing import Any, Mapping, Sequence


class Unit(object):
    """
    Class to store physical and lifting state information, meaning the identifier, the position, the charges,
    the velocity, and the time stamp of a particle.

    A particle can be a point mass or a composite point object. This class is used as a messaging format across the
    application. The identifier allows the state handler to incorporate unit information into the global state.

    Attributes
    ----------
    identifier : Any
        The identifier of the particle this unit corresponds to.
    position : Sequence[float]
        The position of the particle in the global physical state.
    charge : Mapping[str, float] or None
        A map from the name onto the value of the charge of the particle.
    velocity : Sequence[float] or None
        The velocity of the particle in the global lifting state. None if the velocity is 0 in all directions.
    time_stamp : float or None
        The time-stamp of the particle in the global lifting state.
    """

    def __init__(self, identifier: Any, position: Sequence[float], charge: Mapping[str, float] = None,
                 velocity: Sequence[float] = None, time_stamp: float = None) -> None:
        """
        The constructor of the Unit class.

        Parameters
        ----------
        identifier : Any
            The identifier of the particle this unit corresponds to.
        position : Sequence[float]
            The initial position of the particle.
        charge : Mapping[str, float] or None, optional
             A map from the name onto the value of the charge of the particle.
        velocity : Sequence[float] or None, optional
            The velocity of the particle.
        time_stamp : float or None, optional
            The time stamp of the particle.
        """
        self.identifier = identifier
        self.position = position
        self.charge = charge
        self.velocity = velocity
        self.time_stamp = time_stamp
