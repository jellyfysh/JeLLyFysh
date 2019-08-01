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
"""Module for helper functions used in the event handlers."""
import logging
from typing import Sequence, Tuple


_logger = logging.getLogger(__name__)


def analyse_velocity(velocity: Sequence[float]) -> Tuple[int, float]:
    """
    Return the direction of motion and the speed of the given velocity.

    The velocity must have only a single nonzero component (the speed in the direction of motion).

    Parameters
    ----------
    velocity : Sequence[float]
        The velocity.

    Returns
    -------
    (int, float)
        The direction of motion, the speed.

    Raises
    ------
    AssertionError
        If more than one component of the velocity is nonzero.
    """
    direction_of_motions = [index for index, component in enumerate(velocity)
                            if component != 0.0]
    assert len(direction_of_motions) == 1
    return direction_of_motions[0], velocity[direction_of_motions[0]]


def bounding_potential_warning(event_handler_name: str, bounding_derivative: float, real_derivative: float) -> None:
    """
    Log a warning if the real derivative is greater than zero and larger than the bounding derivative.

    The logging message includes the name of the event handler.

    Parameters
    ----------
    event_handler_name : str
        The event handler name.
    bounding_derivative : float
        The bounding potential derivative.
    real_derivative : float
        The real potential derivative.
    """
    if real_derivative > 0 and bounding_derivative < real_derivative:
        _logger.warning("In the event handler {0} the bounding event rate {1} "
                        "is not bigger than the real event rate {2}"
                        .format(event_handler_name, bounding_derivative, real_derivative))
