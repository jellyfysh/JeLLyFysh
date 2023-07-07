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
"""Module for the FibonacciSphere class"""
import logging
import math
from random import gauss
from typing import Sequence
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base import vectors


class FibonacciSphere(object):
    """See http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/"""
    def __init__(self, number_of_directions: int = 10, epsilon: float = 0.36) -> None:
        self.init_arguments = lambda: {"number_of_directions": number_of_directions, "epsilon": epsilon}
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           number_of_directions=number_of_directions, epsilon=epsilon)
        self._number_of_directions = number_of_directions
        golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
        self._directions = []
        for i in range(number_of_directions):
            phi = 2.0 * math.pi * i / golden_ratio
            theta = math.acos(1.0 - 2.0 * (i + epsilon) / (number_of_directions - 1.0 + 2.0 * epsilon))
            self._directions.append([math.cos(phi) * math.sin(theta), math.sin(phi) * math.sin(theta), math.cos(theta)])
        self._random_directions = [[] for _ in range(number_of_directions)]

    def init_arguments(self):
        raise NotImplementedError

    @property
    def number_of_directions(self):
        return self._number_of_directions

    def yield_directions(self):
        yield from self._directions

    def get_closest_direction_index(self, direction: Sequence[float]):
        assert abs(vectors.norm_sq(direction) - 1.0) < 1.0e-13
        min_cosine_distance = math.inf
        min_index = None
        for index, fibonacci_direction in enumerate(self._directions):
            cosine_distance = 1.0 - vectors.dot(direction, fibonacci_direction)
            if cosine_distance < min_cosine_distance:
                min_cosine_distance = cosine_distance
                min_index = index
        return min_index

    def get_random_direction_about_index(self, direction_index: int) -> Sequence[float]:
        assert direction_index < self._number_of_directions
        while not self._random_directions[direction_index]:
            random_direction = [gauss(mu=0.0, sigma=1.0), gauss(mu=0.0, sigma=1.0), gauss(mu=0.0, sigma=1.0)]
            norm = vectors.norm(random_direction)
            random_direction = [r / norm for r in random_direction]
            relevant_direction_index = self.get_closest_direction_index(random_direction)
            self._random_directions[relevant_direction_index].append(random_direction)
        return self._random_directions[direction_index].pop()
