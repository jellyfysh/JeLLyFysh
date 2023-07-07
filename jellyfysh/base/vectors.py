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
"""Module for functions acting on vectors."""
import math
import random
from typing import List, Sequence


def norm(vector: Sequence[float]) -> float:
    """
    Return the euclidean norm of the vector.

    Parameters
    ----------
    vector : Sequence[float]
        The vector.

    Returns
    -------
    float
        The norm.
    """
    return sum(component * component for component in vector) ** 0.5


def norm_sq(vector: Sequence[float]) -> float:
    """
    Return the square of the euclidean norm of the vector.

    Parameters
    ----------
    vector : Sequence[float]
        The vector.

    Returns
    -------
    float
        The squared norm.
    """
    return sum(component * component for component in vector)


def dot(vector_one: Sequence[float], vector_two: Sequence[float]) -> float:
    """
    Return the dot product of the two vectors.

    Parameters
    ----------
    vector_one : Sequence[float]
        The first vector.
    vector_two : Sequence[float]
        The second vector.

    Returns
    -------
    float
        The dot product.

    Raises
    ------
    AssertionError
        If the vectors have different lengths.
    """
    assert len(vector_one) == len(vector_two)
    return sum(x * y for x, y in zip(vector_one, vector_two))


def normalize(vector: Sequence[float], new_norm: float = 1.0) -> List[float]:
    """
    Return the vector normalized to a given norm.

    Parameters
    ----------
    vector : Sequence[float]
        The vector to be normalized. Will not be changed.
    new_norm : float, optional
        The desired norm.

    Returns
    -------
    List[float]
        A copy of the vector normalized to new_norm.
    """
    assert new_norm > 0
    old_norm = norm(vector)
    new_vector = [value / old_norm * new_norm for value in vector]
    return new_vector


def angle_between_two_vectors(vector_one: Sequence[float], vector_two: Sequence[float]) -> float:
    """
    Return the angle between two vectors.

    Parameters
    ----------
    vector_one : Sequence[float]
        The first vector.
    vector_two : Sequence[float]
        The second vector.

    Returns
    -------
    float
        The angle.
    """
    return math.acos(dot(vector_one, vector_two) / norm(vector_one) / norm(vector_two))


def random_vector_on_unit_sphere(dimension: int) -> List[float]:
    """
    Return a random vector on a unit sphere in the given dimension.

    Parameters
    ----------
    dimension : int
        The dimension.

    Returns
    -------
    List[float]
        The random vector.
    """
    while True:
        vector = [random.uniform(-1, 1) for _ in range(dimension)]
        vector_norm = norm(vector)
        if 0.0 < vector_norm <= 1.0:
            break
    vector = normalize(vector)
    return vector
