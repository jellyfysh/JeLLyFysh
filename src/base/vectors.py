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
"""Module for functions acting on vectors."""
import math
from operator import itemgetter
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


def copy_vector_with_replaced_component(vector: Sequence[float], component_to_replace: int,
                                        replacement_value: float) -> List[float]:
    """
    Return a copy of the vector where the given component is replaced by the given value.

    If the given index of the component is too large, the vector will just be copied.

    Parameters
    ----------
    vector : Sequence[float]
        The vector.
    component_to_replace : int
        The index of the component which should be replaced.
    replacement_value : float
        The value to insert at the wanted component.

    Returns
    -------
    List[float]
        The copied vector with the component replaced.
    """
    return [value if index != component_to_replace else replacement_value for index, value in enumerate(vector)]


def displacement_until_new_norm_sq_component_positive(old_vector: Sequence[float], norm_sq_of_new_vector: float,
                                                      translation_direction: int) -> float:
    """
    Return the displacement which has to be subtracted from the given component of the vector so that the squared norm
    equals the given norm.

    The component of the vector must be > 0.0.

    Parameters
    ----------
    old_vector : Sequence[float]
        The vector.
    norm_sq_of_new_vector : float
        The wanted squared norm.
    translation_direction : int
        The index of the component which should be changed.

    Returns
    -------
    float
        The displacement.

    Raises
    ------
    AssertionError
        If the component of the vector is not greater than zero.
    """
    assert old_vector[translation_direction] > 0.0
    return old_vector[translation_direction] - math.sqrt(norm_sq_of_new_vector - sum(
        [value ** 2 for index, value in enumerate(old_vector) if index != translation_direction]))


def displacement_until_new_norm_sq_component_negative(old_vector: Sequence[float], norm_sq_of_new_vector: float,
                                                      translation_direction: int) -> float:
    """
    Return the displacement which has to be subtracted from the given component of the vector so that the squared norm
    equals the given norm.

    The component of the vector must be <= 0.0.

    Parameters
    ----------
    old_vector : Sequence[float]
        The vector.
    norm_sq_of_new_vector : float
        The wanted squared norm.
    translation_direction : int
        The index of the component which should be changed.

    Returns
    -------
    float
        The displacement.

    Raises
    ------
    AssertionError
        If the component of the vector is not smaller than or equal zero.
    """
    assert old_vector[translation_direction] <= 0.0
    return old_vector[translation_direction] + math.sqrt(norm_sq_of_new_vector - sum(
        [value ** 2 for index, value in enumerate(old_vector) if index != translation_direction]))


_permutations_3d = [itemgetter(*[0, 1, 2]), itemgetter(*[1, 2, 0]), itemgetter(*[2, 0, 1])]


def permutation_3d(vector: Sequence[float], main_direction: int) -> Sequence[float]:
    """
    Return the vector rotated until the component at the given main direction is first.

    Parameters
    ----------
    vector : Sequence[float]
        The vector.
    main_direction : int
        The main direction.

    Returns
    -------
    Sequence[float]
        The rotated vector.

    """
    assert len(vector) == 3
    assert 0 <= main_direction <= 2
    return _permutations_3d[main_direction](vector)


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
