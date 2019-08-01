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
"""Module for the HypercubicPeriodicBoundaries class."""
from typing import List, MutableSequence, Sequence
from setting import hypercubic_setting as setting
from .periodic_boundaries import PeriodicBoundaries


class HypercubicPeriodicBoundaries(PeriodicBoundaries):
    """
    This class implements periodic boundaries for a hypercubic setting.

    An instance of this class is used for the periodic boundaries in the HypercubicSetting class.
    """
    @staticmethod
    def correct_position(position: MutableSequence[float]) -> None:
        """
        Correct the given position vector in place for periodic boundaries.

        Parameters
        ----------
        position : MutableSequence[float]
            The position vector.
        """
        for index, entry in enumerate(position):
            position[index] = HypercubicPeriodicBoundaries.correct_position_entry(entry, index)

    @staticmethod
    def correct_position_entry(position_entry: float, _: int) -> float:
        """
        Return the position entry at the given component corrected for periodic boundaries.

        Parameters
        ----------
        position_entry : float
            The position entry.
        _ : int
            The index of the position entry within the position vector (not used in this class).

        Returns
        -------
        float
            The position entry corrected for periodic boundaries.
        """
        return position_entry % setting.system_length

    @staticmethod
    def separation_vector(reference_position: Sequence[float],
                          target_position: Sequence[float]) -> List[float]:
        """
        Return the shortest separation vector of the target position divided by the reference position.

        Parameters
        ----------
        reference_position : Sequence[float]
            The reference position.
        target_position : Sequence[float]
            The target position.

        Returns
        -------
        List[float]
            The shortest separation vector target_position - reference_position, possibly corrected for periodic
            boundaries.
        """
        separation = [target_position[index] - reference_position[index] for index in
                      range(setting.dimension)]
        HypercubicPeriodicBoundaries.correct_separation(separation)
        return separation

    @staticmethod
    def correct_separation(separation: MutableSequence[float]) -> None:
        """
        Correct the given separation vector in place for periodic boundaries.

        Parameters
        ----------
        separation : MutableSequence[float]
            The separation vector.
        """
        for index, entry in enumerate(separation):
            separation[index] = HypercubicPeriodicBoundaries.correct_separation_entry(entry, index)

    @staticmethod
    def correct_separation_entry(separation_entry: float, _: int) -> float:
        """
        Return the separation entry at the given component corrected for periodic boundaries.

        Parameters
        ----------
        separation_entry : float
            The separation entry.
        _ : int
            The index of the position entry within the position vector (not used in this class).

        Returns
        -------
        float
            The separation entry corrected for periodic boundaries.
        """
        return ((separation_entry + setting.system_length_over_two) % setting.system_length
                - setting.system_length_over_two)

    @staticmethod
    def next_image(position_entry: float, _: int) -> float:
        """
        Return the translated position in the next image in the given direction.

        Parameters
        ----------
        position_entry : float
            The starting position entry.
        _ : int
            The direction in which the position entry should be translated (not used in this class).

        Returns
        -------
        float
            The position entry translated in the next image in the given direction.
        """
        return position_entry + setting.system_length
