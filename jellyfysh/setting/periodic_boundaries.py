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
"""Module for the abstract PeriodicBoundaries class and the default PeriodicBoundariesNotImplemented class."""
from abc import ABCMeta, abstractmethod
from typing import MutableSequence, Sequence


class PeriodicBoundaries(metaclass=ABCMeta):
    """
    Abstract class for periodic boundaries used in the setting package.

    This class implements several methods which are used when considering periodic boundary conditions.
    Usually, the inheriting class should not be instantiated by the user (via the configuration file), but in a setter
    class of some setting. For example, the hypercubic periodic boundaries instance is created in the constructor of
    the HypercubicSetting class.
    """

    @staticmethod
    @abstractmethod
    def correct_position(position: MutableSequence[float]) -> None:
        """
        Correct the given position vector in place for periodic boundaries.

        Parameters
        ----------
        position : MutableSequence[float]
            The position vector.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def correct_position_entry(position_entry: float, index: int) -> float:
        """
        Return the position entry at the given component corrected for periodic boundaries.

        Parameters
        ----------
        position_entry : float
            The position entry.
        index : int
            The index of the position entry within the position vector.

        Returns
        -------
        float
            The position entry corrected for periodic boundaries.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def separation_vector(reference_position: Sequence[float],
                          target_position: Sequence[float]) -> MutableSequence[float]:
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
        MutableSequence[float]
            The shortest separation vector target_position - reference_position, possibly corrected for periodic
            boundaries.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def correct_separation(separation: MutableSequence[float]) -> None:
        """
        Correct the given separation vector in place for periodic boundaries.

        Parameters
        ----------
        separation : MutableSequence[float]
            The separation vector.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def correct_separation_entry(separation_entry: float, index: int) -> float:
        """
        Return the separation entry at the given component corrected for periodic boundaries.

        Parameters
        ----------
        separation_entry : float
            The separation entry.
        index : int
            The index of the position entry within the position vector.

        Returns
        -------
        float
            The separation entry corrected for periodic boundaries.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def next_image(position_entry: float, direction: int) -> None:
        """
        Return the translated position in the next image in the given direction.

        Parameters
        ----------
        position_entry : float
            The starting position entry.
        direction : int
            The direction in which the position entry should be translated.

        Returns
        -------
        float
            The position entry translated in the next image in the given direction.
        """
        raise NotImplementedError


class PeriodicBoundariesNotImplemented(PeriodicBoundaries):
    """
    This class implements periodic boundaries but each method raises an exception.

    An instance of this class is the default instance for the periodic boundaries in the Setting class.
    """
    @staticmethod
    def correct_position(position: MutableSequence[float]) -> None:
        """
        Correct the given position vector in place for periodic boundaries.

        Parameters
        ----------
        position : MutableSequence[float]
            The position vector.

        Raises
        ------
        NotImplementedError
            On each call of this method.
        """
        raise NotImplementedError("Periodic boundaries not implemented!")

    @staticmethod
    def correct_position_entry(position_entry: float, index: int) -> float:
        """
        Return the position entry at the given component corrected for periodic boundaries.

        Parameters
        ----------
        position_entry : float
            The position entry.
        index : int
            The index of the position entry within the position vector.

        Returns
        -------
        float
            The position entry corrected for periodic boundaries.

        Raises
        ------
        NotImplementedError
            On each call of this method.
        """
        raise NotImplementedError("Periodic boundaries not implemented!")

    @staticmethod
    def separation_vector(reference_position: Sequence[float],
                          target_position: Sequence[float]) -> MutableSequence[float]:
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
        MutableSequence[float]
            The shortest separation vector target_position - reference_position, possibly corrected for periodic
            boundaries.

        Raises
        ------
        NotImplementedError
            On each call of this method.
        """
        raise NotImplementedError("Periodic boundaries not implemented!")

    @staticmethod
    def correct_separation(separation: MutableSequence[float]) -> None:
        """
        Correct the given separation vector in place for periodic boundaries.

        Parameters
        ----------
        separation : MutableSequence[float]
            The separation vector.

        Raises
        ------
        NotImplementedError
            On each call of this method.
        """
        raise NotImplementedError("Periodic boundaries not implemented!")

    @staticmethod
    def correct_separation_entry(separation_entry: float, index: int) -> float:
        """
        Return the separation entry at the given component corrected for periodic boundaries.

        Parameters
        ----------
        separation_entry : float
            The separation entry.
        index : int
            The index of the position entry within the position vector.

        Returns
        -------
        float
            The separation entry corrected for periodic boundaries.

        Raises
        ------
        NotImplementedError
            On each call of this method.
        """
        raise NotImplementedError("Periodic boundaries not implemented!")

    @staticmethod
    def next_image(position_entry: float, direction: int) -> float:
        """
        Return the translated position in the next image in the given direction.

        Parameters
        ----------
        position_entry : float
            The starting position entry.
        direction : int
            The direction in which the position entry should be translated.

        Returns
        -------
        float
            The position entry translated in the next image in the given direction.

        Raises
        ------
        NotImplementedError
            On each call of this method.
        """
        raise NotImplementedError("Periodic boundaries not implemented!")
