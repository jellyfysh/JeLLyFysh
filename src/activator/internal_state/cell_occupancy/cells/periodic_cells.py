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
"""Module for the abstract PeriodicCells class."""
from abc import ABCMeta, abstractmethod
from typing import Any
from .cells import Cells


class PeriodicCells(Cells, metaclass=ABCMeta):
    """
    Abstract class for a general periodic cell geometry.

    Each implementation should define all Cells method plus the methods here under consideration of translational
    invariance.
    Similar to the Cells class, cells are accessed by identifiers. The specific form of these (integer, tuple...) is
    left open so they are of type Any in this class.
    """

    def __init__(self) -> None:
        """
        The (currently empty) constructor of the abstract Cells class.
        """
        super().__init__()

    @abstractmethod
    def relative_cell(self, cell: Any, reference_cell: Any) -> Any:
        """
        Return the cell identifier with the same distance to the origin cell as the cell to the reference cell.

        This is the inverse method of the translate method. This method takes periodic boundaries into account.

        Parameters
        ----------
        cell : Any
            The cell which gets mapped onto a cell with the same distance to the origin cell as the cell to the
            reference_cell.
        reference_cell : Any
            The cell which gets mapped onto the origin cell.

        Returns
        -------
        Any
            The cell identifier with the same distance to the origin cell as the cell to the reference cell.
        """
        raise NotImplementedError

    @abstractmethod
    def translate(self, cell: Any, relative_cell: Any) -> Any:
        """
        Return the cell identifier with the same distance to the given cell as the given relative cell to the origin.

        This is the inverse method of the relative_cell method. This method takes periodic boundaries into account.

        Parameters
        ----------
        cell : Any
            The cell from which the resulting cell has the same distance as the relative_cell to the origin cell.
        relative_cell : Any
            The cell whose distance to the origin cell matters.

        Returns
        -------
        Any
            The cell identifier with the same distance to the cell as the relative_cell to the origin cell.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def zero_cell(self) -> Any:
        """
        Return the cell identifier which is located at the origin.

        Returns
        -------
        Any
            The cell identifier which is located at the origin.
        """
        raise NotImplementedError
