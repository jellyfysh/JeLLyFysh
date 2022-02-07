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
"""Module for the abstract InputHandler class."""
from abc import ABCMeta, abstractmethod
from typing import Any


class InputHandler(metaclass=ABCMeta):
    """
    Abstract class for an input handler used in the input-output handler.

    An input handler creates the initial global physical state. It should also set up the setting package so that
    all possible global state identifiers can be created from all JF modules.
    """

    def __init__(self) -> None:
        """The constructor of the InputHandler class."""
        pass

    @abstractmethod
    def read(self) -> Any:
        """
        Return the initial global physical state.

        The returned object is passed to the initialize method of the state handler of the run. The precise format
        should therefore be suiting for the state handler.

        Returns
        -------
        Any
            The initial global physical state.
        """
        raise NotImplementedError
