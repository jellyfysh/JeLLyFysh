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
"""Module for abstract StartOfRunEventHandler class."""
from abc import ABCMeta
from typing import Any
from jellyfysh.event_handler import EventHandler


class StartOfRunEventHandler(EventHandler, metaclass=ABCMeta):
    """
    Abstract event handler base class used to start a run.

    At the beginning of a run, the activator specifically returns an instance of this class to be run by the mediator.
    In each run, there should be only a single instance of a StartOfRunEventHandler.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        The constructor of the StartOfRunEventHandler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)
