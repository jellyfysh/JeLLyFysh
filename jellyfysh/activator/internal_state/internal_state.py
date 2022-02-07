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
"""Module for the abstract InternalState class."""
from abc import ABCMeta, abstractmethod
from typing import Any
from jellyfysh.base.initializer import Initializer


class InternalState(Initializer, metaclass=ABCMeta):
    """
    Abstract class for a general internal state.

    An internal state should map (possibly several) identifiers of the global state onto identifiers of the internal
    state and vice versa. The initialize and update methods keep the internal state consistent with the global state.

    The specific type of the internal state identifiers depends on the implementation of this class. It is therefore of
    type Any in this class. Likewise, the specific type of the global state identifiers depends on the used state
    handler. Thus, global state identifiers are also of type Any in this class.
    """

    def __init__(self) -> None:
        """
        The (currently empty) constructor of the abstract Cells class.
        """
        super().__init__()

    @abstractmethod
    def initialize(self, extracted_global_state: Any) -> None:
        """
        Initialize the internal state based on the full extracted global state from the state handler.

        Extends the initialize method of the Initializer class. Use this method once in the beginning of the run to
        initialize the internal state. Only after a call of this method, other public methods of this class can be
        called without raising an error. The precise format of the argument is specified by the used state handler.

        Parameters
        ----------
        extracted_global_state : Any
            The full extracted global state from the state handler.
        """
        super().initialize()

    @abstractmethod
    def __getitem__(self, internal_state_identifier: Any) -> Any:
        """
        Return the stored global state identifier(s) based on an internal state identifier.

        Parameters
        ----------
        internal_state_identifier : Any
            The internal state identifier.

        Returns
        -------
        Any
           The global state identifier(s) associated with the internal state identifier.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, extracted_active_global_state: Any) -> None:
        """
        Update the internal state based on the extracted active global state.

        Use this method to keep the internal state consistent with the global state.
        The precise format of the extracted_active_global_state is specified by the used state handler.

        Parameters
        ----------
        extracted_active_global_state : Any
            The active global state information.
        """
        raise NotImplementedError
