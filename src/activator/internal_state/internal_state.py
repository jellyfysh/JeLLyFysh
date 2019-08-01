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
"""Module for the abstract InternalState class."""
from abc import ABCMeta, abstractmethod
from typing import Any
from base.initializer import Initializer


class InternalState(Initializer, metaclass=ABCMeta):
    """
    Abstract class for a general internal state.

    An internal state should be accessible via some identifier, meaning that it returns identifiers of the global
    state given such an internal state identifier. Also it provides an initializing method and an update
    method to keep the internal state consistent with the global state.
    The specific form of the internal state identifier (integer, tuple...) is left open and are therefore of type Any in
    this class. The same is true for the global state identifiers.
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
        called without raising an error. The precise format of the argument is specified by the activator class which
        uses the internal state.

        Parameters
        ----------
        extracted_global_state : Any
            The full extracted global state from the state handler.
        """
        super().initialize()

    @abstractmethod
    def __getitem__(self, internal_state_identifier: Any) -> Any:
        """
        Return the stored global state identifier based on an internal state identifier.

        Parameters
        ----------
        internal_state_identifier : Any
            The internal state identifier.

        Returns
        -------
        Any
           The global state identifier associated with the internal state identifier.
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
