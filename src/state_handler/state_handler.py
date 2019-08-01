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
"""Module for the abstract StateHandler class."""
from abc import ABCMeta, abstractmethod
from typing import Any
from base.initializer import Initializer
from .lifting_state import LiftingState
from .physical_state import PhysicalState


class StateHandler(Initializer, metaclass=ABCMeta):
    """
    Abstract class for a general state handler to be used in the mediator.

    In JF, the state handler manages the global state, that is the combination of the global physical state (all
    positions and charges of point masses and composite point objects) and the global lifting state (all velocities and
    time-stamps of the last time-slicing of point masses and composite point objects). The state handler provides
    methods to extract and insert parts of the global state. Using these methods, the mediator will construct in-states
    for event handlers and insert the returned out-states into the global state. Also, the state handler is able to
    extract the full global state, which is needed in the output handlers. Similarly it can be initialized by the
    full global state created by an input handler. Finally, the state handler can extract the active global state, which
    is used on the one hand by the activator and on the other hand by some event handlers to time-slice all active point
    masses and composite point objects.

    The precise format of the identifiers used in the state handler and the returned parts of the global state depend
    on the used physical state. Therefore we use the type Any throughout this class.
    """

    def __init__(self, physical_state: PhysicalState, lifting_state: LiftingState) -> None:
        """
        The constructor of the TreeStateHandler class.

        Parameters
        ----------
        physical_state : state_handler.physical_state.PhysicalState
            The class to store the physical state.
        lifting_state : state_handler.lifting_state.LiftingState
            The class to store the lifting state.
        """
        super().__init__()
        self._physical_state = physical_state
        self._lifting_state = lifting_state

    @abstractmethod
    def initialize(self, global_state: Any) -> None:
        """
        Initialize the state handler with the given full global physical state.

        Extends the initialize method of the Initializer class. This method must be extended and used once in the
        beginning of the run to initialize the state handler. Only after a call of this method, other public methods of
        this class can be called without raising an error. The precise format of the argument is specified by the used
        input handler.

        Parameters
        ----------
        global_state : Any
            The full global state.
        """
        super().initialize()

    @abstractmethod
    def extract_from_global_state(self, identifier: Any) -> Any:
        """
        Extract a part of the global state based on a global state identifier.

        The precise format of the identifier depends on the implementation of used physical state. The format of the
        returned part of the global state is specified in the implementation of the state handler. The identifier is
        returned by the activator and the extracted part of the global state will be sent to the event handlers by the
        mediator.
        The extracted part of the global state should be changeable by the event handlers without changing the global
        state itself. Therefore positions and velocities should be copied.

        Parameters
        ----------
        identifier : Any
            The global state identifier.

        Returns
        -------
        Any
            The extracted part of the global state corresponding the global state identifier.
        """
        raise NotImplementedError

    @abstractmethod
    def insert_into_global_state(self, extracted_global_state: Any) -> None:
        """
        Insert an extracted part of the global state into the global state.

        The precise format of the extracted global state depends on the implementation of the state handler.
        It is given by the out-states of the event handlers which changed internally the extracted global state they
        received by the method extract_from_global_state via the mediator.

        Parameters
        ----------
        extracted_global_state : Any
            The part of the global state which should be inserted into the global state.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_active_global_state(self) -> Any:
        """
        Extract the active part of the global state.

        The active part of the global state is the part which appears in the global lifting state and which therefore
        has nonzero velocities and time-stamps.
        The precise format of the extracted active global state depends on the implementation of the state handler.
        It is passed on to the activator, which uses this information to determine the event handlers and their in-state
        identifiers to run next by the mediator. Also this method can be useful when an event handler wants to
        time-slice the full active global state.
        The extracted part of the global state should be changeable by the event handlers without changing the global
        state itself. Therefore positions and velocities should be copied.

        Returns
        -------
        Any
            The active global state.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_global_state(self) -> Any:
        """
        Extract the full global state.

        The precise format of the extracted full global state depends on the implementation of the state handler.
        The output of this method is given to the output handlers. Since these do not change this information,
        the positions and velocities do not have to be copied.

        Returns
        -------
        Any
            The full global state.
        """
        raise NotImplementedError

    @abstractmethod
    def update_logging(self) -> None:
        """
        Update the logging of this class.

        This method is called in resume.py which might be run with a different logging level compared to the run which
        created the dump. This method must then ensure that this class logs on the correct level.
        """
        raise NotImplementedError
