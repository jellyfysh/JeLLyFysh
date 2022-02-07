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
""""Module for abstract CellBoundingPotentialEventHandler class."""
from abc import ABCMeta
from typing import Any
from jellyfysh.activator.internal_state.cell_occupancy.cells import PeriodicCells
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.initializer import Initializer
from jellyfysh.potential.cell_bounding_potential import CellBoundingPotential


class CellBoundingPotentialEventHandler(Initializer, metaclass=ABCMeta):
    """
    This event handler uses a cell bounding potential and initializes it properly.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.

    Since in this version of JeLLyFysh, cell separations are only defined for periodic cell systems (i.e., with taking
    periodic boundary conditions into account), a cell bounding potential can only be initialized with an instance of
    the 'PeriodicCells' class. The same restriction thus holds for this event handler (see 'initialize' method).
    """

    def __init__(self, bounding_potential: CellBoundingPotential, **kwargs: Any) -> None:
        """
        The constructor of the CellBoundingPotentialEventHandler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        bounding_potential : potential.cell_bounding_potential.CellBoundingPotential
            The cell bounding potential.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the bounding potential is not an instance of a cell bounding potential.
        """
        super().__init__(**kwargs)
        self._cells = None
        self._active_cell = None
        self._relative_cell = None
        self._bounding_potential = bounding_potential
        if not isinstance(self._bounding_potential, CellBoundingPotential):
            raise ConfigurationError("The event handler {0} can only be used "
                                     "with the class CellBoundingPotential!".format(self.__class__.__name__))

    def initialize(self, cells: PeriodicCells, calculate_lower_bound: bool) -> None:
        """
        Initialize the cell bounding potential.

        This is done by handing the cells to the cell bounding potential, and by telling it whether it needs to
        determine a lower bound on the derivatives for all not excluded cell separations, or not.

        Since in this version of JeLLyFysh, cell separations are only defined for periodic cell systems (i.e., with
        taking periodic boundary conditions into account), a cell bounding potential can only be initialized with an
        instance of the 'PeriodicCells' class. The same restriction thus holds for this event handler.

        Extends the initialize method of the abstract Initializer class. This method is called once in the beginning of
        the run by the activator. Only after a call of this method, other public methods of this class can be called
        without raising an error.

        Parameters
        ----------
        cells : activator.internal_state.cell_occupancy.cells.Cells
            The cell system.
        calculate_lower_bound : bool
            Whether the cell bounding potential needs to compute a lower bound or not.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the cell system is not an instance of a periodic cell system.
        """
        super().initialize()
        if not isinstance(cells, PeriodicCells):
            raise ConfigurationError("The event handler {0} can only be initialized with an instance of the "
                                     "'PeriodicCells' class.".format(self.__class__.__name__))
        self._cells = cells
        self._bounding_potential.initialize(cells, calculate_lower_bound)

    @property
    def bounding_potential(self) -> CellBoundingPotential:
        """
        Return the cell bounding potential.

        Returns
        -------
        potential.cell_bounding_potential.CellBoundingPotential
            The cell bounding potential.
        """
        return self._bounding_potential
