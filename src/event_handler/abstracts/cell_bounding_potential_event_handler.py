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
""""Module for abstract CellBoundingPotentialEventHandler class."""
from abc import ABCMeta
from typing import Any, Sequence, Union
from activator.internal_state.cell_occupancy.cells import Cells
from base.exceptions import ConfigurationError
from base.initializer import Initializer
import base.node as node
from potential.cell_bounding_potential import CellBoundingPotential


class CellBoundingPotentialEventHandler(Initializer, metaclass=ABCMeta):
    """
    This event handler uses a cell bounding potential and initializes it properly.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
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
        self._check_bounding_potential()

    def initialize(self, cells: Cells, extracted_global_state: Sequence[node.Node], charge: Union[str, None]) -> None:
        """
        Initialize the cell bounding potential.

        This is done by handing the cells to the potential and also by extracting the maximum charge from the extracted
        global state. If the charge is None, the potential is initialized without a charge.
        Extends the initialize method of the abstract Initializer class. This method is called once in the beginning of
        the run by the activator. Only after a call of this method, other public methods of this class can be called
        without raising an error.

        Parameters
        ----------
        cells : activator.internal_state.cell_occupancy.cells.Cells
            The cell system.
        extracted_global_state : Sequence[base.node.Node]
            The extracted global state of the tree state handler.
        charge : str or None
            The relevant charge for this event handler.
        """
        super().initialize()
        self._cells = cells
        max_charge = (max(abs(leaf_node.value.charge[charge])
                          for root_cnode in extracted_global_state
                          for leaf_node in node.yield_leaf_nodes(root_cnode))
                      if charge is not None else 1.0)
        self._bounding_potential.initialize(cells, max_charge, charge is not None)

    def initialize_with_initialized_potential(self, cells: Cells, initialized_potential: CellBoundingPotential) -> None:
        """
        Initialize this event handler by replacing the potential with an initialized cell bounding potential.

        Adds a second initialize method relevant to the abstract Initializer class. This method is called once in the
        beginning of the run by the activator. Only after a call of this method, other public methods of this class can
        be called without raising an error.

        Parameters
        ----------
        cells : activator.internal_state.cell_occupancy.cells.cells.Cells
            The cell system.
        initialized_potential : potential.cell_bounding_potential.CellBoundingPotential
            The initialized cell bounding potential.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the bounding potential is not an instance of a cell bounding potential.
        """
        super().initialize()
        self._cells = cells
        self._bounding_potential = initialized_potential
        self._check_bounding_potential()

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

    def _check_bounding_potential(self) -> None:
        """Check if the bounding potential is an instance of a cell bounding potential and raise an exception if not."""
        if not isinstance(self._bounding_potential, CellBoundingPotential):
            raise ConfigurationError("The event handler {0} can only be used "
                                     "with the class CellBoundingPotential!".format(self.__class__.__name__))
