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
"""Module for the CellBoundingPotential class."""
from copy import copy
import logging
from activator.internal_state.cell_occupancy.cells import Cells
from base.initializer import Initializer
from base.logging import log_init_arguments
from estimator import Estimator
import setting
from .potential import InvertiblePotential


# noinspection PyMethodOverriding
class CellBoundingPotential(InvertiblePotential, Initializer):
    """
    This class implements a cell bounding potential.

    It uses an estimator in order to determine upper bounds for the derivatives for possible separations, when two units
    are located in two different cells. This is determined for all pairs of cells which are not excluded in the cell
    system.
    """

    def __init__(self, estimator: Estimator):
        """
        The constructor of the CellBoundingPotential class.

        Parameters
        ----------
        estimator : estimator.Estimator
            The estimator.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           estimator=estimator.__class__.__name__)
        super().__init__()
        self._estimator = estimator
        self._derivative_bounds = None
        self._bounding_event_rate = None
        self._max_charge_product = None

    def initialize(self, cells: Cells, max_charge: float, calculate_lower_bound: bool) -> None:
        """
        Initialize this class by using the estimator to determine upper bounds for the derivatives for all not excluded
        cell separations.

        Extends the initialize method of the abstract Initializer class. This method is called once in the beginning of
        the run by the activator. Only after a call of this method, other public methods of this class can be called
        without raising an error.
        The estimator is initialized here with the maximum relevant charge in the system. When this class does not
        take charges into account, the estimator only needs to determine upper bounds of the derivative, otherwise
        also lower bounds have to be computed.

        Parameters
        ----------
        cells : activator.internal_state.cell_occupancy.cells.Cells
            The cell system.
        max_charge : float
            The maximum relevant charge.
        calculate_lower_bound : bool
            Whether the estimator should calculate a lower bound for the derivative.
        """
        super().initialize()
        self._max_charge_product = max_charge * max_charge
        self._estimator.initialize(max_charge)
        self._derivative_bounds = ([], []) if calculate_lower_bound else ([],)
        print("Initializing the cell bounding potential in the class {0}.".format(self.__class__.__name__))
        for cell in cells.yield_cells():
            for bound_array in self._derivative_bounds:
                bound_array.append([None for _ in range(setting.dimension)])

            if cell not in cells.excluded_cells(0):
                cell_lower_corner = copy(cells.cell_min(cell))
                cell_upper_corner = copy(cells.cell_max(cell))
                for d in range(setting.dimension):
                    while cell_upper_corner[d] < cell_lower_corner[d]:
                        cell_upper_corner[d] = setting.periodic_boundaries.next_image(cell_upper_corner[d], d)

                ref_cell_lower_corner = copy(cells.cell_min(0))
                ref_cell_upper_corner = copy(cells.cell_max(0))
                for d in range(setting.dimension):
                    while ref_cell_upper_corner[d] < ref_cell_lower_corner[d]:
                        ref_cell_upper_corner[d] = setting.periodic_boundaries.next_image(ref_cell_upper_corner[d], d)

                lower_corner = [cell_lower_corner[d] - ref_cell_upper_corner[d] for d in range(setting.dimension)]
                upper_corner = [cell_upper_corner[d] - ref_cell_lower_corner[d] for d in range(setting.dimension)]

                for direction in range(setting.dimension):
                    bounds = self._estimator.derivative_bound(
                        lower_corner, upper_corner, direction, calculate_lower_bound=calculate_lower_bound)
                    for index, bound in enumerate(bounds):
                        self._derivative_bounds[index][cell][direction] = bound

        if not calculate_lower_bound:
            self._derivative_bounds = self._derivative_bounds[0]
            # noinspection PyAttributeOutsideInit
            self.displacement = self._displacement_without_charges

        print("Finished initialization of the cell bounding potential in the class {0}."
              .format(self.__class__.__name__))

    def derivative(self, direction: int, cell_separation: int, charge_one: float, charge_two: float) -> float:
        """
        Return the derivative of the potential along a direction for a given cell separation.

        This method should only be called after the displacement method, since the derivative is determined there.

        Parameters
        ----------
        direction : int
            The direction of the derivative.
        cell_separation : int
            The cell separation.
        charge_one : float
            The charge c_i.
        charge_two : float
            The charge c_j.

        Returns
        -------
        float
            The derivative.
        """
        return self._bounding_event_rate

    def displacement(self, direction: int, cell_separation: int, charge_one: float, charge_two: float,
                     potential_change: float) -> float:
        """
        Return the displacement of the active unit until the cumulative event rate of the potential equals the given
        potential change.

        This method first computes the bounding event rate by multiplying the relevant bound of the derivative by the
        ratio of the charge product and the maximum charge product used in the initialize method. The relevant bound is
        the upper bound if the charge product is greater than zero and the lower bound otherwise.
        The displacement is then just the potential change divided by the constant bounding event rate.
        This method might return a displacement which places the active unit out of its cell which changes the cell
        separation. However, this is not relevant because a cell boundary event should be triggered before.

        Parameters
        ----------
        direction : int
            The direction of the derivative.
        cell_separation : int
            The cell separation.
        charge_one : float
            The charge c_i.
        charge_two : float
            The charge c_j.
        potential_change : float
         The sampled potential change

        Returns
        -------
        float
            The displacement of the active unit i where the cumulative event rate equals the sampled potential change.
        """
        charge_product = charge_one * charge_two
        if charge_product > 0:
            self._bounding_event_rate = (self._derivative_bounds[0][cell_separation][direction] * charge_product
                                         / self._max_charge_product)
        else:
            self._bounding_event_rate = (self._derivative_bounds[1][cell_separation][direction] * charge_product
                                         / self._max_charge_product)

        return potential_change / self._bounding_event_rate if self._bounding_event_rate > 0 else float('inf')

    def _displacement_without_charges(self, direction: int, cell_separation: int, charge_one: float, charge_two: float,
                                      potential_change: float) -> float:
        """
        Return the displacement of the active unit until the cumulative event rate of the potential equals the given
        potential change when no charges are taken into account.

        This method replaces the displacement method, if this class was initialized without storing lower bounds for
        the derivatives. The charges should then be one.
        The displacement is then just the potential change divided by the constant bounding event rate.
        This method might return a displacement which places the active unit out of its cell which changes the cell
        separation. However, this is not relevant because a cell boundary event should be triggered before.

        Parameters
        ----------
        direction : int
            The direction of the derivative.
        cell_separation : int
            The cell separation.
        charge_one : float
            The charge c_i.
        charge_two : float
            The charge c_j.
        potential_change : float
         The sampled potential change

        Returns
        -------
        float
            The displacement of the active unit i where the cumulative event rate equals the sampled potential change.

        Raises
        ------
        AssertionError
            If the two charges are not one.
        """
        assert charge_one == charge_two == 1.0
        self._bounding_event_rate = self._derivative_bounds[cell_separation][direction]
        return potential_change / self._bounding_event_rate if self._bounding_event_rate > 0 else float('inf')
