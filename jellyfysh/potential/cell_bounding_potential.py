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
"""Module for the CellBoundingPotential class."""
import logging
from typing import Tuple, Union
from jellyfysh.activator.internal_state.cell_occupancy.cells import PeriodicCells
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.initializer import Initializer
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.estimator import Estimator
import jellyfysh.setting as setting
from .abstracts import StandardVelocityInvertiblePotential


# noinspection PyMethodOverriding
class CellBoundingPotential(StandardVelocityInvertiblePotential, Initializer):
    """
    This class implements a cell bounding potential.

    This potential only allows for standard velocities (i.e., velocities parallel to one of the cartesian coordinate
    axes going in the positive direction) of the active unit.

    It uses an estimator in order to determine upper and lower bounds for the derivatives for possible cell separations.
    Since in this version of JeLLyFysh, cell separations are only defined for periodic cell systems (i.e., with taking
    periodic boundary conditions into account), this class can only be initialized with an instance of the
    'PeriodicCells' class (see 'initialize' method). The upper and lower bounds for the derivatives are determined for
    all cell separations which are not excluded in the cell system.

    This bounding potential can be used with any number of point masses involved in the factor (e.g., pair factors
    involving two point masses or molecular factors involving all point masses of two composite point objects), and for
    any number of active point masses. The used estimator must just be suited for the factor treated in the event
    handler that uses this cell bounding potential.
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

    def initialize(self, cells: PeriodicCells, calculate_lower_bound: bool) -> None:
        """
        Initialize this class by using the estimator to determine upper bounds for the derivatives for all not excluded
        cell separations.

        Since in this version of JeLLyFysh, cell separations are only defined for periodic cell systems (i.e., with
        taking periodic boundary conditions into account), this class can only be initialized with an instance of the
        'PeriodicCells' class.

        The lower bounds of the estimator are only relevant if the event handler that uses this potential considers
        charges.

        Extends the initialize method of the abstract Initializer class. This method is called once in the beginning of
        the run by the activator. Only after a call of this method, other public methods of this class can be called
        without raising an error.

        Parameters
        ----------
        cells : activator.internal_state.cell_occupancy.cells.Cells
            The cell system.
        calculate_lower_bound : bool
            Whether the lower bounds for the derivative returned by estimator should be stored.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the cell system is not an instance of a periodic cell system.
        """
        super().initialize()
        if not isinstance(cells, PeriodicCells):
            raise ConfigurationError("The potential {0} can only be initialized with an instance of the "
                                     "'PeriodicCells' class.".format(self.__class__.__name__))
        self._derivative_bounds = ({}, {}) if calculate_lower_bound else ({},)
        print("Initializing the cell bounding potential in the class {0}.".format(self.__class__.__name__))
        for cell in cells.yield_cells():
            if cell not in cells.nearby_cells(cells.zero_cell):
                cell_separation = cells.relative_cell(cell, cells.zero_cell)
                for bound_array in self._derivative_bounds:
                    bound_array[cell_separation] = [None for _ in range(setting.dimension)]
                lower_corner = [cell.cell_min[direction] - cells.zero_cell.cell_max[direction]
                                for direction in range(setting.dimension)]
                upper_corner = [cell.cell_max[direction] - cells.zero_cell.cell_min[direction]
                                for direction in range(setting.dimension)]

                for direction in range(setting.dimension):
                    bounds = self._estimator.derivative_bound(
                        lower_corner, upper_corner, direction, calculate_lower_bound=calculate_lower_bound)
                    for index, bound in enumerate(bounds):
                        self._derivative_bounds[index][cell_separation][direction] = bound

        if not calculate_lower_bound:
            self._derivative_bounds = self._derivative_bounds[0]
            # noinspection PyAttributeOutsideInit
            self.standard_velocity_displacement = self._standard_velocity_displacement_without_charges

        print("Finished initialization of the cell bounding potential in the class {0}."
              .format(self.__class__.__name__))

    def standard_velocity_derivative(self, direction: int, cell_separation: int,
                                     active_charges: Union[float, Tuple[float]],
                                     target_charges: Union[float, Tuple[float]]) -> float:
        """
        Return the space derivative of the potential along a positive direction parallel to one of the cartesian axes
        for the given cell separation and charges of the active and the target point masses.

        This method should only be called after the displacement method, since the derivative is determined there.

        Parameters
        ----------
        direction : int
            The direction of the derivative.
        cell_separation : int
            The cell separation.
        active_charges : float or Tuple[float]
            The charges of the active point masses.
        target_charges : float or Tuple[float]
            The charges of the target point masses.

        Returns
        -------
        float
            The space derivative.
        """
        return self._bounding_event_rate

    def standard_velocity_displacement(self, direction: int, cell_separation: int,
                                       active_charges: Union[float, Tuple[float]],
                                       target_charges: Union[float, Tuple[float]], potential_change: float) -> float:
        """
        Return the required displacement in space of the active unit along the positive direction of motion parallel to
        one of the cartesian axes where the cumulative event rate of the potential equals the given potential change.

        This method first computes the bounding event rate by using the stored derivative bounds of the estimator, and
        the multiplicative correction factor returned by the estimator based on the charges of the active point masses
        and the target point masses. The relevant bound is the upper bound if the correction factor is greater than zero
        and the lower bound otherwise.
        The displacement is then just the potential change divided by the constant bounding event rate.
        This method might return a displacement which places the active unit out of its cell which changes the cell
        separation. However, this is not relevant because a cell boundary event should be triggered before.

        If there is only a single active and/or target point mass, the single charge should be given as a float.
        Otherwise the charges should be given as tuples.

        Parameters
        ----------
        direction : int
            The direction of motion of the active unit.
        cell_separation : int
            The cell separation.
        active_charges : float or Tuple[float]
            The charges of the active point masses.
        target_charges : float or Tuple[float]
            The charges of the target point masses.
        potential_change : float
         The sampled potential change

        Returns
        -------
        float
            The required displacement in space of the active unit along its direction of motion where the cumulative
            event rate equals the sampled potential change.
        """
        charge_product = self._estimator.charge_correction_factor(active_charges, target_charges)
        if charge_product > 0.0:
            self._bounding_event_rate = self._derivative_bounds[0][cell_separation][direction] * charge_product
        else:
            self._bounding_event_rate = self._derivative_bounds[1][cell_separation][direction] * charge_product

        return potential_change / self._bounding_event_rate if self._bounding_event_rate > 0 else float('inf')

    def _standard_velocity_displacement_without_charges(self, direction: int, cell_separation: int, charge_one: float,
                                                        charge_two: float, potential_change: float) -> float:
        """
        Return the required displacement in space of the active unit along the positive direction of motion parallel to
        one of the cartesian axes where the cumulative event rate of the potential equals the given potential change
        when no charges are taken into account.

        This method replaces the standard_velocity_displacement method, if this class was initialized without storing
        lower bounds for the derivatives. The charges should then be one.
        The displacement is then just the potential change divided by the constant bounding event rate.
        This method might return a displacement which places the active unit out of its cell which changes the cell
        separation. However, this is not relevant because a cell boundary event should be triggered before.

        Parameters
        ----------
        direction : int
            The direction of motion of the active unit.
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
            The required displacement in space of the active unit along its direction of motion where the cumulative
            event rate equals the sampled potential change.

        Raises
        ------
        AssertionError
            If the two charges are not one.
        """
        assert charge_one == charge_two == 1.0
        self._bounding_event_rate = self._derivative_bounds[cell_separation][direction]
        return potential_change / self._bounding_event_rate if self._bounding_event_rate > 0 else float('inf')
