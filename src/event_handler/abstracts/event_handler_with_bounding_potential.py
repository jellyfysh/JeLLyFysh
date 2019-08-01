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
"""Modules for useful abstract event handler base classes with bounding potentials."""
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import random
from typing import Any, Sequence
from base.exceptions import ConfigurationError
from base.unit import Unit
from event_handler.helper_functions import bounding_potential_warning
from lifting import Lifting
from potential import Potential
import setting
from .abstracts import SingleActiveLeafUnitEventHandler
from .composite_objects import CompositeObjectsEventHandler


class EventHandlerWithBoundingPotential(SingleActiveLeafUnitEventHandler, metaclass=ABCMeta):
    """
    This event handler base class assumes a bounding event rate and deals with an interaction between a single active
    and a single target leaf unit via a general potential expecting a single separation.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    It relies on the fact, that the bounding event rate is determined in the inheriting class. Then the out-state can
    be determined by first confirming the event. If it is confirmed, the active and target leaf unit just exchange their
    velocities.
    """

    def __init__(self, potential: Potential, **kwargs: Any):
        """
        The constructor of the EventHandlerWithBoundingPotential class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        potential : potential.Potential
            The potential between the active and target leaf unit.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the potential expects more than one separation.
        """
        super().__init__(**kwargs)
        self._bounding_event_rate = None
        self._potential = potential
        if self._potential.number_separation_arguments != 1:
            raise ConfigurationError("The event handler {0} expects a potential "
                                     "which handles exactly one separation!".format(self.__class__.__name__))

    def _calculate_out_state_of_two_leaf_unit_bounding_potential(self, separation: Sequence[float],
                                                                 potential_charges: Sequence[float]) -> None:
        """
        Calculate the out-state for the given separation and charges of the active and target leaf units.

        This method uses the ratio of the potential event rate and the _bounding_event_rate attribute to confirm the
        event. If the event is confirmed, the velocities of the active and target leaf unit are exchanged.

        Parameters
        ----------
        separation : Sequence[float]
            The separation between the active and target leaf unit.
        potential_charges : Sequence[float]
            The charges of the active and target leaf units.

        Raises
        ------
        AssertionError
            If the _leaf_units attribute contains more than two leaf units.
        """
        assert len(self._leaf_units) == 2
        real_derivative = self._potential.derivative(self._direction_of_motion, separation, *potential_charges)
        if real_derivative > 0:
            bounding_potential_warning(self.__class__.__name__, self._bounding_event_rate, real_derivative)
            if random.uniform(0, self._bounding_event_rate) < real_derivative:
                self._exchange_velocity(self._leaf_cnodes[self._active_leaf_unit_index],
                                        self._leaf_cnodes[self._active_leaf_unit_index ^ 1])


class TwoCompositeObjectBoundingPotentialEventHandler(EventHandlerWithBoundingPotential, CompositeObjectsEventHandler,
                                                      metaclass=ABCMeta):
    """
    This event handler base class assumes a bounding event rate and deals with a interaction between leaf units int two
    composite objects where a single leaf unit is active via a general potential expecting a single separation.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    All units in the different composite objects interact pairwise. This class can include a charge. The values of these
    are passed to the potential.
    In order to determine the out-state, this class relies on a lifting scheme.
    """

    def __init__(self, potential: Potential, lifting: Lifting, charge: str = None, **kwargs: Any) -> None:
        """
        The constructor of the TwoCompositeObjectBoundingPotentialEventHandler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        potential : potential.Potential
            The potential between all pairs of units in different composite objects.
        lifting : lifting.Lifting
            The lifting scheme.
        charge : str or None, optional
            The charge this event handler passes to the potential. If None, the potential just gets one as the charges.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the number of nodes per root node is one and therefore no composite objects are present in the run.
        base.exceptions.ConfigurationError
            If the potential expects more than one separation.
        base.exceptions.ConfigurationError
            If the charge is not None but the potential expects more than two charges.
        """
        super().__init__(potential=potential, **kwargs)
        self._lifting = lifting
        if not setting.number_of_nodes_per_root_node > 1:
            raise ConfigurationError("Class {0} can only be used when composite point objects are present!"
                                     .format(self.__class__.__name__))
        if self._potential.number_separation_arguments != 1:
            raise ConfigurationError("The event handler {0} expects a potential "
                                     "which handles exactly one separation!".format(self.__class__.__name__))
        if charge is None:
            self._potential_charges = (lambda unit_one, unit_two:
                                       tuple(1.0 for _ in range(self._potential.number_charge_arguments)))
        else:
            if self._potential.number_charge_arguments == 2:
                self._potential_charges = lambda unit_one, unit_two: (unit_one.charge[charge], unit_two.charge[charge])
            else:
                raise ConfigurationError("The event handler {0} was initialized with a charge which is not None,"
                                         " but its potential {1} "
                                         "expects not exactly 2 charges."
                                         .format(self.__class__.__name__, self._potential.__class__.__name__))

    @staticmethod
    def _not_same_composite_object(identifier_one: Sequence[int], identifier_two: Sequence[int]) -> bool:
        """Return if the two identifiers belong to the same composite object or not."""
        return identifier_one[0] != identifier_two[0]

    def _fill_lifting(self, local_units: Sequence[Unit], target_units: Sequence[Unit],
                      active_leaf_unit_derivative: float,
                      target_composite_object_factor_derivatives: Sequence[float]) -> None:
        """
        Fill the lifting scheme.

        Some of the needed derivatives are arguments of this method, since they have been usually computed to confirm
        the event in the first place. This is the sum of the derivatives between the active leaf unit with all
        units in the target composite object. Therefore the derivative between a leaf unit in the target composite
        object and the active leaf unit has also been computed.

        Parameters
        ----------
        local_units : Sequence[base.unit.Unit]
            Units within the same composite object as the active leaf unit.
        target_units : Sequence[base.unit.Unit]
            Units within the target composite object.
        active_leaf_unit_derivative : float
            The summed derivative between the active leaf unit and all leaf units in the target composite particle.
        target_composite_object_factor_derivatives : Sequence[float]
            The sequence of derivatives between each unit in the target composite object and the active leaf unit.
        """
        self._lifting.reset()
        local_composite_object_factor_derivatives = [0.0] * len(local_units)

        for index_1, local_unit in enumerate(local_units):
            if local_unit is self._active_leaf_unit:
                local_composite_object_factor_derivatives[index_1] = active_leaf_unit_derivative
                continue
            for index_2, target_unit in enumerate(target_units):
                pairwise_derivative = self._potential.derivative(
                    self._direction_of_motion,
                    setting.periodic_boundaries.separation_vector(local_unit.position, target_unit.position),
                    *self._potential_charges(local_unit, target_unit))
                local_composite_object_factor_derivatives[index_1] += pairwise_derivative
                target_composite_object_factor_derivatives[index_2] -= pairwise_derivative

        if local_units[0].identifier[0] < target_units[0].identifier[0]:
            for index_1, local_unit in enumerate(local_units):
                self._lifting.insert(local_composite_object_factor_derivatives[index_1],
                                     local_unit.identifier, local_unit is self._active_leaf_unit)
            for index_2, target_unit in enumerate(target_units):
                self._lifting.insert(target_composite_object_factor_derivatives[index_2],
                                     target_unit.identifier, False)
        else:
            for index_2, target_unit in enumerate(target_units):
                self._lifting.insert(target_composite_object_factor_derivatives[index_2],
                                     target_unit.identifier, False)
            for index_1, local_unit in enumerate(local_units):
                self._lifting.insert(local_composite_object_factor_derivatives[index_1],
                                     local_unit.identifier, local_unit is self._active_leaf_unit)


class EventHandlerWithPiecewiseConstantBoundingPotential(SingleActiveLeafUnitEventHandler, metaclass=ABCMeta):
    """
    Abstract event handler base class for a piecewise constant bounding potential.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    This class dynamically constructs such a bounding potential by comparing two event rates:
    1. The event rate with the active unit at the present location.
    2. The event rate with the active unit displaced by some value.
    The maximum of these event rates plus an offset then yields the bounding event rate. By tuning the offset and
    the displacement of the active unit, one can balance between the confirmation rate of events and the number
    of events where the real event rate was in fact larger than the bounding event rate.
    This class assumes that there is a single active leaf unit. To be open for the maximum number of potentials,
    the inheriting class should provide methods which get the relevant separations and charges from all leaf units for
    the potential it uses.
    """

    def __init__(self, potential: Potential, offset: float, max_displacement: float, **kwargs: Any):
        """
        The constructor of the EventHandlerWithPiecewiseConstantBoundingPotential class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        potential : potential.Potential
            The potential between the leaf units.
        offset : float
            The offset.
        max_displacement : float
            The maximum displacement by which the active unit is displaced to determine the bounding event rate.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the maximum displacement is not larger than zero.
        """
        super().__init__(**kwargs)
        self._potential = potential
        self._offset = offset
        if not max_displacement > 0.0:
            raise ConfigurationError("Please use a value for max_displacement > 0.0 in the class {0}."
                                     .format(self.__class__.__name__))
        self._max_displacement = max_displacement
        self._bounding_event_rate = None

    def _displacement_from_piecewise_constant_bounding_potential(self, potential_change: float) -> float:
        """
        Calculate the displacement based on the constant bounding event rate.

        The constant bounding event rate is determined dynamically. The ratio of the sampled potential change and this
        bounding event rate then yields the displacement.

        Parameters
        ----------
        potential_change : float
            The sampled potential change.

        Returns
        -------
        float:
            The displacement of the active unit i where the cumulative event rate equals the sampled potential change.
        """
        separations = self._get_separations(self._leaf_units)
        charges = self._get_charges(self._leaf_units)
        derivatives = self._potential.derivative(self._direction_of_motion, *separations, *charges)
        try:
            # noinspection PyUnresolvedReferences
            # Some potentials return several derivatives (for example bending potential)
            derivative_one = derivatives[self._active_leaf_unit_index]
        except TypeError:
            derivative_one = derivatives

        advanced_active_unit = deepcopy(self._leaf_units[self._active_leaf_unit_index])
        advanced_active_unit.position[self._direction_of_motion] += self._max_displacement
        advanced_active_unit.position[self._direction_of_motion] = (
            setting.periodic_boundaries.correct_position_entry(
                advanced_active_unit.position[self._direction_of_motion], self._direction_of_motion))

        advanced_units = self._leaf_units.copy()
        advanced_units[self._active_leaf_unit_index] = advanced_active_unit
        separations = self._get_separations(advanced_units)
        derivatives = self._potential.derivative(self._direction_of_motion, *separations, *charges)
        try:
            # noinspection PyUnresolvedReferences
            # Some potentials return several derivatives (for example bending potential)
            derivative_two = derivatives[self._active_leaf_unit_index]
        except TypeError:
            derivative_two = derivatives

        constant_derivative = max(derivative_one, derivative_two) + self._offset
        if constant_derivative <= 0.0:
            self._bounding_event_rate = None
            return self._max_displacement
        elif potential_change / constant_derivative < self._max_displacement:
            self._bounding_event_rate = constant_derivative
            return potential_change / constant_derivative
        else:
            self._bounding_event_rate = None
            return self._max_displacement

    def _event_rate_from_piecewise_constant_bounding_potential(self) -> float:
        """
        Return the bounding event rate.

        This method should only be called after the _displacement_from_piecewise_constant_bounding_potential method,
        since the bounding event rate is determined there.

        Returns
        -------
        float
            The bounding event rate.
        """

        return self._bounding_event_rate

    @abstractmethod
    def _get_separations(self, units: Sequence[Unit]) -> Sequence[Sequence[float]]:
        """
        Return the sequence of separations relevant for the potential.

        Parameters
        ----------
        units : Sequence[base.unit.Unit]
            The sequence of leaf units.

        Returns
        -------
        Sequence[Sequence[float]]
            The sequence of separations.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_charges(self, units: Sequence[Unit]) -> Sequence[float]:
        """
        Return the sequence of charges relevant for the potential.

        Parameters
        ----------
        units : Sequence[base.unit.Unit]
            The sequence of leaf units.

        Returns
        -------
        Sequence[float]
            The sequence of charges.
        """
        raise NotImplementedError
