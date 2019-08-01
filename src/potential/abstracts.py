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
"""Module for the abstract MexicanHatPotential class."""
from abc import ABCMeta, abstractmethod
from typing import Any, MutableSequence, Sequence
from base.exceptions import ConfigurationError
from base import vectors
from .potential import InvertiblePotential


# noinspection PyMethodOverriding
class MexicanHatPotential(InvertiblePotential, metaclass=ABCMeta):
    """
    Abstract class for invertible potentials which are shaped like a mexican hat.

    A mexican hat potential depends on a single separation and is characterized by an absolute value separation
    |r_ij| = |r_j - r_i| = r_0 where the potential has a minimum. Therefore the potential is monotonically decreasing
    within the range [0, r_0] for the absolute values of the separation and monotonically increasing for absolute values
    of the separation which are larger than r_0. To enforce that, the general multiplicative prefactor should be greater
    than zero. This class assumes that i is the active unit.

    An inheriting class must be able to compute the potential itself and to invert the potential inside and outside the
    range [0, r_0].
    """

    def __init__(self, prefactor: float, equilibrium_separation: float, **kwargs: Any) -> None:
        """
        The constructor of the MexicanHatPotential class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        prefactor : float
            A general multiplicative prefactor of the potential.
        equilibrium_separation : float
            The absolute value of the separation r_0 where the potential has a minimum.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the general multiplicative prefactor is smaller than or equal to zero.
        base.exceptions.ConfigurationError
            If the equilibrium separation is smaller than or equal to zero.
        """
        super().__init__(prefactor=prefactor, **kwargs)
        self._equilibrium_separation = equilibrium_separation
        self._equilibrium_separation_squared = equilibrium_separation ** 2
        if not self._prefactor > 0.0:
            raise ConfigurationError("The potential {0} can only be used for cases "
                                     "where the prefactor"
                                     " is greater than 0.0!".format(self.__class__.__name__))
        if not self._equilibrium_separation > 0.0:
            raise ConfigurationError("The potential {0} can only be used for an equilibrium separation which is greater"
                                     " than 0.0!".format(self.__class__.__name__))

    def displacement(self, direction: int, separation: MutableSequence[float], potential_change: float) -> float:
        """
        Return the displacement of the active unit i until the cumulative event rate of the potential equals the given
        potential change.

        This method determines whether the active unit is behind or in front of the target unit, and whether the
        the absolute value of the separation is larger than r_0 ('outside sphere') or smaller ('inside sphere').
        This is used to call one of the four corresponding methods.

        Parameters
        ----------
        direction : int
            The direction of motion of the active unit i.
        separation : MutableSequence[float]
            The separation vector r_ij.
        potential_change : float
            The sampled potential change.

        Returns
        -------
        float
            The displacement of the active unit i where the cumulative event rate equals the sampled potential change.
        """
        norm_of_separation = vectors.norm(separation)
        # Active unit is outside of the minimum potential shell
        if norm_of_separation > self._equilibrium_separation:
            # Active unit is in front of target unit
            if separation[direction] < 0.0:
                current_potential = self._potential(separation)
                displacement = self._displacement_front_outside_sphere(
                    direction, current_potential, potential_change, separation)
            # Active unit is behind of target unit
            else:
                displacement = self._displacement_behind_outside_sphere(direction, potential_change, separation)
        # Active unit is inside of the minimum potential shell
        else:
            # Active unit is in front of target unit
            if separation[direction] < 0.0:
                displacement = self._displacement_front_inside_sphere(direction, potential_change, separation)
            # Active unit is behind of target unit
            else:
                current_potential = self._potential(separation)
                displacement = self._displacement_behind_inside_sphere(
                    direction, current_potential, potential_change, separation)
        return displacement

    def _displacement_front_outside_sphere(self, direction: int, current_potential: float, potential_change: float,
                                           separation: Sequence[float]) -> float:
        """
        Return the displacement of the active unit i until the cumulative event rate of the potential equals the given
        potential change, for the case of the active unit in front and outside the potential minimum sphere.

        This method determines how far the active unit can climb the potential.

        Parameters
        ----------
        direction : int
            The direction of motion of the active unit i.
        current_potential : float
            The potential at the separation r_ij.
        potential_change : float
            The sampled potential change.
        separation : MutableSequence[float]
            The separation vector r_ij.

        Returns
        -------
        float
            The displacement of the active unit i where the cumulative event rate equals the sampled potential change.
        """
        norm_of_new_separation = self._invert_potential_outside_minimum(current_potential + potential_change)
        return vectors.displacement_until_new_norm_sq_component_negative(
            separation, norm_of_new_separation * norm_of_new_separation, direction)

    def _displacement_behind_outside_sphere(self, direction: int, potential_change: float,
                                            separation: MutableSequence[float]) -> float:
        """
        Return the displacement of the active unit i until the cumulative event rate of the potential equals the given
        potential change, for the case of the active unit behind and outside the potential minimum sphere.

        This method checks if the active unit can reach the potential minimum sphere. If so, the active unit can be
        placed on the edge (the derivative of the potential is negative until there). Then the method corresponding
        to the case where the active unit is behind and inside the sphere is used.
        If the active unit cannot reach the potential minimum sphere, the derivative is negative until the active unit
        is at the same level as the target unit. Then the method for the case where the active unit is in front and
        outside the sphere is used.

        Parameters
        ----------
        direction : int
            The direction of motion of the active unit i.
        potential_change : float
            The sampled potential change.
        separation : MutableSequence[float]
            The separation vector r_ij.

        Returns
        -------
        float
            The displacement of the active unit i where the cumulative event rate equals the sampled potential change.
        """
        # Active unit can reach potential minimum sphere
        try:
            displacement = vectors.displacement_until_new_norm_sq_component_positive(
                separation, self._equilibrium_separation_squared, direction)
            separation[direction] -= displacement
            current_potential = self._potential(separation)
            displacement += self._displacement_behind_inside_sphere(
                direction, current_potential, potential_change, separation)
        # Active unit cannot reach potential minimum sphere
        except ValueError:
            displacement = separation[direction]
            separation[direction] = 0.0
            current_potential = self._potential(separation)
            displacement += self._displacement_front_outside_sphere(
                direction, current_potential, potential_change, separation)
        return displacement

    def _displacement_front_inside_sphere(self, direction: int, potential_change: float,
                                          separation: Sequence[float]) -> float:
        """
        Return the displacement of the active unit i until the cumulative event rate of the potential equals the given
        potential change, for the case of the active unit behind and outside the potential minimum sphere.

        This method places the active unit on the edge of the potential minimum sphere (the derivative of the potential
        is negative until there). Then the method for the case where the active unit is in front and
        outside the sphere is used.

        Parameters
        ----------
        direction : int
            The direction of motion of the active unit i.
        potential_change : float
            The sampled potential change.
        separation : MutableSequence[float]
            The separation vector r_ij.

        Returns
        -------
        float
            The displacement of the active unit i where the cumulative event rate equals the sampled potential change.
        """
        displacement = vectors.displacement_until_new_norm_sq_component_negative(
            separation, self._equilibrium_separation_squared, direction)
        separation[direction] -= displacement
        # Use the sampled potential change to travel uphill
        current_potential = self._potential(separation)
        displacement += self._displacement_front_outside_sphere(
            direction, current_potential, potential_change, separation)
        return displacement

    def _displacement_behind_inside_sphere(self, direction: int, current_potential: float, potential_change: float,
                                           separation: MutableSequence[float]) -> float:
        """
        Return the displacement of the active unit i until the cumulative event rate of the potential equals the given
        potential change, for the case of the active unit in front and outside the potential minimum sphere.

        First, this method determines how far the active unit can climb the potential hill inside the sphere. If it
        cannot climb the full hill, the displacement is returned. If it can, the active unit can be placed on the edge
        of the potential minimum sphere (the derivative of the potential is negative until there). Then the method for
        the case where the active unit is in front and outside the sphere is used.

        Parameters
        ----------
        direction : int
            The direction of motion of the active unit i.
        current_potential : float
            The potential at the separation r_ij.
        potential_change : float
            The sampled potential change.
        separation : MutableSequence[float]
            The separation vector r_ij.

        Returns
        -------
        float
            The displacement of the active unit i where the cumulative event rate equals the sampled potential change.
        """
        separation_at_maximum_inside = vectors.copy_vector_with_replaced_component(separation, direction, 0.0)
        maximum_potential_inside = self._potential(separation_at_maximum_inside)
        potential_difference = maximum_potential_inside - current_potential
        # Active unit cannot climb potential hill
        if potential_change < potential_difference:
            norm_of_new_separation = self._invert_potential_inside_minimum(
                current_potential + potential_change)
            displacement = vectors.displacement_until_new_norm_sq_component_positive(
                separation, norm_of_new_separation * norm_of_new_separation, direction)
        else:
            displacement = separation[direction]
            separation[direction] = 0.0
            potential_change -= potential_difference
            displacement += self._displacement_front_inside_sphere(direction, potential_change, separation)
        return displacement

    @abstractmethod
    def _potential(self, separation: Sequence[float]) -> float:
        """
        Return the potential for the given separation.

        Parameters
        ----------
        separation : Sequence[float]
            The separation vector r_ij.

        Returns
        -------
        float
            The potential.
        """
        raise NotImplementedError

    @abstractmethod
    def _invert_potential_inside_minimum(self, potential: float) -> float:
        """
        Return the absolute value of the separation r where the potential equals the given potential with r <= r_0.

        Parameters
        ----------
        potential : float
            The wanted potential.

        Returns
        -------
        float:
            The absolute value of the separation.
        """
        raise NotImplementedError

    @abstractmethod
    def _invert_potential_outside_minimum(self, potential: float) -> float:
        """
        Return the absolute value of the separation r where the potential equals the given potential with r >= r_0.

        Parameters
        ----------
        potential : float
            The wanted potential.

        Returns
        -------
        float:
            The absolute value of the separation.
        """
        raise NotImplementedError
