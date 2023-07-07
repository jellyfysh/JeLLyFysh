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
"""Module for abstract potential classes."""
from abc import ABCMeta, abstractmethod
import inspect
from math import sqrt
from typing import Any, Sequence, Tuple
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base import vectors
from .potential import Potential, InvertiblePotential


class StandardVelocityPotential(Potential, metaclass=ABCMeta):
    """
    Abstract class for potentials that only allow standard velocities (i.e., velocities parallel to one of the cartesian
    coordinate axes going in the positive direction) of the active unit.

    This abstract class implements the derivative method of the abstract Potential class. Here, the direction of motion
    and the speed of the active unit is determined first. Then, the abstract standard_velocity_derivative method is
    called. An inheriting class has to overwrite this method that should return the derivative of the potential along a
    the direction of motion that is parallel to one of the cartesian axes.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        The constructor of the StandardVelocityPotential class.

        The base Potential class defines the number_separation_arguments and number_charge_arguments properties which
        use the corresponding self._number_separation_arguments and self._number_charge_arguments attributes. These
        properties return the number of separation and charge arguments, respectively, that the derivative method
        expects. They are used by event handlers to check whether the potential has the expected form. An inheriting
        class should initialize the private properties to the correct values. If they are not set, this class tries to
        initialize the number of separation and charge arguments of the derivative method. Since this class effectively
        replaces the derivative method by the standard_velocity_derivative method, this class inspects the argument
        names of the standard_velocity_derivative method and not the derivative method (see the overridden
        _inspect_number_separation_and_charge_arguments method). Still, for this to work, each separation argument
        should begin with 'separation' and similarly each charge argument should begin with 'charge'.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)

    def _inspect_number_separation_and_charge_arguments(self):
        """Overrides the method in the base Potential class so that the number of separation and charge arguments is
           inspected using the standard_velocity_derivative method instead of the derivative method."""
        derivative_signature = inspect.signature(self.standard_velocity_derivative)
        number_derivative_arguments = len(derivative_signature.parameters)
        number_separation_arguments = len([parameter_name for parameter_name in derivative_signature.parameters
                                           if "separation" in parameter_name])
        number_charge_arguments = len([parameter_name for parameter_name in derivative_signature.parameters
                                       if "charge" in parameter_name])
        if number_separation_arguments + number_charge_arguments + 1 == number_derivative_arguments:
            self._number_separation_arguments = number_separation_arguments
            self._number_charge_arguments = number_charge_arguments

    def derivative(self, velocity: Sequence[float], *args: Any, **kwargs: Any) -> float:
        """
        Return the directional time derivative along a given velocity vector of the active unit for certain separations
        and charges.

        This method just extracts the direction of motion, which has to be parallel to one of the cartesian axes, out of
        the velocity and then uses the standard_velocity_derivative method. This method returns the space derivative
        along the given cartesian axis. This space derivative is multiplied by the absolute value of the velocity for
        the returned time derivative.

        Note that all further arguments of the this method (like the separations and charges) are simply passed through.

        Parameters
        ----------
        velocity : Sequence[float]
            The velocity of the active unit along which the directional derivative is computed.
        args : Any
            Additional args which are passed to the standard_velocity_derivative method.
        kwargs : Any
            Additional kwargs which are passed to the standard_velocity_derivative method.

        Returns
        -------
        float
            The directional time derivative.

        Raises
        ------
        AssertionError
            If the velocity is not in the positive direction parallel to one of the cartesian axes.
        """
        direction_of_motion, speed = self._analyse_velocity(velocity)
        return self.standard_velocity_derivative(direction_of_motion, *args, **kwargs) * speed

    # noinspection PyMethodMayBeStatic
    def _analyse_velocity(self, velocity: Sequence[float]) -> Tuple[int, float]:
        """
        Return the direction of motion and the speed of the given velocity.

        The velocity must have only a single nonzero component larger than zero (the speed in the direction of motion).

        Parameters
        ----------
        velocity : Sequence[float]
            The velocity.

        Returns
        -------
        (int, float)
            The direction of motion, the speed.

        Raises
        ------
        AssertionError
            If the velocity is not in the positive direction parallel to one of the cartesian axes.

        """
        direction_of_motions = [index for index, component in enumerate(velocity) if component != 0.0]
        assert len(direction_of_motions) == 1
        assert velocity[direction_of_motions[0]] > 0.0
        return direction_of_motions[0], velocity[direction_of_motions[0]]

    @abstractmethod
    def standard_velocity_derivative(self, direction: int, separations, charges=None) -> float:
        """
        Return the space derivative of the potential along a positive direction parallel to one of the cartesian axes
        for the given separations and charges.

        How the separation vectors between the active and the target units are defined should be clearly indicated in
        the inheriting potential so that the event handler which uses this potential is implemented in the right way.

        When overwriting this method, each separation should appear as its own argument. The same is true for the
        charges. For the latter, also no charges is a possibility. For example a potential depending on two
        separations and three charges would define this method as
        'def standard_velocity_derivative(self, direction, separation_one, separation_two, charge_one, charge_two,
        charge_three)'.

        Parameters
        ----------
        direction : int
            The direction of the derivative.
        separations
            All the separations needed to calculate the derivative.
        charges : optional
            All the charges needed to calculate the derivative.

        Returns
        -------
        float
            The space derivative.
        """
        raise NotImplementedError


class StandardVelocityInvertiblePotential(StandardVelocityPotential, InvertiblePotential, metaclass=ABCMeta):
    """
    Abstract class for invertible potentials that only allow standard velocities (i.e., velocities parallel to one of
    the cartesian coordinate axes going in the positive direction) of the active unit.

    This abstract class implements the displacement method of the abstract Potential class. Here, the direction of
    motion and the speed of the active unit is determined first. Then, the abstract standard_velocity_displacement
    method is called. An inheriting class has to overwrite this method that should return the displacement in space of
    the active unit along direction of motion that is parallel to one of the cartesian axes until the cumulative event
    rate of the potential equals the given potential change. The space displacement is then converted into a time
    displacement by using the speed of the active unit.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        The constructor of the StandardVelocityInvertiblePotential class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)

    def _inspect_potential_change_argument(self):
        """Overrides the method in the base InvertiblePotential class so that the standard_velocity_displacement method
           is used instead of the displacement method to inspect whether a potential change argument is required."""
        displacement_signature = inspect.signature(self.standard_velocity_displacement)
        number_displacement_arguments = len(displacement_signature.parameters)
        if self.number_separation_arguments + self.number_charge_arguments + 1 == number_displacement_arguments:
            self._potential_change_required = False
        if self.number_separation_arguments + self.number_charge_arguments + 2 == number_displacement_arguments:
            if "potential_change" in [parameter_name for parameter_name in displacement_signature.parameters]:
                self._potential_change_required = True

    def displacement(self, velocity: Sequence[float], *args, **kwargs) -> float:
        """
        Return the required time displacement of the active unit along its velocity where the cumulative event rate of
        the potential equals the given potential change.

        This method just extracts the direction of motion, which has to be parallel to one of the cartesian axes, out of
        the velocity and then uses the standard_velocity_displacement method. Here, the returned space displacement is
        converted into a time displacement by using the speed of the active unit. Note that all further arguments of the
        this method (like the separations and charges) are simply passed through.

        Parameters
        ----------
        velocity : Sequence[float]
            The velocity of the active unit.
        args : Any
            Additional args which are passed to the standard_velocity_displacement method.
        kwargs : Any
            Additional kwargs which are passed to the standard_velocity_displacement method.

        Returns
        -------
        float
            The required time displacement of the active unit along its velocity where the cumulative event rate equals
            the sampled potential change.

        Raises
        ------
        AssertionError
            If the velocity is not in the positive direction parallel to one of the cartesian axes.
        """
        direction_of_motion, speed = self._analyse_velocity(velocity)
        return self.standard_velocity_displacement(direction_of_motion, *args, **kwargs) / speed

    @abstractmethod
    def standard_velocity_displacement(self, direction: int, separations, charges=None,
                                       potential_change: float = None) -> float:
        """
        Return the required displacement in space of the active unit along the positive direction of motion parallel to
        one of the cartesian axes where the cumulative event rate of the potential equals the given potential change.

        How a movement of the active unit induces a change of the separation should be clearly indicated in the
        inheriting invertible potential so that the event handler which uses this potential is implemented in the right
        way.

        Note that some potentials do not require a potential change because they contain infinite large steps (e.g., the
        hard-sphere potential). Thus, the potential change argument is only optional.

        When overwriting this method, each separation should appear as its own argument. The same is true for the
        charges. For the latter, also no charges is a possibility. For example a potential depending on one
        separations and two charges would define this method as
        'def standard_velocity_displacement(self, direction, separation, charge_one, charge_two, potential_change)'.

        Parameters
        ----------
        direction : int
            The direction of motion of the active unit.
        separations
            All the separations needed to calculate the displacement.
        charges : optional
            All the charges needed to calculate the displacement.
        potential_change : float, optional
            The sampled potential change.

        Returns
        -------
        float
            The required displacement in space of the active unit along its direction of motion where the cumulative
            event rate equals the sampled potential change.
        """
        raise NotImplementedError


# noinspection PyMethodOverriding
class MexicanHatPotential(InvertiblePotential, metaclass=ABCMeta):
    """
    Abstract class for invertible potentials which are shaped like a mexican hat.

    A mexican hat potential depends on on a single separation and is characterized by an equilibrium separation
    |r_ij| = |r_j - r_i| = r_0 where the potential has a minimum. Therefore, the potential is monotonically decreasing
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
        self._minimum_potential = self._potential(self._equilibrium_separation)
        if not self._prefactor > 0.0:
            raise ConfigurationError("The potential {0} can only be used for cases "
                                     "where the prefactor"
                                     " is greater than 0.0!".format(self.__class__.__name__))
        if not self._equilibrium_separation > 0.0:
            raise ConfigurationError("The potential {0} can only be used for an equilibrium separation which is greater"
                                     " than 0.0!".format(self.__class__.__name__))

    def displacement(self, velocity: Sequence[float], separation: Sequence[float], potential_change: float) -> float:
        """
        Return the required time displacement of the active unit along its velocity where the cumulative event rate of
        the potential equals the given potential change.

        This method determines whether the movement of the active unit lets the separation become smaller ('behind') or
        bigger ('front'), and whether the the absolute value of the separation is larger than r_0 ('outside sphere') or
        smaller ('inside sphere'). This is used to call one of the four corresponding methods.

        Parameters
        ----------
        velocity : Sequence[float]
            The velocity of the active unit.
        separation : Sequence[float]
            The separation vector r_ij.
        potential_change : float
            The sampled potential change.

        Returns
        -------
        float
            The required time displacement of the active unit along its velocity where the cumulative event rate equals
            the sampled potential change.

        Raises
        ------
        AssertionError
            If the velocity is zero.
        """
        assert any(entry != 0.0 for entry in velocity)
        velocity_squared = vectors.norm_sq(velocity)
        separation_squared = vectors.norm_sq(separation)
        separation_dot_velocity = vectors.dot(separation, velocity)
        # Active unit is outside of the minimum potential shell.
        if separation_squared >= self._equilibrium_separation_squared:
            # Norm of separation becomes bigger.
            if separation_dot_velocity <= 0.0:
                return self._displacement_front_outside_sphere(
                    velocity_squared, separation_squared, separation_dot_velocity,
                    self._potential(sqrt(separation_squared)), potential_change)
            # Norm of separation becomes smaller.
            else:
                return self._displacement_behind_outside_sphere(
                    velocity_squared, separation_squared, separation_dot_velocity, potential_change)
        # Active unit is inside of the minimum potential shell
        else:
            # Norm of separation becomes bigger.
            if separation_dot_velocity <= 0.0:
                return self._displacement_front_inside_sphere(
                    velocity_squared, separation_squared, separation_dot_velocity, potential_change)
            # Norm of separation becomes smaller.
            else:
                return self._displacement_behind_inside_sphere(
                    velocity_squared, separation_squared, separation_dot_velocity,
                    self._potential(sqrt(separation_squared)), potential_change)

    def _displacement_front_outside_sphere(self, velocity_squared: float, separation_squared: float,
                                           separation_dot_velocity: float, current_potential: float,
                                           potential_change: float) -> float:
        """
        Return the required time displacement of the active unit i along its velocity until the cumulative event rate of
        the potential equals the given potential change, for the case of the active unit in front and outside the
        potential minimum sphere.

        This method determines how far the active unit can climb the potential hill.

        Parameters
        ----------
        velocity_squared : float
            The square of velocity of the active unit i.
        separation_squared : float
            The square of the separation vector r_ij.
        separation_dot_velocity : float
            The dot product of the separation vector r_ij and the velocity vector of the active unit i.
        current_potential : float
            The potential at the separation r_ij.
        potential_change : float
            The sampled potential change.

        Returns
        -------
        float
            The time displacement of the active unit i where the cumulative event rate equals the sampled potential
            change.
        """
        norm_of_new_separation = self._invert_potential_outside_minimum(current_potential + potential_change)
        sqrt_term = (separation_dot_velocity * separation_dot_velocity
                     - velocity_squared * (separation_squared - norm_of_new_separation * norm_of_new_separation))
        return (separation_dot_velocity + sqrt(sqrt_term)) / velocity_squared

    def _displacement_front_inside_sphere(self, velocity_squared: float, separation_squared: float,
                                          separation_dot_velocity: float, potential_change: float) -> float:
        """
        Return the required time displacement of the active unit i along its velocity until the cumulative event rate of
        the potential equals the given potential change, for the case of the active unit in front and inside the
        potential minimum sphere.

        This method places the active unit on the edge of the potential minimum sphere (the derivative of the potential
        is negative until there). Then the method for the case where the active unit is in front and
        outside the sphere is used.

        Parameters
        ----------
        velocity_squared : float
            The square of velocity of the active unit i.
        separation_squared : float
            The square of the separation vector r_ij.
        separation_dot_velocity : float
            The dot product of the separation vector r_ij and the velocity vector of the active unit i.
        potential_change : float
            The sampled potential change.

        Returns
        -------
        float
            The time displacement of the active unit i where the cumulative event rate equals the sampled potential
            change.
        """
        sqrt_term = (separation_dot_velocity * separation_dot_velocity
                     - velocity_squared * (separation_squared - self._equilibrium_separation_squared))
        displacement = (separation_dot_velocity + sqrt(sqrt_term)) / velocity_squared
        return displacement + self._displacement_front_outside_sphere(
            velocity_squared, self._equilibrium_separation_squared,
            separation_dot_velocity - displacement * velocity_squared, self._minimum_potential, potential_change)

    def _displacement_behind_inside_sphere(self, velocity_squared: float, separation_squared: float,
                                           separation_dot_velocity: float, current_potential: float,
                                           potential_change: float) -> float:
        """
        Return the required time displacement of the active unit i along its velocity until the cumulative event rate of
        the potential equals the given potential change, for the case of the active unit behind and inside the potential
        minimum sphere.

        First, this method determines how far the active unit can climb the potential hill inside the sphere. If it
        cannot climb the full hill, the corresponding displacement is returned. If it can, the active unit can be placed
        on the edge of the potential minimum sphere (the derivative of the potential is negative until there). Then the
        method for the case where the active unit is in front and outside the sphere is used.

        Parameters
        ----------
        velocity_squared : float
            The square of velocity of the active unit i.
        separation_squared : float
            The square of the separation vector r_ij.
        separation_dot_velocity : float
            The dot product of the separation vector r_ij and the velocity vector of the active unit i.
        current_potential : float
            The potential at the separation r_ij.
        potential_change : float
            The sampled potential change.

        Returns
        -------
        float
            The time displacement of the active unit i where the cumulative event rate equals the sampled potential
            change.
        """
        max_displacement = separation_dot_velocity / velocity_squared
        minimum_separation_squared = separation_squared - max_displacement * max_displacement * velocity_squared
        maximum_potential_inside = self._potential(sqrt(minimum_separation_squared))
        potential_difference = maximum_potential_inside - current_potential
        # Active unit cannot climb potential hill.
        if potential_change < potential_difference:
            norm_of_new_separation = self._invert_potential_inside_minimum(
                current_potential + potential_change)
            sqrt_term = (separation_dot_velocity * separation_dot_velocity
                         - velocity_squared * (separation_squared - norm_of_new_separation * norm_of_new_separation))
            return (separation_dot_velocity - sqrt(sqrt_term)) / velocity_squared
        # Active unit can climb potential hill.
        else:
            return max_displacement + self._displacement_front_inside_sphere(
                velocity_squared, minimum_separation_squared,
                separation_dot_velocity - max_displacement * velocity_squared, potential_change - potential_difference)

    def _displacement_behind_outside_sphere(self, velocity_squared: float, separation_squared: float,
                                            separation_dot_velocity: float, potential_change: float) -> float:
        """
        Return the required time displacement of the active unit i along its velocity until the cumulative event rate of
        the potential equals the given potential change, for the case of the active unit behind and outside the
        potential minimum sphere.

        This method checks if the active unit can reach the potential minimum sphere. If so, the active unit can be
        placed on the edge (the derivative of the potential is negative until there). Then the method corresponding
        to the case where the active unit is behind and inside the sphere is used.
        If the active unit cannot reach the potential minimum sphere, the derivative is negative until the smallest
        possible separation is reached. Then the method for the case where the active unit is in front and
        outside the sphere is used.

        Parameters
        ----------
        velocity_squared : float
            The square of velocity of the active unit i.
        separation_squared : float
            The square of the separation vector r_ij.
        separation_dot_velocity : float
            The dot product of the separation vector r_ij and the velocity vector of the active unit i.
        potential_change : float
            The sampled potential change.

        Returns
        -------
        float
            The time displacement of the active unit i where the cumulative event rate equals the sampled potential
            change.
        """
        sqrt_term = (separation_dot_velocity * separation_dot_velocity
                     - velocity_squared * (separation_squared - self._equilibrium_separation_squared))
        # Active unit can reach potential minimum sphere
        if sqrt_term >= 0.0:
            displacement = (separation_dot_velocity - sqrt(sqrt_term)) / velocity_squared
            return displacement + self._displacement_behind_inside_sphere(
                velocity_squared, self._equilibrium_separation_squared,
                separation_dot_velocity - displacement * velocity_squared, self._minimum_potential, potential_change)
        # Active unit cannot reach potential minimum sphere.
        else:
            displacement = separation_dot_velocity / velocity_squared
            minimum_separation_squared = separation_squared - displacement * displacement * velocity_squared
            minimum_potential = self._potential(sqrt(minimum_separation_squared))
            return displacement + self._displacement_front_outside_sphere(
                velocity_squared, minimum_separation_squared, separation_dot_velocity - displacement * velocity_squared,
                minimum_potential, potential_change)

    @abstractmethod
    def _potential(self, norm_of_separation: float) -> float:
        """
        Return the potential for the given absolute value of the separation.

        Parameters
        ----------
        norm_of_separation : float
            The absolute value |r_ij| of the separation vector.

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
