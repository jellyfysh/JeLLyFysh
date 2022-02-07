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
"""Module for the abstract Potential and InvertiblePotential classes."""
from abc import ABCMeta, abstractmethod
import inspect
from typing import Any, Sequence
from jellyfysh.base.exceptions import ConfigurationError


class Potential(metaclass=ABCMeta):
    """
    Abstract class for potentials used in the event handlers.

    A general potential only provides a method, which calculates the directional derivative along a given velocity
    vector of the active unit. The potentials in JFV can depend on several separations (between the active and target
    units) and charges. Note that, for the case of periodic boundaries, periodicity is not taken into account by the
    potentials but by the event handlers which use these potentials. An exception are merged-image potentials.
    """

    def __init__(self, prefactor: float = 1.0, **kwargs: Any) -> None:
        """
        The constructor of the Potential class.

        This class defines the number_separation_arguments and number_charge_arguments properties which use the
        corresponding self._number_separation_arguments and self._number_charge_arguments attributes. These properties
        return the number of separation and charge arguments, respectively, that the derivative method expects. They
        are used by event handlers to check whether the potential has the expected form. An inheriting class should
        initialize the private properties to the correct values. If they are not set, this class tries to initialize the
        number of separation and charge arguments of the derivative method by inspecting the argument names of this
        method (see _inspect_number_separation_and_charge_arguments method). For this to work, each separation argument
        should begin with 'separation' and similarly each charge argument should begin with 'charge'.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        prefactor : float, optional
            A general multiplicative prefactor of the potential.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the prefactor equals 0.
        """
        if prefactor == 0.0:
            raise ConfigurationError("Give a prefactor unequal 0.0 as the prefactor for the potential {0}."
                                     .format(self.__class__.__name__))
        self._prefactor = prefactor
        self._number_separation_arguments = None
        self._number_charge_arguments = None
        super().__init__(**kwargs)

    @property
    def number_separation_arguments(self) -> int:
        """
        Return the number of separation arguments the derivative method expects.

        For this method to work, the self._number_separation_arguments attribute must have been set.
        If both self._number_separation_arguments and self._number_charge_arguments (used for the
        number_charge_arguments property) attributes were not set, this method tries to inspect the number of separation
        and charge arguments of the derivative method by inspecting the argument names. For this to work, each
        separation argument should begin with 'separation' and similarly each charge argument should begin with
        'charge'.

        Returns
        -------
        int
            The number of separation arguments.

        Raises
        ------
        NotImplementedError
            If self._number_separation_arguments was not initialized and could also not be inspected.
        """
        if self._number_separation_arguments is None and self._number_charge_arguments is None:
            self._inspect_number_separation_and_charge_arguments()
        if self._number_separation_arguments is None:
            raise NotImplementedError("The number of separation and charge arguments of the potential {0} "
                                      "were not set and could not be determined automatically. Make sure each "
                                      "separation argument contains the string 'separation' and each charge "
                                      "argument contains the string 'charge'. Otherwise, set the "
                                      "self._number_separation_arguments and self._number_charge_arguments attributes "
                                      "yourself.".format(self.__class__.__name__))
        return self._number_separation_arguments

    @property
    def number_charge_arguments(self):
        """
        Return the number of charge arguments the derivative method expects.

        For this method to work, the self._number_separation_arguments attribute must have been set.
        If both self._number_separation_arguments and self._number_charge_arguments (used for the
        number_charge_arguments property) attributes were not set, this method tries to inspect the number of separation
        and charge arguments of the derivative method by inspecting the argument names. For this to work, each
        separation argument should begin with 'separation' and similarly each charge argument should begin with
        'charge'.

        Returns
        -------
        int
            The number of charge arguments.

        Raises
        ------
        NotImplementedError
            If self._number_charge_arguments was not initialized and could also not be inspected.
        """
        if self._number_separation_arguments is None and self._number_charge_arguments is None:
            self._inspect_number_separation_and_charge_arguments()
        if self._number_charge_arguments is None:
            raise NotImplementedError("The number of separation and charge arguments of the potential {0} "
                                      "were not set and could not be determined automatically. Make sure each "
                                      "separation argument contains the string 'separation' and each charge "
                                      "argument contains the string 'charge'. Otherwise, set the "
                                      "self._number_separation_arguments and self._number_charge_arguments attributes "
                                      "yourself.".format(self.__class__.__name__))
        return self._number_charge_arguments

    def _inspect_number_separation_and_charge_arguments(self):
        """Try to inspect the number of separation and charge arguments of the derivative method."""
        derivative_signature = inspect.signature(self.derivative)
        number_derivative_arguments = len(derivative_signature.parameters)
        number_separation_arguments = len([parameter_name for parameter_name in derivative_signature.parameters
                                           if "separation" in parameter_name])
        number_charge_arguments = len([parameter_name for parameter_name in derivative_signature.parameters
                                       if "charge" in parameter_name])
        if number_separation_arguments + number_charge_arguments + 1 == number_derivative_arguments:
            self._number_separation_arguments = number_separation_arguments
            self._number_charge_arguments = number_charge_arguments

    @abstractmethod
    def derivative(self, velocity: Sequence[float], separations, charges=None) -> float:
        """
        Return the directional time derivative along a given velocity vector of the active unit for the given
        separations and charges.

        How the separation vectors between the active and the target units are defined should be clearly indicated in
        the inheriting potential so that the event handler which uses this potential is implemented in the right way.

        When overwriting this method, each separation should appear as its own argument. The same is true for the
        charges. For the latter, also no charges is a possibility. For example a potential depending on two
        separations and three charges would define this method as
        'def derivative(self, velocity, separation_one, separation_two, charge_one, charge_two, charge_three)'.

        Parameters
        ----------
        velocity : Sequence[float]
            The velocity of the active unit along which the directional derivative is computed.
        separations
            All the separations needed to calculate the derivative.
        charges : optional
            All the charges needed to calculate the derivative.

        Returns
        -------
        float
            The directional time derivative.
        """
        raise NotImplementedError


class InvertiblePotential(Potential, metaclass=ABCMeta):
    """
    Abstract class for invertible potentials used in the event handlers.

    An invertible potential is a potential whose event rate can be integrated in closed form along a straight-line
    trajectory. Based on a sampled wanted cumulative event rate, the needed displacement of the active unit can be
    provided.
    """

    def __init__(self, prefactor: float = 1.0, **kwargs: Any) -> None:
        """
        The constructor of the InvertiblePotential class.

        This class defines the potential_change_required property that uses the corresponding
        self._potential_change_required attribute. This property determines whether the displacement method expects
        a potential change argument, and is used by event handlers to check whether a potential change is sampled or
        not. If the self._potential_change_required attribute is not set, this class tries to initialize it by
        inspecting the arguments of the displacement method (see _inspect_potential_change_argument method). See the
        docstring of the potential_change_required property for requirements of the inspection.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        prefactor : float, optional
            A general multiplicative prefactor of the potential.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(prefactor=prefactor, **kwargs)
        self._potential_change_required = None

    @abstractmethod
    def displacement(self, velocity: Sequence[float], separations, charges=None,
                     potential_change: float = None) -> float:
        """
        Return the required time displacement of the active unit along its velocity where the cumulative event rate of
        the potential equals the given potential change.

        How a movement of the active unit induces a change of the separation should be clearly indicated in the
        inheriting invertible potential so that the event handler which uses this potential is implemented in the right
        way.

        Note that some potentials do not require a potential change because they contain infinite large steps (e.g., the
        hard-sphere potential). Thus, the potential change argument is only optional.

        When overwriting this method, each separation should appear as its own argument. The same is true for the
        charges. For the latter, also no charges is a possibility. For example a potential depending on one
        separation, two charges, and a potential change would define this method as
        'def displacement(self, velocity, separation, charge_one, charge_two, potential_change)'.

        Parameters
        ----------
        velocity : Sequence[float]
            The velocity of the active unit.
        separations
            All the separations needed to calculate the displacement.
        charges : optional
            All the charges needed to calculate the displacement.
        potential_change : float, optional
            The sampled potential change.

        Returns
        -------
        float
            The required time displacement of the active unit along its velocity where the cumulative event rate equals
            the sampled potential change.
        """
        raise NotImplementedError

    @property
    def potential_change_required(self) -> bool:
        """
        Return whether the displacement method requires a potential change.

        For this method to work, the self._potential_change_required attribute must have been set. If this attribute has
        not been set, this method tries to inspect the argument names of the 'displacement' method. Here, it compares
        the real number of arguments to the expected number of arguments. The minimum number of expected arguments are
        one velocity argument, self.number_separation_arguments separation arguments, and self.number_charge_arguments
        charge arguments. If the real number of arguments is equal to this minimum number of expected arguments, it is
        assumed that no potential change is required. If the real number of arguments is equal to the minimum number
        of expected arguments plus one, it is checked whether one of the arguments has the name 'potential_change'. Only
        then it is assumed that a potential change is required. In every other case, the inspection fails and the
        self._potential_change_required should be set elsewhere.

        Returns
        -------
        bool
            Whether a potential change argument is required.

        Raises
        ------
        NotImplementedError
            If self._potential_change_required was not initialized and could also not be inspected.
        """
        if self._potential_change_required is None:
            self._inspect_potential_change_argument()
        if self._potential_change_required is None:
            raise NotImplementedError("In the potential {0}, it was not set whether a potential change is required. "
                                      "Moreover, this could not be determined automatically. If no potential change is "
                                      "required, make sure that the number of arguments of the 'displacement' method "
                                      "is equal to the number of separation arguments plus the number of charge "
                                      "arguments plus one. If a potential change is required, only a single additional "
                                      "argument named 'potential_change' should appear. Otherwise, set the "
                                      "self._potential_change_required attribute yourself."
                                      .format(self.__class__.__name__))
        return self._potential_change_required

    def _inspect_potential_change_argument(self):
        """Try to inspect the potential change argument of the displacement method."""
        displacement_signature = inspect.signature(self.displacement)
        number_displacement_arguments = len(displacement_signature.parameters)
        if self.number_separation_arguments + self.number_charge_arguments + 1 == number_displacement_arguments:
            self._potential_change_required = False
        if self.number_separation_arguments + self.number_charge_arguments + 2 == number_displacement_arguments:
            if "potential_change" in [parameter_name for parameter_name in displacement_signature.parameters]:
                self._potential_change_required = True
