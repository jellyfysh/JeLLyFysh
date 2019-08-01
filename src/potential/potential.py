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
"""Module for the abstract Potential and InvertiblePotential classes."""
from abc import ABCMeta, abstractmethod
import inspect
from typing import Any
from base.exceptions import ConfigurationError


class Potential(metaclass=ABCMeta):
    """
    Abstract class for potentials used in the event handlers.

    A general potential only provides a method, which calculates the derivative along a given direction. The potentials
    in JFV can depend on several separations and charges. Note that, for the case of periodic boundaries, periodicity
    is not taken into account by the potentials but by the event handlers which use these potentials. An exception
    are merged-image potentials.
    """

    def __init__(self, prefactor: float = 1.0, **kwargs: Any) -> None:
        """
        The constructor of the Potential class.

        This constructor tries to initialize the number of separation and charge arguments the derivative method
        gets by inspecting the argument names of this method. For this to work, each separation argument should begin
        with 'separation' and similarly each charge argument should begin with 'charge'. When using a different
        convention in an inhering class, the attributes self._number_separation_arguments and
        self._number_charges_arguments should be set accordingly on initialization.

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
        derivative_signature = inspect.signature(self.derivative)
        number_derivative_arguments = len(derivative_signature.parameters)
        self._number_separation_arguments = len([parameter_name for parameter_name in derivative_signature.parameters
                                                 if "separation" in parameter_name])
        self._number_charges_arguments = len([parameter_name for parameter_name in derivative_signature.parameters
                                              if "charge" in parameter_name])
        if self._number_separation_arguments + self._number_charges_arguments + 1 != number_derivative_arguments:
            # The number of separation and charges arguments is not enough to fill all arguments of derivative method
            self._number_separation_arguments = None
            self._number_charges_arguments = None
        super().__init__(**kwargs)

    @property
    def number_separation_arguments(self) -> int:
        """
        Return the number of separation arguments the derivative method expects.

        For this method to work, either the inspection described in the constructor must have succeeded or
        self._number_separation_arguments must have been set.

        Returns
        -------
        int
            The number of separation arguments.

        Raises
        ------
        NotImplementedError
            If self._number_separation_arguments is None.
        """
        if self._number_separation_arguments is not None:
            return self._number_separation_arguments
        raise NotImplementedError("The number of separation and charge arguments of the potential {0}"
                                  " could not be determined automatically."
                                  " Make sure each separation argument contains the string 'separation' and each "
                                  "charge argument contains the string 'charge'. Otherwise overwrite this method."
                                  .format(self.__class__.__name__))

    @property
    def number_charge_arguments(self):
        """
        Return the number of charge arguments the derivative method expects.

        For this method to work, either the inspection described in the constructor must have succeeded or
        self._number_charges_arguments must have been set.

        Returns
        -------
        int
            The number of charge arguments.

        Raises
        ------
        NotImplementedError
            If self._number_charges_arguments is None.
        """
        if self._number_charges_arguments is not None:
            return self._number_charges_arguments
        raise NotImplementedError("The number of separation and charge arguments of the potential {0}"
                                  " could not be determined automatically."
                                  " Make sure each separation argument contains the string 'separation' and each "
                                  "charge argument contains the string 'charge'. Otherwise overwrite this method."
                                  .format(self.__class__.__name__))

    @abstractmethod
    def derivative(self, direction: int, separations, charges=None) -> float:
        """
        Return the derivative of the potential along a direction for certain separations and charges.

        When overwriting this method, each separation should appear as its own argument. The same is true for the
        charges. For the latter, also no charges is a possibility. For example a potential depending on two
        separations and three charges would define this method as
        'def derivative(self, direction, separation_one, separation_two, charge_one, charge_two, charge_three)'.

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
            The derivative.
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

    @abstractmethod
    def displacement(self, direction: int, separations, charges=None, potential_change: float = None) -> float:
        """
        Return the displacement of the active unit until the cumulative event rate of the potential equals the given
        potential change.

        How a movement of the active unit induces a change of the separation, depends on the used event handler. The
        potential change argument is optional, since it is not necessary for some models (for example hard-sphere
        simulations).

        When overwriting this method, each separation should appear as its own argument. The same is true for the
        charges. For the latter, also no charges is a possibility. For example a potential depending on one
        separations and two charges would define this method as
        'def displacement(self, direction, separation, charge_one, charge_two, potential_change)'.

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
            The displacement of the active unit where the cumulative event rate equals the sampled potential change.
        """
        raise NotImplementedError
