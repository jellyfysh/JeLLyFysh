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
"""Module for the ChargeValues class."""
import logging
from typing import Sequence
from jellyfysh.base.factory import get_alias
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.strings import to_snake_case


class ChargeValues(object):
    """
    Class which stores charge values and a charge name.

    This class is used in the RandomNodeCreator and the PdbInputHandler class. Both classes regard a tree specified by
    a root node. The root node can have children. This class specifies the charge values for all leaf node particles
    within a single tree.
    """

    def __init__(self, charge_values: Sequence[float], charge_name: str = None) -> None:
        """
        The constructor of the ChargeValues class.

        If the charge name is None, the name of the class is used. If the JF factory included an alias in the classname,
        the alias is used.

        Parameters
        ----------
        charge_values : Sequence[float]
            The sequence of charges for each leaf node particle within a single tree.
        charge_name : str or None, optional
            The name of the charge.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, charge_values=charge_values,
                           charge_name=charge_name)
        # If no charge name is given, we want to use the alias of the .ini file which is included in the
        # __class__.__name__ property in the factory. get_alias() extracts this alias
        self._charge_name = (charge_name if charge_name is not None
                             else to_snake_case(get_alias(self.__class__.__name__)))
        self._charge_values = charge_values

    @property
    def charge_name(self) -> str:
        """
        Return the charge name.

        Returns
        -------
        str
            The charge name.
        """
        return self._charge_name

    def __len__(self) -> int:
        """
        Return the number of particle charges stored in this class.

        Returns
        -------
        int
            The number of particle charges.
        """
        return len(self._charge_values)

    def __getitem__(self, item: int) -> float:
        """
        Return the charge of the given leaf node index.

        Parameters
        ----------
        item : int
            The particle index

        Returns
        -------
        float
            The charge.
        """
        return self._charge_values[item]
