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
"""Module for the RandomNodeCreator class."""
from abc import ABCMeta, abstractmethod
from typing import Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.input_output_handler.input_handler.charge_values import ChargeValues


class RandomNodeCreator(metaclass=ABCMeta):
    """
    A random node creator which creates a single random root node for the random input handler.

    This class is used in the random input handler. It creates a random root node of a tree of height at most two.
    For each node, this class creates a particle which contains a position and a charge. The wanted charges for
    particles on leaf nodes are given by a sequence of ChargeValues objects. These contain every charge for every leaf
    node within a root node and a charge name.
    """

    def __init__(self, charge_values: Sequence[ChargeValues] = ()) -> None:
        """
        The constructor of the RandomNodeCreator class.

        Parameters
        ----------
        charge_values : Sequence[input_output_handler.input_handler.charge_values.ChargeValues], optional
            The sequence of charge values.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the charge values does not contain a charge for each child node within a root node.
        base.exceptions.ConfigurationError:
            If in the sequence of charge values a charge name appears more than once.
        """
        # Check that per charge value there is a charge given for every particle
        for charge_value in charge_values:
            if not len(charge_value) == self.number_of_nodes_per_root_node:
                raise ConfigurationError("Charge {0} is not given for every child of a composite point object."
                                         .format(charge_value.charge_name))
        # Check that each charge value has a different name
        if not len(set(charge_value.charge_name for charge_value in charge_values)) == len(charge_values):
            raise ConfigurationError("Please choose a unique charge name for all the charges used.")
        self._charge_values = charge_values

    @abstractmethod
    def fill_root_node(self, node) -> None:
        """
        Fill the root node with a random composite object or point mass.

        Parameters
        ----------
        node : base.node.Node
            The root node.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def number_of_nodes_per_root_node(self) -> int:
        """
        The number of leaf nodes this class fills a root node with.

        Returns
        -------
        int
            The number of nodes per root node.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def number_of_node_levels(self) -> int:
        """
        The number of node levels this class creates in a root node.

        Returns
        -------
        int
            The number of node levels.
        """
        raise NotImplementedError
