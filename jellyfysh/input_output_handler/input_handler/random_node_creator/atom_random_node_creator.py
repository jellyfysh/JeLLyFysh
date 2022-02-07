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
"""Module for the AtomRandomNodeCreator class."""
import logging
from typing import Sequence
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.particle import Particle
from jellyfysh.input_output_handler.input_handler.charge_values import ChargeValues
import jellyfysh.setting as setting
from .random_node_creator import RandomNodeCreator


class AtomRandomNodeCreator(RandomNodeCreator):
    """
    A random node creator which creates a single random atom root node for the random input handler.

    This class is used in the random input handler. It creates a random atom root node of a tree of height one.
    For the given root node, this class creates a particle which contains a position and a charge. The root node has no
    children. The wanted charges for the particle are given by a sequence of ChargeValues objects. These contain every
    charge and a charge name.
    """

    def __init__(self, charge_values: Sequence[ChargeValues] = ()) -> None:
        """
        The constructor of the AtomRandomNodeCreator class.

        Parameters
        ----------
        charge_values : Sequence[input_output_handler.input_handler.charge_values.ChargeValues], optional
            The sequence of charge values.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           charge_values=[charge_value.__class__.__name__ for charge_value in charge_values])
        super().__init__(charge_values)

    def fill_root_node(self, node: Node) -> None:
        """
        Fill the root node with a random point mass.

        Parameters
        ----------
        node : base.node.Node
            The root node.
        """
        node.value = Particle(setting.random_position(),
                              {charge_value.charge_name: charge_value[0]
                                  for charge_value in self._charge_values})

    @property
    def number_of_nodes_per_root_node(self) -> int:
        """
        The number of leaf nodes this class fills a root node with.

        The number of nodes per root node is one for this class.

        Returns
        -------
        int
            The number of nodes per root node.
        """
        return 1

    @property
    def number_of_node_levels(self) -> int:
        """
        The number of node levels this class creates in a root node.

        The number of node levels is one for this class.

        Returns
        -------
        int
            The number of node levels.
        """
        return 1
