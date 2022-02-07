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
"""Module for the DipoleRandomNodeCreator class."""
import logging
import random
from typing import List, Sequence
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.particle import Particle
from jellyfysh.base.vectors import random_vector_on_unit_sphere
from jellyfysh.input_output_handler.input_handler.charge_values import ChargeValues
import jellyfysh.setting as setting
from .random_node_creator import RandomNodeCreator


class DipoleRandomNodeCreator(RandomNodeCreator):
    """
    A random node creator which creates a single random dipole root node for the random input handler.

    This class is used in the random input handler. It creates a random dipole root node of a tree of height two.
    For the given root node, this class creates a particle which contains the position of the barycenter of the dipole.
    The root node has two children. Each child node contains a particle with the position of the atoms of the dipole.
    The atoms are separated at most by a maximum initial dipole separation, and at least by a minimum dipole separation.
    The wanted charges for the atoms are given by a sequence of ChargeValues objects. These contain every
    charge and a charge name.
    """

    def __init__(self, min_initial_dipole_separation: float = 0.0, max_initial_dipole_separation: float = 0.05,
                 charge_values: Sequence[ChargeValues] = ()) -> None:
        """
        The constructor of the DipoleRandomNodeCreator class.

        Parameters
        ----------
        min_initial_dipole_separation : float, optional
            The minimum initial dipole separation.
        max_initial_dipole_separation : float, optional
            The maximum initial dipole separation.
        charge_values : Sequence[input_output_handler.input_handler.charge_values.ChargeValues], optional
            The sequence of charge values.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           min_initial_dipole_separation=min_initial_dipole_separation,
                           max_initial_dipole_separation=max_initial_dipole_separation,
                           charge_values=[charge_value.__class__.__name__ for charge_value in charge_values])
        super().__init__(charge_values)
        self._min_initial_dipole_separation = min_initial_dipole_separation
        self._max_initial_dipole_separation = max_initial_dipole_separation

    def fill_root_node(self, node) -> None:
        """
        Fill the root node with a random dipole composite object.

        Parameters
        ----------
        node : base.node.Node
            The root node.
        """
        dipole_center = setting.random_position()
        particles = self._create_random_dipole(center=dipole_center)
        for particle in particles:
            node.add_child(Node(particle))
        node.value = Particle(position=dipole_center)

    def _create_random_dipole(self, center: Sequence[float] = None) -> List[Particle]:
        """Create the particles of a dipole molecule at the given center with random positions."""
        if center is None:
            center = setting.random_position()

        random_direction = random_vector_on_unit_sphere(setting.dimension)
        dipole_separation = random.uniform(self._min_initial_dipole_separation / 2.0,
                                           self._max_initial_dipole_separation / 2.0)
        position_one = [center[d] + random_direction[d] * dipole_separation for d in range(setting.dimension)]
        position_two = [center[d] - random_direction[d] * dipole_separation for d in range(setting.dimension)]
        setting.periodic_boundaries.correct_position(position_one)
        setting.periodic_boundaries.correct_position(position_two)
        return [Particle(position_one,
                         {charge_value.charge_name: charge_value[0] for charge_value in self._charge_values}),
                Particle(position_two,
                         {charge_value.charge_name: charge_value[1] for charge_value in self._charge_values})]

    @property
    def number_of_nodes_per_root_node(self) -> int:
        """
        The number of leaf nodes this class fills a root node with.

        The number of nodes per root node is two for this class.

        Returns
        -------
        int
            The number of nodes per root node.
        """
        return 2

    @property
    def number_of_node_levels(self) -> int:
        """
        The number of node levels this class creates in a root node.

        The number of node levels is two for this class.

        Returns
        -------
        int
            The number of node levels.
        """
        return 2
