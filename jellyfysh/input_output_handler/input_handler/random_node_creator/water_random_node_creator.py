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
"""Module for the WaterRandomNodeCreator class."""
import logging
import math
from typing import List, Sequence
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.particle import Particle
from jellyfysh.base import vectors
from jellyfysh.input_output_handler.input_handler.charge_values import ChargeValues
import jellyfysh.setting as setting
from .random_node_creator import RandomNodeCreator


class WaterRandomNodeCreator(RandomNodeCreator):
    """
    A random node creator which creates a single random water root node for the random input handler.

    This class is used in the random input handler. It creates a random water root node of a tree of height two.
    For the given root node, this class creates a particle which contains the position of the barycenter of the water
    molecule. The root node has three children. Each child node contains a particle with the position of the atoms of
    the water molecule. The first and second, as well as the second and third atoms are separated by a bond length.
    The angle spanned by the atoms is given by a bond angle.
    The wanted charges for the atoms are given by a sequence of ChargeValues objects. These contain every
    charge and a charge name.
    """

    def __init__(self, bond_length: float = 1.012, bond_angle: float = 1.9764,
                 charge_values: Sequence[ChargeValues] = ()) -> None:
        """
        The constructor of the WaterRandomNodeCreator class.

        Parameters
        ----------
        bond_length : float, optional
            The bond length.
        bond_angle : float, optional
            The bond angle.
        charge_values : Sequence[input_output_handler.input_handler.charge_values.ChargeValues], optional
            The sequence of charge values.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           bond_length=bond_length, bond_angle=bond_angle,
                           charge_values=[charge_value.__class__.__name__ for charge_value in charge_values])
        super().__init__(charge_values)
        self._bond_length = bond_length
        self._bond_angle = bond_angle

    def fill_root_node(self, node: Node) -> None:
        """
        Fill the root node with a random water composite object.

        Parameters
        ----------
        node : base.node.Node
            The root node.
        """
        molecule_center = setting.random_position()
        particles = self._create_random_water_molecule(molecule_center)
        for particle in particles:
            node.add_child(Node(particle))
        node.value = Particle(position=molecule_center)

    def _create_random_water_molecule(self, center: Sequence[float]) -> List[Particle]:
        """Create the particles of a water molecule at the given center with random positions."""
        molecule_orientation = vectors.random_vector_on_unit_sphere(setting.dimension)
        random_vector = vectors.random_vector_on_unit_sphere(setting.dimension)
        weight = vectors.dot(molecule_orientation, random_vector)
        random_vector = [random_vector[d] - weight * molecule_orientation[d] for d in range(setting.dimension)]
        random_vector = vectors.normalize(random_vector, math.sin(self._bond_angle / 2))
        molecule_orientation = vectors.normalize(molecule_orientation, math.cos(self._bond_angle / 2))
        oh_vector_one = [molecule_orientation[d] + random_vector[d] for d in range(setting.dimension)]
        oh_vector_two = [molecule_orientation[d] - random_vector[d] for d in range(setting.dimension)]

        oh_vector_one = vectors.normalize(oh_vector_one, self._bond_length)
        oh_vector_two = vectors.normalize(oh_vector_two, self._bond_length)

        oxygen_position, hydrogen_position_one, hydrogen_position_two = [], [], []
        for d in range(setting.dimension):
            current_center_component = (oh_vector_one[d] + oh_vector_two[d]) / 3
            oxygen_position.append(center[d] - current_center_component)
            hydrogen_position_one.append(oxygen_position[d] + oh_vector_one[d])
            hydrogen_position_two.append(oxygen_position[d] + oh_vector_two[d])

        setting.periodic_boundaries.correct_position(oxygen_position)
        setting.periodic_boundaries.correct_position(hydrogen_position_one)
        setting.periodic_boundaries.correct_position(hydrogen_position_two)

        return [Particle(hydrogen_position_one,
                         {charge_value.charge_name: charge_value[0] for charge_value in self._charge_values}),
                Particle(oxygen_position,
                         {charge_value.charge_name: charge_value[1] for charge_value in self._charge_values}),
                Particle(hydrogen_position_two,
                         {charge_value.charge_name: charge_value[2] for charge_value in self._charge_values})]

    @property
    def number_of_nodes_per_root_node(self) -> int:
        """
        The number of leaf nodes this class fills a root node with.

        The number of nodes per root node is three for this class.

        Returns
        -------
        int
            The number of nodes per root node.
        """
        return 3

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
