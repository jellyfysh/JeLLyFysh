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
"""Module for the PdbInputHandler class."""
import logging
from typing import List, Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base.particle import Particle
from jellyfysh.input_output_handler.mdanalysis_import import Universe
import jellyfysh.setting as setting
from jellyfysh.setting import hypercuboid_setting
from .charge_values import ChargeValues
from .input_handler import InputHandler


class PdbInputHandler(InputHandler):
    """
    Input handler which reads an initial global physical state for the tree state handler from a .pdb file.

    This class is designed to work together with the TreeStateHandler. Here, the global physical state is given
    by a sequence of trees. Each tree is specified by a root node, which themselves are connected to children nodes.
    Each node contains a Particle object, which stores the position and the charge.
    This input handler allows for a variable number of root nodes with the same number of children. Each tree is of
    height at most two. If the height is two, the root nodes correspond to composite objects and the child nodes to
    point masses. If the height is one, the root nodes correspond to point masses.
    Only point masses can have a charge. The wanted charges are given by a sequence of ChargeValues objects. These
    contain every charge and a charge name.
    To read the .pdb file, this class uses the MDAnalysis package.
    Note that the positions of the point masses in the .pdb file are corrected for periodic boundaries in order to allow
    for different conventions of the origin of the system box (e.g., [0.0]*setting.dimension, which is used in this
    application, vs. [-L/2]*setting.dimension, which is used in .pdb files created by Lammps). The positions of the
    composite point objects are obtained by averaging the positions of its point masses. Here, the positions are
    corrected for periodic boundaries so that the separation vectors between the point masses are the smallest.
    Also, this input handler can only be used when the hypercuboid setting is initialized and in at most three
    dimensions. It will check whether the system lengths in the hypercuboid setting module are the same as the ones in
    the .pdb file.
    """

    def __init__(self, filename: str, charge_values: Sequence[ChargeValues] = ()) -> None:
        """
        The constructor of the PdbInputHandler class.

        The constructor sets the number of root nodes, the number of nodes per root node and the number of node levels
        in the setting package.

        Parameters
        ----------
        filename : str
            The filename of the .pdb file which contains the initial global physical state.
        charge_values : Sequence[input_output_handler.input_handler.charge_values.ChargeValues], optional
            The sequence of charge values.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the filename does not end with .pdb.
        base.exceptions.ConfigurationError
            If the setting package specifies a dimension larger than 3 which cannot be initialized with a .pdb file.
        base.exceptions.ConfigurationError
            If the setting package specifies a dimension smaller than 3 and any of the superfluous system lengths in the
            .pdb file is not 0.0.
        base.exceptions.ConfigurationError
            If the .pdb file specifies composite objects with different numbers of point masses.
        base.exceptions.ConfigurationError
            If the .pdb specifies other system lengths as the ones stored in the setting.hypercuboid_setting module.
        base.exceptions.ConfigurationError
            If the charge values does not contain a charge for each child node within a root node.
        base.exceptions.ConfigurationError:
            If in the sequence of charge values a charge name appears more than once.
        """
        logger = logging.getLogger(__name__)
        log_init_arguments(logger.debug, self.__class__.__name__,
                           filename=filename,
                           charge_values=[charge_value.__class__.__name__ for charge_value in charge_values])
        super().__init__()
        if not filename.endswith(".pdb"):
            raise ConfigurationError("Input filename for input handler {0} should end with .pdb."
                                     .format(self.__class__.__name__))
        if setting.dimension < 3:
            logger.warning("The PDB format only allows for 3 dimensions but the simulation is done in {0} dimensions. "
                           "This class will check if the positions in not used dimensions are 0.0."
                           .format(setting.dimension))
        if setting.dimension > 3:
            raise ConfigurationError("The PDB format does not allow for dimensions > 3.")

        self._filename = filename
        self._universe = Universe(self._filename)
        self._charge_values = charge_values
        if not all(len(residue.atoms) == len(self._universe.residues[0].atoms) for residue in self._universe.residues):
            raise ConfigurationError("Currently only point composite objects with the same number of children are "
                                     "supported!")
        setting.set_number_of_root_nodes(len(self._universe.residues))
        setting.set_number_of_nodes_per_root_node(len(self._universe.residues[0].atoms))
        setting.set_number_of_node_levels(1 if len(self._universe.residues[0].atoms) == 1 else 2)

        system_lengths = self._universe.dimensions[:setting.dimension]
        for index, system_length in enumerate(system_lengths):
            if abs(system_length - hypercuboid_setting.system_lengths[index]) > 1.0e-5:
                raise ConfigurationError("The system lengths {0} stored in the .pdb file {1} are not equal to the "
                                         "system lengths {2} stored in the setting.hypercuboid_setting module."
                                         .format(system_lengths, self._filename, hypercuboid_setting.system_lengths))
        unused_system_lengths = self._universe.dimensions[setting.dimension:3]
        if any(unused_system_length != 0.0 for unused_system_length in unused_system_lengths):
            raise ConfigurationError("The setting.hypercuboid_setting module specifies the dimension {0}<3 but the"
                                     " superfluous system lengths {1} stored in the .pdb file {2} are not 0.0."
                                     .format(setting.dimension, unused_system_lengths, self._filename))

        # Test that per charge value there is a charge given for every point mass
        for charge_value in self._charge_values:
            if len(charge_value) != setting.number_of_nodes_per_root_node:
                raise ConfigurationError("Please give the charge {0} for every child of a composite point object!"
                                         .format(charge_value.charge_name))
        # Test that each charge value has a different name
        if len(set(charge_value.charge_name for charge_value in self._charge_values)) != len(self._charge_values):
            raise ConfigurationError("Each given charge should have a different name!")

    def read(self) -> List[Node]:
        """
        Return the initial global physical state.

        This method creates the root nodes of the trees and fills them with the atom positions given in the .pdb file.
        The charges of the atoms are initialized with the sequence of charge values. The barycenter positions of the
        composite objects are calculated using the atom positions.

        Returns
        -------
        List[base.node.Node]
            The initial global physical state.

        Raises
        ------
        AssertionError
            If the setting package specifies a dimension smaller than 3 and any of the superfluous position entries in
            the .pdb file is not 0.0.
        """
        # self._universe is set to None at the end of this method. If this method is used again, recreate it.
        if self._universe is None:
            self._universe = Universe(self._filename)

        all_nodes = [Node() for _ in range(setting.number_of_root_nodes)]
        for atom in self._universe.atoms:
            # Indices in pdb files start at 1
            atom_index = (atom.id - 1) % setting.number_of_nodes_per_root_node
            # MDAnalysis uses numpy floats -> Cast to float
            position = [float(atom.position[index]) for index in range(setting.dimension)]
            assert all(float(unused_position) == 0.0 for unused_position in atom.position[setting.dimension:3])
            setting.periodic_boundaries.correct_position(position)
            particle = Particle(position=position, charge={charge_value.charge_name: charge_value[atom_index]
                                                           for charge_value in self._charge_values})
            if setting.number_of_node_levels > 1:
                all_nodes[atom.resid - 1].add_child(Node(particle))
            else:
                all_nodes[atom.resid - 1].value = particle
        if setting.number_of_node_levels > 1:
            for node in all_nodes:
                first_position = node.children[0].value.position
                center_position = [entry * node.children[0].weight for entry in first_position]
                for other_point_mass_index in range(1, len(node.children)):
                    other_position = node.children[other_point_mass_index].value.position
                    shortest_separation = setting.periodic_boundaries.separation_vector(first_position, other_position)
                    closest_position = [entry + shortest_separation[index]
                                        for index, entry in enumerate(first_position)]
                    center_position = [entry + closest_position[index] * node.children[other_point_mass_index].weight
                                       for index, entry in enumerate(center_position)]
                setting.periodic_boundaries.correct_position(center_position)
                node.value = Particle(position=center_position)

        # Remove large universe object from storage, and allow pickling of this class (because the Universe class of
        # MDAnalysis cannot be pickled).
        self._universe = None
        return all_nodes
