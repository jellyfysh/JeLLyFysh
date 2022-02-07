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
"""Module for the BondLengthAndAngleOutputHandler class."""
import logging
from typing import Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base import vectors
import jellyfysh.setting as setting
from .output_handler import OutputHandler, HardBufferedTextWriter


class BondLengthAndAngleOutputHandler(OutputHandler):
    """
    Output handler which samples the bond length and the bond angle of water molecules.

    This output handler should receive the extracted global state in its write method. It is designed to work together
    with the TreeStateHandler. Here, the extracted global state is given by a sequence of trees. Each tree is specified
    by a root node, which themselves are connected to children nodes. Each node contains a Unit object.
    This output handler allows for a variable number of root nodes, each with three children. The second child
    should correspond to the oxygen and the first and third children to the hydrogens.
    The bond lengths are the separations between the hydrogens and the oxygen. The bond angle is the angle spanned by
    the three atoms.
    The file for the bond lengths will include '_Length' in the filename. Similarly, the file for the bond angles will
    include '_Angle' in the filename.
    """

    def __init__(self, filename: str):
        """
        The constructor of the BondLengthAndAngleOutputHandler class.

        This class uses a HardBufferedTextWriter to first write the bond lengths and angles to temporary files.

        Parameters
        ----------
        filename : str
            The filename of the file this output handler is connected to.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the number of node levels is not two or the number of nodes per root node is not three.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, filename=filename)
        super().__init__(filename)
        filename_dot_position = self._output_filename.rfind('.')
        bond_length_filename = (self._output_filename[:filename_dot_position]
                                + '_Length' + self._output_filename[filename_dot_position:])
        bond_angle_filename = (self._output_filename[:filename_dot_position]
                               + '_Angle' + self._output_filename[filename_dot_position:])
        self._file_bond_lengths = HardBufferedTextWriter(bond_length_filename)
        self._file_bond_angles = HardBufferedTextWriter(bond_angle_filename)
        if setting.number_of_node_levels != 2 or setting.number_of_nodes_per_root_node != 3:
            raise ConfigurationError("The output handler {0} can only be used if each root node has 3 child nodes."
                                     .format(self.__class__.__name__))

    def write(self, extracted_global_state: Sequence[Node]) -> None:
        """
        Extract the bond lengths and bond angles of all water molecules and write them to the temporary files.

        Parameters
        ----------
        extracted_global_state : Sequence[base.node.Node]
            The extracted global state.

        Raises
        ------
        AssertionError
            If not each root node contains three children.
        """
        super().write(extracted_global_state)
        for root_cnode in extracted_global_state:
            assert len(root_cnode.children) == 3
            hydrogen_one_position = root_cnode.children[0].value.position
            oxygen_position = root_cnode.children[1].value.position
            hydrogen_two_position = root_cnode.children[2].value.position
            vector_oh_one = setting.periodic_boundaries.separation_vector(oxygen_position, hydrogen_one_position)
            vector_oh_two = setting.periodic_boundaries.separation_vector(oxygen_position, hydrogen_two_position)
            print(vectors.norm(vector_oh_one), file=self._file_bond_lengths)
            print(vectors.norm(vector_oh_two), file=self._file_bond_lengths)
            print(vectors.angle_between_two_vectors(vector_oh_one, vector_oh_two), file=self._file_bond_angles)

    def post_run(self):
        """Clean up the output handler."""
        self._file_bond_lengths.close()
        self._file_bond_angles.close()
