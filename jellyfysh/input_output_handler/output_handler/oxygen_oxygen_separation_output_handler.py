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
"""Module for the OxygenOxygenSeparationOutputHandler class."""
import logging
from typing import Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node
from jellyfysh.base import vectors
import jellyfysh.setting as setting
from .output_handler import OutputHandler, HardBufferedTextWriter


class OxygenOxygenSeparationOutputHandler(OutputHandler):
    """
    Output handler which samples the shortest separations of the oxygens of water molecules.

    This output handler should receive the extracted global state in its write method. It is designed to work together
    with the TreeStateHandler. Here, the extracted global state is given by a sequence of trees. Each tree is specified
    by a root node, which themselves are connected to children nodes. Each node contains a Unit object.
    This output handler allows for a variable number of root nodes, each with three children. The second child
    should correspond to the oxygen.
    """

    def __init__(self, filename: str) -> None:
        """
        The constructor of the OxygenOxygenSeparationOutputHandler class.

        This class uses a HardBufferedTextWriter to first write the separations to a temporary file.

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
        self._file = HardBufferedTextWriter(filename)
        if setting.number_of_node_levels != 2 or setting.number_of_nodes_per_root_node != 3:
            raise ConfigurationError("The output handler {0} can only be used if each root node has 3 child nodes."
                                     .format(self.__class__.__name__))

    def write(self, extracted_global_state: Sequence[Node]) -> None:
        """
        Extract the shortest separations of the second child of each root node and write it to the temporary file.

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
        assert all(len(root_cnode.children) == 3 for root_cnode in extracted_global_state)
        oxygen_positions = [root_cnode.children[1].value.position for root_cnode in extracted_global_state]
        for first_index, first_oxygen_position in enumerate(oxygen_positions):
            for second_index in range(first_index + 1, len(oxygen_positions)):
                oxygen_separation = setting.periodic_boundaries.separation_vector(first_oxygen_position,
                                                                                  oxygen_positions[second_index])
                print(vectors.norm(oxygen_separation), file=self._file)

    def post_run(self) -> None:
        """Clean up the output handler."""
        self._file.close()
