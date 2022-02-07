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
"""Module for the SeparationOutputHandler class."""
import logging
from typing import Sequence
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node, yield_leaf_nodes
from jellyfysh.base import vectors
import jellyfysh.setting as setting
from .output_handler import OutputHandler, HardBufferedTextWriter


class SeparationOutputHandler(OutputHandler):
    """
    Output handler which samples the shortest separations between leaf units in different composite objects.

    This output handler should receive the extracted global state in its write method. It is designed to work together
    with the TreeStateHandler. Here, the extracted global state is given by a sequence of trees. Each tree is specified
    by a root node, which themselves are connected to children nodes. Each node contains a Unit object.
    This output handler allows for a variable number of root nodes, each with the same number of children. Then, the
    output handler samples the shortest separations between all leaf unit pairs in different composite objects.
    If the extracted global state contains composite objects, this output handler creates a file for each possible pair
    of leaf units between two composite objects and includes the pair in the filename.
    """

    def __init__(self, filename: str) -> None:
        """
        The constructor of the SeparationOutputHandler class.

        This class uses a HardBufferedTextWriter to first write the separations to temporary files.

        Parameters
        ----------
        filename : str
            The filename of the file this output handler is connected to.

        Raises
        ------
        AssertionError
            If the filename does not contain a file format.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, filename=filename)
        super().__init__(filename)
        self._files = []
        split_filename = filename.split(".")
        assert len(split_filename) == 2
        for number in range(setting.number_of_nodes_per_root_node):
            if setting.number_of_nodes_per_root_node > 1:
                self._files.append(HardBufferedTextWriter("{0}_1{1}.{2}"
                                                          .format(split_filename[0],
                                                                  number + setting.number_of_nodes_per_root_node + 1,
                                                                  split_filename[1])))
            else:
                self._files.append(HardBufferedTextWriter(filename))

    def write(self, extracted_global_state: Sequence[Node]) -> None:
        """
        Extract the shortest separations between all leaf unit pairs in different composite objects and write them to
        the temporary files.

        Parameters
        ----------
        extracted_global_state : Sequence[base.node.Node]
            The extracted global state.
        """
        super().write(extracted_global_state)
        for first_root_cnode_index, first_root_cnode in enumerate(extracted_global_state):
            for second_root_cnode_index in range(first_root_cnode_index + 1, len(extracted_global_state)):
                second_root_cnode = extracted_global_state[second_root_cnode_index]
                for first_leaf_node in yield_leaf_nodes(first_root_cnode):
                    for second_leaf_node in yield_leaf_nodes(second_root_cnode):
                        first_leaf_identifier = first_leaf_node.value.identifier[-1]
                        second_leaf_identifier = second_leaf_node.value.identifier[-1]
                        identifier_distance = (abs(first_leaf_identifier - second_leaf_identifier)
                                               if setting.number_of_node_levels > 1 else 0)
                        separation = setting.periodic_boundaries.separation_vector(first_leaf_node.value.position,
                                                                                   second_leaf_node.value.position)
                        print(vectors.norm(separation), file=self._files[identifier_distance])

    def post_run(self):
        """Clean up the output handler."""
        for file in self._files:
            file.close()
