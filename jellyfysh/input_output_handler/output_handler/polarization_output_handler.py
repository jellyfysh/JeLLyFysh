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
"""Module for the PolarizationOutputHandler class."""
import logging
from typing import Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node, yield_closest_leaf_unit_positions
import jellyfysh.setting as setting
from .output_handler import OutputHandler, HardBufferedTextWriter


class PolarizationOutputHandler(OutputHandler):
    """
    Output handler which samples the polarization of charge neutral composite point objects.

    This output handler should receive the extracted global state in its write method. It is designed to work together
    with the TreeStateHandler. Here, the extracted global state is given by a sequence of trees. Each tree is specified
    by a root node, which themselves are connected to children nodes. Each node contains a Unit object.
    This output handler allows for a variable number of root nodes, each with more than one children. The relevant
    charges of the point masses within a single composite point object should add up to zero.
    The polarization is calculated by summing the polarization of all composite point objects. The polarization of a
    single composite point object is given by the sum of the charges times the positions of the point masses within it.
    In order to take take care of periodic boundary conditions, the output handler computes the shortest separation
    vectors between the position of the root cnode (i.e., the average position of all point masses) and the point
    masses. THe separation vectors are then added to the position of the root cnode. By this, the positions of the point
    masses are the one which are the closest to the position of the root cnode.
    """

    def __init__(self, filename: str, charge: str) -> None:
        """
        The constructor of the PolarizationOutputHandler class.

        This class uses a HardBufferedTextWriter to first write the polarization to a temporary file.

        Parameters
        ----------
        filename : str
            The filename of the file this output handler is connected to.
        charge : str
            The charge used to calculate the polarization.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the number of node levels is not two or the number of nodes per root node is not three.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, filename=filename, charge=charge)
        super().__init__(filename)
        self._file = HardBufferedTextWriter(filename)
        self._charge = charge
        if setting.number_of_node_levels != 2 or setting.number_of_nodes_per_root_node == 1:
            raise ConfigurationError("The output handler {0} can only be used with charge neutral composite point"
                                     " objects.".format(self.__class__.__name__))
        print("# Polarization Vector", file=self._file)

    def write(self, extracted_global_state: Sequence[Node]) -> None:
        """
        Extract the polarization and write it to the temporary files.

        Parameters
        ----------
        extracted_global_state : Sequence[base.node.Node]
            The extracted global state.

        Raises
        ------
        AssertionError
            If not each root node is charge neutral.
        """
        super().write(extracted_global_state)
        polarization = [0.0 for _ in range(setting.dimension)]
        for root_cnode in extracted_global_state:
            assert sum(child.value.charge[self._charge] for child in root_cnode.children) == 0.0
            for leaf_unit, position in yield_closest_leaf_unit_positions(root_cnode):
                for index, entry in enumerate(position):
                    polarization[index] += leaf_unit.charge[self._charge] * entry
        print("\t".join(map(str, polarization)), file=self._file)

    def post_run(self) -> None:
        """Clean up the output handler."""
        self._file.close()
