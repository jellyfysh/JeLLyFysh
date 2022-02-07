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
"""Module for the PdbOutputHandler class."""
import logging
from typing import Sequence
import warnings
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node, yield_closest_leaf_unit_positions
from jellyfysh.base.uuid import get_uuid
from jellyfysh.input_output_handler.mdanalysis_import import Writer
from jellyfysh.setting import hypercuboid_setting as setting
from .abstracts import MDAnalysisOutputHandler


class PdbOutputHandler(MDAnalysisOutputHandler):
    """
    Output handler which writes the extracted global state into a .pdb file.

    This output handler should receive the extracted global state in its write method. It is designed to work together
    with the TreeStateHandler. Here, the extracted global state is given by a sequence of trees. Each tree is specified
    by a root node, which themselves are connected to children nodes. Each node contains a Unit object.
    This output handler allows for a variable number of root nodes, each with the same number of children. For each
    leaf node, the names of the corresponding point masses can be given on initialization of this class. The same is
    true for the bonds of leaf nodes of a single root node.

    Note that the positions of point masses are corrected for periodic boundary conditions so that they are the closest
    to the position of the composite point object they belong to.
    The writing to .pdb files can only be used if the hypercuboid setting is initialized and in at most three
    dimensions (see MDAnalysisOutputHandler class).
    This class writes the extracted global state into a new .pdb file on each call of the write method. For this, the
    filename includes a counter which starts at a given integer.
    """

    def __init__(self, filename: str, names_within_composite_object: Sequence[str] = (),
                 bonds_within_composite_object: Sequence[int] = (), starting_integer: int = 0) -> None:
        """
        The constructor of the PdbOutputHandler class.

        Parameters
        ----------
        filename : str
            The filename of the file this output handler is connected to.
        names_within_composite_object : Sequence[str], optional
            The sequence of names of the point masses within a composite object.
        bonds_within_composite_object : Sequence[int], optional
            The sequence of bonds of the point masses within the composite object. In the sequence, the bonds should be
            given in pairs of two. The point masses are numbered as they appear in the children sequence of the root
            node.
        starting_integer : int
            The starting integer of the file counter.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the hypercuboid setting is not initialized.
        base.exceptions.ConfigurationError
            If the filename does not end with .pdb.
        base.exceptions.ConfigurationError
            If the setting package specifies a dimension larger than 3 which cannot be initialized with a .pdb file.
        base.exceptions.ConfigurationError
            If the bonds_within_composite_object sequence is not divisible by two.
        base.exceptions.ConfigurationError
            If the names_within_composite_object sequence does not specify a name for each point mass (if it is not
            empty).
        """
        logger = logging.getLogger(__name__)
        log_init_arguments(logger.debug, self.__class__.__name__, filename=filename,
                           names_within_composite_object=names_within_composite_object,
                           bonds_within_composite_object=bonds_within_composite_object,
                           starting_integer=starting_integer)
        if not filename.endswith(".pdb"):
            raise ConfigurationError("Output filename for output handler {0} should end with .pdb."
                                     .format(self.__class__.__name__))
        super().__init__(filename, names_within_composite_object, bonds_within_composite_object)
        self._filename_without_ending = filename[:-4]
        self._current_file_index = starting_integer

    def write(self, extracted_global_state: Sequence[Node]) -> None:
        """
        Write the extracted global state to a .pdb file.

        Parameters
        ----------
        extracted_global_state : Sequence[base.node.Node]
            The extracted global state.
        """
        super().write(extracted_global_state)

        for root_cnode in extracted_global_state:
            for leaf_unit, position in yield_closest_leaf_unit_positions(root_cnode):
                self._get_atom(self._universe, leaf_unit.identifier).position = (
                        list(position) + [0.0 for _ in range(3 - setting.dimension)])
        with Writer(self._filename_without_ending + str(self._current_file_index) + ".pdb",
                    self._number_of_atoms, multiframe=False,
                    remarks="RUN IDENTIFICATION HASH: {0}".format(get_uuid())) as writer:
            # MDAnalysis prints UserWarnings that not so important attributes weren't set which we ignore
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                writer.write(self._universe.atoms)
        self._current_file_index += 1

    def post_run(self) -> None:
        """Clean up the output handler."""
        pass
