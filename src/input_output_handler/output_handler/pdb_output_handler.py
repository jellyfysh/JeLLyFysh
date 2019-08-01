# JeLLFysh - a Python application for all-atom event-chain Monte Carlo - https://github.com/jellyfysh
# Copyright (C) 2019 The JeLLyFysh organization
# (see the AUTHORS file for the full list of authors)
#
# This file is part of JeLLyFysh.
#
# JeLLyFysh is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either > version 3 of the License, or (at your option) any
# later version.
#
# JeLLyFysh is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with JeLLyFysh in the LICENSE file.
# If not, see <https://www.gnu.org/licenses/>.
#
# If you use JeLLyFysh in published work, please cite the following reference (see [Hoellmer2019] in References.bib):
# Philipp Hoellmer, Liang Qin, Michael F. Faulkner, A. C. Maggs, Werner Krauth
# JeLLyFysh-Version1.0 -- a Python application for all-atom event-chain Monte Carlo,
# arXiv e-prints: 1907.12502 (2019), https://arxiv.org/abs/1907.12502
#
"""Module for the PdbOutputHandler class."""
import logging
from typing import Sequence
import warnings
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.node import Node, yield_leaf_nodes
from base.uuid import get_uuid
from input_output_handler.mdanalysis_import import Universe, Writer
from setting import hypercuboid_setting as setting
from .output_handler import OutputHandler


class PdbOutputHandler(OutputHandler):
    """
    Output handler which writes the extracted global state into a .pdb file.

    This output handler should receive the extracted global state in its write method. It is designed to work together
    with the TreeStateHandler. Here, the extracted global state is given by a sequence of trees. Each tree is specified
    by a root node, which themselves are connected to children nodes. Each node contains a Unit object.
    This output handler allows for a variable number of root nodes, each with the same number of children. For each
    leaf node, the names of the corresponding point masses can be given on initialization of this class. The same is
    true for the bonds of leaf nodes of a single root node.
    The writing to .pdb files can only be used if the hypercuboid setting is initialized and in at most three
    dimensions.
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
        if not setting.initialized():
            raise ConfigurationError("The class {0} can only be used in a hypercuboid setting."
                                     .format(self.__class__.__name__))
        if not filename.endswith(".pdb"):
            raise ConfigurationError("Output filename for output handler {0} should end with .pdb."
                                     .format(self.__class__.__name__))
        if setting.dimension < 3:
            logger.warning("PDB format uses 3 dimensions. Not used dimensions will be set to length 0.0.")
        if setting.dimension > 3:
            raise ConfigurationError("PDB format cannot be used for dimensions > 3.")
        super().__init__(filename)
        self._names_within_composite_object = names_within_composite_object
        self._filename_without_ending = filename[:-4]
        self._current_file_index = starting_integer
        self._number_of_atoms = setting.number_of_root_nodes * setting.number_of_nodes_per_root_node
        # noinspection PyArgumentEqualDefault
        self._universe = Universe.empty(n_atoms=self._number_of_atoms, n_residues=setting.number_of_root_nodes,
                                        n_segments=1, atom_resindex=[index // setting.number_of_nodes_per_root_node
                                                                     for index in range(self._number_of_atoms)],
                                        residue_segindex=[0] * setting.number_of_root_nodes, trajectory=True,
                                        velocities=False, forces=False)
        # dimensions = [x, y, z, alpha, beta, gamma]
        self._universe.dimensions = ([setting.system_lengths[index] for index in range(setting.dimension)]
                                     + [0.0 for _ in range(3 - setting.dimension)]
                                     + [0.0 for _ in range(3)])
        if setting.number_of_node_levels == 1:
            self._get_atom = lambda identifier: self._universe.atoms[identifier[0]]
        else:
            self._get_atom = lambda identifier: self._universe.atoms[
                identifier[0] * setting.number_of_nodes_per_root_node + identifier[1]]

        self._universe.add_TopologyAttr("resids",
                                        values=[index + 1 for index in range(len(self._universe.residues))])

        if bonds_within_composite_object:
            if len(bonds_within_composite_object) % 2 != 0:
                raise ConfigurationError("The list of bonds should be divisible by two!")

            bonds = sum(([index + composite_index * setting.number_of_nodes_per_root_node
                          for index in bonds_within_composite_object]
                         for composite_index in range(setting.number_of_root_nodes)), [])
            bonds_iterator = iter(bonds)
            self._universe.add_TopologyAttr("bonds",
                                            values=[index_tuple for index_tuple in zip(bonds_iterator, bonds_iterator)])

        # PDBWriter currently not supports changing the record type from ATOM to HETATM although the attribute exists
        # self._universe.add_TopologyAttr("record_types")
        # for atom in self._universe.atoms:
        #    atom.record_type = "HETATM"

        if names_within_composite_object:
            if len(names_within_composite_object) != setting.number_of_nodes_per_root_node:
                raise ConfigurationError("Please give a name for each point mass within a composite point object!")
            self._universe.add_TopologyAttr(
                "resnames", values=["".join(names_within_composite_object)] * len(self._universe.residues))
            self._universe.add_TopologyAttr(
                "names", values=[names_within_composite_object[atom.ix % setting.number_of_nodes_per_root_node]
                                 for atom in self._universe.atoms])

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
            for leaf_node in yield_leaf_nodes(root_cnode):
                self._get_atom(leaf_node.value.identifier).position = leaf_node.value.position
        with Writer(self._filename_without_ending + str(self._current_file_index) + ".pdb",
                    self._number_of_atoms, multiframe=False,
                    remarks="RUN IDENTIFICATION HASH: {0}".format(get_uuid())) as writer:
            # MDAnalysis prints UserWarnings that not so important attributes weren't set which we ignore
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                writer.write(self._universe.atoms)
        self._current_file_index += 1

    def post_run(self):
        """Clean up the output handler."""
        pass
