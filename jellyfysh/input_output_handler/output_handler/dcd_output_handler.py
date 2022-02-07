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
"""Module for the DcdOutputHandler class."""
import logging
from os import rename
from typing import Any, MutableMapping, Sequence
from jellyfysh.input_output_handler.mdanalysis_import import Writer
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.node import Node, yield_closest_leaf_unit_positions
from jellyfysh.base.uuid import get_uuid
from jellyfysh.setting import hypercuboid_setting as setting
from .abstracts import MDAnalysisOutputHandler
from .pdb_output_handler import PdbOutputHandler


class DcdOutputHandler(MDAnalysisOutputHandler):
    """
    Output handler which writes the trajectory of the leaf units in the extracted global state into a .dcd file.

    This output handler should receive the extracted global state in its write method. It is designed to work together
    with the TreeStateHandler. Here, the extracted global state is given by a sequence of trees. Each tree is specified
    by a root node, which themselves are connected to children nodes. Each node contains a Unit object.
    This output handler allows for a variable number of root nodes, each with the same number of children. For each
    leaf node, the names of the corresponding point masses can be given on initialization of this class. The same is
    true for the bonds of leaf nodes of a single root node.

    This output handler stores the subsequent positions of the leaf units in the extracted global states of each call
    of the write method in a single trajectory in the .dcd file. The first call of the write method further uses the
    PdbOutputHandler to store the topology of the extracted global state (i.e., the connections of the leaf units) in a
    .pdb file.  For this, the same filename with exchanged file suffixes is used.
    Note that the positions of point masses are corrected for periodic boundary conditions so that they are the closest
    to the position of the composite point object they belong to.
    The writing to .dcd files can only be used if the hypercuboid setting is initialized and in at most three
    dimensions (see MDAnalysisOutputHandler class).
    """

    def __init__(self, filename: str, names_within_composite_object: Sequence[str] = (),
                 bonds_within_composite_object: Sequence[int] = ()) -> None:
        """
        The constructor of the DcdOutputHandler class.

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

        Raises
        ------
        base.exceptions.ConfigurationError
            If the hypercuboid setting is not initialized.
        base.exceptions.ConfigurationError
            If the filename does not end with .dcd.
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
                           bonds_within_composite_object=bonds_within_composite_object)
        if not filename.endswith(".dcd"):
            raise ConfigurationError("Output filename for output handler {0} should end with .dcd."
                                     .format(self.__class__.__name__))
        super().__init__(filename, names_within_composite_object, bonds_within_composite_object)
        self._filename_without_ending = filename[:-4]
        self._pdb_file_created = False
        self._bonds_within_composite_object = bonds_within_composite_object
        self._names_within_composite_object = names_within_composite_object
        self._writer = Writer(filename, self._number_of_atoms, lengthunit="angstrom",
                              remarks="RUN IDENTIFICATION HASH: {0}".format(get_uuid()), format="LAMMPS")

    def write(self, extracted_global_state: Sequence[Node]) -> None:
        """
        Write the topology of the extracted global state into a .pdb file, followed by writing the positions of the
        leaf units in the extracted global state into a .dcd file.

        Note that the positions of the leaf units are written into a .dcd file in the _write_trajectory method. This
        method is replaced by _write_trajectory after the first call so that the .pdb file is created only once.

        Parameters
        ----------
        extracted_global_state : Sequence[base.node.Node]
            The extracted global state.
        """
        pdb_output_handler = PdbOutputHandler(self._filename_without_ending + ".pdb",
                                              names_within_composite_object=self._names_within_composite_object,
                                              bonds_within_composite_object=self._bonds_within_composite_object)
        pdb_output_handler.write(extracted_global_state)
        pdb_output_handler.post_run()
        # PdbOutputHandler adds counter in filename
        rename(self._filename_without_ending + "0" + ".pdb", self._filename_without_ending + ".pdb")
        self._pdb_file_created = True
        # noinspection PyAttributeOutsideInit
        self.write = self._write_trajectory
        self._write_trajectory(extracted_global_state)

    def _write_trajectory(self, extracted_global_state: Sequence[Node]) -> None:
        """
        Write the positions of the leaf units in the extracted global state into a .dcd file.

        This method replaces the write method after the first call.

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
        self._writer.write(self._universe)

    def post_run(self) -> None:
        """Clean up the output handler."""
        self._writer.close()

    def __getstate__(self) -> MutableMapping[str, Any]:
        """
        Return a state of this class that can be pickled.

        This method removes the self._writer attribute from the self.__dict__ dictionary so that
        it can be pickled (because objects of MDAnalysis are mostly not well pickled).

        Returns
        -------
        MutableMapping[str, Any]
            The state that can be pickled.
        """
        super_instance_dictionary = super().__getstate__()
        del super_instance_dictionary["_writer"]
        return super_instance_dictionary

    def __setstate__(self, state: MutableMapping[str, Any]) -> None:
        """
        Use the state dictionary to initialize this class.

        This method creates the self._writer attribute that was deleted in the __getstate__ method. Since MDAnalysis
        does not allow to continue writing into a .dcd file, this method has to start a new .dcd file. If the old
        .dcd file already contained data (because a pdb file was already created before the dump), the new .dcd file
        ends with '_after_dump.dcd' in order to avoid loss of data.

        Parameters
        ----------
        state : MutableMapping[str, Any]
            The state.
        """
        super().__setstate__(state)
        if self._pdb_file_created:
            self._writer = Writer(self._filename_without_ending + "_after_dump.dcd", self._number_of_atoms,
                                  lengthunit="angstrom", remarks="RUN IDENTIFICATION HASH: {0}".format(get_uuid()),
                                  format="LAMMPS")
            self.write = self._write_trajectory
        else:
            self._writer = Writer(self._filename_without_ending + ".dcd", self._number_of_atoms, lengthunit="angstrom",
                                  remarks="RUN IDENTIFICATION HASH: {0}".format(get_uuid()), format="LAMMPS")
