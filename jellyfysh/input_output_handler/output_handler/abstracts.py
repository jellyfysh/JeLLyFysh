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
"""Module for the abstract MDAnalysisOutputHandler class."""
from abc import ABCMeta
import logging
from typing import Any, MutableMapping, Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.input_output_handler.mdanalysis_import import Universe
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.setting import hypercuboid_setting as setting
from .output_handler import OutputHandler


class MDAnalysisOutputHandler(OutputHandler, metaclass=ABCMeta):
    """
    Abstract output handler class that creates and stores a universe in MDAnalysis. This universe can be used to use
    MDAnalysis writer classes in an inheriting output handler.

    This output handler is designed to work together with the TreeStateHandler. Here, the extracted global state is
    given by a sequence of trees. Each tree is specified by a root node, which themselves are connected to children
    nodes. Each node contains a Unit object. This output handler allows for a variable number of root nodes, each with
    the same number of children. For each leaf node, the names of the corresponding point masses can be given on
    initialization of this class. The same is true for the bonds of leaf nodes of a single root node.
    """
    def __init__(self, filename: str, names_within_composite_object: Sequence[str] = (),
                 bonds_within_composite_object: Sequence[int] = ()) -> None:
        """
        The constructor of the MDAnalysisOutputHandler class.

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
            If the setting package specifies a dimension larger than 3 which cannot be used within MDAnalysis.
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
        if not setting.initialized():
            raise ConfigurationError("The class {0} can only be used in a hypercuboid setting."
                                     .format(self.__class__.__name__))
        if setting.dimension < 3:
            logger.warning("MDAnalysis only allows for 3 dimensions but the simulation is done in {0} dimensions. "
                           "This class will set positions in not used dimensions to 0.0.".format(setting.dimension))
        if setting.dimension > 3:
            raise ConfigurationError("MDAnalysis does not allow for dimensions > 3.")
        super().__init__(filename)
        self._names_within_composite_object = names_within_composite_object
        self._number_of_atoms = setting.number_of_root_nodes * setting.number_of_nodes_per_root_node
        # dimensions = [x, y, z, alpha, beta, gamma]
        self._dimensions = ([setting.system_lengths[index] for index in range(setting.dimension)]
                            + [0.0 for _ in range(3 - setting.dimension)] + [90.0 for _ in range(3)])

        if setting.number_of_node_levels == 1:
            self._get_atom = lambda universe, identifier: universe.atoms[identifier[0]]
        else:
            self._get_atom = lambda universe, identifier: universe.atoms[
                identifier[0] * setting.number_of_nodes_per_root_node + identifier[1]]

        if bonds_within_composite_object:
            if len(bonds_within_composite_object) % 2 != 0:
                raise ConfigurationError("The list of bonds should be divisible by two!")
        self._bonds_within_composite_object = bonds_within_composite_object

        if names_within_composite_object:
            if len(names_within_composite_object) != setting.number_of_nodes_per_root_node:
                raise ConfigurationError("Please give a name for each point mass within a composite point object!")
        self._names_within_composite_object = names_within_composite_object

        self._universe = self._create_universe_with_topologies(setting.number_of_root_nodes,
                                                               setting.number_of_nodes_per_root_node)

    def _create_universe_with_topologies(self, number_of_root_nodes, number_of_nodes_per_root_node) -> Universe:
        """Create a MDAnalysis universe with all topologies that can be set by the JF application."""
        # noinspection PyArgumentEqualDefault
        universe = Universe.empty(n_atoms=self._number_of_atoms, n_residues=number_of_root_nodes,
                                  n_segments=1, atom_resindex=[index // number_of_nodes_per_root_node
                                                               for index in range(self._number_of_atoms)],
                                  residue_segindex=[0] * number_of_root_nodes, trajectory=True,
                                  velocities=False, forces=False)
        # dimensions = [x, y, z, alpha, beta, gamma]
        universe.dimensions = self._dimensions
        universe.add_TopologyAttr("resids", values=[index + 1 for index in range(len(universe.residues))])

        if self._bonds_within_composite_object:
            bonds = sum(([index + composite_index * number_of_nodes_per_root_node
                          for index in self._bonds_within_composite_object]
                         for composite_index in range(number_of_root_nodes)), [])
            bonds_iterator = iter(bonds)
            universe.add_TopologyAttr("bonds",
                                      values=[index_tuple for index_tuple in zip(bonds_iterator, bonds_iterator)])

        # PDBWriter currently not supports changing the record type from ATOM to HETATM although the attribute exists
        # universe.add_TopologyAttr("record_types")
        # for atom in universe.atoms:
        #    atom.record_type = "HETATM"

        if self._names_within_composite_object:
            universe.add_TopologyAttr(
                "resnames", values=["".join(self._names_within_composite_object)] * len(universe.residues))
            universe.add_TopologyAttr(
                "names", values=[self._names_within_composite_object[atom.ix % number_of_nodes_per_root_node]
                                 for atom in universe.atoms])

        # The positions of the atoms in the universe always have to be three dimensional.
        for atom in universe.atoms:
            atom.position = [0.0, 0.0, 0.0]

        return universe

    def __copy__(self):
        """No shallow copies can be created of this class."""
        raise NotImplementedError

    # noinspection PyDefaultArgument
    def __deepcopy__(self, _={}):
        """No deep copies can be created of this class."""
        raise NotImplementedError

    def __getstate__(self) -> MutableMapping[str, Any]:
        """
        Return a state of this class that can be pickled.

        This method removes the self._universe attribute from the self.__dict__ dictionary so that it can be pickled
        (because objects of MDAnalysis are mostly not well pickled). Moreover, attributes storing the number of root
        nodes, and the number of nodes per root nodes are added (see __setstate__ method for an explanation why these
        are required).

        Returns
        -------
        MutableMapping[str, Any]
            The state that can be pickled.
        """
        instance_dictionary = self.__dict__.copy()
        del instance_dictionary["_universe"]
        instance_dictionary["_number_of_root_nodes"] = setting.number_of_root_nodes
        instance_dictionary["_number_of_nodes_per_root_node"] = setting.number_of_nodes_per_root_node
        return instance_dictionary

    def __setstate__(self, state: MutableMapping[str, Any]) -> None:
        """
        Use the state dictionary to initialize this class.

        This method creates the self._universe attribute that was deleted in the __getstate__ method. For this,
        the number of root nodes, and the number of nodes per root nodes is required. Since it is not guaranteed that
        the setting package is initialized when this method is used, these numbers were stored as attributes in the
        __getstate__ method.

        Parameters
        ----------
        state : MutableMapping[str, Any]
            The state.
        """
        number_of_root_nodes = state["_number_of_root_nodes"]
        del state["_number_of_root_nodes"]
        number_of_nodes_per_root_node = state["_number_of_nodes_per_root_node"]
        del state["_number_of_nodes_per_root_node"]
        self.__dict__.update(state)
        self._universe = self._create_universe_with_topologies(number_of_root_nodes, number_of_nodes_per_root_node)
