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
"""Module for the FactorTypeMaps class."""
from abc import ABCMeta, abstractmethod
from copy import copy
import logging
import re
from typing import Any, Iterable, Tuple
from jellyfysh.base.exceptions import FactorSetError
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.state_handler.tree_state_handler import StateId
import jellyfysh.setting as setting


_logger = logging.getLogger(__name__)


class FactorTypeMaps(object):
    """
    This class implements the parsing of factor type maps out of a file as a singleton.

    The class stores the class attribute _instance, which gets built once as an instance of the __FactorTypeMaps class
    defined within this class. Any attributes defined in the instance, will also be accessible on the instance of this
    FactorTypeMaps class.

    The file containing the factor type maps, gives the interaction between units on leaf nodes for two equal composite
    point objects. For this, it numbers the units on the leaf nodes as they appear in the TreeStateHandler.
    Each line in the file starts with a comma separated list in square brackets of the numbers of leaf node units which
    should be involved in a factor. The name of the factor in CamelCase follows, separated with a comma and a
    whitespace, after the list.
    Local factors (i.e. factors which only concern numbers of units within a molecule) must only be given once.
    For non local factors, the point masses within the same composite point object will not interact!

    For example for water created by the WaterRandomNodeCreator in the RandomInputHandler, there are three leaf node
    units per root node, which get the number 0, 1, and 2. The leaf node units in the second water molecule get the
    numbers 3, 4, and 5.
    Within the first water molecule, the two hydrogens have the numbers 0 and 2 and the oxygen the number 1.
    The SPC/FW model includes a harmonic factor between the hydrogens and the oxygens and a bending factor involving
    all three. Between two water molecules, the two oxygens are involved in a Lennard Jones factor. All units in
    different molecules interact via the Coulomb factor.
    The file would therefore look like:

    [0, 1], Harmonic
    [1, 2], Harmonic
    [0, 1, 2], Bending
    [1, 4], LennardJones
    [0, 3], Coulomb
    [0, 4], Coulomb
    [0, 5], Coulomb
    [1, 3], Coulomb
    [1, 4], Coulomb
    [1, 5], Coulomb
    [2, 3], Coulomb
    [2, 4], Coulomb
    [2, 5], Coulomb

    For each of the different factors, a _FactorTypeMap will be constructed. Such a factor type map defines a method
    which yields the identifiers of the units involved in the given factor, based on an active unit identifier.
    Assume that the active unit has the identifier (0, 1) (the oxygen of the 0th water molecule) and that there are 3
    water molecules. Then the factor type map for LennardJones would generate ((0,1), (1, 1)) and ((0, 1), (2, 1)).

    If one wants to combine the Coulomb factor in a single factor, one could write:

    [0, 1, 2, 3, 4, 5], Coulomb

    For more examples, see the files in the directory config_files/factor_set_files/.
    A factor type map file can contain single line comments starting with '#'.

    The factor type maps can be used by the FactorTypeMapInStateTagger, which will assume the same factors given for
    two composite point objects between all composite point objects. In this class, also the mapping from the factor
    name given here onto the event handlers which handle the in-states will take place.
    """

    _instance = None

    def __init__(self, filename: str) -> None:
        """
        The constructor of the FactorTypeMaps class.

        If _instance is still None (nobody has called this constructor yet), construct an instance of the
        __FactorTypeMaps class and assign _instance to this.
        If _instance is already set (that means the constructor has been already called), check if the filename
        argument is the same as before. It is not allowed to try to construct several FactorTypeMaps reading from
        different files.

        Parameters
        ----------
        filename : str
            The filename out of which the factor type maps should be parsed.

        Raises
        ------
        AttributeError
            If it is tried to construct this class a second time with a different filename.
        """
        log_init_arguments(_logger.debug, self.__class__.__name__, filename=filename)
        if not FactorTypeMaps._instance:
            FactorTypeMaps._instance = FactorTypeMaps.__FactorTypeMaps(filename)
        else:
            if FactorTypeMaps._instance.filename != filename:
                raise AttributeError("Class {0} is created as a singleton and should only be created for one filename."
                                     .format(self.__class__.__name__))

    def __getattr__(self, item: Any) -> Any:
        """
        Any attribute will be looked up in the _instance.

        Parameters
        ----------
        item : Any
            The attribute.

        Returns
        -------
        Any
            The attribute of _instance.
        """
        return getattr(FactorTypeMaps._instance, item)

    def __getitem__(self, item: Any) -> Any:
        """
        A call of __getitem__ will be sent to _instance.

        Parameters
        ----------
        item : Any
            The item to get.

        Returns
        -------
        Any
            The result of __getitem__(item) used on _instance.
        """
        return FactorTypeMaps._instance.__getitem__(item)

    class __FactorTypeMaps(object):
        """
        The real FactorTypeMaps class used in _instance in the singleton class FactorTypeMaps.
        """

        def __init__(self, filename: str) -> None:
            """
            The constructor of the __FactorTypeMaps class.

            The given file will be parsed in this constructor. For each different factor in the file an instance of
            _FactorTypeMap will be constructed. These can be accessed via the __getitem__ method.

            Parameters
            ----------
            filename : str
                The filename out of which the factor type maps should be parsed.

            Raises
            ------
            base.exception.FactorSetError
                If a line in the given file does not follow the format described in FactorTypeMaps or if a given number
                in the file exceeds 2 * setting.number_of_nodes_per_root_node
            """
            self._factors = {}
            self._instantiate_factor_type_maps(filename)
            self._filename = filename

        def _instantiate_factor_type_maps(self, filename: str) -> None:
            line_pattern = re.compile(r"""
                (\[(?:[0-9]+,\ )*[0-9]+\]) # Check if list of integers in form [1, 2, 10, ...] exists
                ,\s                        # List should be followed by a comma and a whitespace
                ((?:[A-Z][a-z]*)+)         # Checks for factor type string (CamelCase)""", flags=re.VERBOSE)
            list_pattern = re.compile(r"[0-9]+")
            with open(filename) as file:
                for line in file:
                    if line.startswith("#"):
                        continue
                    match = line_pattern.match(line)
                    if match is None:
                        raise FactorSetError("Line {0} does not match pattern '[index, index, ...], Factor'"
                                             .format(line))
                    if len(match.groups()) != 2:
                        raise FactorSetError("Something went wrong. Please contact the developers!")
                    factor = match.group(2)
                    leaf_unit_indices = []
                    integer_list = list_pattern.findall(match.group(1))
                    for integer in integer_list:
                        if int(integer) >= 2 * setting.number_of_nodes_per_root_node:
                            raise FactorSetError("Given point mass index {0} is too large for two composite point "
                                                 "objects of length {1}"
                                                 .format(integer, setting.number_of_nodes_per_root_node))
                        leaf_unit_indices.append(int(integer))
                    if factor not in self._factors.keys():
                        self._factors[factor] = _FactorTypeMap()
                    if all(leaf_unit_index < setting.number_of_nodes_per_root_node for leaf_unit_index in
                           leaf_unit_indices):
                        self._factors[factor].local = True
                    else:
                        self._factors[factor].local = False
                    self._factors[factor].append_to_map(leaf_unit_indices)

        def __getitem__(self, factor: str):
            """
            Return the factor type map for the given factor.

            If the factor has not been defined in the file, a default factor type map with all leaf node units between
            different composite point objects interacting is returned.

            Parameters
            ----------
            factor : str
                The name of the factor appearing in the parsed file.

            Returns
            -------
            activator.tagger.factor_type_maps._FactorTypeMap
                The FactorTypeMap for the given factor.
            """
            try:
                return self._factors[factor]
            except KeyError:
                _logger.warning("No factor type map given for factor {0}. "
                                "Falling back on default map which maps from the active leaf unit onto all other leaf "
                                "units which are not in the same composite point object.".format(factor))
                return _AllLeafUnitFactorTypeMap()

        @property
        def filename(self) -> str:
            """
            Return the filename of the file out of which the factor type maps have been parsed.

            Returns
            -------
            str
                The filename.
            """
            return self._filename


class _FactorTypeMapAbstractClass(metaclass=ABCMeta):
    """
    The abstract class for a factor type map.

    A factor type map should be able to generate the identifiers involved in a factor given an active unit identifier.
    """

    def __init__(self) -> None:
        """
        The (currently empty) constructor of the _FactorTypeMapAbstractClass.
        """
        pass

    @abstractmethod
    def yield_factor_identifier(self, active_unit_identifier: StateId) -> Iterable[Tuple[StateId, ...]]:
        """
        Generate the in-state identifiers for the factor according to the parsed file.

        Parameters
        ----------
        active_unit_identifier : activator.tag_activator.StateId
            The identifier of an active unit.

        Yields
        ------
        Tuple[activator.tag_activator.StateId, ...]
            The factor index set including the identifier of the active unit.
        """
        raise NotImplementedError


# noinspection PyMissingOrEmptyDocstring,PyMissingTypeHints
class _FactorTypeMap(_FactorTypeMapAbstractClass):
    """
    This class implements a _FactorTypeMap built from a file.

    Should only be constructed by the FactorTypeMaps class.
    """
    def __init__(self):
        super().__init__()
        self._local = None
        self._map = {}
        if setting.number_of_nodes_per_root_node == 1:
            self._yield_factor_identifier_non_local = self._yield_factor_identifier_no_composite_objects

    def yield_factor_identifier(self, active_unit_identifier):
        raise NotImplementedError("Please set local!")

    # noinspection PyAttributeOutsideInit
    def local(self, value):
        if self._local is None:
            self._local = value
            if self._local:
                self.yield_factor_identifier = self._yield_factor_identifier_local
            else:
                self.yield_factor_identifier = self._yield_factor_identifier_non_local
        else:
            if value != self._local:
                raise AttributeError("Local member is not allowed to be changed.")

    local = property(fset=local)

    def append_to_map(self, indices):
        for index in indices:
            if index >= setting.number_of_nodes_per_root_node:
                continue
            if index not in self._map.keys():
                self._map[index] = []
            self._map[index].append(copy(indices))

    def _yield_factor_identifier_local(self, active_identifier):
        assert len(active_identifier) == 2
        assert active_identifier[0] < setting.number_of_root_nodes
        for target_leaf_node_number_list in self._map[active_identifier[1]]:
            yield tuple((active_identifier[0], target_leaf_node) for target_leaf_node in target_leaf_node_number_list)

    def _yield_factor_identifier_non_local(self, active_identifier):
        assert len(active_identifier) == 2
        assert active_identifier[0] < setting.number_of_root_nodes
        assert active_identifier[1] < setting.number_of_nodes_per_root_node
        for other_root in range(setting.number_of_root_nodes):
            if other_root == active_identifier[0]:
                continue
            if active_identifier[1] in self._map.keys():
                for target_leaf_node_number_list in self._map[active_identifier[1]]:
                    yield tuple((active_identifier[0], target_leaf_node)
                                if target_leaf_node < setting.number_of_nodes_per_root_node
                                else (other_root, target_leaf_node - setting.number_of_nodes_per_root_node)
                                for target_leaf_node in target_leaf_node_number_list)

    @staticmethod
    def _yield_factor_identifier_no_composite_objects(active_identifier):
        assert len(active_identifier) == 1
        assert active_identifier[0] < setting.number_of_root_nodes
        yield from _AllLeafUnitFactorTypeMap.yield_factor_identifier_no_composite_objects(active_identifier)

    @property
    def map(self):
        return self._map


# noinspection PyMissingOrEmptyDocstring
class _AllLeafUnitFactorTypeMap(_FactorTypeMapAbstractClass):
    """
    This class implements a factor type map where the factor includes all leaf units of different composite point
    objects.

    Should only be constructed by the FactorTypeMaps class.
    """
    def __init__(self):
        super().__init__()
        if setting.number_of_nodes_per_root_node == 1:
            self.yield_factor_identifier = self.yield_factor_identifier_no_composite_objects
        else:
            self.yield_factor_identifier = self._yield_factor_identifier_composite_objects

    def yield_factor_identifier(self, active_unit_identifier):
        raise NotImplementedError("Something went wrong. Please contact the developers.")

    @staticmethod
    def yield_factor_identifier_no_composite_objects(active_unit_identifier):
        assert len(active_unit_identifier) == 1
        assert active_unit_identifier[0] < setting.number_of_root_nodes
        for root_node_number in range(setting.number_of_root_nodes):
            target_tuple = (root_node_number,)
            if target_tuple == active_unit_identifier:
                continue
            # noinspection PyRedundantParentheses
            yield (active_unit_identifier, target_tuple)

    @staticmethod
    def _yield_factor_identifier_composite_objects(active_unit_identifier):
        assert len(active_unit_identifier) == 2
        assert active_unit_identifier[0] < setting.number_of_root_nodes
        assert active_unit_identifier[1] < setting.number_of_nodes_per_root_node
        for other_root_number in range(setting.number_of_root_nodes):
            if other_root_number == active_unit_identifier[0]:
                continue
            for leaf_node_number in range(setting.number_of_nodes_per_root_node):
                # noinspection PyRedundantParentheses
                yield (active_unit_identifier, (other_root_number, leaf_node_number))
