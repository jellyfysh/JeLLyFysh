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
"""Module for the TreeLiftingState class."""
import logging
from typing import Iterable, Sequence, Tuple, Union
from base.logging import log_init_arguments
import setting
from .lifting_state import LiftingState


class TreeLiftingState(LiftingState):
    """
    The tree lifting state implements a lifting state for the tree state handler.

    This class is designed to work together with the TreeStateHandler. The global state identifiers are then tuples of
    integers, where the tuples can have different lengths.
    These identifiers are mapped onto the velocity and the time stamp in a simple dictionary.
    """

    def __init__(self) -> None:
        """
        The constructor of the TreeLiftingState class.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__)
        super().__init__()
        self._lifting_dictionary = {}
        self._lifted_identifiers = {identifier_length: set()
                                    for identifier_length in range(1, setting.number_of_node_levels + 1)}
        if setting.number_of_node_levels == 1:
            self.yield_independent_lifted_identifiers = self._yield_independent_lifted_identifiers_simple

    def set(self, identifier: Tuple[int, ...], velocity: Sequence[float], time_stamp: float) -> None:
        """
        Store the given velocity and time stamp for the global state identifier.

        If the velocity and the time stamp are None, the global state identifier is deleted from the global lifting
        state.

        Parameters
        ----------
        identifier : Tuple[int, ...]
            The global state identifier.
        velocity : Sequence[float] or None
            The velocity.
        time_stamp : float or None
            The time stamp.
        """
        if velocity is not None:
            self._lifting_dictionary[identifier] = (velocity, time_stamp)
            self._lifted_identifiers[len(identifier)].add(identifier)
        else:
            self._delete(identifier)

    def get(self, identifier: Tuple[int, ...]) -> Union[Tuple[Sequence[float], float], Tuple[None, None]]:
        """
        Return the velocity and the time stamp for the global state identifier.

        If the global state identifier is not stored within the global lifting state, this method returns (None, None).

        Parameters
        ----------
        identifier : Tuple[int, ...]
            The global state identifier.

        Returns
        -------
        (Sequence[float], float) or (None, None)
            The velocity, the time stamp.
        """
        return self._lifting_dictionary.get(identifier, (None, None))

    def yield_independent_lifted_identifiers(self) -> Iterable[Tuple[int, ...]]:
        """
        Generate all independent lifted identifiers stored in the global lifting state.

        For the TreeStateHandler independent velocities means the following: If a point mass is active,
        this induces a velocity of the composite point object it belongs to (velocity multiplied by the weight of the
        node of the point mass). Similarly a nonzero velocity of a composite point object leads to the fact, that all
        point masses of this composite point object have the same velocity. For the first case, only the identifier of
        the point mass should be returned, for the latter case only the identifier of the composite point object.
        This method checks for each stored global state identifier on the composite point object level, if all its
        children are also active. If so, the identifier of the composite point object is generated, otherwise the
        identifier of the point masses.
        This is only relevant when composite point objects are involved in the run. If this is not the case, this method
        is replaced by _yield_independent_lifted_identifiers_simple.

        Yields
        ------
        Tuple[int, ...]
            The independently lifted global state identifiers.
        """
        for lifted_root_identifier in self._lifted_identifiers[1]:
            lifted_identifiers = []
            for leaf_unit_identifier in range(setting.number_of_nodes_per_root_node):
                identifier = lifted_root_identifier + (leaf_unit_identifier,)
                if identifier in self._lifted_identifiers[2]:
                    lifted_identifiers.append(identifier)
            if len(lifted_identifiers) == setting.number_of_nodes_per_root_node:
                yield lifted_root_identifier
            else:
                yield from lifted_identifiers

    def _yield_independent_lifted_identifiers_simple(self) -> Iterable[Tuple[int, ...]]:
        """
        Generate all independent lifted identifiers stored in the global lifting state.

        This method replaces yield_independent_lifted_identifiers when only point masses are involved in the run. Then
        all lifted identifiers are generated.

        Yields
        ------
        Tuple[int, ...]
            The independently lifted global state identifiers.
        """
        yield from self._lifting_dictionary.keys()

    def _delete(self, identifier: Tuple[int, ...]) -> None:
        """Delete a global state identifier out of the global lifting state."""
        if identifier in self._lifting_dictionary.keys():
            del self._lifting_dictionary[identifier]
            self._lifted_identifiers[len(identifier)].remove(identifier)
