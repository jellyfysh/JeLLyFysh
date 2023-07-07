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
"""Module for useful abstract event handler base classes with common functions used in composite objects."""
from abc import ABCMeta
from copy import copy
from typing import Any
from .abstracts import LeavesEventHandler


class CompositeObjectsEventHandler(LeavesEventHandler, metaclass=ABCMeta):
    """
    This event handler base class can extract the leaf units present in two different composite objects.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.
    The _leaf_units attribute should contain the same number of units in two different composite objects and only a
    single composite object can have a nonzero velocity.

    Attributes
    ----------
    _local_leaf_units : Sequence[base.unit.Unit]
        The leaf units in the composite object with nonzero velocity.
    _target_leaf_units : Sequence[base.unit.Unit]
        The leaf units in the composite object with zero velocity.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        The constructor of the CompositeObjectsEventHandler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)
        self._local_leaf_units = None
        self._target_leaf_units = None

    def _construct_leaf_units_of_composite_objects(self) -> None:
        """
        Construct the local and target leaf units stored in the _leaf_units attribute.

        Raises
        ------
        AssertionError:
            If the _leaf_units attribute is None.
        AssertionError:
            If the number of leaf units is not divisible by two.
        """
        assert self._leaf_units is not None
        assert len(self._leaf_units) % 2 == 0
        sorted_leaf_units = sorted(self._leaf_units, key=lambda unit: unit.identifier)

        if all(leaf_unit.time_stamp is None for leaf_unit in sorted_leaf_units[len(sorted_leaf_units) // 2:]):
            self._local_leaf_units = sorted_leaf_units[:len(sorted_leaf_units) // 2]
            self._target_leaf_units = sorted_leaf_units[len(sorted_leaf_units) // 2:]
        else:
            assert all(leaf_unit.time_stamp is None for leaf_unit in sorted_leaf_units[:len(sorted_leaf_units) // 2])
            self._target_leaf_units = sorted_leaf_units[:len(sorted_leaf_units) // 2]
            self._local_leaf_units = sorted_leaf_units[len(sorted_leaf_units) // 2:]


class CompositeObjectsLifting(CompositeObjectsEventHandler, metaclass=ABCMeta):
    """
    This event handler base class adds a method to pass the velocity of an active composite object to a target composite
    object.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    its point masses consistent.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        The constructor of the CompositeObjectsLifting class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)

    def _pass_composite_object_velocity(self) -> None:
        """
        Pass the velocity of the active composite object stored in the _local_leaf_units attribute to the target
        composite object stored in the _target_leaf_units attribute.

        This method keeps the velocities of the branches consistent.

        Raises
        ------
        AssertionError
            If not all units in the _local_leaf_units attribute are nonzero.
        AssertionError
            If not all units in the _target_leaf_units are None.
        """
        assert all(leaf_unit.velocity == self._local_leaf_units[0].velocity
                   for leaf_unit in self._local_leaf_units)
        assert all(leaf_unit.velocity is None for leaf_unit in self._target_leaf_units)
        velocity = self._local_leaf_units[0].velocity
        negative_velocity = [-component for component in velocity]
        for leaf_cnode in self._leaf_cnodes:
            if leaf_cnode.value in self._local_leaf_units:
                self._register_velocity_change_leaf_cnode(leaf_cnode, negative_velocity)
                leaf_cnode.value.velocity = None
                leaf_cnode.value.time_stamp = None
            else:
                leaf_cnode.value.velocity = velocity.copy()
                leaf_cnode.value.time_stamp = copy(self._event_time)
                self._register_velocity_change_leaf_cnode(leaf_cnode, velocity)
        self._commit_non_leaf_velocity_changes()
