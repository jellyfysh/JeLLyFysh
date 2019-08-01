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
"""Module for useful abstract event handler base classes."""
from abc import ABCMeta
from typing import Any, Sequence
from base.node import Node, yield_leaf_nodes
from base.unit import Unit
from event_handler import EventHandler
from event_handler.helper_functions import analyse_velocity
import setting


class BasicEventHandler(EventHandler, metaclass=ABCMeta):
    """
    The most basic abstract event handler classes for in-states built from the TreeStateHandler.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    This class provides methods to store the in-state, to time-slice an unit and to time-slice all units present on all
    cnodes in the in-state.

    Attributes
    ----------
    _event_time : float
        The stored event time.
    _state : Sequence[base.node.Node]
        The in-state.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        The constructor of the BasicEventHandler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)
        self._event_time = None
        self._state = None

    def _store_in_state(self, in_state: Sequence[Node]) -> None:
        """
        Store the in-state in the _state attribute.

        Parameters
        ----------
        in_state : Sequence[base.node.Node]
            The in-state.
        """
        self._state = in_state

    def _time_slice_unit(self, unit: Unit) -> None:
        """
        Time-slice the unit to the _event_time attribute.

        Parameters
        ----------
        unit : base.unit.Unit
            The unit.
        """
        if unit.velocity is not None:
            for d in range(setting.dimension):
                unit.position[d] = setting.periodic_boundaries.correct_position_entry(
                    unit.position[d] + unit.velocity[d] * (self._event_time - unit.time_stamp), d)
            unit.time_stamp = self._event_time

    def _time_slice_all_units_in_state(self) -> None:
        """Time-slice all units in the _state attribute to the _event_time attribute."""
        # TODO Remove the often unnecessary loop of the complete self._state?
        for cnode in self._state:
            self._time_slice_subtree_units(cnode)

    def _time_slice_subtree_units(self, cnode: Node) -> None:
        """
        Time-slice the unit on the cnode and all units on descendants to the _event_time attribute.

        Parameters
        ----------
        cnode : base.node.Node
            The cnode.
        """
        self._time_slice_unit(cnode.value)
        for child in cnode.children:
            self._time_slice_subtree_units(child)


class LeavesEventHandler(BasicEventHandler, metaclass=ABCMeta):
    """
    This event handler base class acts on leaf units in the in-state built from the TreeStateHandler.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    This class provides methods to construct the leaf cnodes and leaf units and to keep each branch consistent when
    the velocity of a leaf unit changes.

    Attributes
    ----------
    _leaf_cnodes : Sequence[base.node.Node]
        The leaf cnodes present in the _state attribute.
    _leaf_units : Sequence[base.unit.Unit]
        The leaf units present in the _state attribute.
    _non_leaf_velocity_changes : Mapping[state_handler.tree_state_handler.StateId, Sequence[float]]
        A map from identifiers of non leaf units onto the corresponding velocity change of this unit.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        The constructor of the LeavesEventHandler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)
        self._leaf_cnodes = None
        self._leaf_units = None
        self._non_leaf_velocity_changes = {}

    def _construct_leaf_cnodes(self) -> None:
        """Extract the _leaf_cnodes and _leaf_units from the _state attribute."""
        assert self._state is not None
        self._leaf_cnodes = []
        self._leaf_units = []
        for cnode in self._state:
            for leaf_cnode in yield_leaf_nodes(cnode):
                self._leaf_cnodes.append(leaf_cnode)
                self._leaf_units.append(leaf_cnode.value)

    def _register_velocity_change_leaf_cnode(self, leaf_cnode: Node, leaf_velocity_change: Sequence[float]) -> None:
        """
        Register a velocity change of a leaf cnode and store the velocity changes of parent cnodes in the
        _non_leaf_velocity_changes attribute.

        Parameters
        ----------
        leaf_cnode:
            The leaf cnode.
        leaf_velocity_change: Sequence[float]
            The velocity change.
        """
        parent_cnode = leaf_cnode.parent
        if parent_cnode is None:
            return
        velocity_change = [component * leaf_cnode.weight for component in leaf_velocity_change]
        while parent_cnode is not None:
            identifier = parent_cnode.value.identifier
            try:
                for d in range(setting.dimension):
                    self._non_leaf_velocity_changes[identifier][d] += velocity_change[d]
            except KeyError:
                self._non_leaf_velocity_changes[identifier] = velocity_change.copy()
            for d in range(setting.dimension):
                leaf_velocity_change[d] *= parent_cnode.weight
            parent_cnode = parent_cnode.parent

    def _commit_non_leaf_velocity_changes(self) -> None:
        """
        Change the velocity in the units of non leaf cnodes stored in the _non_leaf_velocity_changes attribute and
        reset this attribute.
        """
        for cnode in self._state:
            self._commit_sub_tree_non_leaf_velocity_change(cnode)
        self._non_leaf_velocity_changes = {}

    def _commit_sub_tree_non_leaf_velocity_change(self, cnode: Node) -> None:
        """
        Change the velocity for the unit on the cnode and all descendants with the velocity changes stored in the
        _non_leaf_velocity_changes attribute.

        Units which already have a velocity before the velocity change will be time-sliced to the _event_time attribute.
        Also, if a velocity component is smaller than 1e-6 after the velocity change, it will be regarded as zero.

        Parameters
        ----------
        cnode : base.node.Node
            The cnode.
        """
        unit = cnode.value
        if unit.identifier in self._non_leaf_velocity_changes.keys():
            if unit.velocity is None:
                unit.velocity = self._non_leaf_velocity_changes[unit.identifier].copy()
                unit.time_stamp = self._event_time
            else:
                self._time_slice_unit(cnode.value)
                for d in range(setting.dimension):
                    unit.velocity[d] += self._non_leaf_velocity_changes[unit.identifier][d]
                if all(component < 1e-6 for component in unit.velocity):
                    unit.velocity = None
                    unit.time_stamp = None
        for child in cnode.children:
            self._commit_sub_tree_non_leaf_velocity_change(child)


class SingleActiveLeafUnitEventHandler(LeavesEventHandler, metaclass=ABCMeta):
    """
    This event handler base class extracts a single active leaf unit and acts on this.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    This class provides methods to extract the active leaf unit and to exchange the velocity between two leaf units.

    Attributes
    ----------
    _active_leaf_unit_index : int
        The index of the active leaf unit (cnode) within the _leaf_units (_leaf_cnodes) attribute.
    _direction_of_motion : int
        The direction of motion of the active leaf unit.
    _active_leaf_unit : base.unit.Unit
        The active leaf unit.
    _speed: float
        The absolute value of the active leaf unit's velocity.
    """
    def __init__(self, **kwargs: Any) -> None:
        """
        The constructor of the SingleActiveLeafUnitEventHandler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)
        self._active_leaf_unit_index = None
        self._direction_of_motion = None
        self._active_leaf_unit = None
        self._speed = None

    def _extract_active_leaf_unit(self) -> None:
        """
        Extract the active leaf unit, the direction of motion and the speed from the _leaf_units attribute.

        Raises
        ------
        AssertionError
            If more than one leaf unit in the _leaf_units attribute is active.
        AssertionError
            If the velocity of the active leaf unit is not aligned with an axis and the direction of motion cannot be
            represented as an integer.
        """
        assert self._leaf_units is not None
        active_leaf_units = [(index, unit) for index, unit in enumerate(self._leaf_units) if unit.velocity is not None]
        assert len(active_leaf_units) == 1
        self._active_leaf_unit_index = active_leaf_units[0][0]
        self._active_leaf_unit = self._leaf_units[self._active_leaf_unit_index]
        self._direction_of_motion, self._speed = analyse_velocity(active_leaf_units[0][1].velocity)

    def _exchange_velocity(self, cnode_with_active_unit: Node, target_cnode: Node) -> None:
        """
        Exchange the velocity between the cnode with the active unit and the target cnode.

        The velocity of target unit must be zero before exchanging. This method also keeps the branches of the nodes
        consistent.

        Parameters
        ----------
        cnode_with_active_unit : base.node.Node
            The cnode which stores the active unit.
        target_cnode : base.node.Node
            The cnode which gets the velocity of the active unit.
        """
        active_unit = cnode_with_active_unit.value
        target_unit = target_cnode.value
        self._register_velocity_change_leaf_cnode(cnode_with_active_unit,
                                                  [-component for component in active_unit.velocity])
        self._register_velocity_change_leaf_cnode(target_cnode, active_unit.velocity)
        target_unit.velocity = active_unit.velocity
        target_unit.time_stamp = active_unit.time_stamp
        active_unit.velocity = None
        active_unit.time_stamp = None
        self._commit_non_leaf_velocity_changes()


class EventHandlerWithOutputHandler(BasicEventHandler, metaclass=ABCMeta):
    """
    This event handler base class adds an output handler property so that it can be connected to one.

    At the beginning of the run, the mediator checks, if the output handler exists.
    """
    def __init__(self, output_handler: str, **kwargs: Any) -> None:
        """
        The constructor of the EventHandlerWithOutputHandler class.

        This class is designed for cooperative inheritance, meaning that it passes through all unused kwargs in the
        init to the next class in the MRO via super.

        Parameters
        ----------
        output_handler : str
            The name of the output handler.
        kwargs : Any
            Additional kwargs which are passed to the __init__ method of the next class in the MRO.
        """
        super().__init__(**kwargs)
        self._output_handler = output_handler

    @property
    def output_handler(self) -> str:
        """
        Return the output handler.

        Returns
        -------
        str
            The output handler.
        """
        return self._output_handler
