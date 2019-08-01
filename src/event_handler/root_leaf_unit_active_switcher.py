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
"""Module for the RootLeafUnitActiveSwitcher class."""
from enum import Enum
import logging
import random
from typing import Sequence
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from base.node import Node
import setting
from .abstracts import LeavesEventHandler
from .helper_functions import analyse_velocity


class _Modes(Enum):
    leaf_unit_active = 0,
    root_unit_active = 1


class RootLeafUnitActiveSwitcher(LeavesEventHandler):
    """
    Event handler which switches between the modes where a single leaf unit is active and a single composite object is
    active in a fixed interval.

    This class is designed to work together with the TreeStateHandler. Here, the in-states are branches of cnodes
    containing units. Also, the event handlers are responsible for keeping the time-slicing of composite objects and
    it's point masses consistent.
    For both modes only a direction of motion in positive direction along an axis is allowed.
    This class implements both directions, but a single instance can only treat one. Therefore it makes sense, to have
    two instances of this class per run which activate and deactivate each other (plus other event handlers relevant
    only in one of the modes).
    """

    def __init__(self, chain_length: float, aim_mode: str) -> None:
        """
        The constructor of the RootLeafUnitActiveSwitcher class.

        The aim mode can have two values:
        1. leaf_unit_active: Switches from the composite object motion to the motion of a single leaf unit.
        2. root_unit_active: Switches from the motion of a single leaf unit to the composite object motion.

        Parameters
        ----------
        chain_length : float
            The time interval of the switching of the modes.
        aim_mode : str
            The aim mode.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the number of nodes per root node is one and therefore no composite objects are present in the run.
        base.exceptions.ConfigurationError:
            If the chain length is not greater than zero.
        base.exceptions.ConfigurationError:
            If the aim_mode was not one of the allowed options.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, chain_length=chain_length,
                           aim_mode=aim_mode)
        super().__init__()
        if not setting.number_of_node_levels > 1:
            raise ConfigurationError("Event handler {0} should only be used when composite point objects are involved."
                                     .format(self.__class__.__name__))
        if not chain_length > 0.0:
            raise ConfigurationError("The chain_length in the event handler {0} has to be > 0.0."
                                     .format(self.__class__.__name__))
        self._chain_length = chain_length
        try:
            self._aim_mode = _Modes[aim_mode]
        except KeyError:
            raise ConfigurationError("Invalid aim mode {0} given in event handler {1}. "
                                     "Only supported aim modes are {2}".format(aim_mode, self.__class__.__name__,
                                                                               [i for i in _Modes]))
        self.send_out_state = getattr(self, "_send_out_state_" + self._aim_mode.name)

    def send_event_time(self, root_cnode_of_active_unit: Sequence[Node]) -> float:
        """
        Return the candidate event time.

        The in-state consists of the branch of the currently independent active unit. The next candidate event time
        is the time stamp of this unit plus the chain length. This makes it possible, to deactivate this event handler
        and also trash and create events from it.

        Parameters
        ----------
        root_cnode_of_active_unit : Sequence[base.node.Node]
            The in-state.

        Returns
        -------
        float
            The candidate event time.

        Raises
        ------
        AssertionError
            If the in-state contains not exactly one branch.
        """
        assert len(root_cnode_of_active_unit) == 1
        self._event_time = root_cnode_of_active_unit[0].value.time_stamp + self._chain_length
        return self._event_time

    def _send_out_state_leaf_unit_active(self, active_root_cnode: Sequence[Node]) -> Sequence[Node]:
        """
        Return the out-state.

        This method replaces the send_out_state method for the aim_mode leaf_unit_active.
        This method receives the branch of the active composite object. It extracts the speed and the direction of
        motion, chooses a random leaf unit within the composite object and makes it active.

        Parameters
        ----------
        active_root_cnode : Sequence[base.node.Node]
            The branch of the active composite object.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.

        Raises
        ------
        AssertionError
            If not exactly one branch is given in the argument.
        AssertionError
            If not all leaf units in the active composite object have the same induced velocity.
        """
        assert len(active_root_cnode) == 1
        self._store_in_state(active_root_cnode)
        self._time_slice_all_units_in_state()
        self._construct_leaf_cnodes()
        assert all(self._leaf_units[0].velocity == leaf_unit.velocity for leaf_unit in self._leaf_units)
        direction_of_motion, speed = analyse_velocity(self._leaf_units[0].velocity)
        new_active_leaf_cnode = random.choice(self._leaf_cnodes)
        velocity_change = [0.0 for _ in range(setting.dimension)]
        velocity_change[direction_of_motion] -= speed
        for cnode in self._leaf_cnodes:
            if cnode is not new_active_leaf_cnode:
                self._register_velocity_change_leaf_cnode(cnode, velocity_change)
                cnode.value.velocity = None
                cnode.value.time_stamp = None
        self._commit_non_leaf_velocity_changes()
        return self._state

    def _send_out_state_root_unit_active(self, root_cnode_of_active_leaf_unit: Sequence[Node]) -> Sequence[Node]:
        """
        Return the out-state.

        This method replaces the send_out_state method for the aim_mode root_unit_active.
        This method receives the branch of the composite object the active leaf unit belongs to. It extracts the speed
        and the direction of motion and makes the full composite object active.

        Parameters
        ----------
        root_cnode_of_active_leaf_unit : Sequence[base.node.Node]
            The branch of the composite object the active leaf unit belongs to.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.

        Raises
        ------
        AssertionError
            If not exactly one branch is given in the argument.
        AssertionError
            If not only one leaf unit in the composite object has a nonzero velocity.
        """
        assert len(root_cnode_of_active_leaf_unit) == 1
        self._store_in_state(root_cnode_of_active_leaf_unit)
        self._time_slice_all_units_in_state()
        self._construct_leaf_cnodes()
        active_leaf_unit = [leaf_unit for leaf_unit in self._leaf_units if leaf_unit.velocity is not None]
        assert len(active_leaf_unit) == 1
        active_leaf_unit = active_leaf_unit[0]
        direction_of_motion, speed = analyse_velocity(active_leaf_unit.velocity)
        velocity_change = [0.0 for _ in range(setting.dimension)]
        velocity_change[direction_of_motion] += speed
        for cnode in self._leaf_cnodes:
            if cnode.value is not active_leaf_unit:
                cnode.value.velocity = active_leaf_unit.velocity.copy()
                cnode.value.time_stamp = active_leaf_unit.time_stamp
                self._register_velocity_change_leaf_cnode(cnode, velocity_change)
        self._commit_non_leaf_velocity_changes()
        return self._state

    def send_out_state(self, root_cnode: Sequence[Node]) -> Sequence[Node]:
        """
        Return the out-state.

        This method is replaced by either _send_out_state_leaf_unit_active or _send_out_state_root_unit_active,
        depending on the aim mode.
        The argument for both methods is the branch of a moving composite object, either with independent or with
        induced velocity.

        Parameters
        ----------
        root_cnode : Sequence[base.node.Node]
            The branch of a composite object.

        Returns
        -------
        Sequence[base.node.Node]
            The out-state.
        """
        raise NotImplementedError
