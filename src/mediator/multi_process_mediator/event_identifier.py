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
"""Module for the EventStorage class."""
from typing import Any, List
from event_handler import EventHandler
from scheduler.event_identifier import EventIdentifier


class EventStorage(object):
    # TODO Remove this unnecessary class (the heap scheduler now has the event identifier included)
    """
    Class which creates an event storage.

    For each associated object which gets pushed into the scheduler in the multi-process mediator, this class creates an
    integer (using the scheduler.event_identifier.EventIdentifier class). This class allows to store for each event
    identifier the event handler and additional objects. In the multi-process mediator these additional objects are
    the out-states which are calculated in advance.
    """

    def __init__(self):
        """The constructor of the EventStorage class."""
        self._event_identifier = EventIdentifier()
        self._storage = {}
        self._identifiers = {}

    def push_to_storage(self, event_handler: EventHandler) -> int:
        """
        Return a unique identifier for the event handler and store the event handler in the storage.

        Parameters
        ----------
        event_handler : event_handler.EventHandler
            The event handler.

        Returns
        -------
        int
            The unique identifier.
        """
        new_identifier = self._event_identifier.identifier()
        self._storage[new_identifier] = [event_handler]
        self._identifiers[event_handler] = new_identifier
        return new_identifier

    def append_to_stored_value(self, event_handler: EventHandler, objects_to_store: Any) -> None:
        """
        Append the objects to the storage of the event handler.

        Parameters
        ----------
        event_handler : event_handler.EventHandler
            The event handler.
        objects_to_store : Any
            The objects to store.
        """
        identifier = self._identifiers[event_handler]
        self._storage[identifier].append(objects_to_store)

    def get_event_handler(self, identifier: int) -> List[Any]:
        """
        Return the stored objects corresponding to the event identifier.

        Parameters
        ----------
        identifier : int
            The event identifier.

        Returns
        -------
        List[Any]
            The stored objects.
        """
        return self._storage[identifier]

    def get_event_identifier(self, event_handler: EventHandler) -> int:
        """
        Return the event identifier of the event handler.

        Parameters
        ----------
        event_handler : event_handler.EventHandler
            The event handler.

        Returns
        -------
        int
            The event identifier.
        """
        return self._identifiers[event_handler]

    def delete_identifier(self, identifier: int, event_handler: EventHandler) -> None:
        """
        Delete the event identifier and the event handler from the storage.

        Parameters
        ----------
        identifier : int
            The event identifier.
        event_handler :
            The event handler.
        """
        del self._storage[identifier]
        del self._identifiers[event_handler]
        self._event_identifier.delete_identifier(identifier)
