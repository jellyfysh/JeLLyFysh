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
"""Module for the EventIdentifier class."""


class EventIdentifier(object):
    """
    Class which creates event identifiers.

    For each associated object which gets pushed into the scheduler, this class creates an integer. The integers for
    each event present in the scheduler are guaranteed to be different. When an event is deleted in the scheduler,
    its identifier is also deleted in this class, so that it can be reused.
    """

    def __init__(self):
        """
        The constructor of the EventIdentifier class.
        """
        self._reusable_identifiers = []
        self._counter = 0

    def identifier(self) -> int:
        """
        Return a new event identifier.

        If there are any reusable identifiers (this sequence gets filled in the delete_identifier method), return one
        of these. Otherwise an internal counter gives the next integer which can be used as an event identifier.

        Returns
        -------
        int
            The unique event identifier.
        """
        if self._reusable_identifiers:
            return self._reusable_identifiers.pop()
        self._counter += 1
        return self._counter

    def delete_identifier(self, identifier: int) -> None:
        """
        Delete an event identifier.

        The deleted identifier is marked as reusable, so that the identifier method can return it again.

        Parameters
        ----------
        identifier : int
            The event identifier which should be deleted.

        Raises
        ------
        AssertionError
            If the identifier which should be deleted exceeds the internal counter.
        """
        assert identifier <= self._counter
        self._reusable_identifiers.append(identifier)
