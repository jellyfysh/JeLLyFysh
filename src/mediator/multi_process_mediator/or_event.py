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
"""Module for a function which creates an or event."""
from multiprocessing import Event


def _or_set(self):
    self._set()
    self.changed()


def _or_clear(self):
    self._clear()
    self.changed()


def _orify(event, changed_callback):
    event._set = event.set
    event._clear = event.clear
    event.changed = changed_callback
    event.set = lambda: _or_set(event)
    event.clear = lambda: _or_clear(event)


def create_or_event(*events: Event) -> Event:
    """
    Create an or event that gets set when one of the events in the arguments gets set.

    Parameters
    ----------
    events : Event
        The events for which the or event should be created.

    Returns
    -------
    Event
        The or event.
    """
    or_event = Event()

    def _changed():
        booleans = [e.is_set() for e in events]
        if any(booleans):
            or_event.set()
        else:
            or_event.clear()

    for event in events:
        _orify(event, _changed)
    _changed()
    return or_event
