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
"""Module for the unique identification hash function."""
from typing import Any, MutableMapping
import uuid


_uuid = None


def get_uuid() -> uuid.UUID:
    """
    Return the unique random identification hash of the run.

    On the first call, uuid.uuid4() is used to generate the random identification hash. This hash will be stored and
    returned in all calls.

    Returns
    -------
    uuid.UUID
        The unique random identification hash.
    """
    global _uuid
    if _uuid is None:
        _uuid = uuid.uuid4()
    return _uuid


def getstate() -> MutableMapping[str, Any]:
    """
    Return a state of this module that can be pickled.

    This function stores the _uuid variable so that it can be set explicitly in the setstate function.

    Returns
    -------
    MutableMapping[str, Any]
        The state that can be pickled.
    """
    return {"_uuid": get_uuid()}


def setstate(state: MutableMapping[str, Any]) -> None:
    """
    Use the state dictionary to initialize this module.

    This function sets the _uuid variable to the value that is specified in the state.

    Parameters
    ----------
    state : MutableMapping[str, Any]
        The state.

    Raises
    ------
    AssertionError
        If the state dictionary misses the necessary _uuid key for the initialization of this module.
    """
    assert "_uuid" in state
    assert state["_uuid"] is not None
    global _uuid
    _uuid = state["_uuid"]
