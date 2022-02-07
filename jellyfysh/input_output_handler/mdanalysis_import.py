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
"""
Module to import MDAnalysis.

When using pypy3, there is a bug in MDAnalysis which is solved here using a monkey patch.
After importing this module, MDAnalysis.Universe and MDAnalysis.Writer are usable.
"""
import logging
import sys
# Ignore warnings of MDAnalysis
logging.getLogger("MDAnalysis").setLevel(logging.ERROR)
# noinspection PyUnresolvedReferences
from MDAnalysis import Universe, Writer
logging.getLogger("MDAnalysis").setLevel(logging.WARNING)

# There is a bug in pypy3 (see example below) which leads to errors within MDAnalysis
# This can be solved by the following overwriting of a __new__ method
if sys.implementation.name == "pypy":
    # noinspection PyProtectedMember
    from MDAnalysis.core.groups import _ImmutableBase


    def __new__(cls, *_, **__):
        return object.__new__(cls)


    _ImmutableBase.__new__ = __new__


class _PypyBugClass(object):
    # Class with the bug
    __new__ = object.__new__


class _PypyFixedClass(object):
    # Fixed class
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)


class _PypyBugInheritingClass(_PypyBugClass):
    def __init__(self, value):
        self.value = value


class _PypyFixedInheritingClass(_PypyFixedClass):
    def __init__(self, value):
        self.value = value


if __name__ == '__main__':
    try:
        # Creating an instance of this class fails
        _PypyBugInheritingClass(1)
    except TypeError:
        print("Instantiation failed.")
    # Creating an instance of this class succeeds
    _PypyFixedInheritingClass(1)
