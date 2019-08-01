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
"""Module for the abstract Initializer class."""
from abc import ABCMeta, abstractmethod
import inspect
from base.exceptions import InitializerError


class Initializer(metaclass=ABCMeta):
    """
    Abstract class that blocks all public methods until the initialize method has been called.

    This class should be inherited by any class that needs initialization besides the __init__ method. If a class
    inherits from this class, all its public methods raise an exception until the initialize method has been called.
    It is possible to define several initialize methods. However, one of these methods must be called initialize
    (since this class defines it as an abstract method). All other methods should start with initialize so that these
    do not get blocked.
    """

    def __init__(self, **kwargs):
        """
        The constructor of the abstract Initializer class.

        This class is designed for cooperative inheritance, meaning that it passes all kwargs to the next class in the
        MRO via super().
        """
        super().__init__(**kwargs)
        self._public_methods = [name for (name, value) in inspect.getmembers(self, inspect.ismethod)
                                if not name.startswith("_") and not name.startswith("__")
                                and not name.startswith("initialize")]
        self._block_public_methods()

    @abstractmethod
    def initialize(self, *args, **kwargs):
        """Abstract initialize method with arbitrary arguments which should be extended. Frees the public methods."""
        self._free_public_methods()

    # noinspection PyUnusedLocal
    def _throw_error(self, *args, **kwargs):
        """
        As long as the initialize method has not been called, all public methods are replaced by this method.

        This method prints out the arguments of the initialize method.
        """
        error_strings = []
        initialize_signature = inspect.signature(self.initialize)
        for parameter in initialize_signature.parameters:
            error_strings.append(parameter)
            error_strings.append(", ")
        if error_strings:
            # Delete last ,
            del error_strings[-1]
        raise InitializerError("Please call initialize with the following arguments {0}"
                               " before trying to call a public method of this class: {1}"
                               .format("".join(error_strings), self.__class__.__name__))

    def _block_public_methods(self):
        for method in self._public_methods:
            setattr(self, "_blocked_" + method, getattr(self, method))
            setattr(self, method, self._throw_error)

    def _free_public_methods(self):
        for method in self._public_methods:
            setattr(self, method, getattr(self, "_blocked_" + method))
            delattr(self, "_blocked_" + method)
