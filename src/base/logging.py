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
"""Module for the a logging helper function."""
from typing import Any, Callable


def log_init_arguments(logger_function: Callable[[str], None], class_name: str, **kwargs: Any):
    """
    Use the logger function to log that the class gets initialized with the kwargs.

    The logger function is an argument of this function, so that the right logger is used. In most cases, the logger
    is obtained in the module by calling 'logging.getLogger(__name__)'.
    The kwargs will be part of the logging message with both the key and the value. Also the class name will appear
    in the message.

    Parameters
    ----------
    logger_function : Callable[[str], None]
        The logger method.
    class_name : str
        The class name in whose __init__ method this function is called.
    kwargs : Any
        The arguments of the __init__ method as kwargs.
    """
    logger_function(("Initializing class '{0}' with arguments: ".format(class_name) +
                    ", ".join("{0}: {1}".format(key, value) for key, value in kwargs.items()) + ".")
                    if kwargs else "Initializing class {0}.".format(class_name))
