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
"""Module for the DummyOutputHandler class."""
import logging
from typing import Any
from base.logging import log_init_arguments
from .output_handler import OutputHandler


class DummyOutputHandler(OutputHandler):
    """
    An output handler which does nothing in its write method.

    When output is deactivated in debugging mode in resume.py, an instance of this output handler will replace every
    output handler in the run.
    """

    def __init__(self) -> None:
        """The constructor of the DummyOutputHandler class."""
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__)
        super().__init__(".")

    def write(self, *args: Any) -> None:
        """
        Do nothing with the arguments.

        Parameters
        ----------
        args : Any
            The arguments.
        """
        pass

    def post_run(self) -> None:
        """Clean up the output handler."""
        pass
