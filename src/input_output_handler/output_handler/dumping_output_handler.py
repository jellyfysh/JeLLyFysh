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
"""Module for the DumpingOutputHandler class."""
import logging
import random
import sys
import dill
from base.exceptions import ConfigurationError
from base.logging import log_init_arguments
from mediator import Mediator
import setting
from .output_handler import OutputHandler


class DumpingOutputHandler(OutputHandler):
    """
    Output handler which dumps an entire run into a file.

    The run is dumped by dumping the mediator, the state of the random module and the setting package to a file using
    the dill package. The file can be used in resume.py to resume the run starting from the dumped configuration.
    The run can only be resumed using the same python implementation and version. These are included in the filename
    of the dumping file.
    """

    def __init__(self, filename: str) -> None:
        """
        The constructor of the DumpingOutputHandler class.

        Parameters
        ----------
        filename : str
            The filename of the file this output handler is connected to.

        Raises
        ------
        AssertionError
            If the filename does not contain a file format.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, filename=filename)
        split_filename = filename.split(".")
        if len(split_filename) != 2:
            raise ConfigurationError("The given filename {0} contains more than one '.'.".format(filename))
        # Include python implementation and version (implementation_major_minor_macro in filename)
        dumping_filename = (split_filename[0] + "_" + sys.implementation.name + "_" +
                            "_".join([str(sys.version_info[i]) for i in range(3)])
                            + "." + split_filename[1])

        super().__init__(dumping_filename)

    def write(self, mediator: Mediator) -> None:
        """
        Dump the mediator, the state of the random module and the setting package into the dumping file.

        Parameters
        ----------
        mediator : mediator.Mediator
            The mediator.
        """
        print("Writing dump into file {0}".format(self._output_filename))
        with open(self._output_filename, "wb") as file:
            dill.dump([mediator, setting, random.getstate()], file)

    def post_run(self) -> None:
        """Clean up the output handler."""
        pass
