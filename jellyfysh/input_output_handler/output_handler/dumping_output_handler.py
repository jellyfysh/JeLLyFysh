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
"""Module for the DumpingOutputHandler class."""
import logging
import random
import sys
import cloudpickle
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.logging import log_init_arguments
import jellyfysh.base.uuid as uuid
from jellyfysh.mediator import Mediator
import jellyfysh.setting as setting
from .output_handler import OutputHandler


class DumpingOutputHandler(OutputHandler):
    """
    Output handler which dumps an entire run into a file.

    The run is dumped by dumping the mediator, the state of the random module and the setting package to a file using
    the cloudpickle package. The file can be used in resume.py to resume the run starting from the dumped configuration.
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

        Raises
        ------
        RuntimeError
            If the argument is not a mediator.
        """
        print("Writing dump into file {0}".format(self._output_filename))
        if not isinstance(mediator, Mediator):
            raise RuntimeError("The argument of the 'write' method of the class {0} is not the required mediator "
                               "object. Make sure that this output handler is connected to the DumpingEventHandler."
                               .format(self.__class__.__name__))
        with open(self._output_filename, "wb") as file:
            cloudpickle.dump([mediator, setting.getstate(), uuid.getstate(), random.getstate()], file)
        # sys.exit(2)

    def post_run(self) -> None:
        """Clean up the output handler."""
        pass
