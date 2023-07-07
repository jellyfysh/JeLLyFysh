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
"""Module for the abstract OutputHandler class."""
from abc import ABCMeta, abstractmethod
import os
import shutil
from typing import Any, MutableMapping
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.uuid import get_uuid


class OutputHandler(metaclass=ABCMeta):
    """
    Abstract class for an output handler used in the input-output handler.

    An output handler can serve many purposes, from the output of the global state to the sampling and the dumping of
    the entire run. It is always connected to a file to which it writes. The output file should always include the run
    identification hash which is returned by the get_uuid method in the base.uuid module.
    """

    def __init__(self, output_filename: str) -> None:
        """
        The constructor of the OutputHandler class.

        Parameters
        ----------
        output_filename : str
            The filename of the file this output handler is connected to.

        Raises
        ------
        base.exceptions.ConfigurationError
            If the filename contains a directory path which does not exist.
        """
        self._output_filename = output_filename
        self._counter = 0
        directory_name = os.path.dirname(output_filename)
        if directory_name and not os.path.isdir(directory_name):
            raise ConfigurationError("Output directory '{0}' does not exist.".format(directory_name))

    @abstractmethod
    def write(self, *args: Any) -> None:
        """
        Use the arguments to store output information.

        The arguments could be, for example, the full global state so that the output handler can start its sampling.
        This method is called via the input-output handler in the mediating methods of event handlers in the mediator.
        There, each event handler defines itself, which methods should be passed to this output handler.
        This method increases an internal counter and prints a message after 100 calls. Overwrite this method, if
        this behaviour is unwanted.

        Parameters
        ----------
        args : Any
            The arguments.
        """
        self._counter += 1
        if self._counter % 100 == 0:
            print("{0}: Calculated {1} samples.".format(self.__class__.__name__, self._counter))

    @abstractmethod
    def post_run(self) -> None:
        """
        Clean up the output handler.

        Here, for example, the stored information could be written into the file and the file should be closed.
        This method is called by the mediator via the input-output handler at the end of a run.
        """
        raise NotImplementedError


class HardBufferedTextWriter(object):
    """
    A special object which implements a write method needed for the print method of python.

    An instance of this object can be used as the file kwarg of the print method.
    This object writes to a temporary file (filename + '.tmp'). When close is called, the temporary file is renamed to
    the original filename. This file includes the run identification hash.
    """

    def __init__(self, filename: str) -> None:
        """
        The constructor of the HardBufferedTextWriter class.

        This method opens the temporary file and writes the run identification hash into it.

        Parameters
        ----------
        filename : str
            The filename.
        """
        self._filename = filename
        self._tmp_file = open(filename + ".tmp", "w")
        print("# Run identification hash: {0}".format(get_uuid()), file=self)

    def write(self, string: str) -> None:
        """
        Write the string into the temporary file.

        Parameters
        ----------
        string : str
            The string.
        """
        self._tmp_file.write(string)

    def close(self) -> None:
        """Close the temporary file and rename it into the original filename."""
        self._tmp_file.close()
        os.rename(self._filename + ".tmp", self._filename)

    def __getstate__(self) -> MutableMapping[str, Any]:
        """
        Return a state of this class that can be pickled.

        This method copies the current content of the temporary file to a dump file. For the dump file, the .tmp file
        suffix of the temporary file is replaced by .dump. The self._tmp_file attribute is then removed from the
        self.__dict__ dictionary so that it can be pickled (because writable files cannot be pickled).

        Returns
        -------
        MutableMapping[str, Any]
            The state that can be pickled.
        """
        self._tmp_file.flush()
        shutil.copy2(self._filename + ".tmp", self._filename + ".dump")
        state = self.__dict__.copy()
        del state["_tmp_file"]
        return state

    def __setstate__(self, state: MutableMapping[str, Any]) -> None:
        """
        Use the state dictionary to initialize this class.

        This method copies the dump file, that was created in the __getstate__ method, to a new temporary file. This
        temporary file is then reopened and stored in the self._tmp_file attribute.

        Parameters
        ----------
        state : MutableMapping[str, Any]
            The state.

        Raises
        ------
        If the dump file with the file suffix .dump does not exist.
        """
        self.__dict__.update(state)
        assert os.path.exists(self._filename + ".dump")
        shutil.copy2(self._filename + ".dump", self._filename + ".tmp")
        self._tmp_file = open(self._filename + ".tmp", "a")
