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
"""Module for the InputOutputHandler class."""
import logging
from typing import Any, List, Sequence
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.factory import get_alias
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.base.strings import to_snake_case
from .input_handler import InputHandler
from .output_handler import OutputHandler
from .output_handler.dummy_output_handler import DummyOutputHandler


class InputOutputHandler(object):
    """
    The input-output handler which connects the mediator to the outside world.

    It breaks up into a single input handler and a possible empty sequence of output handlers. The input handler enters
    the initial global physical state into the application. The output handlers serve many purposes, from the output
    of the global state to the sampling and the dumping of the entire run.
    """

    def __init__(self, input_handler: InputHandler, output_handlers: Sequence[OutputHandler] = ()) -> None:
        """
        The constructor of the InputOutputHandler class.

        Parameters
        ----------
        input_handler : input_output_handler.input_handler.InputHandler
            The input handler.
        output_handlers : Sequence[input_output_handler.output_handler.OutputHandler]
            The sequence of output handlers.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__,
                           input_handler=input_handler.__class__.__name__,
                           output_handlers=[output_handler.__class__.__name__ for output_handler in output_handlers])
        self._input_handler = input_handler
        # Event Handlers may refer to the alias (if there is one) of the .ini file which is included in the
        # __class__.__name__ property in the factory. get_alias() extracts this alias
        self._output_handlers_dictionary = {to_snake_case(get_alias(output_handler.__class__.__name__)): output_handler
                                            for output_handler in output_handlers}

    def read(self) -> Any:
        """
        Return the initial global physical state created by the input handler.

        The precise format of the returned object depends on the input handler and is passed to the initialize method
        of the state handler.

        Returns
        -------
        Any
            The initial global physical state.
        """
        return self._input_handler.read()

    def write(self, output_handler: str, *args: Any) -> None:
        """
        Pass the arguments to the output handler.

        The arguments could be for example the full global state so that the output handler can start its sampling.
        This method is called in the mediating methods of event handlers in the mediator. There, each event handler
        defines itself, which methods should be passed to its output handler.
        When the output handlers were built using the JF factory, their names can include an alias. Then the output
        handler name should be this alias.

        Parameters
        ----------
        output_handler : str
            The name of the output handler which should receive the arguments.
        args : Any
            The arguments passed to output handler.

        Raises
        ------
        base.exceptions.ConfigurationError
            If no output handler with the given name exists.
        """
        try:
            self._output_handlers_dictionary[output_handler].write(*args)
        except KeyError:
            raise ConfigurationError("The given output handler {0} does not exist.".format(output_handler))

    def post_run(self) -> None:
        """
        Call the post_run method of all output handlers.

        This method is called by the mediator at the end of a run.
        """
        for output_handler in self._output_handlers_dictionary.values():
            output_handler.post_run()

    def deactivate_output(self) -> None:
        """
        Replace all output handlers with an instance of a DummyOutputHandler.

        These have no output. This method is used in debugging mode in resume.py.
        """
        for name in self._output_handlers_dictionary.keys():
            self._output_handlers_dictionary[name] = DummyOutputHandler()

    @property
    def output_handlers(self) -> List[str]:
        """
        Return the sequence of names of all output handlers.

        Returns
        -------
        List[str]
            The names of all output handlers.
        """
        return list(self._output_handlers_dictionary.keys())
