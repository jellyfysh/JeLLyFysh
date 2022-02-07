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
"""Executable script which runs the JeLLyFysh application based on a dump file created by the DumpingOutputHandler."""
from argparse import ArgumentParser, Namespace
import platform
import random
import sys
import time
from typing import Sequence
import dill
from jellyfysh.base.exceptions import EndOfRun
import jellyfysh.base.uuid as uuid
from jellyfysh.run import add_general_parser_arguments, print_start_message, set_up_logging
import jellyfysh.setting as setting


def parse_options(args: Sequence[str]) -> Namespace:
    """
    Convert argument strings to objects and assign them as attributes of the argparse namespace. Return the populated
    namespace.

    The argument strings can for example be sys.argv[1:] in order to parse the command line arguments. The argument
    parser parses the arguments specified in the add_general_parser_arguments function in run.py. This function also
    adds the dumping file as a required positional argument. If also adds the --no-output option, which replaces
    every output handler by a dummy output handler, which just outputs nothing.

    Parameters
    ----------
    args : Sequence[str]
        The argument strings.

    Returns
    -------
    argparse.Namespace
        The populated argparse namespace.
    """
    parser = ArgumentParser(description="Resume a run of the JeLLyFysh application based on a dumped data file.")
    parser.add_argument("dumping_file", help="specify the dumped data file")
    parser.add_argument('--no-output', action='store_true', help='deactivate output of all output handlers')
    add_general_parser_arguments(parser)
    return parser.parse_args(args)


def main() -> None:
    """
    Use the command line argument to resume a dumped run of the JeLLyFysh application.

    First the command line arguments are parsed, and then the logging is set up. The dumping file specified in the
    argument strings is then read using dill.
    Based on the dumping file, the setting package is initialized, the mediator is restored and the state of the random
    module is set.
    The run method of the mediator is executed until an EndOfRun exception is raised. This invokes the post_run method
    of the mediator and ends the resumed run of the application.
    """
    print_start_message()

    args = parse_options(sys.argv[1:])
    logger = set_up_logging(args)

    logger.info("Resuming run based on the dumping file {0}.".format(args.dumping_file))
    with open(args.dumping_file, "rb") as file:
        mediator, dumped_setting, dumped_uuid, dumped_random_state = dill.load(file)
    mediator.update_logging()
    setting.__dict__.update(dumped_setting.__dict__)
    uuid.__dict__.update(dumped_uuid.__dict__)
    random.setstate(dumped_random_state)

    logger.info("Run identification hash: {0}".format(uuid.get_uuid()))
    logger.info("Underlying platform (determined via platform.platform(aliased=True): {0}"
                .format(platform.platform(aliased=True)))

    if args.no_output:
        logger.info("Deactivating output.")
        mediator.deactivate_output()

    logger.info("Resuming the event-chain Monte Carlo simulation.")
    start_time = time.time()
    try:
        mediator.run()
    except EndOfRun:
        logger.info("EndOfRun exception has been raised.")
    end_time = time.time()

    logger.info("Running the post_run method.")
    mediator.post_run()
    logger.info("Runtime of the resumed simulation: --- %s seconds ---" % (end_time - start_time))


if __name__ == '__main__':
    main()
