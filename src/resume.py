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
"""Executable script which runs the JeLLyFysh application based on a dump file created by the DumpingOutputHandler."""
from argparse import ArgumentParser, Namespace
import platform
import random
import sys
import time
from typing import Sequence
import dill
from base.exceptions import EndOfRun
from base.uuid import get_uuid
from run import add_general_parser_arguments, print_start_message, set_up_logging
import setting


def parse_options(args: Sequence[str]) -> Namespace:
    """
    Convert argument strings to objects and assign them as attributes of the argparse namespace. Return the populated
    namespace.

    The argument strings can for example be sys.argv[1:] in order to parse the command line arguments. The argument
    parser parses the arguments specified in the add_general_parser_arguments function in run.py. This function also
    adds the dumping file as an obligatory positional argument. If also adds the --no-output option, which replaces
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
    parser = ArgumentParser(description="Resume from dumped data")
    parser.add_argument("dumping_file", help="Specify dumped data file")
    parser.add_argument('--no-output', action='store_true', help='Deactivate output')
    add_general_parser_arguments(parser)
    return parser.parse_args(args)


def main(argv: Sequence[str]) -> None:
    """
    Use the argument strings to resume a dumped run of the JeLLyFysh application.

    First the argument strings are parsed, and then the logging is set up. The dumping file specified in the
    argument strings is parsed.
    Based on the dumping file, the setting package is initialized, the mediator is restored and the state of the random
    module is set.
    The run method of the mediator is executed until an EndOfRun exception is raised. This invokes the post_run method
    of the mediator and ends the resumed run of the application.

    Parameters
    ----------
    argv : Sequence[str]
        The argument strings.
    """
    args = parse_options(argv)
    logger = set_up_logging(args)

    logger.info("Run identification hash: {0}".format(get_uuid()))
    logger.info("Underlying platform (determined via platform.platform(aliased=True): {0}"
                .format(platform.platform(aliased=True)))

    logger.info("Resuming run based on the dumping file {0}.".format(args.dumping_file))
    with open(args.dumping_file, "rb") as file:
        mediator, dumped_setting, dumped_random_state = dill.load(file)
    mediator.update_logging()
    setting.__dict__.update(dumped_setting.__dict__)
    random.setstate(dumped_random_state)

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
    print_start_message()
    main(sys.argv[1:])
