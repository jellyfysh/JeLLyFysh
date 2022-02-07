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
"""Executable script which runs the JeLLyFysh application based on a configuration file."""
from argparse import ArgumentParser, Namespace
from configparser import ConfigParser
import logging
import platform
import sys
import time
from typing import Sequence
from jellyfysh.base.exceptions import EndOfRun
from jellyfysh.base import factory
from jellyfysh.base.strings import to_camel_case
from jellyfysh.base.uuid import get_uuid
import jellyfysh.version as version


def add_general_parser_arguments(parser: ArgumentParser) -> None:
    """
    Add parser arguments to the command line argument parser.

    This method adds the following possible arguments:
    1. --version, -V: Print the version of the application and exit.
    2. --verbose, -v: Increase verbosity of logging messages. Multiple -v options increase the verbosity. The maximum
    is 2.
    3. --logfile LOGFILE, -l LOGFILE: Specify the logging file.
    Per default, also the following argument is added:
    4. --help, -h: Show the help message and exit.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser.
    """
    parser.add_argument("-V", "--version", action="version", version="JeLLyFysh application version "
                                                                     + version.__version__)
    parser.add_argument("-v", "--verbose", action="count",
                        help="increase verbosity of logging messages "
                             "(multiple -v options increase the verbosity, the maximum is 2)")
    parser.add_argument("-l", "--logfile", action="store", help="specify the logging file")


def parse_options(args: Sequence[str]) -> Namespace:
    """
    Convert argument strings to objects and assign them as attributes of the argparse namespace. Return the populated
    namespace.

    The argument strings can for example be sys.argv[1:] in order to parse the command line arguments. The argument
    parser parses the arguments specified in the add_general_parser_arguments function. This function also adds the
    configuration file as a required positional argument.

    Parameters
    ----------
    args : Sequence[str]
        The argument strings.

    Returns
    -------
    argparse.Namespace
        The populated argparse namespace.
    """
    parser = ArgumentParser(description="Run the JeLLyFysh application based on a configuration file.")
    parser.add_argument("config_file", help="specify the path to the configuration file of the run")
    add_general_parser_arguments(parser)
    return parser.parse_args(args)


def set_up_logging(parsed_arguments: Namespace) -> logging.Logger:
    """
    Set up the logging based on the populated argparse namespace.

    The level of the root logger is set based on the number of -v arguments parsed. For zero -v arguments, it is set
    to logging.WARNING, for one to logging.INFO and for two to logging.DEBUG. If a log file was specified in the
    parsed arguments, the logging information is written into that file.

    Parameters
    ----------
    parsed_arguments : argparse.Namespace
        The populated argparse namespace.

    Returns
    -------
    logging.Logger
        The initialized root logger.
    """
    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)

    if parsed_arguments.verbose is None:
        logger_level = logging.WARNING
    elif parsed_arguments.verbose == 1:
        logger_level = logging.INFO
    else:
        logger_level = logging.DEBUG

    if parsed_arguments.logfile is not None:
        handler = logging.FileHandler(parsed_arguments.logfile)
    else:
        handler = logging.StreamHandler()
    logger.setLevel(logger_level)
    handler.setLevel(logger_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Don't collect information about where calls were made from (slows down PyPy)
    logging._srcfile = None
    return logger


def read_config(config_file: str) -> ConfigParser:
    """
    Parse the configuration file.

    Parameters
    ----------
    config_file : str
        The filename of the configuration file.

    Returns
    -------
    configparser.ConfigParser
        The parsed configuration file.

    Raises
    ------
    RuntimeError
        If the configuration file does not exist.
    """
    config = ConfigParser()
    if not config.read(config_file):
        raise RuntimeError("Given configuration file does not exist.")
    return config


def print_start_message() -> None:
    """Print the start message which includes the copyright."""
    print("JeLLFysh (version {0}) - a Python application for all-atom event-chain Monte Carlo - "
          "https://github.com/jellyfysh".format(version.__version__))
    print("Copyright (C) 2019, 2022 The JeLLyFysh organization")


def main() -> None:
    """
    Use the command line arguments to run the JeLLyFysh application.

    First the command line arguments are parsed, and then the logging is set up. Afterwards, the configuration file
    specified in the command line is parsed. Based on the configuration file, the setting package is initialized and the
    mediator is constructed by the JeLLyFysh factory.
    The run method of the mediator is executed until an EndOfRun exception is raised. This invokes the post_run method
    of the mediator and ends the run of the application.
    """
    print_start_message()

    args = parse_options(sys.argv[1:])
    logger = set_up_logging(args)

    logger.info("Run identification hash: {0}".format(get_uuid()))
    logger.info("Underlying platform (determined via platform.platform(aliased=True): {0}"
                .format(platform.platform(aliased=True)))

    logger.info("Setting up the run based on the configuration file {0}.".format(args.config_file))
    config = read_config(args.config_file)
    factory.build_from_config(config, to_camel_case(config.get("Run", "setting")), "jellyfysh.setting")
    mediator = factory.build_from_config(config, to_camel_case(config.get("Run", "mediator")),
                                         "jellyfysh.mediator")
    used_sections = factory.used_sections
    for section in config.sections():
        if section not in used_sections and section != "Run":
            logger.warning("The section {0} in the .ini file has not been used!".format(section))

    logger.info("Running the event-chain Monte Carlo simulation.")
    start_time = time.time()
    try:
        mediator.run()
    except EndOfRun:
        logger.info("EndOfRun exception has been raised.")
    end_time = time.time()

    logger.info("Running the post_run method.")
    mediator.post_run()
    logger.info("Runtime of the simulation: --- %s seconds ---" % (end_time - start_time))


if __name__ == '__main__':
    main()
