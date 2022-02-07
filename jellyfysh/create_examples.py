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
"""Executable script which copies the exemplary configuration files of the JeLLyFysh application into the current
working directory."""
from argparse import ArgumentParser, Namespace
from pkg_resources import resource_filename
from os import mkdir
import platform
from shutil import copytree
import sys
from typing import Sequence
from jellyfysh.run import add_general_parser_arguments, print_start_message, set_up_logging


def parse_options(args: Sequence[str]) -> Namespace:
    """
    Convert argument strings to objects and assign them as attributes of the argparse namespace. Return the populated
    namespace.

    The argument strings can for example be sys.argv[1:] in order to parse the command line arguments. The argument
    parser parses the arguments specified in the add_general_parser_arguments function in run.py.

    Parameters
    ----------
    args : Sequence[str]
        The argument strings.

    Returns
    -------
    argparse.Namespace
        The populated argparse namespace.
    """
    parser = ArgumentParser(description="Copy the exemplary configuration files of the JeLLyFysh application into the "
                                        "current working directory.")
    add_general_parser_arguments(parser)
    return parser.parse_args(args)


def main() -> None:
    """
    Copy the exemplary configuration files of the JeLLyFysh application into the jellyfysh-examples directory that will
    be created in the current working directory.

    The command line arguments set the the logging level and the logging file, and further determine whether only the
    version or help window should be printed.
    The data files are accessed using the resource management API of setuptools.
    """
    print_start_message()

    args = parse_options(sys.argv[1:])
    logger = set_up_logging(args)

    logger.info("Underlying platform (determined via platform.platform(aliased=True): {0}"
                .format(platform.platform(aliased=True)))

    print("Creating directory jellyfysh-examples.")
    mkdir("jellyfysh-examples")

    config_files_directory = resource_filename(__name__, "config_files")
    logger.debug("Detected path of the config_files directory: {0}".format(config_files_directory))
    print("Copying the configuration files into the directory jellyfysh-examples/config_files.")
    copytree(config_files_directory, "jellyfysh-examples/config_files")

    output_directory = resource_filename(__name__, "output")
    logger.debug("Detected path of the output directory: {0}".format(output_directory))
    print("Copying the reference data and plotting scripts into the directory jellyfysh-examples/output.")
    copytree(output_directory, "jellyfysh-examples/output")


if __name__ == '__main__':
    main()
