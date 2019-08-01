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
"""Executable script which runs all unittests."""
import importlib
import os
import sys
import unittest
import logging


if __name__ == '__main__':
    # Add the src directory to sys.path so that all imports in the unittests work
    this_directory = os.path.dirname(os.path.abspath(__file__))
    src_directory = os.path.abspath(this_directory + "/../src/")
    sys.path.insert(0, src_directory)

    # Print the start message of the run module
    run_module = importlib.import_module("run")
    run_module.print_start_message()

    # Suppress logging
    logging.disable(logging.ERROR)

    # First run all tests except the ones in test_config_files
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    for entry in os.scandir(this_directory):
        if (entry.is_dir() and entry.name != "test_config_files"
                and os.path.isfile(this_directory + "/" + entry.name + "/__init__.py")):
            test_suite.addTests(loader.discover(this_directory + "/" + entry.name, top_level_dir=this_directory))
    runner = unittest.TextTestRunner()
    runner.run(test_suite)

    # Then run all tests in test_config_files
    config_file_test_suite = loader.discover(this_directory + "/test_config_files", top_level_dir=this_directory)
    runner.run(config_file_test_suite)
