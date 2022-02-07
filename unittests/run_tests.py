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
"""Executable script which runs all unittests."""
import importlib
import os
import unittest
import logging
import jellyfysh.run as run


if __name__ == '__main__':
    # Print the start message of the run module.
    run.print_start_message()

    # If necessary, change working directory to the directory of this file so that all tests can be located.
    if os.path.dirname(__file__) != '':
        os.chdir(os.path.dirname(__file__))

    # Suppress logging.
    logging.disable(logging.ERROR)

    # First run all tests except the ones in test_config_files.
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    test_suite.addTests(loader.loadTestsFromModule(importlib.import_module("test_expanded_test_case")))
    for entry in os.scandir():
        if (entry.is_dir() and entry.name != "test_config_files"
                and os.path.isfile(entry.name + "/__init__.py")):
            test_suite.addTests(loader.discover(entry.name, top_level_dir="."))
    runner = unittest.TextTestRunner()
    runner.run(test_suite)

    # Then run all tests in test_config_files.
    config_file_test_suite = loader.discover("test_config_files", top_level_dir=".")
    runner.run(config_file_test_suite)
