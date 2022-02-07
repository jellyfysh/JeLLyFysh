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
from argparse import Namespace
from configparser import ConfigParser
import contextlib
import logging
import os
from pkg_resources import resource_filename
import unittest
from unittest import mock, SkipTest
import sys
from jellyfysh.activator.tagger.factor_type_maps import FactorTypeMaps
import jellyfysh.run as run
import jellyfysh.setting as setting


class TestHardDiskDipolesCells(unittest.TestCase):
    def setUp(self) -> None:
        self._config = ConfigParser()
        self._ini_file = resource_filename(
            "jellyfysh", "config_files/hard_disk_dipoles/hard_disk_dipoles_cells.ini")
        if not self._config.read(self._ini_file):
            self.fail("Could not read the ini file {0}.".format(self._ini_file))
        # Reduce end of run time
        self._config.set("FinalTimeEndOfRunEventHandler", "end_of_run_time", "10")
        # Replace output file
        self._config.set("PolarizationOutputHandler", "filename", "TestHardDiskDipolesCells.dat")
        # Set factor type maps file
        self._config.set("FactorTypeMaps", "filename",
                         resource_filename("jellyfysh", self._config.get("FactorTypeMaps", "filename")))
        # Set initial configuration file
        self._config.set("PdbInputHandler", "filename",
                         resource_filename("jellyfysh", self._config.get("PdbInputHandler", "filename")))

    def tearDown(self) -> None:
        if "TestHardDiskDipolesCells.dat" in os.listdir("."):
            os.remove("TestHardDiskDipolesCells.dat")
        if "TestHardDiskDipolesCells.dat.tmp" in os.listdir("."):
            os.remove("TestHardDiskDipolesCells.dat.tmp")
        # Reset setting
        setting.reset()
        # Reset factor type maps singleton
        FactorTypeMaps._instance = None

    @mock.patch("jellyfysh.run.read_config")
    @mock.patch("jellyfysh.run.set_up_logging")
    def test_run_power_bounded(self, set_up_logging_mock, read_config_mock):
        sys.argv[1:] = [self._ini_file]
        logger = logging.getLogger("")
        logger.setLevel(logging.ERROR)
        set_up_logging_mock.return_value = logger
        read_config_mock.return_value = self._config
        try:
            from jellyfysh.input_output_handler.input_handler.pdb_input_handler import PdbInputHandler
        except ImportError:
            print("\nSkip test if the .ini file {0} runs without an exception "
                  "because MDAnalysis is not installed...".format(self._ini_file), end="")
            raise SkipTest(
                "Skip unittests for hard-disk dipoles with cells configuration file "
                "because MDAnalysis is not installed.")
        print("\nTest if the .ini file {0} runs without an exception...".format(self._ini_file), end="")
        # Redirect stdout to the null device
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                run.main()
        set_up_logging_mock.assert_called_once_with(Namespace(config_file=self._ini_file, logfile=None, verbose=None))
        read_config_mock.assert_called_once_with(self._ini_file)
        self.assertIn("TestHardDiskDipolesCells.dat", os.listdir("."))


if __name__ == '__main__':
    unittest.main()
