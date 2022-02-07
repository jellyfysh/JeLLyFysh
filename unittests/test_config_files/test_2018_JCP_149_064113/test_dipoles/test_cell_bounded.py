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
from configparser import ConfigParser
import contextlib
import os
from pkg_resources import resource_filename
import unittest
from unittest import mock
import sys
from jellyfysh.activator.tagger.factor_type_maps import FactorTypeMaps
import jellyfysh.run as run
import jellyfysh.setting as setting


class TestCellBounded(unittest.TestCase):
    def setUp(self) -> None:
        self._config = ConfigParser()
        self._ini_file = resource_filename(
            "jellyfysh", "config_files/2018_JCP_149_064113/dipoles/cell_bounded.ini")
        if not self._config.read(self._ini_file):
            self.fail("Could not read the ini file {0}.".format(self._ini_file))
        # Reduce end of run time
        self._config.set("FinalTimeEndOfRunEventHandler", "end_of_run_time", "10")
        # Replace output file
        self._config.set("SeparationOutputHandler", "filename", "TestCellBounded.dat")
        # Set factor type maps file
        self._config.set("FactorTypeMaps", "filename",
                         resource_filename("jellyfysh", self._config.get("FactorTypeMaps", "filename")))

    def tearDown(self) -> None:
        if "TestCellBounded_13.dat" in os.listdir("."):
            os.remove("TestCellBounded_13.dat")
        if "TestCellBounded_14.dat" in os.listdir("."):
            os.remove("TestCellBounded_14.dat")
        if "TestCellBounded_13.dat.tmp" in os.listdir("."):
            os.remove("TestCellBounded_13.dat.tmp")
        if "TestCellBounded_14.dat.tmp" in os.listdir("."):
            os.remove("TestCellBounded_14.dat.tmp")
        # Reset setting
        setting.reset()
        # Reset factor type maps singleton
        FactorTypeMaps._instance = None

    @mock.patch("jellyfysh.run.read_config")
    def test_run_cell_bounded(self, read_config_mock):
        sys.argv[1:] = [self._ini_file]
        read_config_mock.return_value = self._config
        print("\nTest if the .ini file {0} runs without an exception...".format(self._ini_file), end="")
        # Redirect stdout to the null device
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                run.main()
        read_config_mock.assert_called_once_with(self._ini_file)
        self.assertIn("TestCellBounded_13.dat", os.listdir("."))
        self.assertIn("TestCellBounded_14.dat", os.listdir("."))


if __name__ == '__main__':
    unittest.main()
