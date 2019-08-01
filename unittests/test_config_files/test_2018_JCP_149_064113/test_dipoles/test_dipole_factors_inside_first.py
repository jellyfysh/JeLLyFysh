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
from configparser import ConfigParser
import contextlib
import os
import unittest
from unittest import mock
from activator.tagger.factor_type_maps import FactorTypeMaps
import run
import setting


_src_directory = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../../../../src/")
_current_working_directory = os.getcwd()


def setUpModule():
    # Change to the source directory as the current working directory so that the factory finds all relevant files
    os.chdir(_src_directory)


def tearDownModule():
    # Revert everything which was done in setUpModule
    os.chdir(_current_working_directory)


class TestDipoleFactorsInsideFirst(unittest.TestCase):
    def setUp(self) -> None:
        self._config = ConfigParser()
        self._ini_file = "config_files/2018_JCP_149_064113/dipoles/dipole_factors_inside_first.ini"
        if not self._config.read(self._ini_file):
            self.fail("Could not read the ini file {0}.".format(self._ini_file))
        # Reduce end of run time
        self._config.set("FinalTimeEndOfRunEventHandler", "end_of_run_time", "10")
        # Replace output file
        self._config.set("SeparationOutputHandler", "filename", "TestDipoleFactorsInsideFirst.dat")

    def tearDown(self) -> None:
        # Use class method to make sure files are deleted even when a test fails
        if "TestDipoleFactorsInsideFirst_13.dat" in os.listdir("."):
            os.remove("TestDipoleFactorsInsideFirst_13.dat")
        if "TestDipoleFactorsInsideFirst_14.dat" in os.listdir("."):
            os.remove("TestDipoleFactorsInsideFirst_14.dat")
        if "TestDipoleFactorsInsideFirst_13.dat.tmp" in os.listdir("."):
            os.remove("TestDipoleFactorsInsideFirst_13.dat.tmp")
        if "TestDipoleFactorsInsideFirst_14.dat.tmp" in os.listdir("."):
            os.remove("TestDipoleFactorsInsideFirst_14.dat.tmp")
        # Reset setting
        setting.reset()
        # Reset factor type maps singleton
        FactorTypeMaps._instance = None

    @mock.patch("run.read_config")
    def test_run_dipole_factors_inside_first(self, read_config_mock):
        argv = [self._ini_file]
        read_config_mock.return_value = self._config
        print("\nTest if the .ini file {0} runs without an exception...".format(self._ini_file), end="")
        # Redirect stdout to the null device
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                run.main(argv)

        self.assertIn("TestDipoleFactorsInsideFirst_13.dat", os.listdir("."))
        self.assertIn("TestDipoleFactorsInsideFirst_14.dat", os.listdir("."))


if __name__ == '__main__':
    unittest.main()
