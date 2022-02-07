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
from unittest import TestCase, main, SkipTest
import logging
import os
import warnings
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.input_output_handler.input_handler.charge_values import ChargeValues
import jellyfysh.setting as setting
from jellyfysh.setting import hypercubic_setting
try:
    from jellyfysh.input_output_handler.input_handler.pdb_input_handler import PdbInputHandler
except ImportError:
    raise SkipTest("Skip unittests for PdbInputHandler because MDAnalysis is not installed.")

_this_directory = os.path.dirname(os.path.abspath(__file__))
_current_working_directory = os.getcwd()


def setUpModule():
    # Change to this directory as the current working directory so that the pdb input handler finds the test .pdb files
    os.chdir(_this_directory)


def tearDownModule():
    # Revert everything which was done in setUpModule
    os.chdir(_current_working_directory)


class TestPdbInputHandler(TestCase):
    def tearDown(self) -> None:
        # Make sure that the setting module is reset even when a test fails
        setting.reset()

    def test_pdb_file_water_origin_zero(self):
        # Origin of system box in pdb file at [0.0, 0.0, 0.0]
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=10.261)
        electric_charge_values = ChargeValues([0.41, -0.82, 0.41], "electric_charge")
        oxygen_indicator_charge = ChargeValues([0.0, 1.0, 0.0], "oxygen_indicator")
        pdb_input_handler = PdbInputHandler("pdb_test_files/water_test_origin_0.pdb",
                                            [electric_charge_values, oxygen_indicator_charge])
        state = pdb_input_handler.read()
        self.assertEqual(len(state), 4)

        water_one = state[0]
        self.assertIsNone(water_one.parent)
        self.assertEqual(water_one.weight, 1.0)
        self.assertIsNone(water_one.value.charge)
        self.assertEqual(len(water_one.value.position), 3)
        # MDAnalysis only uses c floats (not doubles) -> reduce precision
        self.assertAlmostEqual(water_one.value.position[0], 8.43866666666667, places=6)
        self.assertAlmostEqual(water_one.value.position[1], 5.736333333333333, places=6)
        self.assertAlmostEqual(water_one.value.position[2], 1.384333333333333, places=6)
        self.assertEqual(len(water_one.children), 3)
        h_one = water_one.children[0]
        self.assertIs(h_one.parent, water_one)
        self.assertEqual(h_one.weight, 1.0 / 3.0)
        self.assertEqual(h_one.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_one.value.position), 3)
        self.assertAlmostEqual(h_one.value.position[0], 7.859, places=6)
        self.assertAlmostEqual(h_one.value.position[1], 5.311, places=6)
        self.assertAlmostEqual(h_one.value.position[2], 0.879, places=6)
        self.assertEqual(len(h_one.children), 0)
        o_one = water_one.children[1]
        self.assertIs(o_one.parent, water_one)
        self.assertEqual(o_one.weight, 1.0 / 3.0)
        self.assertEqual(o_one.value.charge, {"electric_charge": -0.82, "oxygen_indicator": 1.0})
        self.assertEqual(len(o_one.value.position), 3)
        self.assertAlmostEqual(o_one.value.position[0], 8.663, places=6)
        self.assertAlmostEqual(o_one.value.position[1], 5.863, places=6)
        self.assertAlmostEqual(o_one.value.position[2], 1.139, places=6)
        self.assertEqual(len(o_one.children), 0)
        h_two = water_one.children[2]
        self.assertIs(h_two.parent, water_one)
        self.assertEqual(h_two.weight, 1.0 / 3.0)
        self.assertEqual(h_two.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_two.value.position), 3)
        self.assertAlmostEqual(h_two.value.position[0], 8.794, places=6)
        self.assertAlmostEqual(h_two.value.position[1], 6.035, places=6)
        self.assertAlmostEqual(h_two.value.position[2], 2.135, places=6)
        self.assertEqual(len(h_two.children), 0)

        water_two = state[1]
        self.assertIsNone(water_two.parent)
        self.assertEqual(water_two.weight, 1.0)
        self.assertIsNone(water_two.value.charge)
        self.assertEqual(len(water_two.value.position), 3)
        self.assertAlmostEqual(water_two.value.position[0], 10.119, places=6)
        self.assertAlmostEqual(water_two.value.position[1], 1.288666666666666, places=6)
        self.assertAlmostEqual(water_two.value.position[2], 10.17866666666667, places=6)
        self.assertEqual(len(water_two.children), 3)
        h_one = water_two.children[0]
        self.assertIs(h_one.parent, water_two)
        self.assertEqual(h_one.weight, 1.0 / 3.0)
        self.assertEqual(h_one.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_one.value.position), 3)
        self.assertAlmostEqual(h_one.value.position[0], 9.969, places=6)
        self.assertAlmostEqual(h_one.value.position[1], 1.519, places=6)
        self.assertAlmostEqual(h_one.value.position[2], 9.446, places=6)
        self.assertEqual(len(h_one.children), 0)
        o_one = water_two.children[1]
        self.assertIs(o_one.parent, water_two)
        self.assertEqual(o_one.weight, 1.0 / 3.0)
        self.assertEqual(o_one.value.charge, {"electric_charge": -0.82, "oxygen_indicator": 1.0})
        self.assertEqual(len(o_one.value.position), 3)
        self.assertAlmostEqual(o_one.value.position[0], 10.068, places=6)
        self.assertAlmostEqual(o_one.value.position[1], 1.315, places=6)
        self.assertAlmostEqual(o_one.value.position[2], 0.160, places=6)
        self.assertEqual(len(o_one.children), 0)
        h_two = water_two.children[2]
        self.assertIs(h_two.parent, water_two)
        self.assertEqual(h_two.weight, 1.0 / 3.0)
        self.assertEqual(h_two.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_two.value.position), 3)
        self.assertAlmostEqual(h_two.value.position[0], 0.059, places=6)
        self.assertAlmostEqual(h_two.value.position[1], 1.032, places=6)
        self.assertAlmostEqual(h_two.value.position[2], 0.408, places=6)
        self.assertEqual(len(h_two.children), 0)

        water_three = state[2]
        self.assertIsNone(water_three.parent)
        self.assertEqual(water_three.weight, 1.0)
        self.assertIsNone(water_three.value.charge)
        self.assertEqual(len(water_three.value.position), 3)
        self.assertAlmostEqual(water_three.value.position[0], 2.045, places=6)
        self.assertAlmostEqual(water_three.value.position[1], 0.3630000000000001, places=6)
        self.assertAlmostEqual(water_three.value.position[2], 0.5586666666666666, places=6)
        self.assertEqual(len(water_three.children), 3)
        h_one = water_three.children[0]
        self.assertIs(h_one.parent, water_three)
        self.assertEqual(h_one.weight, 1.0 / 3.0)
        self.assertEqual(h_one.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_one.value.position), 3)
        self.assertAlmostEqual(h_one.value.position[0], 2.429, places=6)
        self.assertAlmostEqual(h_one.value.position[1], 10.081, places=6)
        self.assertAlmostEqual(h_one.value.position[2], 0.100, places=6)
        self.assertEqual(len(h_one.children), 0)
        o_one = water_three.children[1]
        self.assertIs(o_one.parent, water_three)
        self.assertEqual(o_one.weight, 1.0 / 3.0)
        self.assertEqual(o_one.value.charge, {"electric_charge": -0.82, "oxygen_indicator": 1.0})
        self.assertEqual(len(o_one.value.position), 3)
        self.assertAlmostEqual(o_one.value.position[0], 1.747, places=6)
        self.assertAlmostEqual(o_one.value.position[1], 0.149, places=6)
        self.assertAlmostEqual(o_one.value.position[2], 0.744, places=6)
        self.assertEqual(len(o_one.children), 0)
        h_two = water_three.children[2]
        self.assertIs(h_two.parent, water_three)
        self.assertEqual(h_two.weight, 1.0 / 3.0)
        self.assertEqual(h_two.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_two.value.position), 3)
        self.assertAlmostEqual(h_two.value.position[0], 1.959, places=6)
        self.assertAlmostEqual(h_two.value.position[1], 1.120, places=6)
        self.assertAlmostEqual(h_two.value.position[2], 0.832, places=6)
        self.assertEqual(len(h_two.children), 0)

        water_four = state[3]
        self.assertIsNone(water_four.parent)
        self.assertEqual(water_four.weight, 1.0)
        self.assertIsNone(water_four.value.charge)
        self.assertEqual(len(water_four.value.position), 3)
        self.assertAlmostEqual(water_four.value.position[0], 4.568, places=6)
        self.assertAlmostEqual(water_four.value.position[1], 9.38066666666667, places=6)
        self.assertAlmostEqual(water_four.value.position[2], 10.03266666666667, places=6)
        self.assertEqual(len(water_four.children), 3)
        h_one = water_four.children[0]
        self.assertIs(h_one.parent, water_four)
        self.assertEqual(h_one.weight, 1.0 / 3.0)
        self.assertEqual(h_one.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_one.value.position), 3)
        self.assertAlmostEqual(h_one.value.position[0], 4.283, places=6)
        self.assertAlmostEqual(h_one.value.position[1], 8.790, places=6)
        self.assertAlmostEqual(h_one.value.position[2], 0.340, places=6)
        self.assertEqual(len(h_one.children), 0)
        o_one = water_four.children[1]
        self.assertIs(o_one.parent, water_four)
        self.assertEqual(o_one.weight, 1.0 / 3.0)
        self.assertEqual(o_one.value.charge, {"electric_charge": -0.82, "oxygen_indicator": 1.0})
        self.assertEqual(len(o_one.value.position), 3)
        self.assertAlmostEqual(o_one.value.position[0], 4.247, places=6)
        self.assertAlmostEqual(o_one.value.position[1], 9.615, places=6)
        self.assertAlmostEqual(o_one.value.position[2], 9.967, places=6)
        self.assertEqual(len(o_one.children), 0)
        h_two = water_four.children[2]
        self.assertIs(h_two.parent, water_four)
        self.assertEqual(h_two.weight, 1.0 / 3.0)
        self.assertEqual(h_two.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_two.value.position), 3)
        self.assertAlmostEqual(h_two.value.position[0], 5.174, places=6)
        self.assertAlmostEqual(h_two.value.position[1], 9.737, places=6)
        self.assertAlmostEqual(h_two.value.position[2], 9.530, places=6)
        self.assertEqual(len(h_two.children), 0)

    def test_pdb_file_water_origin_l2(self):
        # Origin of system box in pdb file at [-L/2, -L/2, -L/2]
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=10.261)
        electric_charge_values = ChargeValues([0.41, -0.82, 0.41], "electric_charge")
        oxygen_indicator_charge = ChargeValues([0.0, 1.0, 0.0], "oxygen_indicator")
        with warnings.catch_warnings():
            # Ignore warnings of MDAnalysis
            warnings.simplefilter("ignore", UserWarning)
            pdb_input_handler = PdbInputHandler("pdb_test_files/water_test_origin_l2.pdb",
                                                [electric_charge_values, oxygen_indicator_charge])
            state = pdb_input_handler.read()
        self.assertEqual(len(state), 4)

        water_one = state[0]
        self.assertIsNone(water_one.parent)
        self.assertEqual(water_one.weight, 1.0)
        self.assertIsNone(water_one.value.charge)
        self.assertEqual(len(water_one.value.position), 3)
        # MDAnalysis only uses c floats (not doubles) -> reduce precision
        self.assertAlmostEqual(water_one.value.position[0], 8.43866666666667, places=6)
        self.assertAlmostEqual(water_one.value.position[1], 5.736333333333333, places=6)
        self.assertAlmostEqual(water_one.value.position[2], 1.384333333333333, places=6)
        self.assertEqual(len(water_one.children), 3)
        h_one = water_one.children[0]
        self.assertIs(h_one.parent, water_one)
        self.assertEqual(h_one.weight, 1.0 / 3.0)
        self.assertEqual(h_one.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_one.value.position), 3)
        self.assertAlmostEqual(h_one.value.position[0], 7.859, places=6)
        self.assertAlmostEqual(h_one.value.position[1], 5.311, places=6)
        self.assertAlmostEqual(h_one.value.position[2], 0.879, places=6)
        self.assertEqual(len(h_one.children), 0)
        o_one = water_one.children[1]
        self.assertIs(o_one.parent, water_one)
        self.assertEqual(o_one.weight, 1.0 / 3.0)
        self.assertEqual(o_one.value.charge, {"electric_charge": -0.82, "oxygen_indicator": 1.0})
        self.assertEqual(len(o_one.value.position), 3)
        self.assertAlmostEqual(o_one.value.position[0], 8.663, places=6)
        self.assertAlmostEqual(o_one.value.position[1], 5.863, places=6)
        self.assertAlmostEqual(o_one.value.position[2], 1.139, places=6)
        self.assertEqual(len(o_one.children), 0)
        h_two = water_one.children[2]
        self.assertIs(h_two.parent, water_one)
        self.assertEqual(h_two.weight, 1.0 / 3.0)
        self.assertEqual(h_two.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_two.value.position), 3)
        self.assertAlmostEqual(h_two.value.position[0], 8.794, places=6)
        self.assertAlmostEqual(h_two.value.position[1], 6.035, places=6)
        self.assertAlmostEqual(h_two.value.position[2], 2.135, places=6)
        self.assertEqual(len(h_two.children), 0)

        water_two = state[1]
        self.assertIsNone(water_two.parent)
        self.assertEqual(water_two.weight, 1.0)
        self.assertIsNone(water_two.value.charge)
        self.assertEqual(len(water_two.value.position), 3)
        self.assertAlmostEqual(water_two.value.position[0], 10.119, places=6)
        self.assertAlmostEqual(water_two.value.position[1], 1.288666666666666, places=6)
        self.assertAlmostEqual(water_two.value.position[2], 10.17866666666667, places=6)
        self.assertEqual(len(water_two.children), 3)
        h_one = water_two.children[0]
        self.assertIs(h_one.parent, water_two)
        self.assertEqual(h_one.weight, 1.0 / 3.0)
        self.assertEqual(h_one.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_one.value.position), 3)
        self.assertAlmostEqual(h_one.value.position[0], 9.969, places=6)
        self.assertAlmostEqual(h_one.value.position[1], 1.519, places=6)
        self.assertAlmostEqual(h_one.value.position[2], 9.446, places=6)
        self.assertEqual(len(h_one.children), 0)
        o_one = water_two.children[1]
        self.assertIs(o_one.parent, water_two)
        self.assertEqual(o_one.weight, 1.0 / 3.0)
        self.assertEqual(o_one.value.charge, {"electric_charge": -0.82, "oxygen_indicator": 1.0})
        self.assertEqual(len(o_one.value.position), 3)
        self.assertAlmostEqual(o_one.value.position[0], 10.068, places=6)
        self.assertAlmostEqual(o_one.value.position[1], 1.315, places=6)
        self.assertAlmostEqual(o_one.value.position[2], 0.160, places=6)
        self.assertEqual(len(o_one.children), 0)
        h_two = water_two.children[2]
        self.assertIs(h_two.parent, water_two)
        self.assertEqual(h_two.weight, 1.0 / 3.0)
        self.assertEqual(h_two.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_two.value.position), 3)
        self.assertAlmostEqual(h_two.value.position[0], 0.059, places=6)
        self.assertAlmostEqual(h_two.value.position[1], 1.032, places=6)
        self.assertAlmostEqual(h_two.value.position[2], 0.408, places=6)
        self.assertEqual(len(h_two.children), 0)

        water_three = state[2]
        self.assertIsNone(water_three.parent)
        self.assertEqual(water_three.weight, 1.0)
        self.assertIsNone(water_three.value.charge)
        self.assertEqual(len(water_three.value.position), 3)
        self.assertAlmostEqual(water_three.value.position[0], 2.045, places=6)
        self.assertAlmostEqual(water_three.value.position[1], 0.3630000000000001, places=6)
        self.assertAlmostEqual(water_three.value.position[2], 0.5586666666666666, places=6)
        self.assertEqual(len(water_three.children), 3)
        h_one = water_three.children[0]
        self.assertIs(h_one.parent, water_three)
        self.assertEqual(h_one.weight, 1.0 / 3.0)
        self.assertEqual(h_one.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_one.value.position), 3)
        self.assertAlmostEqual(h_one.value.position[0], 2.429, places=6)
        self.assertAlmostEqual(h_one.value.position[1], 10.081, places=6)
        self.assertAlmostEqual(h_one.value.position[2], 0.100, places=6)
        self.assertEqual(len(h_one.children), 0)
        o_one = water_three.children[1]
        self.assertIs(o_one.parent, water_three)
        self.assertEqual(o_one.weight, 1.0 / 3.0)
        self.assertEqual(o_one.value.charge, {"electric_charge": -0.82, "oxygen_indicator": 1.0})
        self.assertEqual(len(o_one.value.position), 3)
        self.assertAlmostEqual(o_one.value.position[0], 1.747, places=6)
        self.assertAlmostEqual(o_one.value.position[1], 0.149, places=6)
        self.assertAlmostEqual(o_one.value.position[2], 0.744, places=6)
        self.assertEqual(len(o_one.children), 0)
        h_two = water_three.children[2]
        self.assertIs(h_two.parent, water_three)
        self.assertEqual(h_two.weight, 1.0 / 3.0)
        self.assertEqual(h_two.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_two.value.position), 3)
        self.assertAlmostEqual(h_two.value.position[0], 1.959, places=6)
        self.assertAlmostEqual(h_two.value.position[1], 1.120, places=6)
        self.assertAlmostEqual(h_two.value.position[2], 0.832, places=6)
        self.assertEqual(len(h_two.children), 0)

        water_four = state[3]
        self.assertIsNone(water_four.parent)
        self.assertEqual(water_four.weight, 1.0)
        self.assertIsNone(water_four.value.charge)
        self.assertEqual(len(water_four.value.position), 3)
        self.assertAlmostEqual(water_four.value.position[0], 4.568, places=6)
        self.assertAlmostEqual(water_four.value.position[1], 9.38066666666667, places=6)
        self.assertAlmostEqual(water_four.value.position[2], 10.03266666666667, places=6)
        self.assertEqual(len(water_four.children), 3)
        h_one = water_four.children[0]
        self.assertIs(h_one.parent, water_four)
        self.assertEqual(h_one.weight, 1.0 / 3.0)
        self.assertEqual(h_one.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_one.value.position), 3)
        self.assertAlmostEqual(h_one.value.position[0], 4.283, places=6)
        self.assertAlmostEqual(h_one.value.position[1], 8.790, places=6)
        self.assertAlmostEqual(h_one.value.position[2], 0.340, places=6)
        self.assertEqual(len(h_one.children), 0)
        o_one = water_four.children[1]
        self.assertIs(o_one.parent, water_four)
        self.assertEqual(o_one.weight, 1.0 / 3.0)
        self.assertEqual(o_one.value.charge, {"electric_charge": -0.82, "oxygen_indicator": 1.0})
        self.assertEqual(len(o_one.value.position), 3)
        self.assertAlmostEqual(o_one.value.position[0], 4.247, places=6)
        self.assertAlmostEqual(o_one.value.position[1], 9.615, places=6)
        self.assertAlmostEqual(o_one.value.position[2], 9.967, places=6)
        self.assertEqual(len(o_one.children), 0)
        h_two = water_four.children[2]
        self.assertIs(h_two.parent, water_four)
        self.assertEqual(h_two.weight, 1.0 / 3.0)
        self.assertEqual(h_two.value.charge, {"electric_charge": 0.41, "oxygen_indicator": 0.0})
        self.assertEqual(len(h_two.value.position), 3)
        self.assertAlmostEqual(h_two.value.position[0], 5.174, places=6)
        self.assertAlmostEqual(h_two.value.position[1], 9.737, places=6)
        self.assertAlmostEqual(h_two.value.position[2], 9.530, places=6)
        self.assertEqual(len(h_two.children), 0)

    def test_pdb_file_atom_origin_zero(self):
        # Origin of system box in pdb file at [0, 0, 0]
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        electric_charge_values = ChargeValues([1.0], "electric_charge")
        # This file also contains positions outside the system box which should be corrected for periodic boundaries
        pdb_input_handler = PdbInputHandler("pdb_test_files/atom_test_origin_0.pdb", [electric_charge_values])
        state = pdb_input_handler.read()
        self.assertEqual(len(state), 4)

        atom_one = state[0]
        self.assertIsNone(atom_one.parent)
        self.assertEqual(atom_one.weight, 1.0)
        self.assertEqual(atom_one.value.charge, {"electric_charge": 1.0})
        self.assertEqual(len(atom_one.value.position), 3)
        # MDAnalysis only uses c floats (not doubles) -> reduce precision
        self.assertAlmostEqual(atom_one.value.position[0], 0.983, places=6)
        self.assertAlmostEqual(atom_one.value.position[1], 0.032, places=6)
        self.assertAlmostEqual(atom_one.value.position[2], 0.714, places=6)
        self.assertEqual(len(atom_one.children), 0)

        atom_two = state[1]
        self.assertIsNone(atom_two.parent)
        self.assertEqual(atom_two.weight, 1.0)
        self.assertEqual(atom_two.value.charge, {"electric_charge": 1.0})
        self.assertEqual(len(atom_two.value.position), 3)
        # MDAnalysis only uses c floats (not doubles) -> reduce precision
        self.assertAlmostEqual(atom_two.value.position[0], 0.618, places=6)
        self.assertAlmostEqual(atom_two.value.position[1], 0.697, places=6)
        self.assertAlmostEqual(atom_two.value.position[2], 0.848, places=6)
        self.assertEqual(len(atom_two.children), 0)

        atom_three = state[2]
        self.assertIsNone(atom_three.parent)
        self.assertEqual(atom_three.weight, 1.0)
        self.assertEqual(atom_three.value.charge, {"electric_charge": 1.0})
        self.assertEqual(len(atom_three.value.position), 3)
        # MDAnalysis only uses c floats (not doubles) -> reduce precision
        self.assertAlmostEqual(atom_three.value.position[0], 0.312, places=6)
        self.assertAlmostEqual(atom_three.value.position[1], 0.806, places=6)
        self.assertAlmostEqual(atom_three.value.position[2], 0.642, places=6)
        self.assertEqual(len(atom_three.children), 0)

        atom_four = state[3]
        self.assertIsNone(atom_four.parent)
        self.assertEqual(atom_four.weight, 1.0)
        self.assertEqual(atom_four.value.charge, {"electric_charge": 1.0})
        self.assertEqual(len(atom_four.value.position), 3)
        # MDAnalysis only uses c floats (not doubles) -> reduce precision
        self.assertAlmostEqual(atom_four.value.position[0], 0.012, places=6)
        self.assertAlmostEqual(atom_four.value.position[1], 0.923, places=6)
        self.assertAlmostEqual(atom_four.value.position[2], 0.100, places=6)
        self.assertEqual(len(atom_four.children), 0)

    def test_pdb_file_atom_origin_l2(self):
        # Origin of system box in pdb file at [-L/2, -L/2, -L/2]
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        electric_charge_values = ChargeValues([1.0], "electric_charge")
        # This file also contains positions outside the system box which should be corrected for periodic boundaries
        with warnings.catch_warnings():
            # Ignore warnings of MDAnalysis
            warnings.simplefilter("ignore", UserWarning)
            pdb_input_handler = PdbInputHandler("pdb_test_files/atom_test_origin_l2.pdb", [electric_charge_values])
            state = pdb_input_handler.read()
        self.assertEqual(len(state), 4)

        atom_one = state[0]
        self.assertIsNone(atom_one.parent)
        self.assertEqual(atom_one.weight, 1.0)
        self.assertEqual(atom_one.value.charge, {"electric_charge": 1.0})
        self.assertEqual(len(atom_one.value.position), 3)
        # MDAnalysis only uses c floats (not doubles) -> reduce precision
        self.assertAlmostEqual(atom_one.value.position[0], 0.983, places=6)
        self.assertAlmostEqual(atom_one.value.position[1], 0.032, places=6)
        self.assertAlmostEqual(atom_one.value.position[2], 0.714, places=6)
        self.assertEqual(len(atom_one.children), 0)

        atom_two = state[1]
        self.assertIsNone(atom_two.parent)
        self.assertEqual(atom_two.weight, 1.0)
        self.assertEqual(atom_two.value.charge, {"electric_charge": 1.0})
        self.assertEqual(len(atom_two.value.position), 3)
        # MDAnalysis only uses c floats (not doubles) -> reduce precision
        self.assertAlmostEqual(atom_two.value.position[0], 0.618, places=6)
        self.assertAlmostEqual(atom_two.value.position[1], 0.697, places=6)
        self.assertAlmostEqual(atom_two.value.position[2], 0.848, places=6)
        self.assertEqual(len(atom_two.children), 0)

        atom_three = state[2]
        self.assertIsNone(atom_three.parent)
        self.assertEqual(atom_three.weight, 1.0)
        self.assertEqual(atom_three.value.charge, {"electric_charge": 1.0})
        self.assertEqual(len(atom_three.value.position), 3)
        # MDAnalysis only uses c floats (not doubles) -> reduce precision
        self.assertAlmostEqual(atom_three.value.position[0], 0.312, places=6)
        self.assertAlmostEqual(atom_three.value.position[1], 0.806, places=6)
        self.assertAlmostEqual(atom_three.value.position[2], 0.642, places=6)
        self.assertEqual(len(atom_three.children), 0)

        atom_four = state[3]
        self.assertIsNone(atom_four.parent)
        self.assertEqual(atom_four.weight, 1.0)
        self.assertEqual(atom_four.value.charge, {"electric_charge": 1.0})
        self.assertEqual(len(atom_four.value.position), 3)
        # MDAnalysis only uses c floats (not doubles) -> reduce precision
        self.assertAlmostEqual(atom_four.value.position[0], 0.012, places=6)
        self.assertAlmostEqual(atom_four.value.position[1], 0.923, places=6)
        self.assertAlmostEqual(atom_four.value.position[2], 0.100, places=6)
        self.assertEqual(len(atom_four.children), 0)

    def test_two_dimensional_pdb_file(self):
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=2, system_length=1.0)
        # Ignore warnings in logger
        logging.getLogger("jellyfysh.input_output_handler.input_handler.pdb_input_handler").setLevel(logging.ERROR)

        electric_charge_values = ChargeValues([-1.0], "electric_charge")
        # This file also contains positions outside the system box which should be corrected for periodic boundaries
        pdb_input_handler = PdbInputHandler("pdb_test_files/atom_test_2D.pdb", [electric_charge_values])
        state = pdb_input_handler.read()
        self.assertEqual(len(state), 2)

        atom_one = state[0]
        self.assertIsNone(atom_one.parent)
        self.assertEqual(atom_one.weight, 1.0)
        self.assertEqual(atom_one.value.charge, {"electric_charge": -1.0})
        self.assertEqual(len(atom_one.value.position), 2)
        # MDAnalysis only uses c floats (not doubles) -> reduce precision
        self.assertAlmostEqual(atom_one.value.position[0], 0.983, places=6)
        self.assertAlmostEqual(atom_one.value.position[1], 0.032, places=6)
        self.assertEqual(len(atom_one.children), 0)

        atom_two = state[1]
        self.assertIsNone(atom_two.parent)
        self.assertEqual(atom_two.weight, 1.0)
        self.assertEqual(atom_two.value.charge, {"electric_charge": -1.0})
        self.assertEqual(len(atom_two.value.position), 2)
        # MDAnalysis only uses c floats (not doubles) -> reduce precision
        self.assertAlmostEqual(atom_two.value.position[0], 0.618, places=6)
        self.assertAlmostEqual(atom_two.value.position[1], 0.697, places=6)
        self.assertEqual(len(atom_two.children), 0)

    def test_two_dimensional_pdb_file_contains_three_dimensional_system_box_raises_error(self):
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=2, system_length=1.0)
        # Ignore warnings in logger
        logging.getLogger("jellyfysh.input_output_handler.input_handler.pdb_input_handler").setLevel(logging.ERROR)
        with self.assertRaises(ConfigurationError):
            PdbInputHandler("pdb_test_files/atom_test_2D_wrong_system_box.pdb")

    def test_two_dimensional_pdb_file_contains_three_dimensional_position_raises_error(self):
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=2, system_length=1.0)
        # Ignore warnings in logger
        logging.getLogger("jellyfysh.input_output_handler.input_handler.pdb_input_handler").setLevel(logging.ERROR)
        pdb_input_handler = PdbInputHandler("pdb_test_files/atom_test_2D_wrong_positions.pdb")
        with self.assertRaises(AssertionError):
            pdb_input_handler.read()

    def test_non_existing_file_raises_error(self):
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=10.261)
        with self.assertRaises(FileNotFoundError):
            PdbInputHandler("not_existing.pdb")

    def test_dimension_to_high_raises_error(self):
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=4, system_length=10.261)
        with self.assertRaises(ConfigurationError):
            PdbInputHandler("pdb_test_files/water_test_origin_0.pdb")

    def test_not_matching_system_lengths_raises_error(self):
        # PDB file has system_length=10.261
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=10.26)
        with self.assertRaises(ConfigurationError):
            PdbInputHandler("pdb_test_files/water_test_origin_0.pdb")

    def test_wrong_file_suffix_raises_error(self):
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=10.261)
        with self.assertRaises(ConfigurationError):
            PdbInputHandler("pdb_test_files/water_test_origin_0.wrong_ending")

    def test_non_equal_composite_point_objects_raises_error(self):
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=10.261)
        with self.assertRaises(ConfigurationError):
            PdbInputHandler("pdb_test_files/test_non_equal_composite_point_objects.pdb")

    def test_no_charge_for_every_point_mass_raises_error(self):
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=10.261)
        # Only specify two charges although three are required
        electric_charge_values = ChargeValues([0.41, -0.82], "electric_charge")
        with self.assertRaises(ConfigurationError):
            PdbInputHandler("pdb_test_files/water_test_origin_0.pdb", [electric_charge_values])

    def test_repeated_charge_name_raises_error(self):
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=10.261)
        # Create two charge values with the same name
        electric_charge_values = ChargeValues([0.41, -0.82, 0.41], "electric_charge")
        oxygen_indicator_charge = ChargeValues([0.0, 1.0, 0.0], "electric_charge")
        with self.assertRaises(ConfigurationError):
            PdbInputHandler("pdb_test_files/water_test_origin_0.pdb", [electric_charge_values, oxygen_indicator_charge])


if __name__ == '__main__':
    main()
