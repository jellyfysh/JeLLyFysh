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
from unittest import TestCase, main
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.potential.merged_image_coulomb_potential import MergedImageCoulombPotential
import jellyfysh.setting as setting
from jellyfysh.setting import hypercubic_setting


class TestMergedImageCoulombPotential(TestCase):
    def setUpSystemLengthOne(self) -> None:
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        # Set these so the settings are initialized
        setting.set_number_of_root_nodes(2)
        setting.set_number_of_nodes_per_root_node(2)
        setting.set_number_of_node_levels(1)
        self._potential = MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=6,
                                                      position_cutoff=2, prefactor=1.0)
        self._variant_potential = MergedImageCoulombPotential(alpha=5.0, fourier_cutoff=9,
                                                              position_cutoff=2)

    def setUpSystemLengthTwo(self) -> None:
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=2.0)
        setting.set_number_of_root_nodes(2)
        setting.set_number_of_nodes_per_root_node(2)
        setting.set_number_of_node_levels(1)
        self._potential = MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=6,
                                                      position_cutoff=2, prefactor=1.0)
        self._variant_potential = MergedImageCoulombPotential(alpha=5.0, fourier_cutoff=9,
                                                              position_cutoff=2)

    def tearDown(self) -> None:
        setting.reset()

    def test_direction_zero_positive_charge_product_system_length_one(self):
        self.setUpSystemLengthOne()
        # Value calculated with Mathematica to very high precision
        self.assertAlmostEqual(
            self._potential.derivative([1.0, 0.0, 0.0], [1.0 / 7.0, 1.0 / 8.0, 1.0 / 5.0], 1.0, 1.0),
            6.322464150019139, places=12)
        # The derivative does not depend on the absolute value of the velocity but only on the non-vanishing component.
        self.assertAlmostEqual(
            self._potential.derivative([3.1, 0.0, 0.0], [1.0 / 7.0, 1.0 / 8.0, 1.0 / 5.0], 1.0, 1.0),
            6.322464150019139 * 3.1, places=12)

    def test_direction_zero_negative_charge_product_system_length_one(self):
        self.setUpSystemLengthOne()
        self.assertAlmostEqual(
            self._potential.derivative([1.0, 0.0, 0.0], [1.0 / 7.0, 1.0 / 8.0, 1.0 / 5.0], 1.0, -1.0),
            -6.322464150019139, places=12)
        self.assertAlmostEqual(
            self._potential.derivative([3.1, 0.0, 0.0], [1.0 / 7.0, 1.0 / 8.0, 1.0 / 5.0], 1.0, -1.0),
            -6.322464150019139 * 3.1, places=12)

    def test_direction_one_positive_charge_product_system_length_one(self):
        self.setUpSystemLengthOne()
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 1.0, 0.0], [1.0 / 8.0, 1.0 / 7.0, 1.0 / 5.0], 1.0, 1.0),
            6.322464150019139, places=12)
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 3.1, 0.0], [1.0 / 8.0, 1.0 / 7.0, 1.0 / 5.0], 1.0, 1.0),
            6.322464150019139 * 3.1, places=12)

    def test_direction_one_negative_charge_product_system_length_one(self):
        self.setUpSystemLengthOne()
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 1.0, 0.0], [1.0 / 8.0, 1.0 / 7.0, 1.0 / 5.0], 1.0, -1.0),
            -6.322464150019139, places=12)
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 3.1, 0.0], [1.0 / 8.0, 1.0 / 7.0, 1.0 / 5.0], 1.0, -1.0),
            -6.322464150019139 * 3.1, places=12)

    def test_direction_two_positive_charge_product_system_length_one(self):
        self.setUpSystemLengthOne()
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 0.0, 1.0], [1.0 / 5.0, 1.0 / 8.0, 1.0 / 7.0], 1.0, 1.0),
            6.322464150019139, places=12)
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 0.0, 3.1], [1.0 / 5.0, 1.0 / 8.0, 1.0 / 7.0], 1.0, 1.0),
            6.322464150019139 * 3.1, places=12)

    def test_direction_two_negative_charge_product_system_length_one(self):
        self.setUpSystemLengthOne()
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 0.0, 1.0], [1.0 / 5.0, 1.0 / 8.0, 1.0 / 7.0], 1.0, -1.0),
            -6.322464150019139, places=12)
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 0.0, 3.1], [1.0 / 5.0, 1.0 / 8.0, 1.0 / 7.0], 1.0, -1.0),
            -6.322464150019139 * 3.1, places=12)

    def test_variant_potential_system_length_one(self):
        self.setUpSystemLengthOne()
        self.assertAlmostEqual(
            self._variant_potential.derivative([1.0, 0.0, 0.0], [1.0 / 7.0, 1.0 / 8.0, 1.0 / 5.0], 1.0, 1.0),
            6.322464150019139, places=12)
        self.assertAlmostEqual(
            self._variant_potential.derivative([3.1, 0.0, 0.0], [1.0 / 7.0, 1.0 / 8.0, 1.0 / 5.0], 1.0, 1.0),
            6.322464150019139 * 3.1, places=12)

    def test_direction_zero_positive_charge_product_system_length_two(self):
        self.setUpSystemLengthTwo()
        # Value calculated with Mathematica to very high precision
        self.assertAlmostEqual(
            self._potential.derivative([1.0, 0.0, 0.0], [1.0 / 7.0, 1.0 / 8.0, 1.0 / 5.0], 1.0, 1.0),
            6.742588479793721, places=12)
        self.assertAlmostEqual(
            self._potential.derivative([3.1, 0.0, 0.0], [1.0 / 7.0, 1.0 / 8.0, 1.0 / 5.0], 1.0, 1.0),
            6.742588479793721 * 3.1, places=12)

    def test_direction_zero_negative_charge_product_system_length_two(self):
        self.setUpSystemLengthTwo()
        self.assertAlmostEqual(
            self._potential.derivative([1.0, 0.0, 0.0], [1.0 / 7.0, 1.0 / 8.0, 1.0 / 5.0], 1.0, -1.0),
            -6.742588479793721, places=12)
        self.assertAlmostEqual(
            self._potential.derivative([3.1, 0.0, 0.0], [1.0 / 7.0, 1.0 / 8.0, 1.0 / 5.0], 1.0, -1.0),
            -6.742588479793721 * 3.1, places=12)

    def test_direction_one_positive_charge_product_system_length_two(self):
        self.setUpSystemLengthTwo()
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 1.0, 0.0], [1.0 / 8.0, 1.0 / 7.0, 1.0 / 5.0], 1.0, 1.0),
            6.742588479793721, places=12)
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 3.1, 0.0], [1.0 / 8.0, 1.0 / 7.0, 1.0 / 5.0], 1.0, 1.0),
            6.742588479793721 * 3.1, places=12)

    def test_direction_one_negative_charge_product_system_length_two(self):
        self.setUpSystemLengthTwo()
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 1.0, 0.0], [1.0 / 8.0, 1.0 / 7.0, 1.0 / 5.0], 1.0, -1.0),
            -6.742588479793721, places=12)
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 3.1, 0.0], [1.0 / 8.0, 1.0 / 7.0, 1.0 / 5.0], 1.0, -1.0),
            -6.742588479793721 * 3.1, places=12)

    def test_direction_two_positive_charge_product_system_length_two(self):
        self.setUpSystemLengthTwo()
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 0.0, 1.0], [1.0 / 5.0, 1.0 / 8.0, 1.0 / 7.0], 1.0, 1.0),
            6.742588479793721, places=12)
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 0.0, 3.1], [1.0 / 5.0, 1.0 / 8.0, 1.0 / 7.0], 1.0, 1.0),
            6.742588479793721 * 3.1, places=12)

    def test_direction_two_negative_charge_product_system_length_two(self):
        self.setUpSystemLengthTwo()
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 0.0, 1.0], [1.0 / 5.0, 1.0 / 8.0, 1.0 / 7.0], 1.0, -1.0),
            -6.742588479793721, places=12)
        self.assertAlmostEqual(
            self._potential.derivative([0.0, 0.0, 3.1], [1.0 / 5.0, 1.0 / 8.0, 1.0 / 7.0], 1.0, -1.0),
            -6.742588479793721 * 3.1, places=12)

    def test_variant_potential_system_length_two(self):
        self.setUpSystemLengthTwo()
        self.assertAlmostEqual(
            self._variant_potential.derivative([1.0, 0.0, 0.0], [1.0 / 7.0, 1.0 / 8.0, 1.0 / 5.0], 1.0, 1.0),
            6.742588479793721, places=12)
        self.assertAlmostEqual(
            self._variant_potential.derivative([3.1, 0.0, 0.0], [1.0 / 7.0, 1.0 / 8.0, 1.0 / 5.0], 1.0, 1.0),
            6.742588479793721 * 3.1, places=12)

    def test_number_separation_arguments_is_one(self):
        self.setUpSystemLengthOne()
        self.assertEqual(self._potential.number_separation_arguments, 1)
        self.assertEqual(self._variant_potential.number_separation_arguments, 1)

    def test_number_charge_arguments_is_two(self):
        self.setUpSystemLengthOne()
        self.assertEqual(self._potential.number_charge_arguments, 2)
        self.assertEqual(self._variant_potential.number_charge_arguments, 2)

    def test_velocity_zero_raises_error(self):
        self.setUpSystemLengthOne()
        with self.assertRaises(AssertionError):
            self._potential.derivative([0.0, 0.0, 0.0], [0.7, 0.6, -0.1], 1.0, 1.0)
        with self.assertRaises(AssertionError):
            self._variant_potential.derivative([0.0, 0.0, 0.0], [0.7, 0.6, -0.1], 1.0, 1.0)

    def test_negative_velocity_along_axis_raises_error(self):
        self.setUpSystemLengthOne()
        with self.assertRaises(AssertionError):
            self._potential.derivative([-1.0, 0.0, 0.0], [0.7, 0.6, -0.1], 1.0, 1.0)
        with self.assertRaises(AssertionError):
            self._potential.derivative([0.0, -1.0, 0.0], [0.7, 0.6, -0.1], 1.0, 1.0)
        with self.assertRaises(AssertionError):
            self._potential.derivative([0.0, 0.0, -1.0], [0.7, 0.6, -0.1], 1.0, 1.0)
        with self.assertRaises(AssertionError):
            self._variant_potential.derivative([-1.0, 0.0, 0.0], [0.7, 0.6, -0.1], 1.0, 1.0)
        with self.assertRaises(AssertionError):
            self._variant_potential.derivative([0.0, -1.0, 0.0], [0.7, 0.6, -0.1], 1.0, 1.0)
        with self.assertRaises(AssertionError):
            self._variant_potential.derivative([0.0, 0.0, -1.0], [0.7, 0.6, -0.1], 1.0, 1.0)

    def test_velocity_not_parallel_to_axis_raises_error(self):
        self.setUpSystemLengthOne()
        with self.assertRaises(AssertionError):
            self._potential.derivative([1.0, 3.1, 0.0], [0.7, 0.6, -0.1], 1.0, 1.0)
        with self.assertRaises(AssertionError):
            self._variant_potential.derivative([0.0, 1.0, 3.1], [0.7, 0.6, -0.1], 1.0, 1.0)

    def test_prefactor_zero_raises_error(self):
        self.setUpSystemLengthOne()
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=6, position_cutoff=2,
                                        prefactor=0.0)

    def test_ewald_converge_factor_negative_raises_error(self):
        self.setUpSystemLengthOne()
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=-3.45, fourier_cutoff=6, position_cutoff=2,
                                        prefactor=1.0)

    def test_ewald_converge_factor_zero_raises_error(self):
        self.setUpSystemLengthOne()
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=0, fourier_cutoff=6, position_cutoff=2,
                                        prefactor=1.0)

    def test_ewald_fourier_cutoff_negative_raises_error(self):
        self.setUpSystemLengthOne()
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=-1, position_cutoff=2,
                                        prefactor=1.0)

    def test_ewald_fourier_cutoff_zero_raises_no_error(self):
        self.setUpSystemLengthOne()
        MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=0, position_cutoff=2,
                                    prefactor=1.0)
        self.assertTrue(True)

    def test_ewald_position_cutoff_negative_raises_error(self):
        self.setUpSystemLengthOne()
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=6, position_cutoff=-2,
                                        prefactor=1.0)

    def test_ewald_position_cutoff_zero_raises_no_error(self):
        self.setUpSystemLengthOne()
        MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=6, position_cutoff=0,
                                    prefactor=1.0)
        self.assertTrue(True)

    def test_dimension_unequal_three_raises_error(self):
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=2, system_length=1.0)
        setting.set_number_of_root_nodes(2)
        setting.set_number_of_nodes_per_root_node(2)
        setting.set_number_of_node_levels(1)
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=6, position_cutoff=2,
                                        prefactor=1.0)
        setting.reset()
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=1, system_length=1.0)
        setting.set_number_of_root_nodes(2)
        setting.set_number_of_nodes_per_root_node(2)
        setting.set_number_of_node_levels(1)
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=6, position_cutoff=2,
                                        prefactor=1.0)

    def test_setting_not_initialized_raises_error(self):
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=6, position_cutoff=2,
                                        prefactor=1.0)


if __name__ == '__main__':
    main()
