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
from unittest import TestCase, main
from base.exceptions import ConfigurationError
from potential.merged_image_coulomb_potential import MergedImageCoulombPotential
import setting


class TestMergedImageCoulombPotential(TestCase):
    def setUp(self) -> None:
        setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        # Set these so the settings are initialized
        setting.set_number_of_root_nodes(2)
        setting.set_number_of_nodes_per_root_node(2)
        setting.set_number_of_node_levels(1)
        self._potential = MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=6,
                                                      position_cutoff=2, prefactor=1.0)
        self._variant_potential = MergedImageCoulombPotential(alpha=5.0, fourier_cutoff=9,
                                                              position_cutoff=2)

    def tearDown(self) -> None:
        setting.reset()

    def test_direction_zero_positive_charge_product(self):
        self.assertAlmostEqual(self._potential.derivative(0, [1.0 / 7.0, 1.0 / 8.0, 1.0 / 5.0], 1.0, 1.0),
                               6.322464150019139, places=12)

    def test_direction_zero_negative_charge_product(self):
        self.assertAlmostEqual(self._potential.derivative(0, [1.0 / 7.0, 1.0 / 8.0, 1.0 / 5.0], 1.0, -1.0),
                               -6.322464150019139, places=12)

    def test_direction_one_positive_charge_product(self):
        self.assertAlmostEqual(self._potential.derivative(1, [1.0 / 8.0, 1.0 / 7.0, 1.0 / 5.0], 1.0, 1.0),
                               6.322464150019139, places=12)

    def test_direction_one_negative_charge_product(self):
        self.assertAlmostEqual(self._potential.derivative(1, [1.0 / 8.0, 1.0 / 7.0, 1.0 / 5.0], 1.0, -1.0),
                               -6.322464150019139, places=12)

    def test_direction_two_positive_charge_product(self):
        self.assertAlmostEqual(self._potential.derivative(2, [1.0 / 5.0, 1.0 / 8.0, 1.0 / 7.0], 1.0, 1.0),
                               6.322464150019139, places=12)

    def test_direction_two_negative_charge_product(self):
        self.assertAlmostEqual(self._potential.derivative(2, [1.0 / 5.0, 1.0 / 8.0, 1.0 / 7.0], 1.0, -1.0),
                               -6.322464150019139, places=12)

    def test_variant_potential(self):
        self.assertAlmostEqual(self._variant_potential.derivative(0, [1.0 / 7.0, 1.0 / 8.0, 1.0 / 5.0], 1.0, 1.0),
                               6.322464150019139, places=12)

    def test_prefactor_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=6, position_cutoff=2,
                                        prefactor=0.0)

    def test_ewald_converge_factor_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=-3.45, fourier_cutoff=6, position_cutoff=2,
                                        prefactor=1.0)

    def test_ewald_converge_factor_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=0, fourier_cutoff=6, position_cutoff=2,
                                        prefactor=1.0)

    def test_ewald_fourier_cutoff_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=-1, position_cutoff=2,
                                        prefactor=1.0)

    def test_ewald_fourier_cutoff_zero_raises_no_error(self):
        MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=0, position_cutoff=2,
                                    prefactor=1.0)
        self.assertTrue(True)

    def test_ewald_position_cutoff_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=6, position_cutoff=-2,
                                        prefactor=1.0)

    def test_ewald_position_cutoff_zero_raises_no_error(self):
        MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=6, position_cutoff=0,
                                    prefactor=1.0)
        self.assertTrue(True)

    def test_dimension_unequal_three_raises_error(self):
        setting.reset()
        setting.HypercubicSetting(beta=1.0, dimension=2, system_length=1.0)
        setting.dimension = 2
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=6, position_cutoff=2,
                                        prefactor=1.0)
        setting.reset()
        setting.HypercubicSetting(beta=1.0, dimension=1, system_length=1.0)
        setting.dimension = 1
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=6, position_cutoff=2,
                                        prefactor=1.0)

    def test_setting_not_initialized_raises_error(self):
        # Set one variable to None so that setting is not initialized
        setting.hypercubic_setting.number_of_root_nodes = None
        with self.assertRaises(ConfigurationError):
            MergedImageCoulombPotential(alpha=3.45, fourier_cutoff=6, position_cutoff=2,
                                        prefactor=1.0)


if __name__ == '__main__':
    main()
