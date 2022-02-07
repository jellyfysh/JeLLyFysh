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
from jellyfysh.potential.lennard_jones_potential import LennardJonesPotential


# noinspection PyArgumentEqualDefault
class TestLennardJonesPotential(TestCase):
    def setUp(self) -> None:
        self._potential_one = LennardJonesPotential(prefactor=0.5, characteristic_length=0.3)

    def test_derivative_direction_zero(self):
        self.assertAlmostEqual(self._potential_one.derivative([1.0, 0.0, 0.0], [0.3, -0.2, 0.1]),
                               -0.800414148077271, places=13)
        # The derivative does not depend on the absolute value of the velocity but only on the non-vanishing component.
        self.assertAlmostEqual(self._potential_one.derivative([3.1, 0.0, 0.0], [0.3, -0.2, 0.1]),
                               -0.800414148077271 * 3.1, places=13)

    def test_derivative_direction_one(self):
        self.assertAlmostEqual(self._potential_one.derivative([0.0, 1.0, 0.0], [0.4, 0.44, 0.444]),
                               -0.01036860489492274, places=13)
        self.assertAlmostEqual(self._potential_one.derivative([0.0, 3.1, 0.0], [0.4, 0.44, 0.444]),
                               -0.01036860489492274 * 3.1, places=13)

    def test_derivative_direction_two(self):
        self.assertAlmostEqual(self._potential_one.derivative([0.0, 0.0, 1.0], [0.0, 0.0, 0.1]),
                               3.186458999999995e7, places=6)
        self.assertAlmostEqual(self._potential_one.derivative([0.0, 0.0, 3.1], [0.0, 0.0, 0.1]),
                               3.186458999999995e7 * 3.1, places=6)

    def test_displacement_front_outside_minimum(self):
        self.assertAlmostEqual(
            self._potential_one.displacement([1.0, 0.0, 0.0], [-0.1, 0.2, -0.3], 0.06565670434585313),
            0.2, places=13)
        # The time displacement depends on the value of the non-vanishing component of the velocity.
        self.assertAlmostEqual(
            self._potential_one.displacement([3.1, 0.0, 0.0], [-0.1, 0.2, -0.3], 0.06565670434585313),
            0.2 / 3.1, places=13)

    def test_displacement_inside_minimum(self):
        self.assertAlmostEqual(
            self._potential_one.displacement([0.0, 1.0, 0.0],
                                             [0.30197686974556653, -0.03533473711193214, 0.062348805602448674],
                                             0.08754383137424455), 0.3, places=13)
        self.assertAlmostEqual(
            self._potential_one.displacement([0.0, 3.1, 0.0],
                                             [0.30197686974556653, -0.03533473711193214, 0.062348805602448674],
                                             0.08754383137424455), 0.3 / 3.1, places=13)

        self.assertAlmostEqual(
            self._potential_one.displacement([0.0, 1.0, 0.0],
                                             [0.30197686974556653, 0.06466526288806787, 0.062348805602448674],
                                             0.11804106269028171), 0.4, places=13)
        self.assertAlmostEqual(
            self._potential_one.displacement([0.0, 3.1, 0.0],
                                             [0.30197686974556653, 0.06466526288806787, 0.062348805602448674],
                                             0.11804106269028171), 0.4 / 3.1, places=13)

    def test_displacement_behind_outside_minimum_untrapped(self):
        self.assertAlmostEqual(
            self._potential_one.displacement([0.0, 0.0, 1.0], [0.3, 0.2, 0.1], 0.029207419374592217), 0.25, places=13)
        self.assertAlmostEqual(
            self._potential_one.displacement([0.0, 0.0, 3.1], [0.3, 0.2, 0.1], 0.029207419374592217),
            0.25 / 3.1, places=13)

        self.assertAlmostEqual(
            self._potential_one.displacement([0.0, 0.0, 1.0], [0.3, 0.2, 0.1], 0.11087709005680561),
            float('inf'), places=13)
        self.assertAlmostEqual(
            self._potential_one.displacement([0.0, 0.0, 3.1], [0.3, 0.2, 0.1], 0.11087709005680561),
            float('inf'), places=13)

    def test_displacement_behind_outside_minimum_trapped(self):
        self.assertAlmostEqual(
            self._potential_one.displacement([0.0, 1.0, 0.0], [0.1, 0.4, 0.2], 0.998275404075823), 0.25, places=13)
        self.assertAlmostEqual(
            self._potential_one.displacement([0.0, 3.1, 0.0], [0.1, 0.4, 0.2], 0.998275404075823),
            0.25 / 3.1, places=13)

        self.assertAlmostEqual(
            self._potential_one.displacement([0.0, 1.0, 0.0], [0.15, 0.4, 0.2], 3.1841574116762774), 0.8, places=13)
        self.assertAlmostEqual(
            self._potential_one.displacement([0.0, 3.1, 0.0], [0.15, 0.4, 0.2], 3.1841574116762774),
            0.8 / 3.1, places=13)

        self.assertAlmostEqual(
            self._potential_one.displacement([0.0, 1.0, 0.0], [0.15, 0.4, 0.2], 3.2150592241279984),
            float('inf'), places=13)
        self.assertAlmostEqual(
            self._potential_one.displacement([0.0, 3.1, 0.0], [0.15, 0.4, 0.2], 3.2150592241279984),
            float('inf'), places=13)

    def test_number_separation_arguments_is_one(self):
        self.assertEqual(self._potential_one.number_separation_arguments, 1)

    def test_number_charge_arguments_is_zero(self):
        self.assertEqual(self._potential_one.number_charge_arguments, 0)

    def test_potential_change_required(self):
        self.assertTrue(self._potential_one.potential_change_required)

    def test_velocity_zero_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential_one.displacement([0.0, 0.0, 0.0], [-0.01, 0.1, -0.3], 0.05)
        with self.assertRaises(AssertionError):
            self._potential_one.derivative([0.0, 0.0, 0.0], [0.2, 0.1, -0.3])

    def test_negative_velocity_along_axis_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential_one.displacement([-1.0, 0.0, 0.0], [-0.01, 0.1, -0.3], 0.05)
        with self.assertRaises(AssertionError):
            self._potential_one.displacement([0.0, -1.0, 0.0], [-0.01, 0.1, -0.3], 0.05)
        with self.assertRaises(AssertionError):
            self._potential_one.displacement([0.0, 0.0, -1.0], [-0.01, 0.1, -0.3], 0.05)
        with self.assertRaises(AssertionError):
            self._potential_one.derivative([-3.1, 0.0, 0.0], [0.2, 0.1, -0.3])
        with self.assertRaises(AssertionError):
            self._potential_one.derivative([0.0, -3.1, 0.0], [0.2, 0.1, -0.3])
        with self.assertRaises(AssertionError):
            self._potential_one.derivative([0.0, 0.0, -3.1], [0.2, 0.1, -0.3])

    def test_velocity_not_parallel_to_axis_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential_one.displacement([1.0, 3.1, 0.0], [-0.01, 0.1, -0.3], 0.05)
        with self.assertRaises(AssertionError):
            self._potential_one.derivative([0.0, 1.0, 3.1], [0.2, 0.1, -0.3])

    def test_prefactor_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            LennardJonesPotential(prefactor=-0.5, characteristic_length=1.0)

    def test_prefactor_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            LennardJonesPotential(prefactor=0.0, characteristic_length=1.0)

    def test_characteristic_length_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            LennardJonesPotential(prefactor=1.0, characteristic_length=-0.3)

    def test_characteristic_length_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            LennardJonesPotential(prefactor=1.0, characteristic_length=0.0)


if __name__ == '__main__':
    main()
