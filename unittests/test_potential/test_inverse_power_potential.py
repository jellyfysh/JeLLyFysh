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
from jellyfysh.potential.inverse_power_potential import InversePowerPotential


class TestInversePowerPotential(TestCase):
    def setUp(self) -> None:
        prefactor_power_one = 1.32
        self._potential_power_one = InversePowerPotential(power=1.0, prefactor=prefactor_power_one)
        prefactor_power_two = -0.57
        self._potential_power_two = InversePowerPotential(power=2.0, prefactor=prefactor_power_two)

    def test_derivative_positive_charge_product_power_one(self):
        charge_one = 3.4
        charge_two = 0.3
        separation = [-0.4, 0.2, 0.3]
        self.assertAlmostEqual(
            self._potential_power_one.derivative([1.0, 0.0, 0.0], separation, charge_one, charge_two),
            -3.448554528573553, places=12)
        # The derivative does not depend on the absolute value of the velocity but only on the non-vanishing component.
        separation = [-0.4, 0.2, 0.3]
        self.assertAlmostEqual(
            self._potential_power_one.derivative([3.1, 0.0, 0.0], separation, charge_one, charge_two),
            -3.448554528573553 * 3.1, places=12)

        charge_one = -0.2
        charge_two = -0.9
        separation = [0.0, 0.1, 0.0]
        self.assertAlmostEqual(
            self._potential_power_one.derivative([0.0, 1.0, 0.0], separation, charge_one, charge_two),
            23.76, places=11)
        separation = [0.0, 0.1, 0.0]
        self.assertAlmostEqual(
            self._potential_power_one.derivative([0.0, 3.1, 0.0], separation, charge_one, charge_two),
            23.76 * 3.1, places=11)

        charge_one = 0.5
        charge_two = 0.5
        separation = [0.7, -0.01, 0.0]
        self.assertAlmostEqual(
            self._potential_power_one.derivative([0.0, 0.0, 1.0], separation, charge_one, charge_two),
            0.0, places=13)
        separation = [0.7, -0.01, 0.0]
        self.assertAlmostEqual(
            self._potential_power_one.derivative([0.0, 0.0, 3.1], separation, charge_one, charge_two),
            0.0, places=13)

    def test_derivative_negative_charge_product_power_one(self):
        charge_one = 1.3
        charge_two = -1
        separation = [0.1, 0.2, 0.3]
        self.assertAlmostEqual(
            self._potential_power_one.derivative([1.0, 0.0, 0.0], separation, charge_one, charge_two),
            -3.275859222298002, places=12)
        separation = [0.1, 0.2, 0.3]
        self.assertAlmostEqual(
            self._potential_power_one.derivative([3.1, 0.0, 0.0], separation, charge_one, charge_two),
            -3.275859222298002 * 3.1, places=12)

        charge_one = -2.3
        charge_two = 0.1
        separation = [-0.1, -0.2, -0.3]
        self.assertAlmostEqual(
            self._potential_power_one.derivative([0.0, 1.0, 0.0], separation, charge_one, charge_two),
            1.1591501863516, places=12)
        separation = [-0.1, -0.2, -0.3]
        self.assertAlmostEqual(
            self._potential_power_one.derivative([0.0, 3.1, 0.0], separation, charge_one, charge_two),
            1.1591501863516 * 3.1, places=12)

        charge_one = 0.0
        charge_two = 3.2
        separation = [-0.3, 0.3, 0.3]
        self.assertAlmostEqual(
            self._potential_power_one.derivative([0.0, 0.0, 1.0], separation, charge_one, charge_two),
            0.0, places=13)
        separation = [-0.3, 0.3, 0.3]
        self.assertAlmostEqual(
            self._potential_power_one.derivative([0.0, 0.0, 3.1], separation, charge_one, charge_two),
            0.0, places=13)

    def test_displacement_positive_charge_product_power_one(self):
        # Here we test a repulsive interaction between the active and target unit
        # Active unit is in front of target unit -> infinite displacement
        charge_one = 3.4
        charge_two = 0.3
        separation = [-0.4, 0.2, 0.3]
        self.assertEqual(
            self._potential_power_one.displacement([1.0, 0.0, 0.0], separation, charge_one, charge_two, 3.4),
            float('inf'))
        # The time displacement depends on the value of the non-vanishing component of the velocity.
        separation = [-0.4, 0.2, 0.3]
        self.assertEqual(
            self._potential_power_one.displacement([3.1, 0.0, 0.0], separation, charge_one, charge_two, 3.4),
            float('inf'))

        # Active unit is behind of target unit and cannot climb potential hill
        charge_one = -0.2
        charge_two = -0.9
        separation = [0.1, 0.1, 0.0]
        self.assertAlmostEqual(
            self._potential_power_one.displacement([0.0, 1.0, 0.0], separation, charge_one, charge_two, 0.3),
            0.03367690176487714, places=14)
        separation = [0.1, 0.1, 0.0]
        self.assertAlmostEqual(
            self._potential_power_one.displacement([0.0, 3.1, 0.0], separation, charge_one, charge_two, 0.3),
            0.03367690176487714 / 3.1, places=14)

        # Active unit is behind of target unit and can barely climb potential hill
        charge_one = 0.5
        charge_two = 0.5
        separation = [0.7, -0.01, 0.2]
        potential_difference = 0.01813336994737375
        self.assertEqual(
            self._potential_power_one.displacement([0.0, 0.0, 1.0], separation, charge_one, charge_two,
                                                   potential_difference + 1.e-13),
            float('inf'))
        separation = [0.7, -0.01, 0.2]
        self.assertEqual(
            self._potential_power_one.displacement([0.0, 0.0, 3.1], separation, charge_one, charge_two,
                                                   potential_difference + 1.e-13),
            float('inf'))

        # Active unit is behind of target unit and can barely not climb potential hill
        separation = [0.7, -0.01, 0.2]
        self.assertAlmostEqual(
            self._potential_power_one.displacement([0.0, 0.0, 1.0], separation, charge_one, charge_two,
                                                   potential_difference - 1.e-13),
            0.1999995441730163, places=13)
        separation = [0.7, -0.01, 0.2]
        self.assertAlmostEqual(
            self._potential_power_one.displacement([0.0, 0.0, 3.1], separation, charge_one, charge_two,
                                                   potential_difference - 1.e-13),
            0.1999995441730163 / 3.1, places=13)

    def test_displacement_negative_charge_product_power_one(self):
        # Here we test an attractive interaction between the active and target unit
        # Active unit is in front of target unit
        charge_one = 0.7
        charge_two = -0.6
        separation = [-0.3, -0.7, 0.1]
        self.assertAlmostEqual(
            self._potential_power_one.displacement([1.0, 0.0, 0.0], separation, charge_one, charge_two, 0.3),
            0.808074145460921, places=13)
        separation = [-0.3, -0.7, 0.1]
        self.assertAlmostEqual(
            self._potential_power_one.displacement([3.1, 0.0, 0.0], separation, charge_one, charge_two, 0.3),
            0.808074145460921 / 3.1, places=13)

        # Active unit is in front of target unit and can barely climb potential hill
        charge_one = -1.2
        charge_two = 0.9
        separation = [0.1, -0.2, -0.3]
        potential_difference = 3.810076264703522
        self.assertEqual(
            self._potential_power_one.displacement([0.0, 1.0, 0.0], separation, charge_one, charge_two,
                                                   potential_difference + 1.e-13), float('inf'))
        separation = [0.1, -0.2, -0.3]
        self.assertEqual(
            self._potential_power_one.displacement([0.0, 3.1, 0.0], separation, charge_one, charge_two,
                                                   potential_difference + 1.e-13), float('inf'))

        # Active unit is in front of target unit and can barely not climb potential hill
        separation = [0.1, -0.2, -0.3]
        self.assertAlmostEqual(
            self._potential_power_one.displacement([0.0, 1.0, 0.0], separation, charge_one, charge_two,
                                                   potential_difference - 1.e-13),
            1.420427351499843e13, places=1)
        separation = [0.1, -0.2, -0.3]
        self.assertAlmostEqual(
            self._potential_power_one.displacement([0.0, 3.1, 0.0], separation, charge_one, charge_two,
                                                   potential_difference - 1.e-13),
            1.420427351499843e13 / 3.1, places=1)

        # Active unit is behind of target unit
        charge_one = 0.7
        charge_two = -0.6
        separation = [0.2, -0.7, 0.1]
        self.assertAlmostEqual(
            self._potential_power_one.displacement([1.0, 0.0, 0.0], separation, charge_one, charge_two, 0.3),
            0.2 + 0.901026790768559, places=12)
        separation = [0.2, -0.7, 0.1]
        self.assertAlmostEqual(
            self._potential_power_one.displacement([3.1, 0.0, 0.0], separation, charge_one, charge_two, 0.3),
            (0.2 + 0.901026790768559) / 3.1, places=12)

    def test_derivative_positive_charge_product_power_two(self):
        charge_one = 3.4
        charge_two = 0.3
        separation = [-0.4, 0.2, 0.3]
        self.assertAlmostEqual(
            self._potential_power_two.derivative([1.0, 0.0, 0.0], separation, charge_one, charge_two),
            5.530558858501782, places=12)
        separation = [-0.4, 0.2, 0.3]
        self.assertAlmostEqual(
            self._potential_power_two.derivative([3.1, 0.0, 0.0], separation, charge_one, charge_two),
            5.530558858501782 * 3.1, places=12)

        charge_one = -0.2
        charge_two = -0.9
        separation = [0.0, 0.1, 0.0]
        self.assertAlmostEqual(
            self._potential_power_two.derivative([0.0, 1.0, 0.0], separation, charge_one, charge_two),
            -205.1999999999999, places=10)
        separation = [0.0, 0.1, 0.0]
        self.assertAlmostEqual(
            self._potential_power_two.derivative([0.0, 3.1, 0.0], separation, charge_one, charge_two),
            -205.1999999999999 * 3.1, places=10)

        charge_one = 0.5
        charge_two = 0.5
        separation = [0.7, -0.01, 0.0]
        self.assertAlmostEqual(
            self._potential_power_two.derivative([0.0, 0.0, 1.0], separation, charge_one, charge_two), 0.0, places=13)
        separation = [0.7, -0.01, 0.0]
        self.assertAlmostEqual(
            self._potential_power_two.derivative([0.0, 0.0, 3.1], separation, charge_one, charge_two), 0.0, places=13)

    def test_derivative_negative_charge_product_power_two(self):
        charge_one = 1.3
        charge_two = -1
        separation = [0.1, 0.2, 0.3]
        self.assertAlmostEqual(
            self._potential_power_two.derivative([1.0, 0.0, 0.0], separation, charge_one, charge_two),
            7.561224489795917, places=12)
        separation = [0.1, 0.2, 0.3]
        self.assertAlmostEqual(
            self._potential_power_two.derivative([3.1, 0.0, 0.0], separation, charge_one, charge_two),
            7.561224489795917 * 3.1, places=12)

        charge_one = -2.3
        charge_two = 0.1
        separation = [-0.1, -0.2, -0.3]
        self.assertAlmostEqual(
            self._potential_power_two.derivative([0.0, 1.0, 0.0], separation, charge_one, charge_two),
            -2.675510204081631, places=12)
        separation = [-0.1, -0.2, -0.3]
        self.assertAlmostEqual(
            self._potential_power_two.derivative([0.0, 3.1, 0.0], separation, charge_one, charge_two),
            -2.675510204081631 * 3.1, places=12)

        charge_one = 0.0
        charge_two = 3.2
        separation = [-0.3, 0.3, 0.3]
        self.assertAlmostEqual(
            self._potential_power_two.derivative([0.0, 0.0, 1.0], separation, charge_one, charge_two), 0.0, places=13)
        separation = [-0.3, 0.3, 0.3]
        self.assertAlmostEqual(
            self._potential_power_two.derivative([0.0, 0.0, 3.1], separation, charge_one, charge_two), 0.0, places=13)

    def test_displacement_positive_charge_product_power_two(self):
        # Here we test an attractive interaction between the active and target unit
        # Active unit is in front of target unit
        charge_one = 0.7
        charge_two = 0.6
        separation = [-0.3, -0.7, 0.1]
        self.assertAlmostEqual(
            self._potential_power_two.displacement([1.0, 0.0, 0.0], separation, charge_one, charge_two, 0.3),
            1.027990094958428, places=12)
        separation = [-0.3, -0.7, 0.1]
        self.assertAlmostEqual(
            self._potential_power_two.displacement([3.1, 0.0, 0.0], separation, charge_one, charge_two, 0.3),
            1.027990094958428 / 3.1, places=12)

        # Active unit is in front of target unit and can barely climb potential hill
        charge_one = 1.2
        charge_two = 0.9
        separation = [0.1, -0.2, -0.3]
        potential_difference = 4.397142857142857
        self.assertEqual(
            self._potential_power_two.displacement([0.0, 1.0, 0.0], separation, charge_one, charge_two,
                                                   potential_difference + 1.e-13), float('inf'))
        separation = [0.1, -0.2, -0.3]
        self.assertEqual(
            self._potential_power_two.displacement([0.0, 3.1, 0.0], separation, charge_one, charge_two,
                                                   potential_difference + 1.e-13), float('inf'))

        # Active unit is in front of target unit and can barely not climb potential hill
        separation = [0.1, -0.2, -0.3]
        self.assertAlmostEqual(
            self._potential_power_two.displacement([0.0, 1.0, 0.0], separation, charge_one, charge_two,
                                                   potential_difference - 1.e-13),
            2.476623219758192e6, places=7)
        separation = [0.1, -0.2, -0.3]
        self.assertAlmostEqual(
            self._potential_power_two.displacement([0.0, 3.1, 0.0], separation, charge_one, charge_two,
                                                   potential_difference - 1.e-13),
            2.476623219758192e6 / 3.1, places=7)

        # Active unit is behind of target unit
        charge_one = 0.7
        charge_two = 0.6
        separation = [0.2, -0.7, 0.1]
        self.assertAlmostEqual(
            self._potential_power_two.displacement([1.0, 0.0, 0.0], separation, charge_one, charge_two, 0.3),
            0.2 + 0.915929131809139, places=12)
        separation = [0.2, -0.7, 0.1]
        self.assertAlmostEqual(
            self._potential_power_two.displacement([3.1, 0.0, 0.0], separation, charge_one, charge_two, 0.3),
            (0.2 + 0.915929131809139) / 3.1, places=12)

    def test_displacement_negative_charge_product_power_two(self):
        # Here we test a repulsive interaction between the active and target unit (prefactor is < 0)
        # Active unit is in front of target unit -> infinite displacement
        charge_one = 3.4
        charge_two = -0.3
        separation = [-0.4, 0.2, 0.3]
        self.assertEqual(
            self._potential_power_two.displacement([1.0, 0.0, 0.0], separation, charge_one, charge_two, 3.4),
            float('inf'))
        separation = [-0.4, 0.2, 0.3]
        self.assertEqual(
            self._potential_power_two.displacement([3.1, 0.0, 0.0], separation, charge_one, charge_two, 3.4),
            float('inf'))

        # Active unit is behind of target unit and cannot climb potential hill
        charge_one = -0.2
        charge_two = 0.9
        separation = [0.1, 0.1, 0.0]
        self.assertAlmostEqual(
            self._potential_power_two.displacement([0.0, 1.0, 0.0], separation, charge_one, charge_two, 0.3),
            0.005686545899805995, places=15)
        separation = [0.1, 0.1, 0.0]
        self.assertAlmostEqual(
            self._potential_power_two.displacement([0.0, 3.1, 0.0], separation, charge_one, charge_two, 0.3),
            0.005686545899805995 / 3.1, places=15)

        # Active unit can barely climb potential hill
        charge_one = -0.5
        charge_two = 0.5
        separation = [0.7, -0.01, 0.2]
        potential_difference = 0.02193978406864522
        self.assertEqual(
            self._potential_power_two.displacement([0.0, 0.0, 1.0], separation, charge_one, charge_two,
                                                   potential_difference + 1.e-13), float('inf'))
        separation = [0.7, -0.01, 0.2]
        self.assertEqual(
            self._potential_power_two.displacement([0.0, 0.0, 3.1], separation, charge_one, charge_two,
                                                   potential_difference + 1.e-13), float('inf'))

        # Active unit can barely not climb potential hill
        separation = [0.7, -0.01, 0.2]
        self.assertAlmostEqual(
            self._potential_power_two.displacement([0.0, 0.0, 1.0], separation, charge_one, charge_two,
                                                   potential_difference - 1.e-13),
            0.1999995895413005, places=13)
        separation = [0.7, -0.01, 0.2]
        self.assertAlmostEqual(
            self._potential_power_two.displacement([0.0, 0.0, 3.1], separation, charge_one, charge_two,
                                                   potential_difference - 1.e-13),
            0.1999995895413005 / 3.1, places=13)

    def test_number_separation_arguments_is_one(self):
        self.assertEqual(self._potential_power_one.number_separation_arguments, 1)
        self.assertEqual(self._potential_power_two.number_separation_arguments, 1)

    def test_number_charge_arguments_is_two(self):
        self.assertEqual(self._potential_power_one.number_charge_arguments, 2)
        self.assertEqual(self._potential_power_two.number_charge_arguments, 2)

    def test_potential_change_required(self):
        self.assertTrue(self._potential_power_one.potential_change_required)
        self.assertTrue(self._potential_power_two.potential_change_required)

    def test_velocity_zero_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential_power_one.displacement([0.0, 0.0, 0.0], [0.7, 0.6, -0.1], 1.0, 1.0, 0.1)
        with self.assertRaises(AssertionError):
            self._potential_power_one.derivative([0.0, 0.0, 0.0], [0.7, 0.6, -0.1], 1.0, 1.0)
        with self.assertRaises(AssertionError):
            self._potential_power_two.displacement([0.0, 0.0, 0.0], [0.7, 0.6, -0.1], 1.0, 1.0, 0.1)
        with self.assertRaises(AssertionError):
            self._potential_power_two.derivative([0.0, 0.0, 0.0], [0.7, 0.6, -0.1], 1.0, 1.0)

    def test_negative_velocity_along_axis_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential_power_one.displacement([-1.0, 0.0, 0.0], [0.7, 0.6, -0.1], 1.0, 1.0, 0.1)
        with self.assertRaises(AssertionError):
            self._potential_power_one.derivative([0.0, -1.0, 0.0], [0.7, 0.6, -0.1], 1.0, 1.0)
        with self.assertRaises(AssertionError):
            self._potential_power_two.displacement([0.0, 0.0, -1.0], [0.7, 0.6, -0.1], 1.0, 1.0, 0.1)
        with self.assertRaises(AssertionError):
            self._potential_power_two.derivative([-1.0, 0.0, 0.0], [0.7, 0.6, -0.1], 1.0, 1.0)

    def test_velocity_not_parallel_to_axis_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential_power_one.displacement([1.0, 3.1, 0.0], [0.7, 0.6, -0.1], 1.0, 1.0, 0.1)
        with self.assertRaises(AssertionError):
            self._potential_power_one.derivative([0.0, 3.1, 1.0], [0.7, 0.6, -0.1], 1.0, 1.0)
        with self.assertRaises(AssertionError):
            self._potential_power_two.displacement([1.0, 3.1, 2.0], [0.7, 0.6, -0.1], 1.0, 1.0, 0.1)
        with self.assertRaises(AssertionError):
            self._potential_power_two.derivative([3.1, 0.0, 1.0], [0.7, 0.6, -0.1], 1.0, 1.0)

    def test_negative_power_raises_error(self):
        with self.assertRaises(ConfigurationError):
            InversePowerPotential(power=-3.2, prefactor=1)

    def test_power_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            InversePowerPotential(power=0, prefactor=1)

    def test_prefactor_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            InversePowerPotential(power=1, prefactor=0)


if __name__ == '__main__':
    main()
