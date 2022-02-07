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
from jellyfysh.potential.displaced_even_power_potential import DisplacedEvenPowerPotential


# noinspection PyArgumentEqualDefault
class TestDisplacedEvenPowerPotential(TestCase):
    def setUp(self) -> None:
        self._potential = DisplacedEvenPowerPotential(equilibrium_separation=0.1, power=2, prefactor=0.5)

    def test_displacement_active_unit_behind_outside_potential_minimum_sphere_can_reach_sphere(self):
        height_potential_hill = 0.0006921428753625723
        self._potential._prefactor = 0.5 * 0.5 * 1.3
        # Active unit can climb potential hill without problem
        self.assertAlmostEqual(self._potential.displacement([1.0, 0.0, 0.0], [0.3, 0.05, -0.02], 0.3),
                               1.358290719101263, places=12)
        # The time displacement depends on the value of the non-vanishing component of the velocity.
        self.assertAlmostEqual(self._potential.displacement([3.1, 0.0, 0.0], [0.3, 0.05, -0.02], 0.3),
                               1.358290719101263 / 3.1, places=13)

        # Active unit cannot climb potential hill by far
        self.assertAlmostEqual(self._potential.displacement([0.0, 1.0, 0.0], [0.05, 0.3, -0.02], 0.0002),
                               0.2475214783762321, places=13)
        self.assertAlmostEqual(self._potential.displacement([0.0, 3.1, 0.0], [0.05, 0.3, -0.02], 0.0002),
                               0.2475214783762321 / 3.1, places=13)

        # Active unit can barely climb potential hill
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 1.0], [-0.02, 0.05, 0.3], height_potential_hill + 1.e-13), 0.3842621560389638, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 3.1], [-0.02, 0.05, 0.3], height_potential_hill + 1.e-13), 0.3842621560389638 / 3.1, places=13)

        # Active unit can barely not climb potential hill
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [0.3, -0.02, 0.05], height_potential_hill - 1.e-13), 0.2999994007887726, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [0.3, -0.02, 0.05], height_potential_hill - 1.e-13), 0.2999994007887726 / 3.1, places=13)

    def test_displacement_active_unit_behind_outside_potential_minimum_sphere_cannot_reach_sphere(self):
        # Test in all directions
        self._potential._prefactor = 0.5 * 2.1 * 1.1
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [0.3, 0.1, 0.02], 0.4), 0.980898606235689, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [0.3, 0.1, 0.02], 0.4), 0.980898606235689 / 3.1, places=13)

        self._potential._prefactor = 0.5 * 0.4 * 0.4
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 1.0, 0.0], [-0.1, 0.2, 0.0001], 0.002), 0.4379553878391659, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 3.1, 0.0], [-0.1, 0.2, 0.0001], 0.002), 0.4379553878391659 / 3.1, places=13)

        self._potential._prefactor = 0.5
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 1.0], [0.09, -0.09, 0.001], 0.1), 0.5340601455693887, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 3.1], [0.09, -0.09, 0.001], 0.1), 0.5340601455693887 / 3.1, places=13)

    def test_displacement_active_unit_behind_inside_potential_minimum_sphere(self):
        height_potential_hill = 0.0000895627250851899
        self._potential._prefactor = 0.5 * 0.2 * 0.3
        # Active unit can climb potential without problem
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [0.07, 0.03, -0.03], 0.1), 1.99445647542066, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [0.07, 0.03, -0.03], 0.1), 1.99445647542066 / 3.1, places=13)

        # Active unit cannot climb potential hill by far
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 1.0, 0.0], [0.03, 0.07, -0.03], 0.000005), 0.004869943963535905, places=15)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 3.1, 0.0], [0.03, 0.07, -0.03], 0.000005), 0.004869943963535905 / 3.1, places=15)

        # Active unit can barely climb potential hill
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 1.0], [-0.03, 0.03, 0.07], height_potential_hill + 1.e-13), 0.1605558675719981, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 3.1], [-0.03, 0.03, 0.07], height_potential_hill + 1.e-13), 0.1605558675719981 / 3.1, places=13)

        # Active unit can barely not climb potential hill
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [0.07, -0.03, 0.03], height_potential_hill - 1.e-13), 0.06999843272252484, places=14)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [0.07, -0.03, 0.03], height_potential_hill - 1.e-13), 0.06999843272252484 / 3.1, places=14)

    def test_displacement_active_unit_in_front_inside_potential_minimum_sphere(self):
        self._potential._prefactor = 0.5 * 1.3 * 0.7
        # Test in all directions
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [-0.03, 0.04, 0.05], 0.1), 0.5351917072710099, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [-0.03, 0.04, 0.05], 0.1), 0.5351917072710099 / 3.1, places=13)

        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 1.0, 0.0], [0.04, -0.03, 0.05], 0.1), 0.5351917072710099, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 3.1, 0.0], [0.04, -0.03, 0.05], 0.1), 0.5351917072710099 / 3.1, places=13)

        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 1.0], [0.05, 0.04, -0.03], 0.1), 0.5351917072710099, places=13)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 3.1], [0.05, 0.04, -0.03], 0.1), 0.5351917072710099 / 3.1, places=13)

        # Potential bug if the active particle is exactly on the potential minimum sphere
        potential = DisplacedEvenPowerPotential(equilibrium_separation=1.012, power=2, prefactor=529.581)
        self.assertAlmostEqual(potential.displacement(
            [0.0, 0.0, 1.0], [0.8450574268193556, -0.5567961434649575, 0.0], 0.5323176176410311),
            0.2552936081560725, places=13)
        self.assertAlmostEqual(potential.displacement(
            [0.0, 0.0, 3.1], [0.8450574268193556, -0.5567961434649575, 0.0], 0.5323176176410311),
            0.2552936081560725 / 3.1, places=13)

    def test_displacement_active_unit_in_front_outside_potential_minimum_sphere(self):
        self._potential._prefactor = 0.5 * 0.22 * 0.11
        # Test in all directions
        self.assertAlmostEqual(self._potential.displacement(
            [1.0, 0.0, 0.0], [-0.01, 0.1, -0.3], 0.05), 2.110827419070795, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [3.1, 0.0, 0.0], [-0.01, 0.1, -0.3], 0.05), 2.110827419070795 / 3.1, places=12)

        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 1.0, 0.0], [0.1, -0.01, -0.3], 0.05), 2.110827419070795, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 3.1, 0.0], [0.1, -0.01, -0.3], 0.05), 2.110827419070795 / 3.1, places=12)

        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 1.0], [-0.3, 0.1, -0.01], 0.05), 2.110827419070795, places=12)
        self.assertAlmostEqual(self._potential.displacement(
            [0.0, 0.0, 3.1], [-0.3, 0.1, -0.01], 0.05), 2.110827419070795 / 3.1, places=12)

    def test_derivative(self):
        self._potential._prefactor = 0.5 * 0.7 * 0.9
        # Test in all directions
        self.assertAlmostEqual(self._potential.derivative(
            [1.0, 0.0, 0.0], [0.2, 0.1, -0.3]), -0.0923250835190345, places=14)
        # The derivative does not depend on the absolute value of the velocity but only on the non-vanishing component.
        self.assertAlmostEqual(self._potential.derivative(
            [3.1, 0.0, 0.0], [0.2, 0.1, -0.3]), -0.0923250835190345 * 3.1, places=14)

        self.assertAlmostEqual(self._potential.derivative(
            [0.0, 1.0, 0.0], [0.1, 0.2, -0.3]), -0.0923250835190345, places=14)
        self.assertAlmostEqual(self._potential.derivative(
            [0.0, 3.1, 0.0], [0.1, 0.2, -0.3]), -0.0923250835190345 * 3.1, places=14)

        self.assertAlmostEqual(self._potential.derivative(
            [0.0, 0.0, 1.0], [-0.3, 0.1, 0.2]), -0.0923250835190345, places=14)
        self.assertAlmostEqual(self._potential.derivative(
            [0.0, 0.0, 3.1], [-0.3, 0.1, 0.2]), -0.0923250835190345 * 3.1, places=14)

    def test_number_separation_arguments_is_one(self):
        self.assertEqual(self._potential.number_separation_arguments, 1)

    def test_number_charge_arguments_is_zero(self):
        self.assertEqual(self._potential.number_charge_arguments, 0)

    def test_potential_change_required(self):
        self.assertTrue(self._potential.potential_change_required)

    def test_velocity_zero_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential.displacement([0.0, 0.0, 0.0], [-0.01, 0.1, -0.3], 0.05)
        with self.assertRaises(AssertionError):
            self._potential.derivative([0.0, 0.0, 0.0], [0.2, 0.1, -0.3])

    def test_negative_velocity_along_axis_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential.displacement([-1.0, 0.0, 0.0], [-0.01, 0.1, -0.3], 0.05)
        with self.assertRaises(AssertionError):
            self._potential.displacement([0.0, -1.0, 0.0], [-0.01, 0.1, -0.3], 0.05)
        with self.assertRaises(AssertionError):
            self._potential.displacement([0.0, 0.0, -1.0], [-0.01, 0.1, -0.3], 0.05)
        with self.assertRaises(AssertionError):
            self._potential.derivative([-3.1, 0.0, 0.0], [0.2, 0.1, -0.3])
        with self.assertRaises(AssertionError):
            self._potential.derivative([0.0, -3.1, 0.0], [0.2, 0.1, -0.3])
        with self.assertRaises(AssertionError):
            self._potential.derivative([0.0, 0.0, -3.1], [0.2, 0.1, -0.3])

    def test_velocity_not_parallel_to_axis_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential.displacement([1.0, 3.1, 0.0], [-0.01, 0.1, -0.3], 0.05)
        with self.assertRaises(AssertionError):
            self._potential.derivative([0.0, 1.0, 3.1], [0.2, 0.1, -0.3])

    def test_prefactor_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            DisplacedEvenPowerPotential(equilibrium_separation=0.1, power=2, prefactor=-0.5)

    def test_prefactor_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            DisplacedEvenPowerPotential(equilibrium_separation=0.1, power=2, prefactor=0.0)

    def test_power_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            DisplacedEvenPowerPotential(equilibrium_separation=0.1, power=0, prefactor=1.0)

    def test_power_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            DisplacedEvenPowerPotential(equilibrium_separation=0.1, power=-1, prefactor=1.0)

    def test_power_odd_raises_error(self):
        with self.assertRaises(ConfigurationError):
            DisplacedEvenPowerPotential(equilibrium_separation=0.1, power=1, prefactor=1.0)
        with self.assertRaises(ConfigurationError):
            DisplacedEvenPowerPotential(equilibrium_separation=0.1, power=3, prefactor=1.0)

    def test_equilibrium_separation_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            DisplacedEvenPowerPotential(equilibrium_separation=0.0, power=2, prefactor=1.0)

    def test_equilibrium_separation_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            DisplacedEvenPowerPotential(equilibrium_separation=-0.3, power=2, prefactor=1.0)


if __name__ == '__main__':
    main()
