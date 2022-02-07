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
from jellyfysh.potential.hard_dipole_potential import HardDipolePotential


class TestHardSpherePotential(TestCase):
    def setUp(self) -> None:
        self._potential = HardDipolePotential(0.8, 1.2)

    def test_displacement(self):
        # First test velocities aligned with coordinate axes.
        separation = [0.7, 0.6]
        velocity = [1.0, 0.0]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 0.1708497377870817, places=13)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 0.8, places=13)
        velocity = [0.0, 1.0]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 0.212701665379258, places=13)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 0.8, places=13)
        velocity = [-1.0, 0.0]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 0.3392304845413265, places=13)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 1.2, places=12)
        velocity = [0.0, -1.0]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 0.3746794344808964, places=13)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 1.2, places=12)

        # Now test general velocities.
        # Test velocity * separation > 0.0 and active unit can collide with the inner sphere.
        velocity = [0.3, 0.4]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 0.250806661517033, places=13)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 0.8, places=13)
        velocity = [0.6, -0.1]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 0.3572548243300255, places=13)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 0.8, places=13)
        velocity = [0.7, 0.6]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 0.1322781687253752, places=13)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 0.8, places=13)
        # The following velocities lie on the edges of the region where the active unit can collide with the inner
        # sphere.
        velocity = [-0.1123903774256137, 0.589379676492266]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 0.7637620340313499, places=13)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 0.8, places=13)
        velocity = [0.5652566847512938, -0.2012085493809256]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 0.7635783963430719, places=13)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 0.8, places=13)

        # Now test velocity * separation > 0.0 but the active unit cannot collide with the inner sphere.
        # The following velocities lie on the edges of the region where the active unit cannot collide with the inner
        # sphere.
        velocity = [-0.1123903774258101, 0.5893796764922285]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 2.254474600825496, places=12)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 1.2, places=12)
        velocity = [0.565256678044342, -0.201208568222815]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 2.25447456721811, places=12)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 1.2, places=12)
        # The following velocities are away from the edges.
        velocity = [0.5, -0.3]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 1.909004654941586, places=12)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 1.2, places=12)
        velocity = [-0.2, 0.6]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 1.883229162597338, places=12)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 1.2, places=12)

        # Finally, test velocity * separation < 0.0 where spheres cannot collide.
        velocity = [-0.6, 0.6]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 0.825726009552976, places=13)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 1.2, places=12)
        velocity = [0.5, -0.6]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 0.967213114754099, places=13)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 1.2, places=12)
        velocity = [-0.1, -0.2]
        displacement = self._potential.displacement(velocity, separation)
        self.assertAlmostEqual(displacement, 1.322499389946279, places=12)
        self.assertAlmostEqual(sum((entry - velocity[index] * displacement) ** 2
                                   for index, entry in enumerate(separation)) ** 0.5, 1.2, places=12)

    def test_number_separation_arguments_is_one(self):
        self.assertEqual(self._potential.number_separation_arguments, 1)

    def test_number_charge_arguments_is_zero(self):
        self.assertEqual(self._potential.number_charge_arguments, 0)

    def test_potential_change_not_required(self):
        self.assertFalse(self._potential.potential_change_required)

    def test_velocity_zero_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential.displacement([0.0, 0.0], [0.7, 0.6])

    def test_too_small_separation_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential.displacement([1.0, 0.0], [0.8 - 1e-13, 0.0])

    def test_too_large_separation_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential.displacement([1.0, 0.0], [0.0, 1.2 + 1e-13])

    def test_derivative_raises_error(self):
        with self.assertRaises(NotImplementedError):
            self._potential.derivative([1.0, 0.0], [0.7, 0.6])

    def test_minimum_separation_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            HardDipolePotential(-1.0, 1.2)

    def test_minimum_separation_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            HardDipolePotential(0.0, 1.2)

    def test_maximum_separation_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            HardDipolePotential(0.8, -1.0)

    def test_maximum_separation_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            HardDipolePotential(0.8, 0.0)

    def test_minimum_separation_larger_than_maximum_separation_raises_error(self):
        with self.assertRaises(ConfigurationError):
            HardDipolePotential(1.2, 0.8)

    def test_minimum_separation_equal_to_maximum_separation_raises_error(self):
        with self.assertRaises(ConfigurationError):
            HardDipolePotential(0.8, 0.8)


if __name__ == '__main__':
    main()
