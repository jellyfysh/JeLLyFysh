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
from jellyfysh.potential.hard_sphere_potential import HardSpherePotential


class TestHardSpherePotential(TestCase):
    def setUp(self) -> None:
        self._potential = HardSpherePotential(0.4)

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
        self.assertEqual(self._potential.displacement([-1.0, 0.0], [0.7, 0.6]), float("inf"))
        self.assertEqual(self._potential.displacement([0.0, -1.0], [0.7, 0.6]), float("inf"))

        # Now test general velocities.
        # Test velocity * separation > 0.0 and spheres can collide.
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
        # The following velocities lie on the edges of the region where the spheres can collide.
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

        # Now test velocity * separation > 0.0 but spheres cannot collide.
        # The following velocities lie on the edges of the region where the spheres cannot collide.
        self.assertEqual(self._potential.displacement([-0.1123903774258101, 0.5893796764922285], [0.7, 0.6]),
                         float("inf"))
        self.assertEqual(self._potential.displacement([0.5652566813977844, -0.2012085588019645], [0.7, 0.6]),
                         float("inf"))
        # The following velocities are away from the edges.
        self.assertEqual(self._potential.displacement([0.5, -0.3], [0.7, 0.6]), float("inf"))
        self.assertEqual(self._potential.displacement([-0.2, 0.6], [0.7, 0.6]), float("inf"))

        # Finally, test velocity * separation < 0.0 where spheres cannot collide.
        self.assertEqual(self._potential.displacement([-0.6, 0.6], [0.7, 0.6]), float("inf"))
        self.assertEqual(self._potential.displacement([0.5, -0.6], [0.7, 0.6]), float("inf"))
        self.assertEqual(self._potential.displacement([-0.1, -0.2], [0.7, 0.6]), float("inf"))

    def test_number_separation_arguments_is_one(self):
        self.assertEqual(self._potential.number_separation_arguments, 1)

    def test_number_charge_arguments_is_zero(self):
        self.assertEqual(self._potential.number_charge_arguments, 0)

    def test_potential_change_not_required(self):
        self.assertFalse(self._potential.potential_change_required)

    def test_velocity_zero_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential.displacement([0.0, 0.0], [1.0, 1.0])

    def test_too_small_separation_raises_error(self):
        with self.assertRaises(AssertionError):
            self._potential.displacement([1.0, 0.0], [0.8 - 1e-13, 0.0])

    def test_derivative_raises_error(self):
        with self.assertRaises(NotImplementedError):
            self._potential.derivative([1.0, 0.0], [1.0, 1.0])

    def test_radius_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            HardSpherePotential(-1.0)

    def test_radius_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            HardSpherePotential(0.0)


if __name__ == '__main__':
    main()
