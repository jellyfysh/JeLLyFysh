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
from math import cos, sin
from unittest import TestCase, main
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.potential.bending_potential import BendingPotential
import jellyfysh.setting as setting
from jellyfysh.setting import hypercubic_setting


class TestBendingPotential(TestCase):
    def setUp(self) -> None:
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        self._potential = BendingPotential(equilibrium_angle=1.5, prefactor=2.5)

    def tearDown(self) -> None:
        setting.reset()

    def test_equilibrium_position(self):
        separation_one = [1.0, 0.0, 0.0]
        separation_two = [cos(1.5), sin(1.5), 0.0]
        # The derivative does not depend on the absolute value of the velocity but only on the non-vanishing component.
        self.assertAlmostEqual(self._potential.derivative([1.0, 0.0, 0.0], separation_one, separation_two)[0], 0.0,
                               places=13)
        self.assertAlmostEqual(self._potential.derivative([2.0, 0.0, 0.0], separation_one, separation_two)[1], 0.0,
                               places=13)
        self.assertAlmostEqual(self._potential.derivative([3.0, 0.0, 0.0], separation_one, separation_two)[2], 0.0,
                               places=13)
        self.assertAlmostEqual(self._potential.derivative([0.0, 1.0, 0.0], separation_one, separation_two)[0], 0.0,
                               places=13)
        self.assertAlmostEqual(self._potential.derivative([0.0, 2.0, 0.0], separation_one, separation_two)[1], 0.0,
                               places=13)
        self.assertAlmostEqual(self._potential.derivative([0.0, 3.0, 0.0], separation_one, separation_two)[2], 0.0,
                               places=13)
        self.assertAlmostEqual(self._potential.derivative([0.0, 0.0, 1.0], separation_one, separation_two)[0], 0.0,
                               places=13)
        self.assertAlmostEqual(self._potential.derivative([0.0, 0.0, 2.0], separation_one, separation_two)[1], 0.0,
                               places=13)
        self.assertAlmostEqual(self._potential.derivative([0.0, 0.0, 3.0], separation_one, separation_two)[2], 0.0,
                               places=13)

    def test_derivative_returns_three_derivatives(self):
        separation_one = [1.0, 0.0, 0.0]
        separation_two = [cos(1.5), sin(1.5), 0.0]
        derivatives = self._potential.derivative([1.0, 0.0, 0.0], separation_one, separation_two)
        self.assertEqual(len(derivatives), 3)

    def test_non_equilibrium_position(self):
        separation_one = [1.0, 0.0, 0.0]
        separation_two = [0.7847523203428, 0.3446800291675, 0.490292012]
        self.assertAlmostEqual(self._potential.derivative([1.0, 0.0, 0.0], separation_one, separation_two)[2],
                               1.3027748792464318, places=12)
        self.assertAlmostEqual(self._potential.derivative([3.1, 0.0, 0.0], separation_one, separation_two)[2],
                               1.3027748792464318 * 3.1, places=12)
        self.assertAlmostEqual(self._potential.derivative([0.0, 1.0, 0.0], separation_one, separation_two)[2],
                               -0.9810545747373272, places=13)
        self.assertAlmostEqual(self._potential.derivative([0.0, 3.1, 0.0], separation_one, separation_two)[2],
                               -0.9810545747373272 * 3.1, places=13)
        self.assertAlmostEqual(self._potential.derivative([0.0, 0.0, 1.0], separation_one, separation_two)[2],
                               -1.395506500598621, places=12)
        self.assertAlmostEqual(self._potential.derivative([0.0, 0.0, 3.1], separation_one, separation_two)[2],
                               -1.395506500598621 * 3.1, places=12)

    def test_number_separation_arguments_is_two(self):
        self.assertEqual(self._potential.number_separation_arguments, 2)

    def test_number_charge_arguments_is_zero(self):
        self.assertEqual(self._potential.number_charge_arguments, 0)

    def test_velocity_zero_raises_error(self):
        separation_one = [1.0, 0.0, 0.0]
        separation_two = [cos(1.5), sin(1.5), 0.0]
        with self.assertRaises(AssertionError):
            self._potential.derivative([0.0, 0.0, 0.0], separation_one, separation_two)

    def test_negative_velocity_along_axis_raises_error(self):
        separation_one = [1.0, 0.0, 0.0]
        separation_two = [cos(1.5), sin(1.5), 0.0]
        with self.assertRaises(AssertionError):
            self._potential.derivative([-1.0, 0.0, 0.0], separation_one, separation_two)
        with self.assertRaises(AssertionError):
            self._potential.derivative([0.0, -1.0, 0.0], separation_one, separation_two)
        with self.assertRaises(AssertionError):
            self._potential.derivative([0.0, 0.0, -1.0], separation_one, separation_two)

    def test_velocity_not_parallel_to_axis_raises_error(self):
        separation_one = [1.0, 0.0, 0.0]
        separation_two = [cos(1.5), sin(1.5), 0.0]
        with self.assertRaises(AssertionError):
            self._potential.derivative([1.0, 2.0, 0.0], separation_one, separation_two)

    def test_prefactor_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            BendingPotential(equilibrium_angle=1.5, prefactor=0.0)


if __name__ == '__main__':
    main()
