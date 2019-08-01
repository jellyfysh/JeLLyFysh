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
import setting


class TestPeriodicSystem(TestCase):
    def tearDown(self):
        setting.reset()

    def assertAlmostEqualList(self, list_one, list_two, places=7):
        self.assertEqual(len(list_one), len(list_two))
        [self.assertAlmostEqual(list_one[i], list_two[i], places=places) for i in range(len(list_one))]

    def test_assert_almost_equal_list(self):
        list_one = [0.12345678]
        list_two = [0.12345676]
        self.assertAlmostEqualList(list_one, list_two)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualList(list_one, list_two, places=8)
        list_one = [0.00000000001, 0.1231, 0.57162]
        list_two = [0.00000000009, 0.1231, 0.57162]
        self.assertAlmostEqualList(list_one, list_two, places=9)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualList(list_one, list_two, places=10)

    def test_correct_periodic_boundary_position(self):
        setting.HypercuboidSetting(beta=1.0, dimension=3, system_lengths=[1.0, 2.0, 3.0])
        position = [0.1, 1.4, 2.9]
        setting.periodic_boundaries.correct_position(position)
        self.assertEqual(position, [0.1, 1.4, 2.9])
        position = [0.0, 2.0, 1.0]
        setting.periodic_boundaries.correct_position(position)
        self.assertAlmostEqualList(position, [0.0, 0.0, 1.0], places=13)
        position = [-0.3, 1.4, 3.9]
        setting.periodic_boundaries.correct_position(position)
        self.assertAlmostEqualList(position, [0.7, 1.4, 0.9], places=13)
        position = [-10.0, 102.231, 2.999999]
        setting.periodic_boundaries.correct_position(position)
        self.assertAlmostEqualList(position, [0.0, 0.231, 2.999999], places=13)

    def test_correct_periodic_boundary_position_entry(self):
        setting.HypercuboidSetting(beta=1.0, dimension=3, system_lengths=[1.0, 2.0, 3.0])
        position = [0.1, 1.4, 2.9]
        position[0] = setting.periodic_boundaries.correct_position_entry(position[0], 0)
        self.assertEqual(position, [0.1, 1.4, 2.9])
        position = [0.0, 0.4, 33.0]
        position[1] = setting.periodic_boundaries.correct_position_entry(position[1], 1)
        self.assertAlmostEqualList(position, [0.0, 0.4, 33.0], places=13)
        position = [-0.3, 1.4, 3.9]
        position[2] = setting.periodic_boundaries.correct_position_entry(position[2], 2)
        self.assertAlmostEqualList(position, [-0.3, 1.4, 0.9], places=13)
        position = [-10.0, 102.231, 0.999999]
        position[1] = setting.periodic_boundaries.correct_position_entry(position[1], 1)
        self.assertAlmostEqualList(position, [-10.0, 0.231, 0.999999], places=13)

    def test_separation_vector(self):
        setting.HypercuboidSetting(beta=1.0, dimension=3, system_lengths=[1.0, 2.0, 3.0])
        position_one = [0.5, 1.6, 2.3]
        position_two = [0.2, 0.1, 0.9]
        self.assertAlmostEqualList(setting.periodic_boundaries.separation_vector(position_one, position_two),
                                   [-0.3, 0.5, -1.4], places=13)

    def test_correct_periodic_boundary_separation(self):
        setting.HypercuboidSetting(beta=1.0, dimension=4, system_lengths=[1.0, 2.0, 3.0, 4.0])
        separation = [-0.5, -0.8, 1.4, 2.0]
        setting.periodic_boundaries.correct_separation(separation)
        self.assertAlmostEqualList(separation, [-0.5, -0.8, 1.4, -2.0], places=13)
        separation = [-0.6, 1.3, 0.4, 1.9]
        setting.periodic_boundaries.correct_separation(separation)
        self.assertAlmostEqualList(separation, [0.4, -0.7, 0.4, 1.9], places=13)
        setting.system_length = None
        setting.system_length_over_two = None
        setting.dimension = None

    def test_correct_periodic_boundary_separation_entry(self):
        setting.HypercuboidSetting(beta=1.0, dimension=4, system_lengths=[1.0, 2.0, 3.0, 4.0])
        separation = [-0.5, -1.3, 2.4, 1.5]
        separation[3] = setting.periodic_boundaries.correct_separation_entry(separation[3], 3)
        self.assertAlmostEqualList(separation, [-0.5, -1.3, 2.4, 1.5], places=13)
        separation = [-0.6, 1.3, 2.4, 6.9]
        separation[0] = setting.periodic_boundaries.correct_separation_entry(separation[0], 0)
        self.assertAlmostEqualList(separation, [0.4, 1.3, 2.4, 6.9], places=13)

    def test_next_image(self):
        setting.HypercuboidSetting(beta=1.0, dimension=3, system_lengths=[1.0, 2.0, 3.0])
        self.assertEqual(setting.periodic_boundaries.next_image(0.4, 0), 1.4)
        self.assertEqual(setting.periodic_boundaries.next_image(0.6, 1), 2.6)
        self.assertEqual(setting.periodic_boundaries.next_image(2.0, 2), 5.0)


if __name__ == '__main__':
    main()
