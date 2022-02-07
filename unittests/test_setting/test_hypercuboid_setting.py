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
import os
import sys
from unittest import TestCase, main, mock
import jellyfysh.setting as setting
from jellyfysh.setting import hypercubic_setting
from jellyfysh.setting import hypercuboid_setting
_unittest_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_unittest_directory_added_to_path = False
if _unittest_directory not in sys.path:
    sys.path.append(_unittest_directory)
    _unittest_directory_added_to_path = True
# noinspection PyUnresolvedReferences
from expanded_test_case import ExpandedTestCase


def tearDownModule():
    if _unittest_directory_added_to_path:
        sys.path.remove(_unittest_directory)


class TestHypercubicSetting(TestCase):
    def setUp(self) -> None:
        hypercuboid_setting.HypercuboidSetting(beta=2, dimension=5, system_lengths=[1.0, 2.0, 3.0, 4.0, 5.0])

    def tearDown(self):
        setting.reset()

    def test_setting_beta(self):
        self.assertEqual(setting.beta, 2)

    def test_hypercubic_setting_beta(self):
        # Hypercubic setting is not a similar setting and should not be initialized
        self.assertIsNone(hypercubic_setting.beta)

    def test_hypercuboid_setting_beta(self):
        self.assertEqual(hypercuboid_setting.beta, 2)

    def test_setting_dimension(self):
        self.assertEqual(setting.dimension, 5)

    def test_hypercubic_setting_dimension(self):
        self.assertIsNone(hypercubic_setting.dimension)

    def test_hypercuboid_setting_dimension(self):
        self.assertEqual(hypercuboid_setting.dimension, 5)

    def test_setting_periodic_boundaries(self):
        self.assertIsInstance(setting.periodic_boundaries, hypercuboid_setting.HypercuboidPeriodicBoundaries)

    def test_hypercubic_setting_periodic_boundaries(self):
        self.assertIsNone(hypercubic_setting.periodic_boundaries)

    def test_hypercuboid_setting_periodic_boundaries(self):
        self.assertIsInstance(hypercuboid_setting.periodic_boundaries,
                              hypercuboid_setting.HypercuboidPeriodicBoundaries)

    @mock.patch("jellyfysh.setting.hypercuboid_setting.random.uniform")
    def test_setting_random_position(self, random_mock):
        random_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5]
        random_position = setting.random_position()
        self.assertEqual(random_position, [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(random_mock.call_args_list,
                         [mock.call(0.0, 1.0), mock.call(0.0, 2.0), mock.call(0.0, 3.0), mock.call(0.0, 4.0),
                          mock.call(0.0, 5.0)])

    def test_hypercubic_setting_random_position(self):
        self.assertIsNone(hypercubic_setting.random_position)

    @mock.patch("jellyfysh.setting.hypercuboid_setting.random.uniform")
    def test_hypercuboid_setting_random_position(self, random_mock):
        random_mock.side_effect = [0.9, 0.0, 0.1, 0.2, 0.3]
        random_position = hypercuboid_setting.random_position()
        self.assertEqual(random_position, [0.9, 0.0, 0.1, 0.2, 0.3])
        self.assertEqual(random_mock.call_args_list,
                         [mock.call(0.0, 1.0), mock.call(0.0, 2.0), mock.call(0.0, 3.0), mock.call(0.0, 4.0),
                          mock.call(0.0, 5.0)])

    def test_hypercubic_setting_system_length(self):
        self.assertIsNone(hypercubic_setting.system_length)

    def test_hypercuboid_setting_system_length(self):
        self.assertEqual(hypercuboid_setting.system_lengths, (1.0, 2.0, 3.0, 4.0, 5.0))

    def test_hypercubic_setting_system_length_over_two(self):
        self.assertIsNone(hypercubic_setting.system_length_over_two)

    def test_hypercuboid_setting_system_length_over_two(self):
        self.assertEqual(hypercuboid_setting.system_lengths_over_two, (0.5, 1.0, 1.5, 2.0, 2.5))

    def test_setting_number_of_root_nodes_none(self):
        self.assertIsNone(setting.number_of_root_nodes)

    def test_hypercubic_setting_number_of_root_nodes_none(self):
        self.assertIsNone(hypercubic_setting.number_of_root_nodes)

    def test_hypercuboid_setting_number_of_root_nodes_none(self):
        self.assertIsNone(hypercuboid_setting.number_of_root_nodes)

    def test_setting_number_of_nodes_per_root_node_none(self):
        self.assertIsNone(setting.number_of_nodes_per_root_node)

    def test_hypercubic_setting_number_of_nodes_per_root_node_none(self):
        self.assertIsNone(hypercubic_setting.number_of_nodes_per_root_node)

    def test_hypercuboid_setting_number_of_nodes_per_root_node_none(self):
        self.assertIsNone(hypercuboid_setting.number_of_nodes_per_root_node)

    def test_setting_number_of_node_levels_none(self):
        self.assertIsNone(setting.number_of_node_levels)

    def test_hypercubic_setting_number_of_node_levels_none(self):
        self.assertIsNone(hypercubic_setting.number_of_node_levels)

    def test_hypercuboid_setting_number_of_node_levels_none(self):
        self.assertIsNone(hypercuboid_setting.number_of_node_levels)

    def test_set_number_of_root_nodes(self):
        setting.set_number_of_root_nodes(2)
        self.assertEqual(setting.number_of_root_nodes, 2)
        self.assertIsNone(hypercubic_setting.number_of_root_nodes)
        self.assertEqual(hypercuboid_setting.number_of_root_nodes, 2)

    def test_set_number_of_nodes_per_root_node(self):
        setting.set_number_of_nodes_per_root_node(2)
        self.assertEqual(setting.number_of_nodes_per_root_node, 2)
        self.assertIsNone(hypercubic_setting.number_of_nodes_per_root_node)
        self.assertEqual(hypercuboid_setting.number_of_nodes_per_root_node, 2)

    def test_set_number_of_node_levels(self):
        setting.set_number_of_node_levels(2)
        self.assertEqual(setting.number_of_node_levels, 2)
        self.assertIsNone(hypercubic_setting.number_of_node_levels)
        self.assertEqual(hypercuboid_setting.number_of_node_levels, 2)

    def test_initialized(self):
        self.assertFalse(hypercubic_setting.initialized())
        self.assertFalse(hypercuboid_setting.initialized())
        setting.set_number_of_root_nodes(3)
        self.assertFalse(hypercubic_setting.initialized())
        self.assertFalse(hypercuboid_setting.initialized())
        setting.set_number_of_nodes_per_root_node(1)
        self.assertFalse(hypercubic_setting.initialized())
        self.assertFalse(hypercuboid_setting.initialized())
        setting.set_number_of_node_levels(1)
        self.assertFalse(hypercubic_setting.initialized())
        self.assertTrue(hypercuboid_setting.initialized())

    def test_reset(self):
        setting.set_number_of_root_nodes(5)
        setting.set_number_of_nodes_per_root_node(3)
        setting.set_number_of_node_levels(2)
        setting.reset()
        self.assertIsNone(setting.beta)
        self.assertIsNone(hypercubic_setting.beta)
        self.assertIsNone(hypercuboid_setting.beta)
        self.assertIsNone(setting.dimension)
        self.assertIsNone(hypercubic_setting.dimension)
        self.assertIsNone(hypercuboid_setting.dimension)
        self.assertIsNone(setting.random_position)
        self.assertIsNone(hypercubic_setting.random_position)
        self.assertIsNone(hypercuboid_setting.random_position)
        self.assertIsNone(setting.periodic_boundaries)
        self.assertIsNone(hypercubic_setting.periodic_boundaries)
        self.assertIsNone(hypercuboid_setting.periodic_boundaries)
        self.assertIsNone(setting.number_of_root_nodes)
        self.assertIsNone(hypercubic_setting.number_of_root_nodes)
        self.assertIsNone(hypercuboid_setting.number_of_root_nodes)
        self.assertIsNone(setting.number_of_nodes_per_root_node)
        self.assertIsNone(hypercubic_setting.number_of_nodes_per_root_node)
        self.assertIsNone(hypercuboid_setting.number_of_nodes_per_root_node)
        self.assertIsNone(setting.number_of_node_levels)
        self.assertIsNone(hypercubic_setting.number_of_node_levels)
        self.assertIsNone(hypercuboid_setting.number_of_node_levels)
        self.assertIsNone(hypercubic_setting.system_length)
        self.assertIsNone(hypercuboid_setting.system_lengths)
        self.assertIsNone(hypercubic_setting.system_length_over_two)
        self.assertIsNone(hypercuboid_setting.system_lengths_over_two)

    def test_negative_beta_raises_error(self):
        setting.reset()
        with self.assertRaises(AttributeError):
            hypercuboid_setting.HypercuboidSetting(beta=-1, dimension=3, system_lengths=[1.0, 2.0, 3.0])

    def test_beta_zero_raises_error(self):
        setting.reset()
        with self.assertRaises(AttributeError):
            hypercuboid_setting.HypercuboidSetting(beta=0, dimension=3, system_lengths=[1.0, 2.0, 3.0])

    def test_negative_system_length_raises_error(self):
        setting.reset()
        with self.assertRaises(AttributeError):
            hypercuboid_setting.HypercuboidSetting(beta=1, dimension=3, system_lengths=[-1.0, 2.0, 3.0])

    def test_system_length_zero_raises_error(self):
        setting.reset()
        with self.assertRaises(AttributeError):
            hypercuboid_setting.HypercuboidSetting(beta=1, dimension=3, system_lengths=[1.0, 0.0, 3.0])

    def test_too_few_system_length_raises_error(self):
        setting.reset()
        with self.assertRaises(AttributeError):
            hypercuboid_setting.HypercuboidSetting(beta=1, dimension=3, system_lengths=[1.0, 2.0])

    def test_too_many_system_length_raises_error(self):
        setting.reset()
        with self.assertRaises(AttributeError):
            hypercuboid_setting.HypercuboidSetting(beta=1, dimension=3, system_lengths=[1.0, 2.0, 3.0, 4.0])

    def test_negative_dimension_raises_error(self):
        setting.reset()
        with self.assertRaises(AttributeError):
            hypercuboid_setting.HypercuboidSetting(beta=1, dimension=-1, system_lengths=[1.0, 2.0, 3.0])

    def test_dimension_zero_raises_error(self):
        setting.reset()
        with self.assertRaises(AttributeError):
            hypercuboid_setting.HypercuboidSetting(beta=1, dimension=0, system_lengths=[1.0, 2.0, 3.0])

    def test_negative_number_of_root_nodes_raises_error(self):
        with self.assertRaises(AttributeError):
            setting.set_number_of_root_nodes(-5)

    def test_number_of_root_nodes_zero_raises_error(self):
        with self.assertRaises(AttributeError):
            setting.set_number_of_root_nodes(0)

    def test_negative_number_of_nodes_per_root_node_raises_error(self):
        with self.assertRaises(AttributeError):
            setting.set_number_of_nodes_per_root_node(-17)

    def test_number_of_nodes_per_root_node_zero_raises_error(self):
        with self.assertRaises(AttributeError):
            setting.set_number_of_nodes_per_root_node(0)

    def test_negative_number_of_node_levels_raises_error(self):
        with self.assertRaises(AttributeError):
            setting.set_number_of_node_levels(-1)

    def test_number_of_node_levels_zero_raises_error(self):
        with self.assertRaises(AttributeError):
            setting.set_number_of_node_levels(0)

    def test_negative_number_of_node_levels_larger_than_two_raises_error(self):
        with self.assertRaises(AttributeError):
            setting.set_number_of_node_levels(3)

    def test_integer_casting_dimension(self):
        setting.reset()
        # noinspection PyTypeChecker
        hypercuboid_setting.HypercuboidSetting(beta=1.0, dimension=2.8, system_lengths=[1.0, 2.0])
        self.assertEqual(setting.dimension, 2)
        self.assertIsNone(hypercubic_setting.dimension)
        self.assertEqual(hypercuboid_setting.dimension, 2)

    def test_integer_casting_number_of_root_nodes(self):
        setting.set_number_of_root_nodes(1.7)
        self.assertEqual(setting.number_of_root_nodes, 1)
        self.assertIsNone(hypercubic_setting.number_of_root_nodes)
        self.assertEqual(hypercuboid_setting.number_of_root_nodes, 1)

    def test_integer_casting_number_of_nodes_per_root_node(self):
        setting.set_number_of_nodes_per_root_node(33.0)
        self.assertEqual(setting.number_of_nodes_per_root_node, 33)
        self.assertIsNone(hypercubic_setting.number_of_nodes_per_root_node)
        self.assertEqual(hypercuboid_setting.number_of_nodes_per_root_node, 33)

    def test_integer_casting_number_of_node_levels(self):
        setting.set_number_of_node_levels(1.3)
        self.assertEqual(setting.number_of_node_levels, 1)
        self.assertIsNone(hypercubic_setting.number_of_node_levels)
        self.assertEqual(hypercuboid_setting.number_of_node_levels, 1)

    def test_new_initialize_hypercubic_setting_raises_error(self):
        with self.assertRaises(AttributeError):
            hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)

    def test_new_initialize_hypercuboid_setting_raises_error(self):
        with self.assertRaises(AttributeError):
            hypercuboid_setting.HypercuboidSetting(beta=1.0, dimension=3, system_lengths=[1.0, 2.0, 3.0])


# Inherit explicitly from TestCase class for Test functionality in PyCharm.
class TestHypercuboidPeriodicBoundaries(ExpandedTestCase, TestCase):
    def tearDown(self):
        setting.reset()

    def test_correct_periodic_boundary_position(self):
        hypercuboid_setting.HypercuboidSetting(beta=1.0, dimension=3, system_lengths=[1.0, 2.0, 3.0])
        position = [0.1, 1.4, 2.9]
        setting.periodic_boundaries.correct_position(position)
        self.assertEqual(position, [0.1, 1.4, 2.9])
        position = [0.0, 2.0, 1.0]
        setting.periodic_boundaries.correct_position(position)
        self.assertAlmostEqualSequence(position, [0.0, 0.0, 1.0], places=13)
        position = [-0.3, 1.4, 3.9]
        setting.periodic_boundaries.correct_position(position)
        self.assertAlmostEqualSequence(position, [0.7, 1.4, 0.9], places=13)
        position = [-10.0, 102.231, 2.999999]
        setting.periodic_boundaries.correct_position(position)
        self.assertAlmostEqualSequence(position, [0.0, 0.231, 2.999999], places=13)

    def test_correct_periodic_boundary_position_entry(self):
        hypercuboid_setting.HypercuboidSetting(beta=1.0, dimension=3, system_lengths=[1.0, 2.0, 3.0])
        position = [0.1, 1.4, 2.9]
        position[0] = setting.periodic_boundaries.correct_position_entry(position[0], 0)
        self.assertEqual(position, [0.1, 1.4, 2.9])
        position = [0.0, 0.4, 33.0]
        position[1] = setting.periodic_boundaries.correct_position_entry(position[1], 1)
        self.assertAlmostEqualSequence(position, [0.0, 0.4, 33.0], places=13)
        position = [-0.3, 1.4, 3.9]
        position[2] = setting.periodic_boundaries.correct_position_entry(position[2], 2)
        self.assertAlmostEqualSequence(position, [-0.3, 1.4, 0.9], places=13)
        position = [-10.0, 102.231, 0.999999]
        position[1] = setting.periodic_boundaries.correct_position_entry(position[1], 1)
        self.assertAlmostEqualSequence(position, [-10.0, 0.231, 0.999999], places=13)

    def test_separation_vector(self):
        hypercuboid_setting.HypercuboidSetting(beta=1.0, dimension=3, system_lengths=[1.0, 2.0, 3.0])
        position_one = [0.5, 1.6, 2.3]
        position_two = [0.2, 0.1, 0.9]
        self.assertAlmostEqualSequence(setting.periodic_boundaries.separation_vector(position_one, position_two),
                                       [-0.3, 0.5, -1.4], places=13)

    def test_correct_periodic_boundary_separation(self):
        hypercuboid_setting.HypercuboidSetting(beta=1.0, dimension=4, system_lengths=[1.0, 2.0, 3.0, 4.0])
        separation = [-0.5, -0.8, 1.4, 2.0]
        setting.periodic_boundaries.correct_separation(separation)
        self.assertAlmostEqualSequence(separation, [-0.5, -0.8, 1.4, -2.0], places=13)
        separation = [-0.6, 1.3, 0.4, 1.9]
        setting.periodic_boundaries.correct_separation(separation)
        self.assertAlmostEqualSequence(separation, [0.4, -0.7, 0.4, 1.9], places=13)
        setting.system_length = None
        setting.system_length_over_two = None
        setting.dimension = None

    def test_correct_periodic_boundary_separation_entry(self):
        hypercuboid_setting.HypercuboidSetting(beta=1.0, dimension=4, system_lengths=[1.0, 2.0, 3.0, 4.0])
        separation = [-0.5, -1.3, 2.4, 1.5]
        separation[3] = setting.periodic_boundaries.correct_separation_entry(separation[3], 3)
        self.assertAlmostEqualSequence(separation, [-0.5, -1.3, 2.4, 1.5], places=13)
        separation = [-0.6, 1.3, 2.4, 6.9]
        separation[0] = setting.periodic_boundaries.correct_separation_entry(separation[0], 0)
        self.assertAlmostEqualSequence(separation, [0.4, 1.3, 2.4, 6.9], places=13)

    def test_next_image(self):
        hypercuboid_setting.HypercuboidSetting(beta=1.0, dimension=3, system_lengths=[1.0, 2.0, 3.0])
        self.assertEqual(setting.periodic_boundaries.next_image(0.4, 0), 1.4)
        self.assertEqual(setting.periodic_boundaries.next_image(0.6, 1), 2.6)
        self.assertEqual(setting.periodic_boundaries.next_image(2.0, 2), 5.0)


if __name__ == '__main__':
    main()
