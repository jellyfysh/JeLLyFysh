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
from unittest import TestCase, main, mock
import setting
from setting import hypercubic_setting
from setting import hypercuboid_setting
from setting.periodic_boundaries.hypercubic_periodic_boundaries import HypercubicPeriodicBoundaries


class TestHypercubicSetting(TestCase):
    def setUp(self) -> None:
        setting.HypercubicSetting(beta=2, dimension=5, system_length=1.5)

    def tearDown(self):
        setting.reset()

    def test_setting_beta(self):
        self.assertEqual(setting.beta, 2)

    def test_hypercubic_setting_beta(self):
        self.assertEqual(hypercubic_setting.beta, 2)

    def test_hypercuboid_setting_beta(self):
        # Hypercuboid setting is a similar setting which should also get initialized
        self.assertEqual(hypercuboid_setting.beta, 2)

    def test_setting_dimension(self):
        self.assertEqual(setting.dimension, 5)

    def test_hypercubic_setting_dimension(self):
        self.assertEqual(hypercubic_setting.dimension, 5)

    def test_hypercuboid_setting_dimension(self):
        self.assertEqual(hypercuboid_setting.dimension, 5)

    def test_setting_periodic_boundaries(self):
        self.assertIsInstance(setting.periodic_boundaries, HypercubicPeriodicBoundaries)

    def test_hypercubic_setting_periodic_boundaries(self):
        self.assertIsInstance(hypercubic_setting.periodic_boundaries, HypercubicPeriodicBoundaries)

    def test_hypercuboid_setting_periodic_boundaries(self):
        self.assertIsInstance(hypercuboid_setting.periodic_boundaries, HypercubicPeriodicBoundaries)

    @mock.patch("setting.hypercubic_setting.random.uniform")
    def test_setting_random_position(self, random_mock):
        random_mock.side_effect = [0.1, 0.2, 0.3, 0.4, 0.5]
        random_position = setting.random_position()
        self.assertEqual(random_position, [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(random_mock.call_args_list, [mock.call(0.0, 1.5) for _ in range(5)])

    @mock.patch("setting.hypercubic_setting.random.uniform")
    def test_hypercubic_setting_random_position(self, random_mock):
        random_mock.side_effect = [0.9, 0.0, 0.1, 0.2, 0.3]
        random_position = hypercubic_setting.random_position()
        self.assertEqual(random_position, [0.9, 0.0, 0.1, 0.2, 0.3])
        self.assertEqual(random_mock.call_args_list, [mock.call(0.0, 1.5) for _ in range(5)])

    @mock.patch("setting.hypercubic_setting.random.uniform")
    def test_hypercuboid_setting_random_position(self, random_mock):
        random_mock.side_effect = [0.9, 0.0, 0.1, 0.2, 0.3]
        random_position = hypercuboid_setting.random_position()
        self.assertEqual(random_position, [0.9, 0.0, 0.1, 0.2, 0.3])
        self.assertEqual(random_mock.call_args_list, [mock.call(0.0, 1.5) for _ in range(5)])

    def test_hypercubic_setting_system_length(self):
        self.assertEqual(hypercubic_setting.system_length, 1.5)

    def test_hypercuboid_setting_system_length(self):
        self.assertEqual(hypercuboid_setting.system_lengths, (1.5, 1.5, 1.5, 1.5, 1.5))

    def test_hypercubic_setting_system_length_over_two(self):
        self.assertEqual(hypercubic_setting.system_length_over_two, 0.75)

    def test_hypercuboid_setting_system_length_over_two(self):
        self.assertEqual(hypercuboid_setting.system_lengths_over_two, (0.75, 0.75, 0.75, 0.75, 0.75))

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
        self.assertEqual(hypercubic_setting.number_of_root_nodes, 2)
        self.assertEqual(hypercuboid_setting.number_of_root_nodes, 2)

    def test_set_number_of_nodes_per_root_node(self):
        setting.set_number_of_nodes_per_root_node(2)
        self.assertEqual(setting.number_of_nodes_per_root_node, 2)
        self.assertEqual(hypercubic_setting.number_of_nodes_per_root_node, 2)
        self.assertEqual(hypercuboid_setting.number_of_nodes_per_root_node, 2)

    def test_set_number_of_node_levels(self):
        setting.set_number_of_node_levels(2)
        self.assertEqual(setting.number_of_node_levels, 2)
        self.assertEqual(hypercubic_setting.number_of_node_levels, 2)
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
        self.assertTrue(hypercubic_setting.initialized())
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
            setting.HypercubicSetting(beta=-1, dimension=3, system_length=1.0)

    def test_beta_zero_raises_error(self):
        setting.reset()
        with self.assertRaises(AttributeError):
            setting.HypercubicSetting(beta=0, dimension=3, system_length=1.0)

    def test_negative_system_length_raises_error(self):
        setting.reset()
        with self.assertRaises(AttributeError):
            setting.HypercubicSetting(beta=1, dimension=3, system_length=-3.2)

    def test_system_length_zero_raises_error(self):
        setting.reset()
        with self.assertRaises(AttributeError):
            setting.HypercubicSetting(beta=1, dimension=3, system_length=0)

    def test_negative_dimension_raises_error(self):
        setting.reset()
        with self.assertRaises(AttributeError):
            setting.HypercubicSetting(beta=1, dimension=-1, system_length=1.0)

    def test_dimension_zero_raises_error(self):
        setting.reset()
        with self.assertRaises(AttributeError):
            setting.HypercubicSetting(beta=1, dimension=0, system_length=1.0)

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
        setting.HypercubicSetting(beta=1.0, dimension=2.8, system_length=1.0)
        self.assertEqual(setting.dimension, 2)
        self.assertEqual(hypercubic_setting.dimension, 2)
        self.assertEqual(hypercuboid_setting.dimension, 2)

    def test_integer_casting_number_of_root_nodes(self):
        setting.set_number_of_root_nodes(1.7)
        self.assertEqual(setting.number_of_root_nodes, 1)
        self.assertEqual(hypercubic_setting.number_of_root_nodes, 1)
        self.assertEqual(hypercuboid_setting.number_of_root_nodes, 1)

    def test_integer_casting_number_of_nodes_per_root_node(self):
        setting.set_number_of_nodes_per_root_node(33.0)
        self.assertEqual(setting.number_of_nodes_per_root_node, 33)
        self.assertEqual(hypercubic_setting.number_of_nodes_per_root_node, 33)
        self.assertEqual(hypercuboid_setting.number_of_nodes_per_root_node, 33)

    def test_integer_casting_number_of_node_levels(self):
        setting.set_number_of_node_levels(1.3)
        self.assertEqual(setting.number_of_node_levels, 1)
        self.assertEqual(hypercubic_setting.number_of_node_levels, 1)
        self.assertEqual(hypercuboid_setting.number_of_node_levels, 1)

    def test_new_initialize_hypercubic_setting_raises_error(self):
        with self.assertRaises(AttributeError):
            setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)

    def test_new_initialize_hypercuboid_setting_raises_error(self):
        with self.assertRaises(AttributeError):
            setting.HypercuboidSetting(beta=1.0, dimension=3, system_lengths=[1.0, 2.0, 3.0])


if __name__ == '__main__':
    main()
