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
import math
from unittest import TestCase, main
from base import vectors


class TestVectors(TestCase):
    def test_norm(self):
        self.assertAlmostEqual(vectors.norm([0.1, 0.2, 0.3]), 0.3741657386773941, places=13)
        self.assertAlmostEqual(vectors.norm([0.1, 0.0, 0.0]), 0.1, places=13)
        self.assertAlmostEqual(vectors.norm([0.0, 0.2, 0.0]), 0.2, places=13)
        self.assertAlmostEqual(vectors.norm([0.0, 0.0, -0.3]), 0.3, places=13)
        self.assertAlmostEqual(vectors.norm([0.4, 1.5, 2.3]), 2.774887385102321, places=12)
        self.assertAlmostEqual(vectors.norm([0.4, 1.5, -2.3]), 2.774887385102321, places=12)
        self.assertAlmostEqual(vectors.norm([0.4, -1.5, 2.3]), 2.774887385102321, places=12)
        self.assertAlmostEqual(vectors.norm([0.4, -1.5, -2.3]), 2.774887385102321, places=12)
        self.assertAlmostEqual(vectors.norm([-0.4, 1.5, -2.3]), 2.774887385102321, places=12)
        self.assertAlmostEqual(vectors.norm([-0.4, 1.5, 2.3]), 2.774887385102321, places=12)
        self.assertAlmostEqual(vectors.norm([-0.4, -1.5, -2.3]), 2.774887385102321, places=12)
        self.assertAlmostEqual(vectors.norm([-0.4, -1.5, 2.3]), 2.774887385102321, places=12)
        self.assertAlmostEqual(vectors.norm([0.1, 0.0]), 0.1, places=13)
        self.assertAlmostEqual(vectors.norm([0.0, 0.0, 0.0, 0.2]), 0.2, places=13)

    def test_norm_sq(self):
        self.assertAlmostEqual(vectors.norm_sq([0.1, 0.2, 0.3]), 0.14, places=13)
        self.assertAlmostEqual(vectors.norm_sq([-0.4, 0.0, 0.0]), 0.16, places=14)
        self.assertAlmostEqual(vectors.norm_sq([0.0, 0.5, 0.0]), 0.25, places=14)
        self.assertAlmostEqual(vectors.norm_sq([0.0, 0.0, 0.6]), 0.36, places=14)
        self.assertAlmostEqual(vectors.norm_sq([0.7, 0.3, 0.2]), 0.6199999999999999, places=13)
        self.assertAlmostEqual(vectors.norm_sq([0.7, 0.3, -0.2]), 0.6199999999999999, places=13)
        self.assertAlmostEqual(vectors.norm_sq([0.7, -0.3, 0.2]), 0.6199999999999999, places=13)
        self.assertAlmostEqual(vectors.norm_sq([0.7, -0.3, -0.2]), 0.6199999999999999, places=13)
        self.assertAlmostEqual(vectors.norm_sq([-0.7, 0.3, 0.2]), 0.6199999999999999, places=13)
        self.assertAlmostEqual(vectors.norm_sq([-0.7, 0.3, -0.2]), 0.6199999999999999, places=13)
        self.assertAlmostEqual(vectors.norm_sq([-0.7, -0.3, 0.2]), 0.6199999999999999, places=13)
        self.assertAlmostEqual(vectors.norm_sq([-0.7, -0.3, -0.2]), 0.6199999999999999, places=13)
        self.assertAlmostEqual(vectors.norm_sq([0.1, 0.0]), 0.01, places=13)
        self.assertAlmostEqual(vectors.norm_sq([0.0, 0.0, 0.0, 0.2]), 0.04, places=13)

    def test_dot(self):
        self.assertEqual(vectors.dot([1, 2, 3], [4, 5, 6]), 32)
        self.assertEqual(vectors.dot([2, 3], [4, 5]), 23)
        self.assertEqual(vectors.dot([5], [-3]), -15)
        with self.assertRaises(AssertionError):
            vectors.dot([1, 2], [1])

    def test_normalize(self):
        self.assertEqual(vectors.normalize([1, 2, 3]), [1 / 14 ** 0.5, 2 / 14 ** 0.5, 3 / 14 ** 0.5])
        self.assertEqual(vectors.normalize([4, -3]), [4 / 25 ** 0.5, -3 / 25 ** 0.5])
        self.assertEqual(vectors.normalize([2]), [1])
        self.assertEqual(vectors.normalize([2], 2), [2])
        self.assertEqual(vectors.normalize([-1, -2], 2.5), [-1 / 5 ** 0.5 * 2.5, -2 / 5 ** 0.5 * 2.5])
        self.assertEqual(vectors.normalize([1, 1, 1], 0.5),
                         [1 / 3 ** 0.5 * 0.5, 1 / 3 ** 0.5 * 0.5, 1 / 3 ** 0.5 * 0.5])
        # Desired norm must be > 0
        with self.assertRaises(AssertionError):
            vectors.normalize([1], -2)
        with self.assertRaises(AssertionError):
            vectors.normalize([1, 2, 3], 0)

    def test_angle_between_two_vectors(self):
        self.assertAlmostEqual(vectors.angle_between_two_vectors([1], [2]), 0.0, places=13)
        self.assertAlmostEqual(vectors.angle_between_two_vectors([1], [-1]), math.pi, places=13)
        self.assertAlmostEqual(vectors.angle_between_two_vectors([0, 1], [1, 0]), math.pi / 2, places=13)
        self.assertAlmostEqual(vectors.angle_between_two_vectors([1, 0], [0, 1]), math.pi / 2, places=13)
        self.assertAlmostEqual(vectors.angle_between_two_vectors([-1, -2.3, 3.4], [5.2, -0.1, 1.3]),
                               1.595081594067948968249, places=13)

    def test_replace_vector_component(self):
        vector = [0, 1, 2]
        vector = vectors.copy_vector_with_replaced_component(vector, 0, 4)
        self.assertEqual(vector, [4, 1, 2])
        vector = vectors.copy_vector_with_replaced_component(vector, 1, 5)
        self.assertEqual(vector, [4, 5, 2])
        vector = vectors.copy_vector_with_replaced_component(vector, 2, 6)
        self.assertEqual(vector, [4, 5, 6])
        vector = vectors.copy_vector_with_replaced_component(vector, 3, 7)
        self.assertEqual(vector, [4, 5, 6])
        vector = vectors.copy_vector_with_replaced_component(vector, -1, 8)
        self.assertEqual(vector, [4, 5, 6])
        vector = [0, 1]
        self.assertEqual(vectors.copy_vector_with_replaced_component(vector, 0, 2.3), [2.3, 1])
        self.assertEqual(vectors.copy_vector_with_replaced_component(vector, 1, 2.3), [0, 2.3])
        self.assertEqual(vectors.copy_vector_with_replaced_component(vector, -1, 2.3), [0, 1])
        self.assertEqual(vectors.copy_vector_with_replaced_component(vector, 2, 2.3), [0, 1])

    def test_displacement_until_new_norm_sq_component_positive(self):
        vector = [0.2, 0.0, 0.0]
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_positive(vector, 0.01, 0), 0.1,
                               places=13)
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_positive(vector, 0.09, 0), -0.1,
                               places=13)
        vector = [0.0, 0.3, 0.0]
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_positive(vector, 0.01, 1), 0.2,
                               places=13)
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_positive(vector, 0.25, 1), -0.2,
                               places=13)
        vector = [0.0, 0.0, 0.1]
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_positive(vector, 0.0, 2), 0.1,
                               places=13)
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_positive(vector, 0.16, 2), -0.3,
                               places=13)

        vector = [0.4, -0.25, 0.17]
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_positive(vector, 0.35 ** 2, 0),
                               0.2236480791145161, places=13)
        vector = [0.17, 0.4, 0.25]
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_positive(vector, 0.35 ** 2, 1),
                               0.2236480791145161, places=13)
        vector = [-0.17, -0.25, 0.4]
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_positive(vector, 0.35 ** 2, 2),
                               0.2236480791145161, places=13)

        vector = [0.1, 0.2, 0.0]
        with self.assertRaises(ValueError):
            vectors.displacement_until_new_norm_sq_component_positive(vector, 0.01, 0)
        vector = [0.0, 0.1, 0.1]
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_positive(vector, 0.1 ** 2, 1), 0.1,
                               places=13)
        with self.assertRaises(ValueError):
            vectors.displacement_until_new_norm_sq_component_positive(vector, 0.099999999999 ** 2, 2)

        with self.assertRaises(AssertionError):
            vectors.displacement_until_new_norm_sq_component_positive([-0.2, 0.0, 0.0], 0.01, 0)
        with self.assertRaises(AssertionError):
            vectors.displacement_until_new_norm_sq_component_positive([0.0, 0.0, 0.0], 0.01, 0)

    def test_displacement_until_new_norm_sq_component_negative(self):
        vector = [-0.2, 0.0, 0.0]
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_negative(vector, 0.01, 0), -0.1,
                               places=13)
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_negative(vector, 0.09, 0), 0.1,
                               places=13)
        vector = [0.0, -0.3, 0.0]
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_negative(vector, 0.01, 1), -0.2,
                               places=13)
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_negative(vector, 0.25, 1), 0.2,
                               places=13)
        vector = [0.0, 0.0, -0.1]
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_negative(vector, 0.0, 2), -0.1,
                               places=13)
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_negative(vector, 0.16, 2), 0.3,
                               places=13)

        vector = [-0.1, 0.1, 0.14]
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_negative(vector, 0.25 ** 2, 0),
                               0.0813835714721705, places=13)
        vector = [-0.1, -0.1, -0.14]
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_negative(vector, 0.25 ** 2, 1),
                               0.0813835714721705, places=13)
        vector = [0.1, -0.14, -0.1]
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_negative(vector, 0.25 ** 2, 2),
                               0.0813835714721705, places=13)

        vector = [-0.1, 0.2, 0.0]
        with self.assertRaises(ValueError):
            vectors.displacement_until_new_norm_sq_component_negative(vector, 0.01, 0)
        vector = [0.0, -0.1, 0.1]
        self.assertAlmostEqual(vectors.displacement_until_new_norm_sq_component_negative(vector, 0.1 ** 2, 1), -0.1,
                               places=13)

        with self.assertRaises(AssertionError):
            vectors.displacement_until_new_norm_sq_component_negative([0.2, 0.0, 0.0], 0.01, 0)

    def test_permutation_3d(self):
        self.assertEqual(vectors.permutation_3d([5, 2, 17], 0), (5, 2, 17))
        self.assertEqual(vectors.permutation_3d([8, 9, 3], 1), (9, 3, 8))
        self.assertEqual(vectors.permutation_3d([-1, 0, 22], 2), (22, -1, 0))
        # Main direction must be 0, 1, 2 (negative numbers are allowed since indices are counted from back)
        with self.assertRaises(AssertionError):
            vectors.permutation_3d([0, 1, 2], 3)
        # Only 3d vectors are allowed
        with self.assertRaises(AssertionError):
            vectors.permutation_3d([0, 1], 0)

    def test_random_vector_on_unit_sphere(self):
        # TODO implement statistical unittest here
        pass


if __name__ == '__main__':
    main()
