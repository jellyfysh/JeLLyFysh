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
from jellyfysh.base.time import Time, inf


class TestTime(TestCase):
    def test_from_float(self):
        self.assertEqual(Time.from_float(1.23171327853541341).quotient, 1.0)
        # We can only check twelve places because the input float has this precision.
        self.assertAlmostEqual(Time.from_float(1.23171327853541341).remainder, 0.23171327853541341, places=12)

        self.assertEqual(Time.from_float(10.0).quotient, 10.0)
        self.assertEqual(Time.from_float(10.0).remainder, 0.0)

        self.assertEqual(Time.from_float(-4528.213891789).quotient, -4529.0)
        # We can only check nine places because the input float has this precision.
        self.assertAlmostEqual(Time.from_float(-4528.213891789).remainder, 1.0 - 0.213891789, places=9)

        self.assertEqual(Time.from_float(float("inf")).quotient, float("inf"))
        self.assertEqual(Time.from_float(float("inf")).remainder, float("inf"))

        self.assertEqual(Time.from_float(-float("inf")).quotient, -float("inf"))
        self.assertEqual(Time.from_float(-float("inf")).remainder, -float("inf"))

    def test_update(self):
        time = Time(3.0, 0.3247823)
        other_time = Time(6.0, 0.1231)
        time.update(other_time)
        self.assertEqual(time.quotient, 6.0)
        self.assertEqual(time.remainder, 0.1231)
        self.assertEqual(other_time.quotient, 6.0)
        self.assertEqual(other_time.remainder, 0.1231)
        self.assertIsNot(time, other_time)

    def test_add(self):
        time = Time(0.0, 0.0)
        result = time + 1.0
        self.assertEqual(time.quotient, 0.0)
        self.assertEqual(time.remainder, 0.0)
        self.assertEqual(result.quotient, 1.0)
        self.assertEqual(result.remainder, 0.0)

        time = Time(57854.0, 0.4536734584357)
        result = time + 1.5426347561237
        self.assertEqual(time.quotient, 57854.0)
        self.assertEqual(time.remainder, 0.4536734584357)
        self.assertEqual(result.quotient, 57855.0)
        self.assertAlmostEqual(result.remainder, 0.9963082145594, places=13)

        time = Time(54678924378.0, 0.3216781233653267)
        result = time + 2.241783567823847
        self.assertEqual(time.quotient, 54678924378.0)
        self.assertEqual(time.remainder, 0.3216781233653267)
        self.assertEqual(result.quotient, 54678924380.0)
        self.assertAlmostEqual(result.remainder, 0.5634616911891737, places=13)

        time = Time(54678924378.0, 0.3216781233653267)
        result = time + 1.0e-13
        self.assertEqual(time.quotient, 54678924378.0)
        self.assertEqual(time.remainder, 0.3216781233653267)
        self.assertEqual(result.quotient, 54678924378.0)
        self.assertAlmostEqual(result.remainder, 0.3216781233654267, places=13)

        time = Time(57854.0, 0.4536734584357)
        result = time + float("inf")
        self.assertEqual(time.quotient, 57854.0)
        self.assertEqual(time.remainder, 0.4536734584357)
        self.assertEqual(result.quotient, float("inf"))
        self.assertEqual(result.remainder, float("inf"))

        time = Time(57854.0, 0.4536734584357)
        result = time + (-float("inf"))
        self.assertEqual(time.quotient, 57854.0)
        self.assertEqual(time.remainder, 0.4536734584357)
        self.assertEqual(result.quotient, -float("inf"))
        self.assertEqual(result.remainder, -float("inf"))

    def test_sub(self):
        time = Time(1.0, 0.0)
        other_time = Time(0.0, 0.0)
        result = time - other_time
        self.assertEqual(time.quotient, 1.0)
        self.assertEqual(time.remainder, 0.0)
        self.assertEqual(other_time.quotient, 0.0)
        self.assertEqual(other_time.remainder, 0.0)
        self.assertEqual(result, 1.0)

        time = Time(57854.0, 0.4536734584357)
        other_time = Time(57854.0, 0.2341521367)
        result = time - other_time
        self.assertEqual(time.quotient, 57854.0)
        self.assertEqual(time.remainder, 0.4536734584357)
        self.assertEqual(other_time.quotient, 57854.0)
        self.assertEqual(other_time.remainder, 0.2341521367)
        self.assertAlmostEqual(result, 0.2195213217357, places=13)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924376.0, 0.4536734584357)
        result = time - other_time
        self.assertEqual(time.quotient, 54678924378.0)
        self.assertEqual(time.remainder, 0.3216781233653267)
        self.assertEqual(other_time.quotient, 54678924376.0)
        self.assertEqual(other_time.remainder, 0.4536734584357)
        self.assertAlmostEqual(result, 1.8680046649296267, places=12)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233652267)
        result = time - other_time
        self.assertEqual(time.quotient, 54678924378.0)
        self.assertEqual(time.remainder,  0.3216781233653267)
        self.assertEqual(other_time.quotient, 54678924378.0)
        self.assertEqual(other_time.remainder, 0.3216781233652267)
        self.assertAlmostEqual(result, 1.0e-13, places=13)

        time = Time(57854.0, 0.4536734584357)
        other_time = Time(float("inf"), float("inf"))
        result = time - other_time
        self.assertEqual(time.quotient, 57854.0)
        self.assertEqual(time.remainder, 0.4536734584357)
        self.assertEqual(other_time.quotient, float("inf"))
        self.assertEqual(other_time.remainder, float("inf"))
        self.assertEqual(result, -float("inf"))

        time = Time(57854.0, 0.4536734584357)
        other_time = Time(-float("inf"), -float("inf"))
        result = time - other_time
        self.assertEqual(time.quotient, 57854.0)
        self.assertEqual(time.remainder, 0.4536734584357)
        self.assertEqual(other_time.quotient, -float("inf"))
        self.assertEqual(other_time.remainder, -float("inf"))
        self.assertEqual(result, float("inf"))

        time = Time(float("inf"), float("inf"))
        other_time = Time(57854.0, 0.4536734584357)
        result = time - other_time
        self.assertEqual(time.quotient, float("inf"))
        self.assertEqual(time.remainder, float("inf"))
        self.assertEqual(other_time.quotient, 57854.0)
        self.assertEqual(other_time.remainder, 0.4536734584357)
        self.assertEqual(result, float("inf"))

        time = Time(-float("inf"), -float("inf"))
        other_time = Time(57854.0, 0.4536734584357)
        result = time - other_time
        self.assertEqual(time.quotient, -float("inf"))
        self.assertEqual(time.remainder, -float("inf"))
        self.assertEqual(other_time.quotient, 57854.0)
        self.assertEqual(other_time.remainder, 0.4536734584357)
        self.assertEqual(result, -float("inf"))

    def test_equal_comparison(self):
        time = Time(1.0, 0.5)
        other_time = Time(1.0, 0.5)
        self.assertTrue(time == other_time)
        self.assertEqual(time, other_time)
        self.assertIsNot(time, other_time)

        time = Time(1.0, 0.6)
        other_time = Time(1.0, 0.5)
        self.assertFalse(time == other_time)
        self.assertNotEqual(time, other_time)
        self.assertIsNot(time, other_time)

        time = Time(1.0, 0.4)
        other_time = Time(1.0, 0.5)
        self.assertFalse(time == other_time)
        self.assertNotEqual(time, other_time)
        self.assertIsNot(time, other_time)

        time = Time(2.0, 0.5)
        other_time = Time(1.0, 0.5)
        self.assertFalse(time == other_time)
        self.assertNotEqual(time, other_time)
        self.assertIsNot(time, other_time)

        time = Time(0.0, 0.5)
        other_time = Time(1.0, 0.5)
        self.assertFalse(time == other_time)
        self.assertNotEqual(time, other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertTrue(time == other_time)
        self.assertEqual(time, other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653266)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertFalse(time == other_time)
        self.assertNotEqual(time, other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924377.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertFalse(time == other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653266)
        self.assertFalse(time == other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924377.0, 0.3216781233653267)
        self.assertFalse(time == other_time)
        self.assertIsNot(time, other_time)

    def test_almost_equal_comparison(self):
        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertAlmostEqual(time, other_time, places=13)

        time = Time(54678924378.0, 0.3216781233653266)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertAlmostEqual(time, other_time, places=13)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653367)
        self.assertAlmostEqual(time, other_time, places=13)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233654267)
        self.assertNotAlmostEqual(time, other_time, places=13)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924377.0, 0.3216781233653267)
        self.assertNotAlmostEqual(time, other_time)

    def test_less_than_comparison(self):
        time = Time(1.0, 0.5)
        other_time = Time(1.0, 0.5)
        self.assertFalse(time < other_time)
        self.assertIsNot(time, other_time)

        time = Time(1.0, 0.6)
        other_time = Time(1.0, 0.5)
        self.assertFalse(time < other_time)
        self.assertIsNot(time, other_time)

        time = Time(1.0, 0.4)
        other_time = Time(1.0, 0.5)
        self.assertTrue(time < other_time)
        self.assertIsNot(time, other_time)

        time = Time(2.0, 0.5)
        other_time = Time(1.0, 0.5)
        self.assertFalse(time < other_time)
        self.assertIsNot(time, other_time)

        time = Time(0.0, 0.5)
        other_time = Time(1.0, 0.5)
        self.assertTrue(time < other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertFalse(time < other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653266)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertTrue(time < other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924377.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertTrue(time < other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653266)
        self.assertFalse(time < other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924377.0, 0.3216781233653267)
        self.assertFalse(time < other_time)
        self.assertIsNot(time, other_time)

    def test_greater_than_comparison(self):
        time = Time(1.0, 0.5)
        other_time = Time(1.0, 0.5)
        self.assertFalse(time > other_time)
        self.assertIsNot(time, other_time)

        time = Time(1.0, 0.6)
        other_time = Time(1.0, 0.5)
        self.assertTrue(time > other_time)
        self.assertIsNot(time, other_time)

        time = Time(1.0, 0.4)
        other_time = Time(1.0, 0.5)
        self.assertFalse(time > other_time)
        self.assertIsNot(time, other_time)

        time = Time(2.0, 0.5)
        other_time = Time(1.0, 0.5)
        self.assertTrue(time > other_time)
        self.assertIsNot(time, other_time)

        time = Time(0.0, 0.5)
        other_time = Time(1.0, 0.5)
        self.assertFalse(time > other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertFalse(time > other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653266)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertFalse(time > other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924377.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertFalse(time > other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653266)
        self.assertTrue(time > other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924377.0, 0.3216781233653267)
        self.assertTrue(time > other_time)
        self.assertIsNot(time, other_time)

    def test_less_equal_comparison(self):
        time = Time(1.0, 0.5)
        other_time = Time(1.0, 0.5)
        self.assertTrue(time <= other_time)
        self.assertIsNot(time, other_time)

        time = Time(1.0, 0.6)
        other_time = Time(1.0, 0.5)
        self.assertFalse(time <= other_time)
        self.assertIsNot(time, other_time)

        time = Time(1.0, 0.4)
        other_time = Time(1.0, 0.5)
        self.assertTrue(time <= other_time)
        self.assertIsNot(time, other_time)

        time = Time(2.0, 0.5)
        other_time = Time(1.0, 0.5)
        self.assertFalse(time <= other_time)
        self.assertIsNot(time, other_time)

        time = Time(0.0, 0.5)
        other_time = Time(1.0, 0.5)
        self.assertTrue(time <= other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertTrue(time <= other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653266)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertTrue(time <= other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924377.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertTrue(time <= other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653266)
        self.assertFalse(time <= other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924377.0, 0.3216781233653267)
        self.assertFalse(time <= other_time)
        self.assertIsNot(time, other_time)

    def test_greater_equal_comparison(self):
        time = Time(1.0, 0.5)
        other_time = Time(1.0, 0.5)
        self.assertTrue(time >= other_time)
        self.assertIsNot(time, other_time)

        time = Time(1.0, 0.6)
        other_time = Time(1.0, 0.5)
        self.assertTrue(time >= other_time)
        self.assertIsNot(time, other_time)

        time = Time(1.0, 0.4)
        other_time = Time(1.0, 0.5)
        self.assertFalse(time >= other_time)
        self.assertIsNot(time, other_time)

        time = Time(2.0, 0.5)
        other_time = Time(1.0, 0.5)
        self.assertTrue(time >= other_time)
        self.assertIsNot(time, other_time)

        time = Time(0.0, 0.5)
        other_time = Time(1.0, 0.5)
        self.assertFalse(time >= other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertTrue(time >= other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653266)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertFalse(time >= other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924377.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653267)
        self.assertFalse(time >= other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924378.0, 0.3216781233653266)
        self.assertTrue(time >= other_time)
        self.assertIsNot(time, other_time)

        time = Time(54678924378.0, 0.3216781233653267)
        other_time = Time(54678924377.0, 0.3216781233653267)
        self.assertTrue(time >= other_time)
        self.assertIsNot(time, other_time)

    def test_inf(self):
        self.assertEqual(inf.quotient, float("inf"))
        self.assertEqual(inf.remainder, float("inf"))


if __name__ == '__main__':
    main()
