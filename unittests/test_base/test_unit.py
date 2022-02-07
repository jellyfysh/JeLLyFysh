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
from jellyfysh.base.time import Time
from jellyfysh.base.unit import Unit


class TestUnit(TestCase):
    def test_unit(self):
        test_position = [1]
        test_velocity = [2]
        test_time_stamp = Time(0.0, 0.4)
        unit = Unit(0, test_position, velocity=test_velocity, time_stamp=test_time_stamp)
        self.assertEqual(unit.identifier, 0)
        self.assertEqual(unit.position, [1])
        self.assertIsNone(unit.charge)
        self.assertEqual(unit.velocity, [2])
        self.assertEqual(unit.time_stamp, Time(0.0, 0.4))

        # The position, velocity and time stamp was not copied
        test_position[0] += 1
        test_velocity[0] /= 2
        test_time_stamp.update(Time(1.0, 0.1))
        self.assertEqual(unit.position, [2])
        self.assertEqual(unit.velocity, [1])
        self.assertEqual(unit.time_stamp, Time(1.0, 0.1))


if __name__ == '__main__':
    main()
