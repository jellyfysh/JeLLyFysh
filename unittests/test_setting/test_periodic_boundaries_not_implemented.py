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
from jellyfysh.setting.periodic_boundaries import PeriodicBoundariesNotImplemented


class TestPeriodicBoundariesNotImplemented(TestCase):
    def setUp(self) -> None:
        self._periodic_boundaries = PeriodicBoundariesNotImplemented()

    def test_correct_periodic_boundary_position_raises_error(self):
        position = [0.1, 0.2, 0.3]
        with self.assertRaises(NotImplementedError):
            self._periodic_boundaries.correct_position(position)

    def test_correct_periodic_boundary_position_direction_raises_error(self):
        position = [0.1, 0.2, 0.3]
        with self.assertRaises(NotImplementedError):
            self._periodic_boundaries.correct_position_entry(position[1], 1)

    def test_separation_vector_raises_error(self):
        position_one = [0.1, 0.2, 0.3]
        position_two = [0.4, 0.5, 0.6]
        with self.assertRaises(NotImplementedError):
            self._periodic_boundaries.separation_vector(position_one, position_two)

    def test_correct_periodic_boundary_separation_raises_error(self):
        separation = [0.1, 0.2, 0.3]
        with self.assertRaises(NotImplementedError):
            self._periodic_boundaries.correct_separation(separation)

    def test_correct_periodic_boundary_separation_direction_raises_error(self):
        separation = [0.1, 0.2, 0.3]
        with self.assertRaises(NotImplementedError):
            self._periodic_boundaries.correct_separation_entry(separation[1], 1)

    def test_next_image_raises_error(self):
        with self.assertRaises(NotImplementedError):
            self._periodic_boundaries.next_image(0.1, 0)


if __name__ == '__main__':
    main()
