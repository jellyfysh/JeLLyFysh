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
from jellyfysh.base.particle import Particle


class TestParticle(TestCase):
    def test_particle(self):
        particle = Particle([1, 2, 3])
        self.assertEqual(particle.position, [1, 2, 3])
        self.assertIsNone(particle.charge)
        particle.position = [4, 5, 6]
        self.assertEqual(particle.position, [4, 5, 6])
        self.assertIsNone(particle.charge)
        particle.charge = {"electric": 1.0}
        self.assertEqual(particle.position, [4, 5, 6])
        self.assertEqual(particle.charge, {"electric": 1.0})

        particle = Particle([1], {"test": -0.5, "charge": 0.2})
        self.assertEqual(particle.position, [1])
        self.assertEqual(particle.charge, {"test": -0.5, "charge": 0.2})


if __name__ == '__main__':
    main()
