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
from unittest import TestCase, mock, main
from jellyfysh.base.exceptions import LiftingSchemeError
from jellyfysh.lifting.inside_first_lifting import InsideFirstLifting


class TestInsideFirstLifting(TestCase):
    def setUp(self):
        self._inside_first = InsideFirstLifting()

    @mock.patch("jellyfysh.lifting.lifting.random.uniform")
    def test_get_active_identifier(self, random_mock):
        random_mock.reset_mock()
        random_mock.return_value = 0.0
        self._inside_first.reset()
        self._inside_first.insert(0.6, (1, 2), False)
        self._inside_first.insert(-0.5, (3, 4), False)
        self._inside_first.insert(-0.5, (5, 6), False)
        self._inside_first.insert(-0.5, (7, 8), False)
        self._inside_first.insert(-0.1, (9, 10), False)
        self._inside_first.insert(0.5, (11, 12), True)
        self._inside_first.insert(0.3, (13, 14), False)
        self._inside_first.insert(0.2, (15, 16), False)
        self.assertTupleEqual(self._inside_first.get_active_identifier(), (5, 6))
        random_mock.assert_called_once_with(0.0, 0.5)

        random_mock.reset_mock()
        random_mock.return_value = 0.4 - 1e-10
        self._inside_first.reset()
        self._inside_first.insert(0.6, (1, 2), False)
        self._inside_first.insert(-0.5, (3, 4), False)
        self._inside_first.insert(-0.5, (5, 6), False)
        self._inside_first.insert(-0.5, (7, 8), False)
        self._inside_first.insert(-0.1, (9, 10), False)
        self._inside_first.insert(0.5, (11, 12), True)
        self._inside_first.insert(0.3, (13, 14), False)
        self._inside_first.insert(0.2, (15, 16), False)

        self.assertTupleEqual(self._inside_first.get_active_identifier(), (5, 6))
        random_mock.assert_called_once_with(0.0, 0.5)

        random_mock.reset_mock()
        random_mock.return_value = 0.4 + 1e-10
        self._inside_first.reset()
        self._inside_first.insert(0.6, (1, 2), False)
        self._inside_first.insert(-0.5, (3, 4), False)
        self._inside_first.insert(-0.5, (5, 6), False)
        self._inside_first.insert(-0.5, (7, 8), False)
        self._inside_first.insert(-0.1, (9, 10), False)
        self._inside_first.insert(0.5, (11, 12), True)
        self._inside_first.insert(0.3, (13, 14), False)
        self._inside_first.insert(0.2, (15, 16), False)

        self.assertTupleEqual(self._inside_first.get_active_identifier(), (7, 8))
        random_mock.assert_called_once_with(0.0, 0.5)

        random_mock.reset_mock()
        random_mock.return_value = 0.48
        self._inside_first.reset()
        self._inside_first.insert(0.6, (1, 2), False)
        self._inside_first.insert(-0.5, (3, 4), False)
        self._inside_first.insert(-0.5, (5, 6), False)
        self._inside_first.insert(-0.5, (7, 8), False)
        self._inside_first.insert(-0.1, (9, 10), False)
        self._inside_first.insert(0.5, (11, 12), True)
        self._inside_first.insert(0.3, (13, 14), False)
        self._inside_first.insert(0.2, (15, 16), False)

        self.assertTupleEqual(self._inside_first.get_active_identifier(), (7, 8))
        random_mock.assert_called_once_with(0.0, 0.5)

        self._inside_first.reset()
        self._inside_first.insert(0.6, (1, 2), False)
        self._inside_first.insert(-0.5, (3, 4), False)
        self._inside_first.insert(-0.5, (5, 6), False)
        self._inside_first.insert(-0.5, (7, 8), False)
        self._inside_first.insert(-0.1, (9, 10), False)
        self._inside_first.insert(0.5, (11, 12), False)
        self._inside_first.insert(0.3, (13, 14), False)
        self._inside_first.insert(0.2, (15, 16), False)

        with self.assertRaises(LiftingSchemeError):
            self._inside_first.get_active_identifier()


if __name__ == '__main__':
    main()
