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
from jellyfysh.lifting.ratio_lifting import RatioLifting


class TestRatioLifting(TestCase):
    def setUp(self):
        self._ratio = RatioLifting()

    @mock.patch("jellyfysh.lifting.lifting.random.uniform")
    def test_get_active_identifier(self, random_mock):

        self._ratio.reset()
        self._ratio.insert(-0.3, (1, 2), False)
        self._ratio.insert(0.6, (3, 4), False)
        self._ratio.insert(0.4, (5, 6), True)
        self._ratio.insert(-0.7, (7, 8), False)

        random_mock.reset_mock()
        random_mock.return_value = 0.2
        return_value = self._ratio.get_active_identifier()
        self.assertTupleEqual(return_value, (1, 2))
        random_mock.assert_called_once_with(0.0, 1.0)

        random_mock.reset_mock()
        random_mock.return_value = 0.0
        return_value = self._ratio.get_active_identifier()
        self.assertTupleEqual(return_value, (1, 2))
        random_mock.assert_called_once_with(0.0, 1.0)

        random_mock.reset_mock()
        random_mock.return_value = 1-1e-10
        return_value = self._ratio.get_active_identifier()
        self.assertTupleEqual(return_value, (7, 8))
        random_mock.assert_called_once_with(0.0, 1.0)

        random_mock.reset_mock()
        random_mock.return_value = 0.3 - 1e-10
        return_value = self._ratio.get_active_identifier()
        self.assertTupleEqual(return_value, (1, 2))
        random_mock.assert_called_once_with(0.0, 1.0)

        random_mock.reset_mock()
        random_mock.return_value = 0.3 + 1e-10
        return_value = self._ratio.get_active_identifier()
        self.assertTupleEqual(return_value, (7, 8))
        random_mock.assert_called_once_with(0.0, 1.0)

        random_mock.reset_mock()
        random_mock.return_value = 1.0
        return_value = self._ratio.get_active_identifier()
        self.assertTupleEqual(return_value, (7, 8))
        random_mock.assert_called_once_with(0.0, 1.0)

        random_mock.reset_mock()
        random_mock.return_value = 1.0 + 1e-10
        return_value = self._ratio.get_active_identifier()
        self.assertTupleEqual(return_value, (7, 8))
        random_mock.assert_called_once_with(0.0, 1.0)

        self._ratio.reset()
        self._ratio.insert(-0.3, (1, 2), False)
        self._ratio.insert(0.6, (3, 4), False)
        self._ratio.insert(0.4, (5, 6), False)
        self._ratio.insert(-0.7, (7, 8), False)
        with self.assertRaises(LiftingSchemeError):
            self._ratio.get_active_identifier()


if __name__ == '__main__':
    main()
