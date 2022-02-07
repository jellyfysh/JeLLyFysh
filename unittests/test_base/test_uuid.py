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
from unittest import TestCase, main, mock
from jellyfysh.base import uuid


class TestUuid(TestCase):
    @mock.patch("jellyfysh.base.uuid.uuid.uuid4")
    def test_get_uuid(self, uuid_function):
        uuid_function.return_value = 12
        identifier = uuid.get_uuid()
        self.assertEqual(identifier, 12)
        # On the first call the uuid function should be called, on the second not
        uuid_function.assert_called_once_with()
        uuid_function.reset_mock()
        identifier = uuid.get_uuid()
        self.assertEqual(identifier, 12)
        uuid_function.assert_not_called()


if __name__ == '__main__':
    main()
