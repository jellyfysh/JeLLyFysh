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
from jellyfysh.base import strings


class TestStrings(TestCase):
    def test_to_camel_case(self):
        self.assertEqual(strings.to_camel_case("some_snake_case_name"), "SomeSnakeCaseName")
        self.assertEqual(strings.to_camel_case("short"), "Short")
        self.assertEqual(strings.to_camel_case("AlreadyCamelCase"), "AlreadyCamelCase")

    def test_to_snake_case(self):
        self.assertEqual(strings.to_snake_case("SomeCamelCaseName"), "some_camel_case_name")
        self.assertEqual(strings.to_snake_case("SomeLARGEName"), "some_large_name")
        self.assertEqual(strings.to_snake_case("Short"), "short")
        self.assertEqual(strings.to_snake_case("already_snake_case"), "already_snake_case")

    def test_to_directory_path(self):
        self.assertEqual(strings.to_directory_path("some.package.path"), "some/package/path")
        self.assertEqual(strings.to_directory_path("short_path"), "short_path")
        self.assertEqual(strings.to_directory_path("already/directory/path"), "already/directory/path")


if __name__ == '__main__':
    main()
