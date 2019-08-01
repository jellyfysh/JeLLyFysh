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
from unittest import TestCase, main
from base.exceptions import InitializerError
from base.initializer import Initializer


class TestClass(Initializer):
    def __init__(self):
        super().__init__()
        self.public_variable = "PublicVariable"

    def initialize(self):
        super().initialize()

    def initialize_alternative(self):
        super().initialize()

    def public_method_one(self):
        pass

    def public_method_two(self, argument):
        pass

    def _private_method(self):
        pass

    def __private_method(self, argument):
        pass

    @property
    def property(self):
        return "Property"


class TestInitializer(TestCase):
    def setUp(self) -> None:
        self._test_class = TestClass()

    def test_public_variable_not_blocked(self):
        self.assertEqual(self._test_class.public_variable, "PublicVariable")

    def test_property_not_blocked(self):
        self.assertEqual(self._test_class.property, "Property")

    def test_public_methods_blocked(self):
        with self.assertRaises(InitializerError):
            self._test_class.public_method_one()
        with self.assertRaises(InitializerError):
            self._test_class.public_method_two(None)

    def test_private_methods_not_blocked(self):
        self._test_class._private_method()
        # noinspection PyUnresolvedReferences
        self._test_class._TestClass__private_method(None)

    def test_public_methods_free_after_initialize(self):
        self._test_class.initialize()
        self._test_class.public_method_one()
        self._test_class.public_method_two(None)

    def test_public_methods_free_after_initialize_alternative(self):
        self._test_class.initialize_alternative()
        self._test_class.public_method_one()
        self._test_class.public_method_two(None)


if __name__ == '__main__':
    main()
