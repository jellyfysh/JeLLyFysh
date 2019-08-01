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
from configparser import ConfigParser
import importlib
import os
import sys
import base.factory as factory
from base.exceptions import ConfigurationError

_path_added = False
_unittest_directory = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/..")
_current_working_directory = os.getcwd()


def setUpModule():
    # Change to the unittest directory as the current working directory so that the factory finds all relevant files
    # Also add the unittest to sys.path, so the importlib finds the modules
    global _path_added
    if _unittest_directory not in sys.path:
        _path_added = True
        sys.path.insert(0, _unittest_directory)
    os.chdir(_unittest_directory)


def tearDownModule():
    # Revert everything which was done in setUpModule
    if _path_added:
        sys.path.remove(_unittest_directory)
    os.chdir(_current_working_directory)


class TestFactory(TestCase):
    def setUp(self) -> None:
        self._config = ConfigParser()

    def tearDown(self) -> None:
        factory.used_sections = []

    def test_build_simple_class_default_constructable(self):
        module = importlib.import_module("test_base.test_classes_for_factory.simple_class_default_constructable")
        # Since the class is default constructable, the section has not to be inside of the config file
        created = factory.build_from_config(self._config, "SimpleClassDefaultConstructable",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.SimpleClassDefaultConstructable)
        self.assertEqual(created.__class__.__name__, "SimpleClassDefaultConstructable")
        self.assertEqual(created.some_integer,
                         module.SimpleClassDefaultConstructable.default_values["some_integer"])
        self.assertEqual(created.some_float,
                         module.SimpleClassDefaultConstructable.default_values["some_float"])
        self.assertEqual(created.some_bool,
                         module.SimpleClassDefaultConstructable.default_values["some_bool"])
        self.assertEqual(created.some_string,
                         module.SimpleClassDefaultConstructable.default_values["some_string"])

    def test_build_simple_class_default_constructable_with_name(self):
        module = importlib.import_module("test_base.test_classes_for_factory.simple_class_default_constructable")
        created = factory.build_from_config(self._config, "SomeSectionName", "test_base.test_classes_for_factory",
                                            "SimpleClassDefaultConstructable")
        self.assertIsInstance(created, module.SimpleClassDefaultConstructable)
        self.assertEqual(created.__class__.__name__, "SomeSectionName (SimpleClassDefaultConstructable)")
        self.assertEqual(created.some_integer,
                         module.SimpleClassDefaultConstructable.default_values["some_integer"])
        self.assertEqual(created.some_float,
                         module.SimpleClassDefaultConstructable.default_values["some_float"])
        self.assertEqual(created.some_bool,
                         module.SimpleClassDefaultConstructable.default_values["some_bool"])
        self.assertEqual(created.some_string,
                         module.SimpleClassDefaultConstructable.default_values["some_string"])

    def test_build_simple_class_default_constructable_with_arguments(self):
        module = importlib.import_module("test_base.test_classes_for_factory.simple_class_default_constructable")
        self._config["SimpleClassDefaultConstructable"] = {"some_integer": "2",
                                                           "some_float": "-1.2",
                                                           "some_bool": "False",
                                                           "some_string": "Test"}
        created = factory.build_from_config(self._config, "SimpleClassDefaultConstructable",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.SimpleClassDefaultConstructable)
        self.assertEqual(created.__class__.__name__, "SimpleClassDefaultConstructable")
        self.assertEqual(created.some_integer, 2)
        self.assertEqual(created.some_float, -1.2)
        self.assertEqual(created.some_bool, False)
        self.assertEqual(created.some_string, "Test")

    def test_allowed_boolean_strings(self):
        module = importlib.import_module("test_base.test_classes_for_factory.simple_class_default_constructable")
        self._config["SimpleClassDefaultConstructable"] = {"some_bool": "1"}
        created = factory.build_from_config(self._config, "SimpleClassDefaultConstructable",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.SimpleClassDefaultConstructable)
        self.assertEqual(created.some_bool, True)

        self._config["SimpleClassDefaultConstructable"] = {"some_bool": "YeS"}
        created = factory.build_from_config(self._config, "SimpleClassDefaultConstructable",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.SimpleClassDefaultConstructable)
        self.assertEqual(created.some_bool, True)

        self._config["SimpleClassDefaultConstructable"] = {"some_bool": "true"}
        created = factory.build_from_config(self._config, "SimpleClassDefaultConstructable",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.SimpleClassDefaultConstructable)
        self.assertEqual(created.some_bool, True)

        self._config["SimpleClassDefaultConstructable"] = {"some_bool": "ON"}
        created = factory.build_from_config(self._config, "SimpleClassDefaultConstructable",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.SimpleClassDefaultConstructable)
        self.assertEqual(created.some_bool, True)

        self._config["SimpleClassDefaultConstructable"] = {"some_bool": "0"}
        created = factory.build_from_config(self._config, "SimpleClassDefaultConstructable",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.SimpleClassDefaultConstructable)
        self.assertEqual(created.some_bool, False)

        self._config["SimpleClassDefaultConstructable"] = {"some_bool": "nO"}
        created = factory.build_from_config(self._config, "SimpleClassDefaultConstructable",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.SimpleClassDefaultConstructable)
        self.assertEqual(created.some_bool, False)

        self._config["SimpleClassDefaultConstructable"] = {"some_bool": "FALSE"}
        created = factory.build_from_config(self._config, "SimpleClassDefaultConstructable",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.SimpleClassDefaultConstructable)
        self.assertEqual(created.some_bool, False)

        self._config["SimpleClassDefaultConstructable"] = {"some_bool": "off"}
        created = factory.build_from_config(self._config, "SimpleClassDefaultConstructable",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.SimpleClassDefaultConstructable)
        self.assertEqual(created.some_bool, False)

    def test_not_allowed_boolean_string_raises_error(self):
        self._config["SimpleClassDefaultConstructable"] = {"some_bool": "of"}
        with self.assertRaises(TypeError):
            factory.build_from_config(self._config, "SimpleClassDefaultConstructable",
                                      "test_base.test_classes_for_factory")

        self._config["SimpleClassDefaultConstructable"] = {"some_bool": "Tru"}
        with self.assertRaises(TypeError):
            factory.build_from_config(self._config, "SimpleClassDefaultConstructable",
                                      "test_base.test_classes_for_factory")

    def test_build_simple_class_not_default_constructable_with_arguments(self):
        module = importlib.import_module("test_base.test_classes_for_factory.simple_class_not_default_constructable")
        self._config["SimpleClassNotDefaultConstructable"] = {"some_integer": "0",
                                                              "some_float": "31.231",
                                                              "some_bool": "No",
                                                              "some_string": "String"}
        created = factory.build_from_config(self._config, "SimpleClassNotDefaultConstructable",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.SimpleClassNotDefaultConstructable)
        self.assertEqual(created.__class__.__name__, "SimpleClassNotDefaultConstructable")
        self.assertEqual(created.some_integer, 0)
        self.assertEqual(created.some_float, 31.231)
        self.assertEqual(created.some_bool, False)
        self.assertEqual(created.some_string, "String")

    def test_build_simple_class_not_default_constructable_missing_arguments_raises_error(self):
        self._config["SimpleClassNotDefaultConstructable"] = {"some_integer": "0",
                                                              "some_float": "31.231",
                                                              "some_bool": "No"}
        with self.assertRaises(ConfigurationError):
            factory.build_from_config(self._config, "SimpleClassNotDefaultConstructable",
                                      "test_base.test_classes_for_factory")

    def test_wrong_argument_raises_error(self):
        self._config["SimpleClassNotDefaultConstructable"] = {"some_integer": "0",
                                                              "some_float": "31.231",
                                                              "some_bool": "No",
                                                              "some_string": "Test",
                                                              "wrong": "wrong"}
        with self.assertRaises(ConfigurationError):
            factory.build_from_config(self._config, "SimpleClassNotDefaultConstructable",
                                      "test_base.test_classes_for_factory")

    def test_build_class_with_list_of_simple_type(self):
        module = importlib.import_module("test_base.test_classes_for_factory.class_with_list_of_simple_type")
        self._config["ClassWithListOfSimpleType"] = {"some_list": "0, 1, 2"}
        created = factory.build_from_config(self._config, "ClassWithListOfSimpleType",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.ClassWithListOfSimpleType)
        self.assertEqual(created.__class__.__name__, "ClassWithListOfSimpleType")
        self.assertEqual(created.some_list, [0, 1, 2])

        # Spaces in comma separated list are optional
        self._config["ClassWithListOfSimpleType"] = {"some_list": "0,1,2"}
        created = factory.build_from_config(self._config, "ClassWithListOfSimpleType",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.ClassWithListOfSimpleType)
        self.assertEqual(created.__class__.__name__, "ClassWithListOfSimpleType")
        self.assertEqual(created.some_list, [0, 1, 2])

        # Several spaces are also allowed
        self._config["ClassWithListOfSimpleType"] = {"some_list": "0,  1,     2"}
        created = factory.build_from_config(self._config, "ClassWithListOfSimpleType",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.ClassWithListOfSimpleType)
        self.assertEqual(created.__class__.__name__, "ClassWithListOfSimpleType")
        self.assertEqual(created.some_list, [0, 1, 2])

        # Tabulators are also fine
        self._config["ClassWithListOfSimpleType"] = {"some_list": "0,\t1,\t2"}
        created = factory.build_from_config(self._config, "ClassWithListOfSimpleType",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.ClassWithListOfSimpleType)
        self.assertEqual(created.__class__.__name__, "ClassWithListOfSimpleType")
        self.assertEqual(created.some_list, [0, 1, 2])

        # Line Breaks are supported as well
        self._config["ClassWithListOfSimpleType"] = {"some_list": "0,\n1,\n2"}
        created = factory.build_from_config(self._config, "ClassWithListOfSimpleType",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.ClassWithListOfSimpleType)
        self.assertEqual(created.__class__.__name__, "ClassWithListOfSimpleType")
        self.assertEqual(created.some_list, [0, 1, 2])

        # Some arbitrary mixture of all this is fine
        self._config["ClassWithListOfSimpleType"] = {"some_list": "0,\n\t 1,  \t  \n2"}
        created = factory.build_from_config(self._config, "ClassWithListOfSimpleType",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.ClassWithListOfSimpleType)
        self.assertEqual(created.__class__.__name__, "ClassWithListOfSimpleType")
        self.assertEqual(created.some_list, [0, 1, 2])

    def test_build_simple_class_in_package(self):
        module = importlib.import_module("test_base.test_classes_for_factory.simple_class_in_package"
                                         ".simple_class_in_package")
        created = factory.build_from_config(self._config, "SimpleClassInPackage",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module.SimpleClassInPackage)
        self.assertEqual(created.__class__.__name__, "SimpleClassInPackage")

    def test_build_custom_class_default_constructable(self):
        module_custom = importlib.import_module("test_base.test_classes_for_factory.custom_class_default_constructable")
        module_simple = importlib.import_module("test_base.test_classes_for_factory.simple_class_default_constructable")
        self._config["CustomClassDefaultConstructable"] = {"simple_class": "simple_class_default_constructable)"}
        created = factory.build_from_config(self._config, "CustomClassDefaultConstructable",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module_custom.CustomClassDefaultConstructable)
        self.assertEqual(created.__class__.__name__, "CustomClassDefaultConstructable")
        self.assertIsInstance(created.simple_class, module_simple.SimpleClassDefaultConstructable)
        self.assertEqual(created.simple_class.__class__.__name__, "SimpleClassDefaultConstructable")
        self.assertEqual(created.simple_class.some_integer,
                         module_simple.SimpleClassDefaultConstructable.default_values["some_integer"])
        self.assertEqual(created.simple_class.some_float,
                         module_simple.SimpleClassDefaultConstructable.default_values["some_float"])
        self.assertEqual(created.simple_class.some_bool,
                         module_simple.SimpleClassDefaultConstructable.default_values["some_bool"])
        self.assertEqual(created.simple_class.some_string,
                         module_simple.SimpleClassDefaultConstructable.default_values["some_string"])

    def test_build_custom_class_default_constructable_with_name(self):
        module_custom = importlib.import_module("test_base.test_classes_for_factory.custom_class_default_constructable")
        module_simple = importlib.import_module("test_base.test_classes_for_factory.simple_class_default_constructable")
        # One can set the name of the simple class within the class
        self._config["CustomClassDefaultConstructable"] = {"simple_class": "test (simple_class_default_constructable)"}
        created = factory.build_from_config(self._config, "CustomClassDefaultConstructable",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module_custom.CustomClassDefaultConstructable)
        self.assertEqual(created.__class__.__name__, "CustomClassDefaultConstructable")
        self.assertIsInstance(created.simple_class, module_simple.SimpleClassDefaultConstructable)
        self.assertEqual(created.simple_class.__class__.__name__, "Test (SimpleClassDefaultConstructable)")
        self.assertEqual(created.simple_class.some_integer,
                         module_simple.SimpleClassDefaultConstructable.default_values["some_integer"])
        self.assertEqual(created.simple_class.some_float,
                         module_simple.SimpleClassDefaultConstructable.default_values["some_float"])
        self.assertEqual(created.simple_class.some_bool,
                         module_simple.SimpleClassDefaultConstructable.default_values["some_bool"])
        self.assertEqual(created.simple_class.some_string,
                         module_simple.SimpleClassDefaultConstructable.default_values["some_string"])

    def test_build_custom_class_default_constructable_with_name_and_arguments(self):
        module_custom = importlib.import_module("test_base.test_classes_for_factory.custom_class_default_constructable")
        module_simple = importlib.import_module("test_base.test_classes_for_factory.simple_class_default_constructable")
        self._config["CustomClassDefaultConstructable"] = {"simple_class": "test (simple_class_default_constructable)"}
        self._config["Test"] = {"some_integer": "-3"}
        created = factory.build_from_config(self._config, "CustomClassDefaultConstructable",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module_custom.CustomClassDefaultConstructable)
        self.assertEqual(created.__class__.__name__, "CustomClassDefaultConstructable")
        self.assertIsInstance(created.simple_class, module_simple.SimpleClassDefaultConstructable)
        self.assertEqual(created.simple_class.__class__.__name__, "Test (SimpleClassDefaultConstructable)")
        self.assertEqual(created.simple_class.some_integer, -3)
        self.assertEqual(created.simple_class.some_float,
                         module_simple.SimpleClassDefaultConstructable.default_values["some_float"])
        self.assertEqual(created.simple_class.some_bool,
                         module_simple.SimpleClassDefaultConstructable.default_values["some_bool"])
        self.assertEqual(created.simple_class.some_string,
                         module_simple.SimpleClassDefaultConstructable.default_values["some_string"])

        # Tabs and several spaces are also allowed
        self._config["CustomClassDefaultConstructable"] = {"simple_class":
                                                           "test \t  (simple_class_default_constructable)"}
        self._config["Test"] = {"some_string": "-3"}
        created = factory.build_from_config(self._config, "CustomClassDefaultConstructable",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module_custom.CustomClassDefaultConstructable)
        self.assertEqual(created.__class__.__name__, "CustomClassDefaultConstructable")
        self.assertIsInstance(created.simple_class, module_simple.SimpleClassDefaultConstructable)
        self.assertEqual(created.simple_class.__class__.__name__, "Test (SimpleClassDefaultConstructable)")
        self.assertEqual(created.simple_class.some_integer,
                         module_simple.SimpleClassDefaultConstructable.default_values["some_integer"])
        self.assertEqual(created.simple_class.some_float,
                         module_simple.SimpleClassDefaultConstructable.default_values["some_float"])
        self.assertEqual(created.simple_class.some_bool,
                         module_simple.SimpleClassDefaultConstructable.default_values["some_bool"])
        self.assertEqual(created.simple_class.some_string, "-3")

    def test_build_custom_class_with_abstract_class(self):
        module_custom = importlib.import_module("test_base.test_classes_for_factory.custom_class_with_abstract_class")
        module_simple = importlib.import_module("test_base.test_classes_for_factory.simple_abstract_class"
                                                ".simple_inheriting_class")
        self._config["CustomClassWithAbstractClass"] = {"abstract_class": "simple_inheriting_class"}
        self._config["SimpleInheritingClass"] = {"some_string": "some_string", "some_integer": "13"}
        created = factory.build_from_config(self._config, "CustomClassWithAbstractClass",
                                            "test_base.test_classes_for_factory")
        self.assertIsInstance(created, module_custom.CustomClassWithAbstractClass)
        self.assertEqual(created.__class__.__name__, "CustomClassWithAbstractClass")
        self.assertIsInstance(created.abstract_class, module_simple.SimpleInheritingClass)
        self.assertEqual(created.abstract_class.some_string, "some_string")
        self.assertEqual(created.abstract_class.some_integer, 13)
        self.assertEqual(created.abstract_class.__class__.__name__, "SimpleInheritingClass")

    def test_non_existing_package_raises_error(self):
        with self.assertRaises(ImportError):
            factory.build_from_config(self._config, "SimpleClassDefaultConstructable", "wrong.package")

    def test_used_sections(self):
        self._config["CustomClassWithAbstractClass"] = {"abstract_class": "simple_inheriting_class"}
        self._config["SimpleInheritingClass"] = {"some_string": "some_string", "some_integer": "13"}
        factory.build_from_config(self._config, "CustomClassWithAbstractClass",
                                  "test_base.test_classes_for_factory")
        factory.build_from_config(self._config, "SomeSectionName", "test_base.test_classes_for_factory",
                                  "SimpleClassDefaultConstructable")
        self.assertEqual(factory.used_sections, ["CustomClassWithAbstractClass", "SimpleInheritingClass",
                                                 "SomeSectionName"])

    def test_get_alias_alias_set(self):
        self.assertEqual(factory.get_alias("Test (SomeClass)"), "Test")

    def test_get_alias_alias_not_set(self):
        self.assertEqual(factory.get_alias("SomeClass"), "SomeClass")

    def test_get_alias_no_whitespace_raises_error(self):
        with self.assertRaises(ConfigurationError):
            factory.get_alias("Test(SomeClass)")

    def test_get_alias_too_many_whitespaces_raises_error(self):
        with self.assertRaises(ConfigurationError):
            factory.get_alias("Test  (SomeClass)")

    def test_get_alias_missing_brackets_raises_error(self):
        with self.assertRaises(ConfigurationError):
            factory.get_alias("Test SomeClass")

    def test_get_alias_empty_alias_raises_error(self):
        with self.assertRaises(ConfigurationError):
            factory.get_alias(" (SomeClass)")

    def test_get_alias_empty_class_raises_error(self):
        with self.assertRaises(ConfigurationError):
            factory.get_alias("Test ()")

    def test_get_alias_empty_string_raises_error(self):
        with self.assertRaises(ConfigurationError):
            factory.get_alias("")


if __name__ == '__main__':
    main()
