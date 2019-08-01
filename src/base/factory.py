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
"""Module for factory functions."""
from collections import abc
from configparser import ConfigParser
from importlib import import_module
from inspect import signature, Parameter
from os.path import isdir
import re
import typing
from base import strings
from .exceptions import ConfigurationError


# TODO In current convention, options of same name in different sections will lead to creation of multiple instances,
#  which should be replaced by using one instance

used_sections = []
"""Sequence of section arguments of the build_from_config function calls."""


def build_from_config(config: ConfigParser, section: str, package: str, class_name: str = None) -> typing.Any:
    """
    Construct the object in a section of the configuration.

    The key/value pairs of the section must correspond to the arguments of the __init__ method of the class which is
    constructed. Keys and values must be in snake_case, section names in CamelCase.

    The parsed values must be converted to the correct types. The factory determines these types based on the obligatory
    type hints in the __init__ method. Currently supported types are bool, float, int, and str. Similar to the
    configparser module, this factory recognizes booleans in the section from yes/no, on/off, true/false and 1/0. User
    defined classes are also be possible (the factory is then called recursively and the package is determined via the
    type hint). If the user defined class also needs some __init__ arguments, these must be defined in a separate
    section. Finally, typing.Sequence and typing.Iterable consisting of one of all these types is also possible. The
    factory creates a list in this case.

    It is possible to assign aliases for user defined classes within the configuration. In the values they should be
    given as 'alias (real_class_name)'. The arguments of the __init__ method for this class are then located in the
    section [Alias]. If the factory function is directly called with an alias section, the real class name must be
    specified in the class_name argument. For aliased classes, this factory will effectively subclass the real class and
    set __class__.__name__ to 'Alias (RealClassName)'. The alias (or the real class name if there is no alias) gets
    extracted from this attribute via the `get_alias` function of this module.

    User defined classes, which should be constructable by this factory, must be defined in their own module with the
    module name as the class name in snake_case. If the type hint refers to an abstract class, the inheriting class
    must be defined in the same package.

    If there exists a package with the same snake_case name as the class name (given via the section or class_name),
    the package is changed to this package before trying to import the module given by the class name.

    This function will append the section argument to the used_sections sequence of this module.

    Parameters
    ----------
    config : configparser.ConfigParser
        The parsed configuration file.
    section : str
        The section to construct in CamelCase.
    package : str
        The package where the object specified in the section is defined.
    class_name : str or None, optional
        If the section is an alias, this is the real class name in CamelCase.

    Returns
    -------
    Any
        The constructed object.

    Raises
    ------
    ImportError
        If the module given by the class name does not exist in the given package.
    base.exceptions.ConfigurationError
        If the section does not specify all needed arguments of the __init__ method to construct the class.
        If the section specifies an argument which does not appear in the __init__ method of the class.
    TypeError
        If a type hint in the __init__ method is unsupported.
        If a type hint specifies a boolean but the corresponding value is not recognized as one.
    """
    used_sections.append(section)
    if class_name is None:
        class_name = section

    try:
        section_config = config[section]
    except KeyError:
        section_config = {}

    module_name = strings.to_snake_case(class_name)
    if isdir(strings.to_directory_path(package) + "/" + module_name):
        package = package + "." + module_name

    try:
        module = import_module(package + "." + module_name)
    except ImportError:
        raise ImportError(
            "Module '{0}' (set by section name) not existing in package '{1}'.".format(module_name, package))

    class_object = getattr(module, class_name)
    init_signature = signature(class_object.__init__)

    call_dictionary = {}
    for option in init_signature.parameters:
        # If option is not written in config, we assume it has a default value. If not, error is thrown on construction
        if option in section_config.keys():
            call_dictionary[option] = _create_object(section_config[option],
                                                     init_signature.parameters[option].annotation, config)
        else:
            if option != "self" and init_signature.parameters[option].default is Parameter.empty:
                raise ConfigurationError(
                    "Missing required argument '{0}' to create object '{1}' as specified in section '{2}'."
                    .format(option, class_name, section))

    for option, value in section_config.items():
        if option not in call_dictionary.keys():
            raise ConfigurationError(
                "Tried to initialize object '{0}' with argument '{1}' in section '{2}', "
                "which is not a possible argument of this class.".format(class_name, option, section))

    if class_name != section:
        # If 'section' is an alias, we want instance.__class__.__name__ to be 'section (class_name)'
        # We use type to dynamically create a class inheriting from class_object with the proper name
        # Dict (last argument) is empty since we do not want to add attributes/methods
        # For dumping with dill to work, we have to set the __module__ to __main__
        # (see https://stackoverflow.com/questions/51016272)
        class_object = type("{0} ({1})".format(section, class_name), (class_object,), {"__module__": "__main__"})
    instance = class_object(**call_dictionary)
    return instance


_class_name_pattern = re.compile(r"(\w+)(?: \(\w+\))*")


def get_alias(class_name: str) -> str:
    """
    Extract the alias from the class name.

    If a configuration specifies an alias for a user defined class, it is included in __class__.__name__ as
    'Alias (RealClassName)'. This function extracts 'Alias' from this pattern. If no alias has be set, it will
    just return 'RealClassName'.

    Parameters
    ----------
    class_name : str
        The class name, most probably extracted via __class__.__name__ attribute.
    Returns
    -------
    str
        If an alias is set, the alias, otherwise the class name.
    """
    class_name_pattern_match = _class_name_pattern.fullmatch(class_name)
    if class_name_pattern_match is None or len(class_name_pattern_match.groups()) != 1:
        raise ConfigurationError("Given class name '{0}' could not be processed.".format(class_name))
    return class_name_pattern_match.group(1)


_simple_types = (bool, float, int, str)
_bool_true_strings = ['1', 'yes', 'true', 'on']
_bool_false_strings = ['0', 'no', 'false', 'off']
"""  
In Python 3.7 the variable __origin__ of parameter type Sequence[Any] will return collections.abc.Sequence.
In Pypy however (which relies on Python 3.5) __origin__ will return typing.Sequence.
Therefore one has two define both types in the list of possible list_types
"""
_list_types = (abc.Sequence, typing.Sequence)

_class_pattern = re.compile(r"<class '([A-Za-z_.]*)\.(?:[A-Za-z_]+)\.(?:[A-Za-z]+)'>")

_named_class_pattern = re.compile(r"(\w+)\s*(?:\((\w+)\))*")


def _create_object(value: str, parameter_type: typing.Any, config: ConfigParser) -> typing.Any:
    value = value.replace("\n", "")
    if parameter_type in _simple_types:
        if parameter_type is bool:
            if value.lower() in _bool_true_strings:
                return True
            if value.lower() in _bool_false_strings:
                return False
            raise TypeError("The given value '{0}' cannot be converted into a boolean. Supported values meaning 'True' "
                            "are {1}, and supported values meaning 'False' are {2} (case insensitive)."
                            .format(value, _bool_true_strings, _bool_false_strings))
        return parameter_type(value)
    try:  # __origin__ and __args__ are defined for Iterable and Sequence annotations
        if parameter_type.__origin__ in _list_types:
            if len(parameter_type.__args__) > 1:
                raise TypeError("Annotate only types with same type inside of sequence/iterable!")
            return_object = []
            for string_value in re.split(r",\s*", value):
                return_object.append(
                    _create_object(string_value, parameter_type.__args__[0], config))
            return return_object
        raise TypeError("Given parameter type {0} is not supported.".format(parameter_type))
    except AttributeError:  # Given type is neither simple nor a sequence -> Custom class
        named_class_pattern_match = _named_class_pattern.match(value)
        if named_class_pattern_match is None or len(named_class_pattern_match.groups()) != 2:
            raise ConfigurationError("Given value '{0}' could not be processed.".format(value))
        class_name = named_class_pattern_match.group(1)
        class_to_build = named_class_pattern_match.group(1)
        if named_class_pattern_match.group(2) is not None:
            class_to_build = named_class_pattern_match.group(2)
        match = _class_pattern.match(str(parameter_type))
        if match is None or len(match.groups()) != 1:
            raise ConfigurationError("Given parameter type '{0}' could not be processed.".format(parameter_type))
        return build_from_config(config, strings.to_camel_case(class_name), match.group(1),
                                 strings.to_camel_case(class_to_build))
