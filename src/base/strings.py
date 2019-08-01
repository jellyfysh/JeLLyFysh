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
"""Module for string conversion functions."""
import re


def to_camel_case(snake_case: str) -> str:
    """
    Convert a snake_case string to CamelCase.

    This is the inverse function of to_snake_case.

    Parameters
    ----------
    snake_case : str
        The string in snake_case.

    Returns
    -------
    str
        The string in CamelCase.
    """
    split_string = snake_case.split("_")
    """
    In order to leave words which are already in camel case invariant, one cannot simply use capitalize.
    This function capitalizes the first letter and makes all other letters lowercase.
    ExampleCamelCase would therefore get Examplecamelcase.
    Instead, we use the trick that uppercase letters have a lower ASCII code than lowercase letters.
    min will loop over the original string without _ and the capitalized string and choose the uppercase letter.
    """
    return "".join(map(min, "".join(split_string), "".join(word.capitalize() for word in split_string)))


def to_snake_case(camel_case: str) -> str:
    """
    Convert a CamelCase string to snake_case.

    This is the inverse function of to_camel_case.

    Parameters
    ----------
    camel_case : str
        The string in CamelCase.

    Returns
    -------
    str
        The string in snake_case.
    """
    # (anything) followed by a capitalized (Word) will be replaced by (anything)_(Word)
    # Needed for edge case of acronym followed by another word (example: HTTPResponse -> HTTP_Response
    temporary_string = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", camel_case)
    # separate small letter or number followed by capital letter with underscore and change everything to lowercase
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", temporary_string).lower()


def to_directory_path(package_path: str) -> str:
    """
    Convert a package path containing dots to a directory path containing slashes.

    Parameters
    ----------
    package_path : str
        The package path.

    Returns
    -------
    str
        The directory path.
    """
    return re.sub(r"\.", r"/", package_path)
