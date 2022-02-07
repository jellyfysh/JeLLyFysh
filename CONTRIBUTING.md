# Contributing to JeLLyFysh

## Pull requests

We would be happy to receive your fixes, extensions for the JeLLyFysh application or solutions to open JeLLyFysh issues, 
and are looking forward to your pull requests! After a successful review of your code, we will merge your code into the 
master branch and add your name to the [AUTHORS.md](AUTHORS.md) file. In addition, your name will appear in the commit 
history of the repository.

For successful pull requests make sure you follow the following points:

- Follow the coding style described below.
- Test your code.
- Run the unittests as described in the [README.md](README.md) file.
- Read the [code of conduct](CODE_OF_CONDUCT.md).
- Include the following license notice at the top of any new file:

```Python3
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
```

## Coding style

In this project, we follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code except for 
the line length limit, which is set to 120 characters. Indent your code with 4 spaces per level. We also use the 
[PEP257](https://www.python.org/dev/peps/pep-0257/) conventions for docstrings. We use the 
[NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html) for the docstrings.

The code must be compatible with any Python version >= 3.6 and should be runnable with PyPy version >= 7.

It goes without saying that we will be happy to assist contributors during the course of their first few pull requests :).
