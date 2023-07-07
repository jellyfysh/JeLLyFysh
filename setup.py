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
"""
Setup script for the JeLLyFysh application using setuptools.

This script allows to install the JeLLyFysh application using setuptools. After you have downloaded and uncompressed the
JeLLyFysh source, or pulled it from git, first choose a Python interpreter that supports version 3.6 or greater. We
recommend using the Python 3.6 compatible PyPy3.6 v7.3.0 (see https://www.pypy.org/download.html), which yields a
significant higher performance of the JeLLyFysh application. We further recommend installing the JeLLyFysh application
into a virtual environment (e.g., using virtualenv, venv, pyenv, ...) This documentation uses pypy3 to call the
interpreter of the virtual environment. If you use a different interpreter, change the code accordingly.

To install the JeLLyFysh application, first ensure that pip is installed. This can be checked by running:

    pypy3 -m pip --help

If this command does not show pip's help message, install it (see, e.g., https://pip.pypa.io/en/stable/installing/.).
Moreover, the C extensions of JeLLyFysh require a C compiler (e.g., gcc or clang).

Next, ensure that pip, setuptools, and wheel are installed and on the latest version by running:

    pypy3 -m pip install --upgrade pip setuptools wheel

Afterwards, you are ready to install the JeLLyFysh application. To do so, navigate into the directory of this script
and execute

    pypy3 -m pip install .

This will first install the dependencies of the JeLLyFysh application (see EXTERNAL_DEPENDENCIES.md). Afterwards, the
jellyfysh package is exported to the site-packages directory of your interpreter. Finally, it creates the following
executables:

    jellyfysh config_file: Run the JeLLyFysh application based on a configuration file (see jellyfysh.run.main).
    jellyfysh-resume: Resume a run of the JeLLyFysh application based on a dumped data file (see jellyfysh.resume.main).
    jellyfysh-examples: Copy the exemplary configuration files of the JeLLyFysh application into the
                        current working directory (see jellyfysh.create_examples.main).

All executables provide the --help command line option. Note that you might have to add the directory where pip installs
executables to your PATH environment variable in order to access these executables.

To remove the JeLLyFysh application, just use

    pypy3 -m pip uninstall jellyfysh

Using 'pypy3 -m pip install .' basically copies the source code into the site-packages directory. Therefore you would
need to reinstall the JeLLyFysh application anytime source code was changed. Fortunately, setuptools provides an
editable mode. This installs the JeLLyFysh application without copying any files. Any changes in source code are
directly active without re-installation. To install the JeLLyFysh application in editable mode, simply use

    pypy3 -m pip install -e .

Of course, any C extensions must still be compiled after any changes to C code. This can be achieved using

    pypy3 setup.py build_ext -i

Removing the JeLLyFysh application installed in editable mode works as described above.
"""
from os import path
import sys
try:
    from setuptools import find_packages, setup
except ModuleNotFoundError:
    print("This script requires that setuptools is installed.", file=sys.stderr)
    sys.exit(1)
try:
    import wheel
except ModuleNotFoundError:
    print("This script requires that wheel is installed.", file=sys.stderr)
    sys.exit(1)

if sys.version_info < (3, 6, 0):
    print("The JeLLyFysh application requires Python version 3.6 or greater.", file=sys.stderr)
    sys.exit(1)

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md")) as file:
    long_description = file.read()

version = {}
with open(path.join(this_directory, "jellyfysh/version.py")) as file:
    exec(file.read(), version)

setup(
    name="JeLLyFysh",
    version=version["__version__"],
    author="The JeLLyFysh organization",
    author_email="werner.krauth@ens.fr",
    description="JeLLFysh - a Python application for all-atom event-chain Monte Carlo",
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords="Monte Carlo algorithm, Irreversible Markov chain, N-body simulation, Event-chain algorithm, "
             "Long-range potentials",
    url="https://github.com/jellyfysh",
    license="GNU General Public License v3 (GPLv3)",
    # See https://pypi.org/classifiers/ for available classifiers
    classifiers=["Development Status :: 4 - Beta",

                 "Environment :: Console",

                 "Intended Audience :: Science/Research",
                 "Topic :: Scientific/Engineering :: Physics",
                 "Topic :: Scientific/Engineering :: Bio-Informatics",
                 "Topic :: Scientific/Engineering :: Chemistry",

                 "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                 "Natural Language :: English",

                 "Programming Language :: Python :: Implementation :: CPython",
                 "Programming Language :: Python :: Implementation :: PyPy",
                 "Programming Language :: Python :: 3.6",
                 "Programming Language :: Python :: 3.7",
                 "Programming Language :: Python :: 3.8",
                 "Programming Language :: Python :: 3.9",
                 "Programming Language :: Python :: 3.10",
                 ],
    python_requires=">=3.6",

    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "jellyfysh = jellyfysh.run:main",
            "jellyfysh-resume = jellyfysh.resume:main",
            "jellyfysh-examples = jellyfysh.create_examples: main",
        ]
    },
    # Adding setuptools_scm to setup_requires achieves that all files tracked by git are automatically added
    include_package_data=True,
    setup_requires=["wheel", "setuptools_scm", "cffi>=1.13.2"],
    install_requires=["cffi>=1.13.2", "cloudpickle>=2.2.1"],
    cffi_modules=[
        "jellyfysh/potential/merged_image_coulomb_potential/merged_image_coulomb_potential_build.py:ffi_builder",
        "jellyfysh/potential/inverse_power_coulomb_bounding_potential/"
        + "inverse_power_coulomb_bounding_potential_build.py:ffi_builder",
        "jellyfysh/scheduler/heap_scheduler/heap_build.py:ffi_builder"
    ],

    extras_require={
        "pdb": ["MDAnalysis>=0.20.1"],
        "plot": ["matplotlib==3.1.0"]
    },
    zip_safe=False
)
