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
Module which sets up the build of the merged_image_coulomb_potential.c C extension using cffi.

The recommended way to compile the C extension merged_image_coulomb_potential.c into a shared library that can
be used by cffi in the MergedImageCoulombPotential class is to run 'pypy3 setup.py build_ext -i' in the root
directory of the JeLLyFysh repository. (Of course, 'pypy3' can be replaced with the Python interpreter of your choice).

Alternatively, this script can be executed from the root directory of the JeLLyFysh repository.
"""
from cffi import FFI
ffi_builder = FFI()

# Basically duplicates the information in merged_image_coulomb_potential.h but is required by cffi.
# See https://cffi.readthedocs.io/en/latest/overview.html#if-you-don-t-have-an-already-installed-c-library-to-call.
ffi_builder.cdef(r"""
struct MergedImageCoulombPotential;
struct Gradient {
    double gx;
    double gy;
    double gz;
};
struct MergedImageCoulombPotential *construct_merged_image_coulomb_potential(int fourier_cutoff, int position_cutoff,
                                                                             double alpha, double system_length);
void destroy_merged_image_coulomb_potential(struct MergedImageCoulombPotential *potential);
size_t estimated_size(struct MergedImageCoulombPotential *potential);
struct MergedImageCoulombPotential *copy_merged_image_coulomb_potential(struct MergedImageCoulombPotential *pot);
struct Gradient gradient(struct MergedImageCoulombPotential *potential, double separation[3]);
double derivative(struct MergedImageCoulombPotential *potential, double velocity[3], double separation[3]);
""")

# First argument is name of the output C extension that is used in merged_image_coulomb_potential.py.
# All paths are relative to the root directory of the JeLLyFysh application.
# The 'm' library is the math library on Unix.
ffi_builder.set_source(
    "jellyfysh.potential.merged_image_coulomb_potential._merged_image_coulomb_potential",
    """
    #include "merged_image_coulomb_potential.h"
    """,
    sources=["jellyfysh/potential/merged_image_coulomb_potential/merged_image_coulomb_potential.c"],
    libraries=['m'], include_dirs=["jellyfysh/potential/merged_image_coulomb_potential"])

if __name__ == "__main__":
    ffi_builder.compile(verbose=True)
