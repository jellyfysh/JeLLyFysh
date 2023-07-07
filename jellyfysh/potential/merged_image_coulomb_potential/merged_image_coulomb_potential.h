/************************************************************************************************************************
 * JeLLFysh - a Python application for all-atom event-chain Monte Carlo - https://github.com/jellyfysh                  *
 * Copyright (C) 2019, 2022 The JeLLyFysh organization                                                                  *
 * (See the AUTHORS.md file for the full list of authors.)                                                              *
 *                                                                                                                      *
 * This file is part of JeLLyFysh.                                                                                      *
 *                                                                                                                      *
 * JeLLyFysh is free software: you can redistribute it and/or modify it under the terms of the GNU General Public       *
 * License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later *
 * version.                                                                                                             *
 *                                                                                                                      *
 * JeLLyFysh is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied      *
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more        *
 * details.                                                                                                             *
 *                                                                                                                      *
 * You should have received a copy of the GNU General Public License along with JeLLyFysh in the LICENSE file.          *
 * If not, see <https://www.gnu.org/licenses/>.                                                                         *
 *                                                                                                                      *
 * If you use JeLLyFysh in published work, please cite the following reference (see [Hoellmer2020] in References.bib):  *
 * Philipp Hoellmer, Liang Qin, Michael F. Faulkner, A. C. Maggs, and Werner Krauth,                                    *
 * JeLLyFysh-Version1.0 -- a Python application for all-atom event-chain Monte Carlo,                                   *
 * Computer Physics Communications, Volume 253, 107168 (2020), https://doi.org/10.1016/j.cpc.2020.107168.               *
 ************************************************************************************************************************/

/** @file merged_image_coulomb_potential.h
 *  @brief Declarations of functions to compute the directional time derivative of the merged image coulomb potential
 *         along a given velocity vector of the active unit.
 *
 *  @author The JeLLyFysh organization.
 *  @bug No known bugs.
 */
#ifndef MERGED_IMAGE_COULOMB_POTENTIAL_H
#define MERGED_IMAGE_COULOMB_POTENTIAL_H

#include <stddef.h> // For size_t.

/** @brief Struct that stores a three-dimensional gradient.
 *
 * Note that using a fixed-size array in this struct (which indirectly allows to return such an array, see
 * https://stackoverflow.com/questions/29088462/why-you-cannot-return-fixed-size-const-array-in-c) led to a bug on
 * some operating systems.
 */
struct Gradient {
    /** The components of the three-dimensional gradient. */
    double gx;
    double gy;
    double gz;
};
struct MergedImageCoulombPotential;

struct MergedImageCoulombPotential *construct_merged_image_coulomb_potential(int fourier_cutoff, int position_cutoff,
                                                                             double alpha, double system_length);
void destroy_merged_image_coulomb_potential(struct MergedImageCoulombPotential *potential);
size_t estimated_size(struct MergedImageCoulombPotential *potential);
struct MergedImageCoulombPotential *copy_merged_image_coulomb_potential(struct MergedImageCoulombPotential *pot);

struct Gradient gradient(struct MergedImageCoulombPotential *potential, double separation[3]);
double derivative(struct MergedImageCoulombPotential *potential, double velocity[3], double separation[3]);

#endif // MERGED_IMAGE_COULOMB_POTENTIAL_H
