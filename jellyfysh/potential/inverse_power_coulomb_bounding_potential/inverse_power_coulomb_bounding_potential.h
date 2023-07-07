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

/** @file inverse_power_coulomb_bounding_potential.h
 *  @brief Declarations of functions to compute the directional time derivative of the inverse power coulomb bounding
 *         potential along a given velocity vector of the active unit, and to compute the required time displacement
 *         along its velocity where the cumulative event rate equals a sampled potential change.
 *
 *  @author The JeLLyFysh organization.
 *  @bug No known bugs.
 */
#ifndef INVERSE_POWER_COULOMB_BOUNDING_POTENTIAL_H
#define INVERSE_POWER_COULOMB_BOUNDING_POTENTIAL_H

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

double derivative(double prefactor_product, double velocity[3], double separation[3]);
struct Gradient gradient(double prefactor_product, double separation[3]);
double displacement(double prefactor_product, double velocity[3], double separation[3], double potential_change,
                    double system_length);

#endif // INVERSE_POWER_COULOMB_BOUNDING_POTENTIAL_H
