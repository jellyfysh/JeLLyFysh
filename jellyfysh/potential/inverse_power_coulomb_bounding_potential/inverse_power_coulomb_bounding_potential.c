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

/** @file inverse_power_coulomb_bounding_potential.c
 *  @brief Definitions of functions to compute the space derivative of the inverse power coulomb bounding potential
 *         along the positive x direction, and to compute the required displacement in space along the positive x
 *         direction where the cumulative event rate equals a sampled potential change.
 *
 *  The inverse power coulomb bounding potential between a target unit j and an active unit i is given by
 *  U_ij = c_i * c_j * k / |r_ij,0|. Here, r_ij,0 = nearest(r_j - r_i) is the minimum separation vector, that is, the
 *  vector between r_i and the closest image of r_j under periodic boundary conditions. The charges of the units are
 *  c_i and c_j, respectively, and k is a prefactor. The functions in this file are explicitly implemented for a cubic
 *  setting with side length L in three dimensions.
 *
 *  @author The JeLLyFysh organization.
 *  @bug No known bugs.
 */
#include "inverse_power_coulomb_bounding_potential.h" // Include declarations.

#include <math.h> // For fabs, floor, fmod, pow, sqrt.


/** @brief Compute the space derivative of the inverse power coulomb bounding potential along the positive x direction
 *         evaluated at the given separation r_ij,0.
 *
 *  @param prefactor_product The product c_i * c_j * k.
 *  @param sx The x component of the separation r_ij,0 where the derivative should be evaluated.
 *  @param sy The y component of the separation r_ij,0 where the derivative should be evaluated.
 *  @param sz The z component of the separation r_ij,0 where the derivative should be evaluated.
 *  @return The space derivative.
 */
double derivative(double prefactor_product, double sx, double sy, double sz) {
    return prefactor_product * sx / pow(sx * sx + sy * sy + sz * sz, 3.0 / 2.0);
}


/** @brief Compute the inverse power coulomb bounding potential evaluated at the given separation r_ij,0.
 *
 *  @param prefactor_product The product c_i * c_j * k.
 *  @param sx The x component of the separation r_ij,0 where the derivative should be evaluated.
 *  @param sy The y component of the separation r_ij,0 where the derivative should be evaluated.
 *  @param sz The z component of the separation r_ij,0 where the derivative should be evaluated.
 *  @return The potential.
 */
double potential(double prefactor_product, double sx, double sy, double sz) {
    return prefactor_product / sqrt(sx * sx + sy * sy + sz * sz);
}


/** @brief Return the required displacement in space of the active unit along the positive direction of motion parallel
 *         to the x-axis where the cumulative event rate of the potential equals the given potential change.
 *
 *  This function first computes the cumulative event rate when the active unit is displaced by the system length L.
 *  The remaining potential change is then considered separately.
 *
 *  @param prefactor_product The product c_i * c_j * k.
 *  @param sx The x component of the current separation r_ij,0.
 *  @param sy The y component of the current separation r_ij,0.
 *  @param sz The z component of the current separation r_ij,0.
 *  @param potential_change The sampled potential change.
 *  @param system_length The system length L of the cubic setting.
 *  @return The displacement.
 */
double displacement(double prefactor_product, double sx, double sy, double sz, double potential_change,
                    double system_length) {
    double system_length_over_two = system_length / 2.0;
    double current_potential = potential(prefactor_product, sx, sy, sz);
    double potential_zero = potential(prefactor_product, 0.0, sy, sz);
    double potential_half_length = potential(prefactor_product, system_length_over_two, sy, sz);
    double potential_change_per_system_length = fabs(potential_zero - potential_half_length);
    double displacement = floor(potential_change / potential_change_per_system_length) * system_length;
    potential_change = fmod(potential_change, potential_change_per_system_length);

    double new_norm;
    if (prefactor_product > 0.0) {
        // Repulsive interaction.
        if (sx <= 0.0) {
            // Active unit is in front of target unit.
            // Travel downhill until interaction with next periodic image.
            displacement += system_length_over_two + sx;
            sx = system_length_over_two;
            current_potential = potential_half_length;
        } else {
            // Active unit is behind target unit.
            if (potential_change >= potential_zero - current_potential) {
                // Active unit can climb potential hill with remaining potential change.
                // Reduce potential_change and travel downhill until interaction with next periodic image.
                potential_change -= (potential_zero - current_potential);
                displacement += sx + system_length_over_two;
                sx = system_length_over_two;
                current_potential = potential_half_length;
            }
        }
        // Compute how much active unit can travel uphill with the given potential change.
        new_norm = prefactor_product / (current_potential + potential_change);
        displacement += (sx - sqrt(new_norm * new_norm - (sy * sy + sz * sz)));
    } else {
        // Attractive interaction
        if (sx > 0.0) {
            // Active unit is behind target unit.
            // Travel downhill until sx vanishes.
            displacement += sx;
            sx = 0.0;
            current_potential = potential_zero;
        } else {
            // Active unit is in front of target unit.
            if (potential_change >= potential_half_length - current_potential) {
                potential_change -= (potential_half_length - current_potential);
                displacement += sx + system_length;
                sx = 0.0;
                current_potential = potential_zero;
            }
        }
        new_norm = prefactor_product / (current_potential + potential_change);
        displacement += (sx + sqrt(new_norm * new_norm - (sy * sy + sz * sz)));
    }
    return displacement;
}
