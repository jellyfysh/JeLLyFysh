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
 *  @brief Definitions of functions to compute the directional time derivative of the inverse power coulomb bounding
 *         potential along a given velocity vector of the active unit, and to compute the required time displacement
 *         along its velocity where the cumulative event rate equals a sampled potential change.
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

#include <math.h> // For fabs, INFINITY, pow, sqrt.
#include <stdbool.h> // For true, false.


/** @brief Return the directional time derivative along a given velocity vector of the active unit for the given
 *         separation r_ij,0.
 *
 *  @param prefactor_product The product c_i * c_j * k.
 *  @param velocity The velocity of the active unit.
 *  @param separation The current separation r_ij,0.
 *  @return The directional time derivative.
 */
double derivative(double prefactor_product, double velocity[3], double separation[3]) {
    double constant_factor = prefactor_product / pow(separation[0] * separation[0] + separation[1] * separation[1]
                                                     + separation[2] * separation[2], 3.0 / 2.0);
    return (velocity[0] * separation[0] + velocity[1] * separation[1] + velocity[2] * separation[2]) * constant_factor;
}


/** @brief  Return the gradient of the inverse power coulomb bounding potential evaluated at the given separation.
 *
 *  @param prefactor_product The product c_i * c_j * k.
 *  @param separation The separation r_ij,0.
 *  @return The gradient with respect to the position r_i of the active unit.
 */
struct Gradient gradient(double prefactor_product, double separation[3]) {
    double constant_factor = prefactor_product / pow(separation[0] * separation[0] + separation[1] * separation[1]
                                                     + separation[2] * separation[2], 3.0 / 2.0);
    return (struct Gradient) {constant_factor * separation[0], constant_factor * separation[1],
                              constant_factor * separation[2]};
}


/** @brief Compute the inverse power coulomb bounding potential evaluated at the given separation r_ij,0.
 *
 *  @param prefactor_product The product c_i * c_j * k.
 *  @param separation The current separation r_ij,0.
 *  @return The potential.
 */
double potential(double prefactor_product, double separation[3]) {
    return prefactor_product /
           sqrt(separation[0] * separation[0] + separation[1] * separation[1] + separation[2] * separation[2]);
}


/** @brief Compute the inverse power coulomb bounding potential evaluated at the given separation norm |r_ij,0|.
 *
 *  @param prefactor_product The product c_i * c_j * k.
 *  @param separation_norm The current separation norm |r_ij,0|.
 *  @return The potential.
 */
double potential_from_norm(double prefactor_product, double separation_norm) {
    return prefactor_product / separation_norm;
}


/** @brief Return the required time displacement of the active unit along its velocity where the cumulative event rate
 *         of the potential equals the given potential change.
 *
 *  @param prefactor_product The product c_i * c_j * k.
 *  @param velocity The velocity of the active unit.
 *  @param separation The current separation r_ij,0.
 *  @param potential_change The sampled potential change.
 *  @param system_length The system length L of the cubic setting.
 *  @return The required time displacement.
 */
double displacement(double prefactor_product, double velocity[3], double separation[3], double potential_change,
                    double system_length) {
    double system_length_over_two = system_length / 2.0;
    double velocity_squared = velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2];
    double separation_squared;
    double separation_dot_velocity;
    bool separation_dot_velocity_positive;
    double current_potential;
    double sqrt_term;
    double displacement_until_next_image;
    double min_displacement_until_next_image;
    int index_next_image;
    double max_displacement;
    double max_potential;
    double new_norm;
    double total_displacement = 0.0;

    if (prefactor_product > 0.0) {
        // Repulsive interaction.
        double min_separation[3];
        // These two variables have to be updated in the loop anytime the separation is changed.
        // This allows to set separation_dot_velocity_positive to true or false without an actual comparison which
        // avoids floating point issues.
        separation_dot_velocity = separation[0] * velocity[0] + separation[1] * velocity[1]
                                  + separation[2] * velocity[2];
        separation_dot_velocity_positive = separation_dot_velocity > 0.0 ? true : false;
        while (1) {
            min_displacement_until_next_image = velocity[0] != 0.0 ?
                                                separation[0] / velocity[0] + system_length_over_two / fabs(velocity[0])
                                                : INFINITY;
            index_next_image = 0;
            displacement_until_next_image = velocity[1] != 0.0 ?
                                            separation[1] / velocity[1] + system_length_over_two / fabs(velocity[1])
                                            : INFINITY;
            if (displacement_until_next_image < min_displacement_until_next_image) {
                min_displacement_until_next_image = displacement_until_next_image;
                index_next_image = 1;
            }
            displacement_until_next_image = velocity[2] != 0.0 ?
                                            separation[2] / velocity[2] + system_length_over_two / fabs(velocity[2])
                                            : INFINITY;
            if (displacement_until_next_image < min_displacement_until_next_image) {
                min_displacement_until_next_image = displacement_until_next_image;
                index_next_image = 2;
            }

            if (separation_dot_velocity_positive) {
                // Norm of separation becomes smaller.
                separation_squared = separation[0] * separation[0] + separation[1] * separation[1]
                                     + separation[2] * separation[2];
                max_displacement = separation_dot_velocity / velocity_squared;
                if (max_displacement <= min_displacement_until_next_image) {
                    min_separation[0] = separation[0] - max_displacement * velocity[0];
                    min_separation[1] = separation[1] - max_displacement * velocity[1];
                    min_separation[2] = separation[2] - max_displacement * velocity[2];
                    current_potential = potential_from_norm(prefactor_product, sqrt(separation_squared));
                    max_potential = potential(prefactor_product, min_separation);
                    if (potential_change >= max_potential - current_potential) {
                        // Active unit can climb potential hill with remaining potential change.
                        // Reduce potential_change and set separation to minimum separation.
                        potential_change -= (max_potential - current_potential);
                        total_displacement += max_displacement;
                        separation[0] = min_separation[0];
                        separation[1] = min_separation[1];
                        separation[2] = min_separation[2];
                        separation_dot_velocity = separation[0] * velocity[0] + separation[1] * velocity[1]
                                                  + separation[2] * velocity[2];
                        // Force that separation_dot_velocity is now negative to avoid floating point issues.
                        separation_dot_velocity_positive = false;
                    } else {
                        new_norm = prefactor_product / (current_potential + potential_change);
                        sqrt_term = separation_dot_velocity * separation_dot_velocity
                                    - velocity_squared * (separation_squared - new_norm * new_norm);
                        total_displacement += (separation_dot_velocity - sqrt(sqrt_term)) / velocity_squared;
                        return total_displacement;
                    }
                } else {
                    min_separation[0] = separation[0] - min_displacement_until_next_image * velocity[0];
                    min_separation[1] = separation[1] - min_displacement_until_next_image * velocity[1];
                    min_separation[2] = separation[2] - min_displacement_until_next_image * velocity[2];
                    current_potential = potential_from_norm(prefactor_product, sqrt(separation_squared));
                    max_potential = potential(prefactor_product, min_separation);
                    if (potential_change >= max_potential - current_potential) {
                        // Active unit can climb potential hill with remaining potential change.
                        // Reduce potential_change and set separation to minimum separation.
                        potential_change -= (max_potential - current_potential);
                        total_displacement += min_displacement_until_next_image;
                        separation[0] = min_separation[0];
                        separation[1] = min_separation[1];
                        separation[2] = min_separation[2];
                        separation[index_next_image] *= -1.0;
                        separation_dot_velocity = separation[0] * velocity[0] + separation[1] * velocity[1]
                                                  + separation[2] * velocity[2];
                        separation_dot_velocity_positive = separation_dot_velocity > 0.0 ? true : false;
                    } else {
                        new_norm = prefactor_product / (current_potential + potential_change);
                        sqrt_term = separation_dot_velocity * separation_dot_velocity
                                    - velocity_squared * (separation_squared - new_norm * new_norm);
                        total_displacement += (separation_dot_velocity - sqrt(sqrt_term)) / velocity_squared;
                        return total_displacement;
                    }
                }
            } else {
                // Norm of separation becomes bigger.
                // Travel down potential hill until interaction with next image.
                total_displacement += min_displacement_until_next_image;
                separation[0] -= min_displacement_until_next_image * velocity[0];
                separation[1] -= min_displacement_until_next_image * velocity[1];
                separation[2] -= min_displacement_until_next_image * velocity[2];
                separation[index_next_image] *= -1.0;
                separation_dot_velocity = separation[0] * velocity[0] + separation[1] * velocity[1]
                                          + separation[2] * velocity[2];
                separation_dot_velocity_positive = separation_dot_velocity > 0.0 ? true : false;
            }
        }
    } else {
        // Attractive interaction.
        double max_separation[3];
        // These two variables have to be updated in the loop anytime the separation is changed.
        // This allows to set separation_dot_velocity_positive to true or false without an actual comparison which
        // avoids floating point issues.
        separation_dot_velocity = separation[0] * velocity[0] + separation[1] * velocity[1]
                                  + separation[2] * velocity[2];
        separation_dot_velocity_positive = separation_dot_velocity > 0.0 ? true : false;
        while (1) {
            min_displacement_until_next_image = velocity[0] != 0.0 ?
                                                separation[0] / velocity[0] + system_length_over_two / fabs(velocity[0])
                                                : INFINITY;
            index_next_image = 0;
            displacement_until_next_image = velocity[1] != 0.0 ?
                                            separation[1] / velocity[1] + system_length_over_two / fabs(velocity[1])
                                            : INFINITY;
            if (displacement_until_next_image < min_displacement_until_next_image) {
                min_displacement_until_next_image = displacement_until_next_image;
                index_next_image = 1;
            }
            displacement_until_next_image = velocity[2] != 0.0 ?
                                            separation[2] / velocity[2] + system_length_over_two / fabs(velocity[2])
                                            : INFINITY;
            if (displacement_until_next_image < min_displacement_until_next_image) {
                min_displacement_until_next_image = displacement_until_next_image;
                index_next_image = 2;
            }
            separation_squared = separation[0] * separation[0] + separation[1] * separation[1]
                                 + separation[2] * separation[2];
            if (separation_dot_velocity_positive) {
                // Norm of separation becomes smaller.
                max_displacement = separation_dot_velocity / velocity_squared;
                if (max_displacement <= min_displacement_until_next_image) {
                    total_displacement += max_displacement;
                    separation[0] -= max_displacement * velocity[0];
                    separation[1] -= max_displacement * velocity[1];
                    separation[2] -= max_displacement * velocity[2];
                    separation_dot_velocity = separation[0] * velocity[0] + separation[1] * velocity[1]
                                              + separation[2] * velocity[2];
                    // Force that separation_dot_velocity is now negative to avoid floating point issues.
                    separation_dot_velocity_positive = false;
                } else {
                    total_displacement += min_displacement_until_next_image;
                    separation[0] -= min_displacement_until_next_image * velocity[0];
                    separation[1] -= min_displacement_until_next_image * velocity[1];
                    separation[2] -= min_displacement_until_next_image * velocity[2];
                    separation[index_next_image] *= -1.0;
                    separation_dot_velocity = separation[0] * velocity[0] + separation[1] * velocity[1]
                                              + separation[2] * velocity[2];
                    separation_dot_velocity_positive = separation_dot_velocity > 0.0 ? true : false;
                }
            } else {
                // Norm of separation becomes bigger.
                max_separation[0] = separation[0] - min_displacement_until_next_image * velocity[0];
                max_separation[1] = separation[1] - min_displacement_until_next_image * velocity[1];
                max_separation[2] = separation[2] - min_displacement_until_next_image * velocity[2];
                current_potential = potential_from_norm(prefactor_product, sqrt(separation_squared));
                max_potential = potential(prefactor_product, max_separation);
                if (potential_change >= max_potential - current_potential) {
                    // Active unit can climb potential hill with remaining potential change.
                    // Reduce potential change and set separation to maximum separation.
                    potential_change -= (max_potential - current_potential);
                    total_displacement += min_displacement_until_next_image;
                    separation[0] = max_separation[0];
                    separation[1] = max_separation[1];
                    separation[2] = max_separation[2];
                    separation[index_next_image] *= -1.0;
                    separation_dot_velocity = separation[0] * velocity[0] + separation[1] * velocity[1]
                                              + separation[2] * velocity[2];
                    separation_dot_velocity_positive = separation_dot_velocity > 0.0 ? true : false;
                } else {
                    new_norm = prefactor_product / (current_potential + potential_change);
                    sqrt_term = separation_dot_velocity * separation_dot_velocity
                                - velocity_squared * (separation_squared - new_norm * new_norm);
                    total_displacement += (separation_dot_velocity + sqrt(sqrt_term)) / velocity_squared;
                    return total_displacement;
                }
            }
        }
    }
}
