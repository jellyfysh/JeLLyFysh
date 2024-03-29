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

/** @file merged_image_coulomb_potential.c
 *  @brief Definitions of functions to compute the space derivative of the merged image coulomb potential along the
 *         positive x direction.
 *
 *  This file contains the functions to create, copy, and destroy a struct containing all parameters to compute the
 *  space derivative of the merged image coulomb potential along the positive x direction in the derivative function.
 *
 *  @author The JeLLyFysh organization.
 *  @bug No known bugs.
 */
#include "merged_image_coulomb_potential.h" // Include declarations.

#include <math.h> // For M_PI, cos, erfc, exp, sqrt, sin.
#include <stdlib.h> // For free, malloc, size_t.
#include <string.h> // For memcpy.


/** @brief Struct that stores all necessary parameters to compute the derivative of the merged image coulomb potential. */
struct MergedImageCoulombPotential {
    /** The cutoff in Fourier space of the Ewald summation. */
    const int fourier_cutoff;
    /** The square of the cutoff in Fourier space of the Ewald summation. */
    const int fourier_cutoff_sq;
    /** The cutoff in position space of the Ewald summation. */
    const int position_cutoff;
    /** The square of the cutoff in position space of the Ewald summation. */
    const int position_cutoff_sq;
    /** The convergence factor alpha of the Ewald summation divided by the system length. */
    const double alpha_over_length;
    /** The square of the convergence factor alpha of the Ewald summation divided by the system length. */
    const double alpha_over_length_sq;
    /** Two times the convergence factor alpha of the Ewald summation divided by the square root of pi. */
    const double two_alpha_over_length_root_pi;
    /** The system length. */
    const double system_length;
    /** Two times pi divided by the system length. */
    const double two_pi_over_length;
    /** Pointer to precomputed factors that speed up the computation of the Ewald summation in Fourier space. */
    double *** const fourier_array;
};


/** @brief Create a MergedImageCoulombPotential struct on the heap.
 *
 *  See merged_image_coulomb_potential.py for a more detailed explanation of the parameters.
 *
 *  @param fourier_cutoff The cutoff in Fourier space of the Ewald summation.
 *  @param position_cutoff The cutoff in position space of the Ewald summation.
 *  @param alpha The convergence factor alpha of the Ewald summation.
 *  @param system_length The system length.
 *  @return The pointer to the MergedImageCoulombPotential struct, or NULL if the necessary memory allocation failed.
 */
struct MergedImageCoulombPotential *construct_merged_image_coulomb_potential(int fourier_cutoff, int position_cutoff,
                                                                             double alpha, double system_length) {
    int i, j, k;
    double coefficient, norm_sq;
    double ***fourier_array;

    fourier_array = malloc((fourier_cutoff + 1) * sizeof(double**));
    if (fourier_array == NULL) return NULL;
    for (i = 0; i < fourier_cutoff + 1; i++) {
        fourier_array[i] = malloc((fourier_cutoff + 1) * sizeof(double *));
        if (fourier_array[i] == NULL) return NULL;
        for (j = 0; j < fourier_cutoff + 1; j++) {
            fourier_array[i][j] = malloc((fourier_cutoff + 1) * sizeof(double));
            if (fourier_array[i][j] == NULL) return NULL;
        }
    }

    for (k = 0; k < fourier_cutoff + 1; k++) {
        for (j = 0; j < fourier_cutoff + 1; j++) {
            for (i = 1; i < fourier_cutoff + 1; i++) {
                if (j == 0 && k == 0) {
                    coefficient = 1.0;
                } else if (k == 0 || j == 0) {
                    coefficient = 2.0;
                } else {
                    coefficient = 4.0;
                }
                norm_sq = i * i + j * j + k * k;
                fourier_array[i][j][k] = 4.0 * i * coefficient / (norm_sq * system_length * system_length)
                                         * exp(- M_PI * M_PI * norm_sq / (alpha * alpha));
            }
        }
    }
    struct MergedImageCoulombPotential *potential = malloc(sizeof(struct MergedImageCoulombPotential));
    if (potential == NULL) return NULL;
    struct MergedImageCoulombPotential init = {fourier_cutoff, fourier_cutoff * fourier_cutoff, position_cutoff,
                                               position_cutoff * position_cutoff, alpha / system_length,
                                               alpha * alpha / (system_length * system_length),
                                               2.0 * alpha / (system_length * sqrt(M_PI)), system_length,
                                               2.0 * M_PI/system_length, fourier_array};
    memcpy(potential, &init, sizeof(*potential));
    return potential;
}


/** @brief Return the estimated size in bytes of a MergedImageCoulombPotential struct on the heap.
 *
 *  @param potential The pointer to the MergedImageCoulombPotential on the heap.
 *  @return The estimated size.
 */
size_t estimated_size(struct MergedImageCoulombPotential *potential) {
    return sizeof(struct MergedImageCoulombPotential)
           + (potential->fourier_cutoff + 1) * (sizeof(double**) + sizeof(double*) + sizeof(double));
}


/** @brief Deallocate the memory of a MergedImageCoulombPotential struct on the heap.
 *
 *  @param potential The pointer to the MergedImageCoulombPotential on the heap.
 *  @return Void.
 */
void destroy_merged_image_coulomb_potential(struct MergedImageCoulombPotential *potential) {
    if (potential) {
        if (potential->fourier_array) {
            int i, j;
            for (i = 0; i < potential->fourier_cutoff + 1; i++) {
                if (potential->fourier_array[i]) {
                    for (j = 0; j < potential->fourier_cutoff + 1; j++) {
                        if (potential->fourier_array[i][j]) {
                            free(potential->fourier_array[i][j]);
                        }
                    }
                    free(potential->fourier_array[i]);
                }
            }
            free(potential->fourier_array);
        }
        free(potential);
    }
}


/** @brief Copy a MergedImageCoulombPotential struct on the heap.
 *
 *  @param potential The pointer to the MergedImageCoulombPotential on the heap that should be copied.
 *  @return The pointer to the copied MergedImageCoulombPotential on the heap, or NULL if memory allocation failed.
 */
struct MergedImageCoulombPotential *copy_merged_image_coulomb_potential(struct MergedImageCoulombPotential *potential) {
    int i, j, k;
    double ***copied_fourier_array;
    copied_fourier_array = malloc((potential->fourier_cutoff + 1) * sizeof(double**));
    if (copied_fourier_array == NULL) return NULL;
    for (i = 0; i < potential->fourier_cutoff + 1; i++) {
        copied_fourier_array[i] = malloc((potential->fourier_cutoff + 1) * sizeof(double *));
        if (copied_fourier_array[i] == NULL) return NULL;
        for (j = 0; j < potential->fourier_cutoff + 1; j++) {
            copied_fourier_array[i][j] = malloc((potential->fourier_cutoff + 1) * sizeof(double));
            if (copied_fourier_array[i][j] == NULL) return NULL;
        }
    }
    for (k = 0; k < potential->fourier_cutoff + 1; k++) {
        for (j = 0; j < potential->fourier_cutoff + 1; j++) {
            for (i = 1; i < potential->fourier_cutoff + 1; i++) {
                copied_fourier_array[i][j][k] = potential->fourier_array[i][j][k];
            }
        }
    }
    struct MergedImageCoulombPotential *copied_potential = malloc(sizeof(struct MergedImageCoulombPotential));
    if (copied_potential == NULL) return NULL;
    struct MergedImageCoulombPotential init = {potential->fourier_cutoff, potential->fourier_cutoff_sq,
                                               potential->position_cutoff, potential->position_cutoff_sq,
                                               potential->alpha_over_length, potential->alpha_over_length_sq,
                                               potential->two_alpha_over_length_root_pi, potential->system_length,
                                               potential->two_pi_over_length, copied_fourier_array};
    memcpy(copied_potential, &init, sizeof(*potential));
    return copied_potential;
}


/** @brief Compute the space derivative of the merged image coulomb potential along the positive x direction evaluated
 *         at the given separation.
 *
 *  @param potential The pointer to the MergedImageCoulombPotential on the heap whose parameters should be used.
 *  @param sx The x component of the separation where the derivative should be evaluated.
 *  @param sy The y component of the separation where the derivative should be evaluated.
 *  @param sz The z component of the separation where the derivative should be evaluated.
 *  @return The space derivative.
 */
double derivative(struct MergedImageCoulombPotential *potential, double sx, double sy, double sz) {
    double derivative = 0.0;

    // First compute the part of the Ewald sum of the derivative in position space along the positive x direction.
    double vector_norm, vector_sq, vector_x, vector_y_sq, vector_z_sq;
    int cutoff_x, cutoff_y;
    int i, j, k;
    for (k = -potential->position_cutoff; k < potential->position_cutoff + 1; k++) {
        vector_z_sq = (sz + k * potential->system_length) * (sz + k * potential->system_length);
        cutoff_y = (int) sqrt(potential->position_cutoff_sq - k * k);
        for (j = -cutoff_y; j < cutoff_y + 1; j++) {
            vector_y_sq = (sy + j * potential->system_length) * (sy + j * potential->system_length);
            cutoff_x = (int) sqrt(potential->position_cutoff_sq - j * j - k * k);
            for (i = -cutoff_x; i < cutoff_x + 1; i++) {
                vector_x = sx + i * potential->system_length;
                vector_sq = vector_x * vector_x + vector_y_sq + vector_z_sq;
                vector_norm = sqrt(vector_sq);
                derivative += vector_x * (potential->two_alpha_over_length_root_pi
                                          * exp(-potential->alpha_over_length_sq * vector_sq)
                                          + erfc(potential->alpha_over_length * vector_norm) / vector_norm) / vector_sq;
            }
        }
    }

    // Then compute the part of the Ewald sum of the derivative in Fourier space along the positive x direction.
    double delta_cos_x = cos(potential->two_pi_over_length * sx);
    double delta_sin_x = sin(potential->two_pi_over_length * sx);
    double delta_cos_y = cos(potential->two_pi_over_length * sy);
    double delta_sin_y = sin(potential->two_pi_over_length * sy);
    double delta_cos_z = cos(potential->two_pi_over_length * sz);
    double delta_sin_z = sin(potential->two_pi_over_length * sz);
    double cos_x = delta_cos_x;
    double sin_x = delta_sin_x;
    double cos_y = 1.0;
    double sin_y = 0.0;
    double cos_z = 1.0;
    double sin_z = 0.0;
    double store_cos_value;

    for (i = 1; i < potential->fourier_cutoff + 1; i++) {
        cutoff_y = (int) sqrt(potential->fourier_cutoff_sq - i * i);
        for (j = 0; j < cutoff_y + 1; j++) {
            cutoff_x = (int) sqrt(potential->fourier_cutoff_sq - i * i - j * j);
            for (k = 0; k < cutoff_x + 1; k++) {
                derivative += potential->fourier_array[i][j][k] * sin_x * cos_y * cos_z;

                if (k != cutoff_x) {
                    store_cos_value = cos_z;
                    cos_z = store_cos_value * delta_cos_z - sin_z * delta_sin_z;
                    sin_z = sin_z * delta_cos_z + store_cos_value * delta_sin_z;
                } else if (j != cutoff_y) {
                    store_cos_value = cos_y;
                    cos_y = store_cos_value * delta_cos_y - sin_y * delta_sin_y;
                    sin_y = sin_y * delta_cos_y + store_cos_value * delta_sin_y;
                    cos_z = 1.0;
                    sin_z = 0.0;
                } else if (i != potential->fourier_cutoff) {
                    store_cos_value = cos_x;
                    cos_x = store_cos_value * delta_cos_x - sin_x * delta_sin_x;
                    sin_x = sin_x * delta_cos_x + store_cos_value * delta_sin_x;
                    cos_y = 1.0;
                    sin_y = 0.0;
                    cos_z = 1.0;
                    sin_z = 0.0;
                }
            }
        }
    }
    return derivative;
}
