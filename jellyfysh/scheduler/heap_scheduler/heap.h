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

/** @file heap.h
 *  @brief Declarations of functions that implement a binary min heap.
 *
 *  @author The JeLLyFysh organization.
 *  @bug No known bugs.
 */
#ifndef HEAP_H
#define HEAP_H

#include <stddef.h> // For size_t.

typedef unsigned int uint;

/** @brief Struct that stores all variables of an entry in the binary min heap.
 *
 *  Note that such an object is returned by the 'root' and 'entry' functions of this C extension. In order to achieve
 *  that Python can read the variables of the struct after the 'root' function was called via cffi, the struct HeapEntry
 *  has to be defined here in the header.
 *
 *  In order to avoid loss of precision during long runs of JF, candidate event times are not stored as simple floats
 *  but as the quotient and remainder of an integer division of the candidate event time with 1 (see base.time.Time
 *  class for more information). Therefore, these two doubles appear in this struct and are used to compare entries in
 *  the heap.
 */
struct HeapEntry {
    /** The quotient of an integer division of the candidate event time with 1. */
    double time_quotient;
    /** The remainder of an integer division of the candidate event time with 1. */
    double time_remainder;
    /** The associated event handler. */
    void *event_handler;
    /** The counter value of this event (see heap.c for more information on lazy deletion). */
    uint counter;
};
struct Heap;

struct Heap *construct_heap();
void destroy_heap(struct Heap *heap);
size_t estimated_size(struct Heap *heap);
size_t insert(struct Heap *heap, double time_quotient, double time_remainder, void *event_handler, uint counter);
struct HeapEntry root(struct Heap *heap, void *scheduler, int (*delete_callback)(void *, void *, uint));
void delete_events(struct Heap *heap, void *event_handler);
struct HeapEntry entry(struct Heap *heap, uint index);

#endif // HEAP_H
