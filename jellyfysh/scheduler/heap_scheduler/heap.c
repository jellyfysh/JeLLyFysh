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

/** @file heap.c
 *  @brief Definitions of functions that implement a binary min heap.
 *
 *  This file contains the functions to create, copy, and destroy a struct containing a binary min heap. Further
 *  functions insert a new entry into the heap, obtain the root entry, and destroy unnecessary entries. Here, an entry
 *  consists of a candidate event time, an event handler, and a counter. The candidate event time is used in the
 *  comparisons when an entry is inserted into the heap. In order to avoid loss of precision during long runs of JF,
 *  candidate event times are not stored as simple floats but as the quotient and remainder of an integer division of
 *  the candidate event time with 1 (see base.time.Time class for more information). Therefore, two doubles are used to
 *  transmit the candidate event time when a new event should be inserted into the heap.
 *
 *  The event handler in the entry of the heap is an associated object. The last element of an entry, the counter, is
 *  used to implement lazy deletion in the heap. Each event handler should be associated with a valid counter that is
 *  initially zero. When an entry is inserted into the heap, the current value of the counter is stored as well.
 *  Trashing an entry of an event handler then just increases the valid counter. On a request of the root entry in the
 *  heap, the heap compares the stored value of the counter in the current root entry with the current value of the
 *  counter for the same event handler. If the current counter value is larger (i.e., the entry of the event handler was
 *  trashed in between), the entry is neglected. This procedure is repeated until a still valid entry is found.
 *
 *  Note that the current values of the counter for an event handler are not stored in the heap here. Instead, the
 *  Python module using this C extension via cffi should store these counter values and provide a callback function
 *  which determines whether the current root entry should be deleted (see 'root' function and heap_scheduler.py for an
 *  example of how it is used).
 *
 *  @author The JeLLyFysh organization.
 *  @bug No known bugs.
 */
#include "heap.h" // Include declarations.

#include <stdlib.h> // For calloc, free, realloc, size_t.


/** @brief Struct that stores the binary min heap.
 *
 *  Note that the heap entries are stored in struct HeapEntry objects. Such an object is returned by the 'root' and
 *  'entry' functions of this C extension. In order to achieve that Python can read the variables of the struct after
 *  the 'root' function was called via cffi, the struct HeapEntry has to be defined in the header. Therefore, see heap.h
 *  for more information on this struct.
 */
struct Heap {
    /** The array of heap entries. */
    struct HeapEntry *heap_entries;
    /** The number of entries in the heap. */
    uint length;
    /** The maximum number of entries this heap can currently store. */
    uint size;
};


/** @brief Create a Heap struct on the heap.
 *
 *  The heap is initialized with length and size equal to zero and heap_entries pointing to NULL. The size of the heap
 *  is dynamically adjusted when new entries are inserted into the heap in the 'insert' function.
 *
 *  @return The pointer to the Heap struct, or NULL if the necessary memory allocation failed.
 */
struct Heap *construct_heap() {
    struct Heap *heap = calloc(1, sizeof(struct Heap));
    if (heap == NULL) return NULL;
    return heap;
}


/** @brief Return the estimated size in bytes of a Heap struct on the heap.
 *
 *  Since the size of the Heap struct is dynamically adjusted in the 'insert' function, this function just returns the
 *  size of the Heap struct itself. The number of bytes allocated for the heap_entries array is returned by the 'insert'
 *  function.
 *
 *  @param heap The pointer to the Heap struct on the heap.
 *  @return The estimated size.
 */
size_t estimated_size(struct Heap *heap) {
    return sizeof(struct Heap);
}


/** @brief Deallocate the memory of a Heap struct on the heap.
 *
 *  @param heap The pointer to the Heap struct on the heap.
 *  @return Void.
 */
void destroy_heap(struct Heap *heap) {
    if (heap) {
        if (heap->heap_entries) {
            free(heap->heap_entries);
        }
        free(heap);
    }
}


/** @brief Insert a new entry into the binary min heap.
 *
 *  If the size of the Heap struct is too small, this function will re-allocate memory for twice the number of heap
 *  entries. The first call of this function allocates memory for 64 heap entries. Note that the heap needs space for
 *  one more entry than its length in the 'root' function.
 *
 *  The new entry is inserted by adding the entry to the end of the heap. Then the added entry is compared with
 *  its parent. If the time of the parent is greater than the time of the new entry, the new entry is swapped with
 *  the parent. This is repeated until the time of the parent is smaller than the time of the new entry, or until
 *  the new entry is the root entry (i.e., the new entry is 'bubbled up' to its final position).
 *
 *  The heap entries are stored in the array as depicted in the following sketch that shows the indices and the
 *  corresponding places in the binary min heap:
 *
 *                                   0
 *                                   1
 *                 2                                 3
 *         4               5                6               7
 *     8       9       10       11      12      13      14      15
 *   16 17   18 19   20  21   22  23  24  25  26  27  28  29  30  31
 *
 *  The zeroth entry is an artificial item with a time equal to minus infinity so that it is always on the top. The
 *  first index corresponds to the root entry with the smallest time. The index of the parent is obtained by using the
 *  integer division parent_index = index // 2 (or index >> 1). Similarly, the two children indices are
 *  children_index_one = 2 * index (or index << 1) and children_index_two = 2 * index + 1.
 *
 *  In order to avoid loss of precision during long runs of JF, candidate event times are not stored as simple floats
 *  but as the quotient and remainder of an integer division of the candidate event time with 1 (see base.time.Time
 *  class for more information). Therefore, these two doubles appear as an argument of this function and are used to
 *  the times.
 *
 *  @param heap The pointer to the Heap struct on the heap.
 *  @param time_quotient The quotient of an integer division of the candidate event time with 1.
 *  @param time_remainder The remainder of an integer division of the candidate event time with 1.
 *  @param event_handler The event handler of the entry that should be inserted into the heap.
 *  @param counter The counter of the entry that should be inserted into the heap.
 *  @return The size in bytes that are currently allocated for the heap entries, or -1 if the necessary memory
 *          allocation failed.
 */
size_t insert(struct Heap *heap, double time_quotient, double time_remainder, void *event_handler, uint counter) {
    // Position of new entry, which gets bubbled up from the end.
    uint position = (heap->length)++;
    // Heap needs space for one more entry than its length in the 'root' function.
    if (heap->length + 1 > heap->size) {
        uint old_size = heap->size;
        heap->size = heap->size ? heap->size * 2 : 64;
        heap->heap_entries = realloc(heap->heap_entries, heap->size * sizeof (struct HeapEntry));
        if (heap->heap_entries == NULL) return -1;
        // Initialize zeroth entry if heap was emtpy before.
        if (!old_size) {
            heap->heap_entries[0].time_quotient = -1.0 / 0.0;
            heap->heap_entries[0].time_remainder = -1.0 / 0.0;
            heap->heap_entries[0].event_handler = NULL;
            heap->heap_entries[0].counter = -1;
            heap->length++;
            position++;
        }
    }
    uint parent_position = position >> 1u;
    while (time_quotient < heap->heap_entries[parent_position].time_quotient ||
              (time_quotient == heap->heap_entries[parent_position].time_quotient
               && time_remainder < heap->heap_entries[parent_position].time_remainder)) {
        // Move parent down as long as its time is smaller than the time of the new entry.
        heap->heap_entries[position] = heap->heap_entries[parent_position];
        position = parent_position;
        parent_position = position >> 1u;
    }
    heap->heap_entries[position].time_quotient = time_quotient;
    heap->heap_entries[position].time_remainder = time_remainder;
    heap->heap_entries[position].event_handler = event_handler;
    heap->heap_entries[position].counter = counter;
    return heap->size * sizeof(struct HeapEntry);
}


/** @brief Bubble the element at the given position down the heap.
 *
 *  This function assumes that the heap entry at the given position is currently also stored at
 *  heap->heap_entries[length] (see root and delete_events functions to see how this is achieved in different cases).
 *
 *  In order to bubble the element at the given position down, its time is compared with the time of its two children.
 *  If the entries are in the correct order, stop. If not, the new entry is swapped with the smaller of its children,
 *  and the procedure is repeated (i.e., the entry is 'bubbled down' to its final position).
 *
 *  In order to bypass actually swapping compared entries, the entry that is bubbled down is cached at
 *  heap_entries[length]. Therefore, the heap_entries array needs always space for one more entry than the length (see
 *  the 'insert' function).
 *
 *  @param heap The pointer to the Heap struct on the heap.
 *  @param position The index of the element that should be bubbled down.
 *  @return Void.
 */
void bubble_down(struct Heap *heap, uint position) {
    // The 'position' variable stores the hypothetical position of the entry that is bubbled down.
    uint compare_position, child_position;
    while (position < heap->length) {
        // heap->heap_entries[heap->length] should contain the entry which gets bubbled down.
        compare_position = heap->length;
        // Position of first child.
        child_position = position << 1u;
        if (child_position < heap->length &&
               (heap->heap_entries[child_position].time_quotient < heap->heap_entries[compare_position].time_quotient ||
                   (heap->heap_entries[child_position].time_quotient
                        == heap->heap_entries[compare_position].time_quotient
                    && heap->heap_entries[child_position].time_remainder
                        < heap->heap_entries[compare_position].time_remainder))) {
            compare_position = child_position;
        }
        // Check if second child is even smaller than first child.
        if (child_position + 1 < heap->length &&
               (heap->heap_entries[child_position + 1].time_quotient
                   < heap->heap_entries[compare_position].time_quotient ||
                   (heap->heap_entries[child_position + 1].time_quotient
                       == heap->heap_entries[compare_position].time_quotient
                    && heap->heap_entries[child_position + 1].time_remainder
                        < heap->heap_entries[compare_position].time_remainder))) {
            compare_position = child_position + 1;
        }
        // Bubble smaller child up or put bubbled down entry at position if no child was smaller.
        heap->heap_entries[position] = heap->heap_entries[compare_position];
        position = compare_position;
    }
}


/** @brief Return the root heap entry.
 *
 *  This heap uses lazy deletion and relies on a callback function which determines whether the current root entry is
 *  still valid. This callback function is called with the scheduler object that was passed to this function, and the
 *  event handler and counter value of the current root entry. The scheduler object can be used to use a class method in
 *  Python as the callback function, and thus to circumvent that cffi only allows for global Python callback methods
 *  (see heap_scheduler.py for an example).
 *
 *  If a root entry needs to be deleted, this is achieved by the following procedure. First the currently last entry in
 *  the heap is placed at the root. Then, the new root entry is bubbled down the heap using the bubble_down function.
 *
 *  If the heap is actually empty after the lazy deletion (i.e., it only contains the artificial 0th item or no item at
 *  all), this function returns a HeapEntry struct with an event handler pointing to NULL, a counter equal to -1, and a
 *  time of minus infinity, i.e., both the quotient and remainder of an integer division of this time with 1 are minus
 *  infinity.
 *
 *  @param heap The pointer to the Heap struct on the heap.
 *  @param scheduler The scheduler object that is passed on the the event_valid_callback function.
 *  @param event_valid_callback The callback function.
 *  @return The current root entry of the heap.
 */
struct HeapEntry root(struct Heap *heap, void *scheduler, int (*event_valid_callback)(void *, void *, uint)) {
    uint child_position;
    uint compare_position;
    uint position;
    // The event_valid_callback function returns true if the entry should be deleted and not be returned.
    while (heap->length > 1 &&
           event_valid_callback(scheduler, heap->heap_entries[1].event_handler, heap->heap_entries[1].counter)) {
        // Replace current root entry with the last entry in the heap and reduce length to delete root entry.
        heap->heap_entries[1] = heap->heap_entries[--(heap->length)];
        // Now the entry on the root position needs to be bubbled down.
        bubble_down(heap, 1);
    }
    return heap->length > 1 ? heap->heap_entries[1] : (struct HeapEntry) {-1.0 / 0.0, -1.0 / 0.0, NULL, -1};
}


/** @brief Delete all events associated with the given event handler.
 *
 *  This function can be used when the valid counter for the given event handler exceeds the range of uint. This method
 *  removes all events associated with the given event handler so that the counter can be reset to 0.
 *
 *  @param heap The pointer to the Heap struct on the heap.
 *  @param event_handler The event handler whose events should be deleted.
 *  @return Void.
 */
void delete_events(struct Heap *heap, void *event_handler) {
    uint current_index = 1;
    while (current_index < heap->length) {
        if (heap->heap_entries[current_index].event_handler == event_handler) {
            heap->heap_entries[current_index] = heap->heap_entries[--(heap->length)];
            continue;
        }
        current_index++;
    }

    // Heapify the heap again by bubbling down all nodes that have children in a reverse manner.
    for (uint index = heap->length / 2; index >= 1; index--) {
        // heap->heap_entries[heap->length] should contain the element that gets bubbled down (see bubble_down function)
        heap->heap_entries[heap->length] = heap->heap_entries[index];
        bubble_down(heap, index);
    }
}


/** @brief Return the heap entry at the given index.
 *
 *  This function can be used iteratively to retrieve all entries that are stored in the heap.
 *
 *  Note that the given index is corrected for the artificial 0th item in the heap, i.e., an index equal to zero will
 *  return the first element. If the given index exceeds the stored heap entries, this function returns a HeapEntry
 *  struct with an event handler pointing to NULL, a counter equal to -1 and a time of minus infinity, i.e., both the
 *  quotient and remainder of an integer division of this time with 1 are minus infinity.
 *
 *  @param heap The pointer to the Heap struct on the heap.
 *  @param index The index.
 *  @return The entry in the heap at the given index.
 */
struct HeapEntry entry(struct Heap *heap, uint index) {
    if (index + 1 < heap->length) {
        return heap->heap_entries[index + 1];
    } else {
        return (struct HeapEntry) {-1.0 / 0.0, -1.0 / 0.0, NULL, -1};
    }
}
