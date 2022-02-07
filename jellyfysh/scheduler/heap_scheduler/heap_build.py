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
Module which sets up the build of the heap.c C extension using cffi.

The recommended way to compile the C extension heap.c into a shared library that can be used by cffi in the
HeapScheduler class is to run 'pypy3 setup.py build_ext -i' in the root directory of the JeLLyFysh repository.
(Of course, 'pypy3' can be replaced with the Python interpreter of your choice).

Alternatively, this script can be executed from the root directory of the JeLLyFysh repository.
"""
from cffi import FFI
ffi_builder = FFI()

# Mostly duplicates the information in heap.h but is required by cffi.
# See https://cffi.readthedocs.io/en/latest/overview.html#if-you-don-t-have-an-already-installed-c-library-to-call.
# Here, the Python callback function 'delete_callback' is also declared. This function is defined in heap_scheduler.py.
ffi_builder.cdef(r"""
typedef unsigned int uint;
struct HeapEntry {
    double time_quotient;
    double time_remainder;
    void *event_handler;
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

extern "Python" int event_valid_callback(void *object, void *event_handler, uint counter);
""")

# First argument is name of the output C extension that is used in heap_scheduler.py.
# All paths are relative to the root directory of the JeLLyFysh application.
ffi_builder.set_source(
    "jellyfysh.scheduler.heap_scheduler._heap",
    """
    #include "heap.h"
    """,
    sources=["jellyfysh/scheduler/heap_scheduler/heap.c"],
    include_dirs=["jellyfysh/scheduler/heap_scheduler"])

if __name__ == "__main__":
    ffi_builder.compile(verbose=True)
