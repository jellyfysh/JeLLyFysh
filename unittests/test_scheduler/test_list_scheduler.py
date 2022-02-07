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
from unittest import main
from jellyfysh.base.exceptions import SchedulerError
from jellyfysh.scheduler.list_scheduler import ListScheduler
try:
    from .test_heap_scheduler import TestHeapScheduler, NotComparableClass
except ImportError:
    # Parent package might not be known (e.g., if this file is directly executed as a script).
    # Then relative imports cannot be used, and we use the following absolute import.
    # noinspection PyUnresolvedReferences
    from test_heap_scheduler import TestHeapScheduler, NotComparableClass


class TestListScheduler(TestHeapScheduler):
    def setUp(self):
        self._scheduler = ListScheduler()

    def test_scheduler_exception_on_trashing_non_existing_event(self):
        self._scheduler.push_event(NotComparableClass(), NotComparableClass())
        with self.assertRaises(SchedulerError):
            self._scheduler.trash_event(NotComparableClass())

    def test_scheduler_exception_on_non_comparable_times(self):
        self._scheduler.push_event(NotComparableClass(), NotComparableClass())
        with self.assertRaises(AttributeError):
            self._scheduler.get_succeeding_event()


if __name__ == '__main__':
    main()
