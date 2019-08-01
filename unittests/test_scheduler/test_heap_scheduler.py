# JeLLFysh - a Python application for all-atom event-chain Monte Carlo - https://github.com/jellyfysh
# Copyright (C) 2019 The JeLLyFysh organization
# (see the AUTHORS file for the full list of authors)
#
# This file is part of JeLLyFysh.
#
# JeLLyFysh is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either > version 3 of the License, or (at your option) any
# later version.
#
# JeLLyFysh is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with JeLLyFysh in the LICENSE file.
# If not, see <https://www.gnu.org/licenses/>.
#
# If you use JeLLyFysh in published work, please cite the following reference (see [Hoellmer2019] in References.bib):
# Philipp Hoellmer, Liang Qin, Michael F. Faulkner, A. C. Maggs, Werner Krauth
# JeLLyFysh-Version1.0 -- a Python application for all-atom event-chain Monte Carlo,
# arXiv e-prints: 1907.12502 (2019), https://arxiv.org/abs/1907.12502
#
from unittest import TestCase, main
from base.exceptions import SchedulerError
from scheduler.heap_scheduler import HeapScheduler


class NotComparableClass(object):
    pass


class TestHeapScheduler(TestCase):
    def setUp(self):
        self._scheduler = HeapScheduler()

    def test_scheduler_simple_ordering(self):
        # Use not comparable classes as objects which are pushed to the scheduler
        # since in the application non comparable event handlers are usually pushed
        not_comparable_instances = [NotComparableClass() for _ in range(5)]
        times = [-0.3, 0, -1, 4.3, 8.777]
        for index, not_comparable_instance in enumerate(not_comparable_instances):
            self._scheduler.push_event(times[index], not_comparable_instance)
        self.assertIs(self._scheduler.get_succeeding_event(), not_comparable_instances[2])

    def test_scheduler_same_times(self):
        not_comparable_instances = [NotComparableClass() for _ in range(5)]
        times = [-0.3, 0, -0.3, 4.3, 8.777]
        for index, not_comparable_instance in enumerate(not_comparable_instances):
            self._scheduler.push_event(times[index], not_comparable_instance)
        # If two objects with the same time are inside of scheduler, we do not care which one is returned
        self.assertTrue(self._scheduler.get_succeeding_event() is not_comparable_instances[0]
                        or self._scheduler.get_succeeding_event() is not_comparable_instances[2])

    def test_scheduler_exceptions(self):
        with self.assertRaises(SchedulerError):
            self._scheduler.trash_event(NotComparableClass())
        self._scheduler.push_event(NotComparableClass(), NotComparableClass())
        with self.assertRaises(TypeError):
            self._scheduler.push_event(NotComparableClass(), NotComparableClass())
        with self.assertRaises(SchedulerError):
            self._scheduler.trash_event(NotComparableClass())

    def test_trash(self):
        not_comparable_instances = [NotComparableClass() for _ in range(5)]
        times = [-0.3, 0, -1, 4.3, 8.777]
        for index, not_comparable_instance in enumerate(not_comparable_instances):
            self._scheduler.push_event(times[index], not_comparable_instance)
        self.assertIs(self._scheduler.get_succeeding_event(), not_comparable_instances[2])
        self._scheduler.trash_event(not_comparable_instances[2])
        self.assertIs(self._scheduler.get_succeeding_event(), not_comparable_instances[0])
        self._scheduler.trash_event(not_comparable_instances[0])
        self._scheduler.trash_event(not_comparable_instances[1])
        self.assertIs(self._scheduler.get_succeeding_event(), not_comparable_instances[3])
        self._scheduler.trash_event(not_comparable_instances[4])
        new_not_comparable_instances = [NotComparableClass() for _ in range(2)]
        new_times = [-2.3, 5.6]
        for index, not_comparable_instance in enumerate(new_not_comparable_instances):
            self._scheduler.push_event(new_times[index], not_comparable_instance)
        self.assertIs(self._scheduler.get_succeeding_event(), new_not_comparable_instances[0])
        self._scheduler.trash_event(new_not_comparable_instances[1])
        self._scheduler.trash_event(new_not_comparable_instances[0])
        self.assertIs(self._scheduler.get_succeeding_event(), not_comparable_instances[3])


if __name__ == '__main__':
    main()
