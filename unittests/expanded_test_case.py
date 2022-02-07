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
from typing import Any, Mapping, Sequence
from unittest import TestCase, main, mock


class ExpandedTestCase(TestCase):
    def assertAlmostEqualSequence(self, sequence_one: Sequence[Any], sequence_two: Sequence[Any], places: int = 7):
        self.assertEqual(len(sequence_one), len(sequence_two))
        [self.assertAlmostEqual(sequence_one[i], sequence_two[i], places=places) for i in range(len(sequence_one))]

    def assertCallArgumentsEqualWithAlmostEqualSequence(self, call_args: mock.call,
                                                        positions_of_sequences_in_args: Sequence[int],
                                                        expected_args: Sequence[Any],
                                                        expected_kwargs: Mapping[str, Any], places: int = 7):
        args, kwargs = call_args
        self.assertEqual(kwargs, expected_kwargs)
        self.assertEqual(len(args), len(expected_args))
        for position_of_sequence_in_args in positions_of_sequences_in_args:
            self.assertGreater(len(expected_args), position_of_sequence_in_args)
        for index in range(len(expected_args)):
            if index in positions_of_sequences_in_args:
                self.assertAlmostEqualSequence(args[index], expected_args[index], places=places)
            else:
                self.assertEqual(args[index], expected_args[index])


if __name__ == '__main__':
    main()
