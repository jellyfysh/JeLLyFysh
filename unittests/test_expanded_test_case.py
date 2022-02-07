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
from unittest import TestCase, main, mock
# noinspection PyUnresolvedReferences
from expanded_test_case import ExpandedTestCase


# Inherit explicitly from TestCase class for Test functionality in PyCharm.
class TestExpandedTestCase(ExpandedTestCase, TestCase):
    def test_assert_almost_equal_sequence(self):
        # Test assertAlmostEqualSequence can be used with lists.
        list_one = [0.12345678]
        list_two = [0.12345676]
        self.assertAlmostEqualSequence(list_one, list_two)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualSequence(list_one, list_two, places=8)

        # Test assertAlmostEqualSequence can also be used with tuples.
        tuple_one = (0.00000000001, 0.1231, 0.57162)
        tuple_two = (0.00000000009, 0.1231, 0.57162)
        self.assertAlmostEqualSequence(tuple_one, tuple_two, places=9)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualSequence(tuple_one, tuple_two, places=10)

        # Different sequence lengths raise an AssertionError.
        list_one = [0.3, 0.4, 0.5]
        list_two = [0.3, 0.4]
        with self.assertRaises(AssertionError):
            self.assertAlmostEqualSequence(list_one, list_two)

        # Non-sequence types raise a TypeError.
        with self.assertRaises(TypeError):
            self.assertAlmostEqualSequence([0, 1], 0)
        with self.assertRaises(TypeError):
            self.assertAlmostEqualSequence("ab", [0, 1])

    def test_assert_call_arguments_equal_with_almost_equal_sequence(self):
        function_mock = mock.Mock()
        function_mock(1, "Test", [0.5, 0.6], (0.00000000001, 0.1231, 0.57162), a=0.3, b="Hallo")
        # All arguments are exactly equal so positions_of_sequences_in_args is empty list.
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            function_mock.call_args_list[0], positions_of_sequences_in_args=[],
            expected_args=[1, "Test", [0.5, 0.6], (0.00000000001, 0.1231, 0.57162)],
            expected_kwargs={"a": 0.3, "b": "Hallo"})

        # The argument at index 3 is different so it does not compare exactly equal.
        with self.assertRaises(AssertionError):
            self.assertCallArgumentsEqualWithAlmostEqualSequence(
                function_mock.call_args_list[0], positions_of_sequences_in_args=[],
                expected_args=[1, "Test", [0.5, 0.6], (0.00000000009, 0.1231, 0.57162)],
                expected_kwargs={"a": 0.3, "b": "Hallo"})

        # The argument at index 3 is almost equal.
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            function_mock.call_args_list[0], positions_of_sequences_in_args=[3],
            expected_args=[1, "Test", [0.5, 0.6], (0.00000000009, 0.1231, 0.57162)],
            expected_kwargs={"a": 0.3, "b": "Hallo"}, places=9)

        # The argument at index 3 disagrees for 10 places.
        with self.assertRaises(AssertionError):
            self.assertCallArgumentsEqualWithAlmostEqualSequence(
                function_mock.call_args_list[0], positions_of_sequences_in_args=[3],
                expected_args=[1, "Test", [0.5, 0.6], (0.00000000009, 0.1231, 0.57162)],
                expected_kwargs={"a": 0.3, "b": "Hallo"}, places=10)

        # One can also only use almost equal comparison for the arguments at indices 2 and 3.
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            function_mock.call_args_list[0], positions_of_sequences_in_args=[2, 3],
            expected_args=[1, "Test", [0.5, 0.6], (0.00000000009, 0.1231, 0.57162)],
            expected_kwargs={"a": 0.3, "b": "Hallo"}, places=9)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            function_mock.call_args_list[0], positions_of_sequences_in_args=[2, 3],
            expected_args=[1, "Test", [0.50000000000001, 0.6], (0.00000000001, 0.1231, 0.57162)],
            expected_kwargs={"a": 0.3, "b": "Hallo"}, places=13)
        with self.assertRaises(AssertionError):
            self.assertCallArgumentsEqualWithAlmostEqualSequence(
                function_mock.call_args_list[0], positions_of_sequences_in_args=[2, 3],
                expected_args=[1, "Test", [0.50000000000001, 0.6], (0.00000000001, 0.1231, 0.57162)],
                expected_kwargs={"a": 0.3, "b": "Hallo"}, places=14)

        # If an index in positions_of_sequence_in_args refers to an argument that is not a sequence, one gets a
        # TypeError.
        with self.assertRaises(TypeError):
            self.assertCallArgumentsEqualWithAlmostEqualSequence(
                function_mock.call_args_list[0], positions_of_sequences_in_args=[0],
                expected_args=[1, "Test", [0.5, 0.6], (0.00000000001, 0.1231, 0.57162)],
                expected_kwargs={"a": 0.3, "b": "Hallo"})

        # If an index in positions_of_sequence is too large for the given numbers of arguments, one gets an
        # AssertionError.
        with self.assertRaises(AssertionError):
            self.assertCallArgumentsEqualWithAlmostEqualSequence(
                function_mock.call_args_list[0], positions_of_sequences_in_args=[2, 4],
                expected_args=[1, "Test", [0.5, 0.6], (0.00000000001, 0.1231, 0.57162)],
                expected_kwargs={"a": 0.3, "b": "Hallo"})

        # If other arguments are forgotten, one also gets an AssertionError.
        with self.assertRaises(AssertionError):
            self.assertCallArgumentsEqualWithAlmostEqualSequence(
                function_mock.call_args_list[0], positions_of_sequences_in_args=[3],
                expected_args=["Test", [0.5, 0.6], (0.00000000001, 0.1231, 0.57162)],
                expected_kwargs={"a": 0.3, "b": "Hallo"})
        with self.assertRaises(AssertionError):
            self.assertCallArgumentsEqualWithAlmostEqualSequence(
                function_mock.call_args_list[0], positions_of_sequences_in_args=[3],
                expected_args=[1, [0.5, 0.6], (0.00000000001, 0.1231, 0.57162)],
                expected_kwargs={"a": 0.3, "b": "Hallo"})

        # Too many arguments raise an AssertionError.
        with self.assertRaises(AssertionError):
            self.assertCallArgumentsEqualWithAlmostEqualSequence(
                function_mock.call_args_list[0], positions_of_sequences_in_args=[3],
                expected_args=[1, "Test", [0.5, 0.6], (0.00000000001, 0.1231, 0.57162), 0],
                expected_kwargs={"a": 0.3, "b": "Hallo"}, places=9)

        # Order of kwargs is not important.
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            function_mock.call_args_list[0], positions_of_sequences_in_args=[3],
            expected_args=[1, "Test", [0.5, 0.6], (0.00000000001, 0.1231, 0.57162)],
            expected_kwargs={"b": "Hallo", "a": 0.3}, places=9)

        # Too much kwargs raise an AssertionError.
        with self.assertRaises(AssertionError):
            self.assertCallArgumentsEqualWithAlmostEqualSequence(
                function_mock.call_args_list[0], positions_of_sequences_in_args=[3],
                expected_args=[1, "Test", [0.5, 0.6], (0.00000000001, 0.1231, 0.57162)],
                expected_kwargs={"a": 0.3, "b": "Hallo", "c": -1}, places=9)

        # Too few kwargs raise an AssertionError.
        with self.assertRaises(AssertionError):
            self.assertCallArgumentsEqualWithAlmostEqualSequence(
                function_mock.call_args_list[0], positions_of_sequences_in_args=[3],
                expected_args=[1, "Test", [0.5, 0.6], (0.00000000001, 0.1231, 0.57162)],
                expected_kwargs={"a": 0.3}, places=9)


if __name__ == '__main__':
    main()
