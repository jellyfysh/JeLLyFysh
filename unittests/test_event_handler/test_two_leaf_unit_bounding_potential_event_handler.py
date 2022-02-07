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
import os
import sys
from unittest import TestCase, main, mock
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.node import Node
from jellyfysh.base.time import Time
from jellyfysh.base.unit import Unit
from jellyfysh.event_handler.two_leaf_unit_bounding_potential_event_handler import \
    TwoLeafUnitBoundingPotentialEventHandler
from jellyfysh.potential import Potential, InvertiblePotential
import jellyfysh.setting as setting
from jellyfysh.setting import hypercubic_setting
_unittest_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_unittest_directory_added_to_path = False
if _unittest_directory not in sys.path:
    sys.path.append(_unittest_directory)
    _unittest_directory_added_to_path = True
# noinspection PyUnresolvedReferences
from expanded_test_case import ExpandedTestCase


def tearDownModule():
    if _unittest_directory_added_to_path:
        sys.path.remove(_unittest_directory)


@mock.patch("jellyfysh.event_handler.abstracts.event_handler_with_bounding_potential.random.uniform")
@mock.patch("jellyfysh.event_handler.two_leaf_unit_bounding_potential_event_handler.random.expovariate")
class TestTwoLeafUnitBoundingPotentialEventHandler(ExpandedTestCase, TestCase):
    def setUp(self) -> None:
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=2, system_length=1.0)
        self._bounding_potential_mock_without_charge = mock.MagicMock(spec_set=InvertiblePotential)
        self._bounding_potential_mock_without_charge.number_separation_arguments = 1
        self._bounding_potential_mock_without_charge.number_charge_arguments = 0
        self._bounding_potential_mock_without_charge.potential_change_required = True
        self._potential_mock_without_charge = mock.MagicMock(spec_set=Potential)
        self._potential_mock_without_charge.number_separation_arguments = 1
        self._potential_mock_without_charge.number_charge_arguments = 0
        self._event_handler_without_charge = TwoLeafUnitBoundingPotentialEventHandler(
            potential=self._potential_mock_without_charge,
            bounding_potential=self._bounding_potential_mock_without_charge)

        self._bounding_potential_mock_with_charge = mock.MagicMock(spec_set=InvertiblePotential)
        self._bounding_potential_mock_with_charge.number_separation_arguments = 1
        self._bounding_potential_mock_with_charge.number_charge_arguments = 2
        self._bounding_potential_mock_with_charge.potential_change_required = True
        self._potential_mock_with_charge = mock.MagicMock(spec_set=Potential)
        self._potential_mock_with_charge.number_separation_arguments = 1
        self._potential_mock_with_charge.number_charge_arguments = 2
        self._event_handler_with_charge = TwoLeafUnitBoundingPotentialEventHandler(
            potential=self._potential_mock_with_charge, bounding_potential=self._bounding_potential_mock_with_charge,
            charge="charge")

    def tearDown(self) -> None:
        setting.reset()

    def _setUpSendEventTime(self, random_expovariate_mock, bounding_potential_mock):
        bounding_potential_mock.displacement.return_value = 0.3
        random_expovariate_mock.return_value = 2

    def _setUpSendOutStateAccept(self, random_uniform_mock, bounding_potential_mock, potential_mock):
        bounding_potential_mock.derivative.return_value = 0.7
        potential_mock.derivative.return_value = 0.5
        random_uniform_mock.return_value = 0.4

    def _setUpSendOutStateReject(self, random_uniform_mock, bounding_potential_mock, potential_mock):
        bounding_potential_mock.derivative.return_value = 0.7
        potential_mock.derivative.return_value = 0.5
        random_uniform_mock.return_value = 0.6

    def test_send_event_time_without_charge(self, random_expovariate_mock, _):
        # Bounding Potential returns time displacement 0.3.
        self._setUpSendEventTime(random_expovariate_mock, self._bounding_potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.25, 0.5],
                                 time_stamp=Time.from_float(1.2)), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.5, 1.0], time_stamp=Time.from_float(1.3)), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))

        event_time = self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        self.assertEqual(self._bounding_potential_mock_without_charge.displacement.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._bounding_potential_mock_without_charge.displacement.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[0.5, 1.0], [-0.4, 0.4], 2], places=13,
            expected_kwargs={})
        self.assertAlmostEqual(event_time, Time.from_float(1.6), places=13)

    def test_send_out_state_without_charge_accept(self, random_expovariate_mock, random_uniform_mock):
        # Bounding potential returns time displacement 0.3.
        self._setUpSendEventTime(random_expovariate_mock, self._bounding_potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.25, 0.5],
                                 time_stamp=Time.from_float(1.2)), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.5, 1.0], time_stamp=Time.from_float(1.3)), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateAccept(random_uniform_mock, self._bounding_potential_mock_without_charge,
                                      self._potential_mock_without_charge)
        out_state = self._event_handler_without_charge.send_out_state()
        # New position of active leaf unit is [0.65, 0.2].
        self.assertEqual(self._bounding_potential_mock_without_charge.derivative.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._bounding_potential_mock_without_charge.derivative.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[0.5, 1.0], [0.45, 0.1]], places=13, expected_kwargs={})
        self.assertEqual(self._potential_mock_without_charge.derivative.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._potential_mock_without_charge.derivative.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[0.5, 1.0], [0.45, 0.1]], places=13, expected_kwargs={})
        random_uniform_mock.assert_called_once_with(0, 0.7)

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(first_cnode.value.position, [0.3, 0.0], places=13)
        self.assertEqual(first_cnode.weight, 1)
        self.assertIsNone(first_cnode.value.velocity)
        self.assertIsNone(first_cnode.value.time_stamp)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 2))
        self.assertAlmostEqualSequence(first_child.value.position, [0.65, 0.2], places=13)
        self.assertEqual(first_child.weight, 0.5)
        self.assertIsNone(first_child.value.velocity)
        self.assertIsNone(first_child.value.time_stamp)
        self.assertIsNone(first_child.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (3,))
        self.assertEqual(second_cnode.value.position, [0.5, 0.6])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [0.25, 0.5])
        self.assertAlmostEqual(second_cnode.value.time_stamp, Time.from_float(1.6), places=13)
        self.assertIsNone(second_cnode.value.charge)
        self.assertEqual(len(second_cnode.children), 1)
        second_child = second_cnode.children[0]
        self.assertIs(second_child.parent, second_cnode)
        self.assertEqual(second_child.children, [])
        self.assertEqual(second_child.value.identifier, (3, 1))
        self.assertEqual(second_child.value.position, [0.1, 0.3])
        self.assertEqual(second_child.weight, 0.5)
        self.assertEqual(second_child.value.velocity, [0.5, 1.0])
        self.assertAlmostEqual(second_child.value.time_stamp, Time.from_float(1.6), places=13)
        self.assertIsNone(second_child.value.charge)

    def test_send_out_state_without_charge_reject(self, random_expovariate_mock, random_uniform_mock):
        # Bounding potential returns time displacement 0.3.
        self._setUpSendEventTime(random_expovariate_mock, self._bounding_potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.25, 0.5],
                                 time_stamp=Time.from_float(1.2)), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.5, 1.0], time_stamp=Time.from_float(1.3)), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateReject(random_uniform_mock, self._bounding_potential_mock_without_charge,
                                      self._potential_mock_without_charge)
        out_state = self._event_handler_without_charge.send_out_state()
        # New position of active leaf unit is [0.65, 0.2].
        self.assertEqual(self._bounding_potential_mock_without_charge.derivative.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._bounding_potential_mock_without_charge.derivative.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[0.5, 1.0], [0.45, 0.1]], places=13, expected_kwargs={})
        self.assertEqual(self._potential_mock_without_charge.derivative.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._potential_mock_without_charge.derivative.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[0.5, 1.0], [0.45, 0.1]], places=13, expected_kwargs={})
        random_uniform_mock.assert_called_once_with(0, 0.7)

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(first_cnode.value.position, [0.3, 0.0], places=13)
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.25, 0.5])
        self.assertAlmostEqual(first_cnode.value.time_stamp, Time.from_float(1.6), places=13)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 2))
        self.assertAlmostEqualSequence(first_child.value.position, [0.65, 0.2], places=13)
        self.assertEqual(first_child.weight, 0.5)
        self.assertEqual(first_child.value.velocity, [0.5, 1.0])
        self.assertAlmostEqual(first_child.value.time_stamp, Time.from_float(1.6), places=13)
        self.assertIsNone(first_child.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (3,))
        self.assertEqual(second_cnode.value.position, [0.5, 0.6])
        self.assertEqual(second_cnode.weight, 1)
        self.assertIsNone(second_cnode.value.velocity)
        self.assertIsNone(second_cnode.value.time_stamp)
        self.assertIsNone(second_cnode.value.charge)
        self.assertEqual(len(second_cnode.children), 1)
        second_child = second_cnode.children[0]
        self.assertIs(second_child.parent, second_cnode)
        self.assertEqual(second_child.children, [])
        self.assertEqual(second_child.value.identifier, (3, 1))
        self.assertEqual(second_child.value.position, [0.1, 0.3])
        self.assertEqual(second_child.weight, 0.5)
        self.assertIsNone(second_child.value.velocity)
        self.assertIsNone(second_child.value.time_stamp)
        self.assertIsNone(second_child.value.charge)

    def test_send_event_time_with_charge(self, random_expovariate_mock, _):
        # Bounding potential returns time displacement 0.3.
        self._setUpSendEventTime(random_expovariate_mock, self._bounding_potential_mock_with_charge)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[-2.0, -3.0],
                                 time_stamp=Time.from_float(1.1), charge={"charge": 3.4}), weight=1)

        event_time = self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        self.assertEqual(self._bounding_potential_mock_with_charge.displacement.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._bounding_potential_mock_with_charge.displacement.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[-2.0, -3.0], [0.0, -0.3], -1.2, 3.4, 2], places=13,
            expected_kwargs={})
        self.assertAlmostEqual(event_time, Time.from_float(1.4), places=13)

    def test_send_out_state_with_charge_accept(self, random_expovariate_mock, random_uniform_mock):
        # Bounding potential returns time displacement 0.3.
        self._setUpSendEventTime(random_expovariate_mock, self._bounding_potential_mock_with_charge)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[-2.0, -3.0],
                                 time_stamp=Time.from_float(1.1), charge={"charge": 3.4}), weight=1)

        self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateAccept(random_uniform_mock, self._bounding_potential_mock_with_charge,
                                      self._potential_mock_with_charge)
        out_state = self._event_handler_with_charge.send_out_state()
        # New position of active leaf unit is [0.9, 1.0].
        self.assertEqual(self._bounding_potential_mock_with_charge.derivative.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._bounding_potential_mock_with_charge.derivative.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[-2.0, -3.0], [-0.4, -0.4], -1.2, 3.4], places=13,
            expected_kwargs={})
        self.assertEqual(self._potential_mock_with_charge.derivative.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._potential_mock_with_charge.derivative.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[-2.0, -3.0], [-0.4, -0.4], -1.2, 3.4], places=13,
            expected_kwargs={})
        random_uniform_mock.assert_called_once_with(0, 0.7)

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.children, [])
        self.assertEqual(first_cnode.value.identifier, (3,))
        self.assertEqual(first_cnode.value.position, [0.5, 0.6])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.charge, {"charge": -1.2})
        self.assertEqual(first_cnode.value.velocity, [-2.0, -3.0])
        self.assertAlmostEqual(first_cnode.value.time_stamp, Time.from_float(1.4), places=13)
        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.children, [])
        self.assertEqual(second_cnode.value.identifier, (1,))
        self.assertAlmostEqualSequence(second_cnode.value.position, [0.9, 0.0], places=13)
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.charge, {"charge": 3.4})
        self.assertIsNone(second_cnode.value.velocity)
        self.assertIsNone(second_cnode.value.time_stamp)

    def test_send_out_state_with_charge_reject(self, random_expovariate_mock, random_uniform_mock):
        # Bounding potential returns time displacement 0.3.
        self._setUpSendEventTime(random_expovariate_mock, self._bounding_potential_mock_with_charge)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[-2.0, -3.0],
                                 time_stamp=Time.from_float(1.1), charge={"charge": 3.4}), weight=1)

        self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateReject(random_uniform_mock, self._bounding_potential_mock_with_charge,
                                      self._potential_mock_with_charge)
        out_state = self._event_handler_with_charge.send_out_state()
        # New position of active leaf unit is [0.9, 1.0].
        self.assertEqual(self._bounding_potential_mock_with_charge.derivative.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._bounding_potential_mock_with_charge.derivative.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[-2.0, -3.0], [-0.4, -0.4], -1.2, 3.4], places=13,
            expected_kwargs={})
        self.assertEqual(self._potential_mock_with_charge.derivative.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._potential_mock_with_charge.derivative.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[-2.0, -3.0], [-0.4, -0.4], -1.2, 3.4], places=13,
            expected_kwargs={})
        random_uniform_mock.assert_called_once_with(0, 0.7)

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.children, [])
        self.assertEqual(first_cnode.value.identifier, (3,))
        self.assertEqual(first_cnode.value.position, [0.5, 0.6])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.charge, {"charge": -1.2})
        self.assertIsNone(first_cnode.value.velocity)
        self.assertIsNone(first_cnode.value.time_stamp)
        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.children, [])
        self.assertEqual(second_cnode.value.identifier, (1,))
        self.assertAlmostEqualSequence(second_cnode.value.position, [0.9, 0.0], places=13)
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.charge, {"charge": 3.4})
        self.assertEqual(second_cnode.value.velocity, [-2.0, -3.0])
        self.assertAlmostEqual(second_cnode.value.time_stamp, Time.from_float(1.4), places=13)

    def test_send_event_time_leaf_units_in_same_composite_object(self, random_expovariate_mock, _):
        # Bounding potential returns time displacement 0.3.
        self._setUpSendEventTime(random_expovariate_mock, self._bounding_potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[-0.5, 0.5],
                                 time_stamp=Time.from_float(1.8)), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.9],
                                         velocity=[-1.0, 1.0], time_stamp=Time.from_float(1.8)), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[-0.5, 0.5],
                                 time_stamp=Time.from_float(1.8)), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 0), position=[0.1, 0.3]), weight=0.5))
        event_time = self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        self.assertEqual(self._bounding_potential_mock_without_charge.displacement.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._bounding_potential_mock_without_charge.displacement.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[-1.0, 1.0], [-0.4, 0.4], 2], places=13,
            expected_kwargs={})
        self.assertAlmostEqual(event_time, Time.from_float(2.1), places=13)

    def test_send_out_state_leaf_units_in_same_composite_object_accept(self, random_expovariate_mock,
                                                                       random_uniform_mock):
        # Bounding potential returns time displacement 0.3.
        self._setUpSendEventTime(random_expovariate_mock, self._bounding_potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[-0.5, 0.5],
                                 time_stamp=Time.from_float(1.8)), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.9],
                                         velocity=[-1.0, 1.0], time_stamp=Time.from_float(1.8)), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[-0.5, 0.5],
                                 time_stamp=Time.from_float(1.8)), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 0), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateAccept(random_uniform_mock, self._bounding_potential_mock_without_charge,
                                      self._potential_mock_without_charge)
        out_state = self._event_handler_without_charge.send_out_state()
        # New position of active leaf unit is [0.2, 0.2].
        self.assertEqual(self._bounding_potential_mock_without_charge.derivative.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._bounding_potential_mock_without_charge.derivative.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[-1.0, 1.0], [-0.1, 0.1]], places=13, expected_kwargs={})
        self.assertEqual(self._potential_mock_without_charge.derivative.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._potential_mock_without_charge.derivative.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[-1.0, 1.0], [-0.1, 0.1]], places=13, expected_kwargs={})
        random_uniform_mock.assert_called_once_with(0, 0.7)

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(first_cnode.value.position, [0.05, 0.95], places=13)
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [-0.5, 0.5])
        self.assertAlmostEqual(first_cnode.value.time_stamp, Time.from_float(2.1), places=13)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 1))
        self.assertAlmostEqualSequence(first_child.value.position, [0.2, 0.2], places=13)
        self.assertEqual(first_child.weight, 0.5)
        self.assertIsNone(first_child.value.velocity)
        self.assertIsNone(first_child.value.time_stamp)
        self.assertIsNone(first_child.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(second_cnode.value.position, [0.05, 0.95], places=13)
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [-0.5, 0.5])
        self.assertAlmostEqual(second_cnode.value.time_stamp, Time.from_float(2.1), places=13)
        self.assertIsNone(second_cnode.value.charge)
        self.assertEqual(len(second_cnode.children), 1)
        second_child = second_cnode.children[0]
        self.assertIs(second_child.parent, second_cnode)
        self.assertEqual(second_child.children, [])
        self.assertEqual(second_child.value.identifier, (0, 0))
        self.assertEqual(second_child.value.position, [0.1, 0.3])
        self.assertEqual(second_child.weight, 0.5)
        self.assertEqual(second_child.value.velocity, [-1.0, 1.0])
        self.assertAlmostEqual(second_child.value.time_stamp, Time.from_float(2.1), places=13)
        self.assertIsNone(second_child.value.charge)

    def test_send_out_state_leaf_units_in_same_composite_object_reject(self, random_expovariate_mock,
                                                                       random_uniform_mock):
        # Bounding potential returns time displacement 0.3.
        self._setUpSendEventTime(random_expovariate_mock, self._bounding_potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[-0.5, 0.5],
                                 time_stamp=Time.from_float(1.8)), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.9],
                                         velocity=[-1.0, 1.0], time_stamp=Time.from_float(1.8)), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[-0.5, 0.5],
                                 time_stamp=Time.from_float(1.8)), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 0), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateReject(random_uniform_mock, self._bounding_potential_mock_without_charge,
                                      self._potential_mock_without_charge)
        out_state = self._event_handler_without_charge.send_out_state()
        # New position of active leaf unit is [0.2, 0.2].
        self.assertEqual(self._bounding_potential_mock_without_charge.derivative.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._bounding_potential_mock_without_charge.derivative.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[-1.0, 1.0], [-0.1, 0.1]], places=13, expected_kwargs={})
        self.assertEqual(self._potential_mock_without_charge.derivative.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._potential_mock_without_charge.derivative.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[-1.0, 1.0], [-0.1, 0.1]], places=13, expected_kwargs={})
        random_uniform_mock.assert_called_once_with(0, 0.7)

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(first_cnode.value.position, [0.05, 0.95], places=13)
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [-0.5, 0.5])
        self.assertAlmostEqual(first_cnode.value.time_stamp, Time.from_float(2.1), places=13)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 1))
        self.assertAlmostEqualSequence(first_child.value.position, [0.2, 0.2], places=13)
        self.assertEqual(first_child.weight, 0.5)
        self.assertEqual(first_child.value.velocity, [-1.0, 1.0])
        self.assertAlmostEqual(first_child.value.time_stamp, Time.from_float(2.1), places=13)
        self.assertIsNone(first_child.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (0,))
        self.assertAlmostEqualSequence(second_cnode.value.position, [0.05, 0.95], places=13)
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [-0.5, 0.5])
        self.assertAlmostEqual(second_cnode.value.time_stamp, Time.from_float(2.1), places=13)
        self.assertIsNone(second_cnode.value.charge)
        self.assertEqual(len(second_cnode.children), 1)
        second_child = second_cnode.children[0]
        self.assertIs(second_child.parent, second_cnode)
        self.assertEqual(second_child.children, [])
        self.assertEqual(second_child.value.identifier, (0, 0))
        self.assertEqual(second_child.value.position, [0.1, 0.3])
        self.assertEqual(second_child.weight, 0.5)
        self.assertIsNone(second_child.value.velocity)
        self.assertIsNone(second_child.value.time_stamp)
        self.assertIsNone(second_child.value.charge)

    def _setUpSendEventTimeInfiniteDisplacement(self, random_expovariate_mock, bounding_potential_mock):
        bounding_potential_mock.displacement.return_value = float('inf')
        random_expovariate_mock.return_value = 2

    def test_send_event_time_infinite_displacement(self, random_expovariate_mock, _):
        # Bounding potential returns infinite time displacement.
        self._setUpSendEventTimeInfiniteDisplacement(random_expovariate_mock, self._bounding_potential_mock_with_charge)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[2.0, 0.0],
                                 time_stamp=Time.from_float(1.3), charge={"charge": 3.4}), weight=1)

        event_time = self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        self.assertEqual(self._bounding_potential_mock_with_charge.displacement.call_count, 1)
        self.assertCallArgumentsEqualWithAlmostEqualSequence(
            self._bounding_potential_mock_with_charge.displacement.call_args_list[0],
            positions_of_sequences_in_args=[1], expected_args=[[2.0, 0.0], [0.0, -0.3], -1.2, 3.4, 2], places=13,
            expected_kwargs={})
        self.assertEqual(event_time, Time.from_float(float('inf')))
        # Send out state not relevant in this case because it will never be called.

    def test_bounding_potential_with_wrong_number_of_separation_arguments_raises_error(self, _, __):
        bounding_potential_mock = mock.MagicMock(spec_set=InvertiblePotential)
        bounding_potential_mock.number_separation_arguments = 2
        bounding_potential_mock.number_charge_arguments = 0
        with self.assertRaises(ConfigurationError):
            TwoLeafUnitBoundingPotentialEventHandler(potential=self._potential_mock_without_charge,
                                                     bounding_potential=bounding_potential_mock)

    def test_potential_with_wrong_number_of_separation_arguments_raises_error(self, _, __):
        potential_mock = mock.MagicMock(spec_set=Potential)
        potential_mock.number_separation_arguments = 2
        potential_mock.number_charge_arguments = 0
        with self.assertRaises(ConfigurationError):
            TwoLeafUnitBoundingPotentialEventHandler(potential=potential_mock,
                                                     bounding_potential=self._bounding_potential_mock_without_charge)

    def test_potential_with_wrong_number_of_charge_arguments_raises_error(self, _, __):
        bounding_potential_mock = mock.MagicMock(spec_set=InvertiblePotential)
        bounding_potential_mock.number_separation_arguments = 1
        bounding_potential_mock.number_charge_arguments = 2
        potential_mock = mock.MagicMock(spec_set=Potential)
        potential_mock.number_separation_arguments = 1
        potential_mock.number_charge_arguments = 1
        with self.assertRaises(ConfigurationError):
            TwoLeafUnitBoundingPotentialEventHandler(
                potential=potential_mock, bounding_potential=bounding_potential_mock, charge="charge")

    def test_bounding_potential_with_wrong_number_of_charge_arguments_raises_error(self, _, __):
        bounding_potential_mock = mock.MagicMock(spec_set=InvertiblePotential)
        bounding_potential_mock.number_separation_arguments = 1
        bounding_potential_mock.number_charge_arguments = 1
        potential_mock = mock.MagicMock(spec_set=Potential)
        potential_mock.number_separation_arguments = 1
        potential_mock.number_charge_arguments = 2
        with self.assertRaises(ConfigurationError):
            TwoLeafUnitBoundingPotentialEventHandler(
                potential=potential_mock, bounding_potential=bounding_potential_mock, charge="charge")

    def test_bounding_potential_with_no_potential_change_required_raises_error(self, _, __):
        bounding_potential_mock = mock.MagicMock(spec_set=InvertiblePotential)
        bounding_potential_mock.number_separation_arguments = 1
        bounding_potential_mock.number_charge_arguments = 2
        bounding_potential_mock.potential_change_required = False
        potential_mock = mock.MagicMock(spec_set=Potential)
        potential_mock.number_separation_arguments = 1
        potential_mock.number_charge_arguments = 2
        with self.assertRaises(ConfigurationError):
            TwoLeafUnitBoundingPotentialEventHandler(
                potential=potential_mock, bounding_potential=bounding_potential_mock, charge="charge")

    def test_missing_charges_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTime(random_expovariate_mock, self._bounding_potential_mock_with_charge)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"other_charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9],
                                 velocity=[2.0, 0.0], time_stamp=Time.from_float(1.3),
                                 charge={"other_charge": 3.4}), weight=1)
        with self.assertRaises(KeyError):
            self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])

    def test_no_leaf_unit_active_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTime(random_expovariate_mock, self._bounding_potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9]), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

    def test_both_leaf_units_active_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTime(random_expovariate_mock, self._bounding_potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6], velocity=[1.0, 0.0],
                                 time_stamp=Time.from_float(0.0)), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[1.0, 0.0],
                                 time_stamp=Time.from_float(0.0)), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

    def test_more_than_two_leaf_units_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTime(random_expovariate_mock, self._bounding_potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[1.0, 0.0],
                                 time_stamp=Time.from_float(0.7)), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                         velocity=[2.0, 0.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        in_state_one.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.8]), weight=0.5))
        in_state_two = Node(Unit(identifier=(2,), position=[0.2, 0.4], velocity=[1.0, 0.0],
                                 time_stamp=Time.from_float(0.7)), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(2, 0), position=[0.5, 0.6],
                                         velocity=[2.0, 0.0], time_stamp=Time.from_float(0.7)), weight=0.5))
        in_state_two.add_child(Node(Unit(identifier=(2, 1), position=[0.3, 0.8]), weight=0.5))
        with self.assertRaises(AssertionError):
            self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

    def test_number_send_event_time_arguments_one(self, _, __):
        self.assertEqual(self._event_handler_without_charge.number_send_event_time_arguments, 1)
        self.assertEqual(self._event_handler_with_charge.number_send_event_time_arguments, 1)

    def test_number_send_out_state_arguments_zero(self, _, __):
        self.assertEqual(self._event_handler_without_charge.number_send_out_state_arguments, 0)
        self.assertEqual(self._event_handler_without_charge.number_send_out_state_arguments, 0)


if __name__ == '__main__':
    main()
