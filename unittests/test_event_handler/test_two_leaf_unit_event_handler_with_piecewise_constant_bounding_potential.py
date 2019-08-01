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
from unittest import TestCase, main, mock
from base.exceptions import ConfigurationError
from base.node import Node
from base.unit import Unit
from event_handler.two_leaf_unit_event_handler_with_piecewise_constant_bounding_potential import \
    TwoLeafUnitEventHandlerWithPiecewiseConstantBoundingPotential
from potential import Potential
import setting


@mock.patch("event_handler.abstracts.event_handler_with_bounding_potential.random.uniform")
@mock.patch("event_handler.two_leaf_unit_event_handler_with_piecewise_constant_bounding_potential.random.expovariate")
class TestTwoLeafUnitEventHandlerWithPiecewiseConstantBoundingPotential(TestCase):
    def setUp(self) -> None:
        setting.HypercubicSetting(beta=1.0, dimension=2, system_length=1.0)
        self._potential_mock_with_charge = mock.MagicMock(spec_set=Potential)
        self._potential_mock_with_charge.number_separation_arguments = 1
        self._potential_mock_with_charge.number_charge_arguments = 2
        self._event_handler_with_charge = TwoLeafUnitEventHandlerWithPiecewiseConstantBoundingPotential(
            potential=self._potential_mock_with_charge, offset=0.1, max_displacement=0.2, charge="charge")
        self._potential_mock_without_charge = mock.MagicMock(spec_set=Potential)
        self._potential_mock_without_charge.number_separation_arguments = 1
        self._potential_mock_without_charge.number_charge_arguments = 0
        self._event_handler_without_charge = TwoLeafUnitEventHandlerWithPiecewiseConstantBoundingPotential(
            potential=self._potential_mock_without_charge, offset=0.05, max_displacement=0.1)

    def tearDown(self) -> None:
        setting.reset()

    def _setUpSendEventTimeSmallerMaxDisplacement(self, random_expovariate_mock, max_displacement, offset,
                                                  potential_mock):
        potential_mock.derivative.side_effect = [0.1, 0.2]
        # Makes sure that displacement = potential_change / (0.2 + offset) < max_displacement
        # Then the bounding event rate is 0.2 + offset
        potential_change = max_displacement * (0.2 + offset) / 2
        random_expovariate_mock.return_value = potential_change
        displacement = potential_change / (0.2 + offset)
        assert displacement < max_displacement
        return displacement

    def _setUpSendOutStateAccept(self, random_mock_uniform, potential_mock):
        # bounding event rate is 0.2 + offset
        potential_mock.derivative.side_effect = [0.2]
        random_mock_uniform.return_value = 0.1

    def _setUpSendOutStateReject(self, random_mock_uniform, offset, potential_mock):
        # bounding event rate is 0.2 + offset
        potential_mock.derivative.side_effect = [0.2]
        random_mock_uniform.return_value = 0.2 + offset / 2

    def _setUpSendEventTimeMaxDisplacement(self, random_expovariate_mock, max_displacement, offset, potential_mock):
        potential_mock.derivative.side_effect = [0.1, 0.2]
        # Makes sure that potential_change / (0.2 + offset) > max_displacement
        # Displacement will then be max_displacement and events will always be rejected
        potential_change = max_displacement * (0.2 + offset) * 2
        random_expovariate_mock.return_value = potential_change

    def _setUpSendEventTimeMaxDisplacementDerivativeNegative(self, random_expovariate_mock, potential_mock):
        # The maximum of these derivatives plus all offsets above will always be smaller than zero
        # Displacement will then be max_displacement and events will always be rejected
        potential_mock.derivative.side_effect = [-0.3, -0.2]
        # This value doesn't matter
        random_expovariate_mock.return_value = None

    def test_send_event_time_without_charge_smaller_max_displacement(self, random_expovariate_mock, _):
        displacement = self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.1, 0.05,
                                                                      self._potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.3), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))

        event_time = self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        # First call has the active unit at its initial position, second call moved it by max_displacement
        calls = [mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - 0.9 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - (0.9 + 0.1) % 1.0 + 0.5) % 1.0 - 0.5])]
        self._potential_mock_without_charge.derivative.assert_has_calls(calls)
        self.assertAlmostEqual(event_time, 1.3 + displacement / 1.0)

    def test_send_event_time_without_charge_max_displacement(self, random_expovariate_mock, _):
        self._setUpSendEventTimeMaxDisplacement(random_expovariate_mock, 0.1, 0.05, self._potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.3), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))

        event_time = self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        # First call has the active unit at its initial position, second call moved it by max_displacement
        calls = [mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - 0.9 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - (0.9 + 0.1) % 1.0 + 0.5) % 1.0 - 0.5])]
        self._potential_mock_without_charge.derivative.assert_has_calls(calls)
        self.assertAlmostEqual(event_time, 1.3 + 0.1 / 1.0)

    def test_send_event_time_without_charge_max_displacement_derivative_negative(self, random_expovariate_mock, _):
        self._setUpSendEventTimeMaxDisplacementDerivativeNegative(random_expovariate_mock,
                                                                  self._potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.3), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))

        event_time = self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        # First call has the active unit at its initial position, second call moved it by max_displacement
        calls = [mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - 0.9 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - (0.9 + 0.1) % 1.0 + 0.5) % 1.0 - 0.5])]
        self._potential_mock_without_charge.derivative.assert_has_calls(calls)
        self.assertAlmostEqual(event_time, 1.3 + 0.1 / 1.0)

    def test_send_out_state_without_charge_smaller_max_displacement_accept(self, random_expovariate_mock,
                                                                           random_uniform_mock):
        displacement = self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.1, 0.05,
                                                                      self._potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.3), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateAccept(random_uniform_mock, self._potential_mock_without_charge)
        out_state = self._event_handler_without_charge.send_out_state()
        new_position = (0.9 + displacement) % 1.0
        calls = [mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - 0.9 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - (0.9 + 0.1) % 1.0 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - new_position + 0.5) % 1.0 - 0.5])]
        self._potential_mock_without_charge.derivative.assert_has_calls(calls)
        random_uniform_mock.assert_called_once_with(0, 0.2 + 0.05)

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [0.2, (0.8 + displacement / 2) % 1.0])
        self.assertEqual(first_cnode.weight, 1)
        self.assertIsNone(first_cnode.value.velocity)
        self.assertIsNone(first_cnode.value.time_stamp)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 2))
        self.assertEqual(first_child.value.position, [0.5, (0.9 + displacement) % 1.0])
        self.assertEqual(first_child.weight, 0.5)
        self.assertIsNone(first_child.value.velocity)
        self.assertIsNone(first_child.value.time_stamp)
        self.assertIsNone(first_child.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (3,))
        self.assertEqual(second_cnode.value.position, [0.5, 0.6])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(second_cnode.value.time_stamp, 1.3 + displacement / 1.0)
        self.assertIsNone(second_cnode.value.charge)
        self.assertEqual(len(second_cnode.children), 1)
        second_child = second_cnode.children[0]
        self.assertIs(second_child.parent, second_cnode)
        self.assertEqual(second_child.children, [])
        self.assertEqual(second_child.value.identifier, (3, 1))
        self.assertEqual(second_child.value.position, [0.1, 0.3])
        self.assertEqual(second_child.weight, 0.5)
        self.assertEqual(second_child.value.velocity, [0.0, 1.0])
        self.assertEqual(second_child.value.time_stamp, 1.3 + displacement / 1.0)
        self.assertIsNone(second_child.value.charge)

    def test_send_out_state_without_charge_smaller_max_displacement_reject(self, random_expovariate_mock,
                                                                           random_uniform_mock):
        displacement = self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.1, 0.05,
                                                                      self._potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.3), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateReject(random_uniform_mock, 0.05, self._potential_mock_without_charge)
        out_state = self._event_handler_without_charge.send_out_state()
        new_position = (0.9 + displacement) % 1.0
        calls = [mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - 0.9 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - (0.9 + 0.1) % 1.0 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - new_position + 0.5) % 1.0 - 0.5])]
        self._potential_mock_without_charge.derivative.assert_has_calls(calls)
        random_uniform_mock.assert_called_once_with(0, 0.2 + 0.05)

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [0.2, (0.8 + displacement / 2) % 1.0])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(first_cnode.value.time_stamp, 1.3 + displacement / 1.0)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 2))
        self.assertEqual(first_child.value.position, [0.5, (0.9 + displacement) % 1.0])
        self.assertEqual(first_child.weight, 0.5)
        self.assertEqual(first_child.value.velocity, [0.0, 1.0])
        self.assertEqual(first_child.value.time_stamp, 1.3 + displacement / 1.0)
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

    def test_send_out_state_without_charge_max_displacement(self, random_expovariate_mock, random_uniform_mock):
        # Event is always rejected, displacement is max_displacement
        self._setUpSendEventTimeMaxDisplacement(random_expovariate_mock, 0.1, 0.05, self._potential_mock_without_charge)
        displacement = 0.1
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.3), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        out_state = self._event_handler_without_charge.send_out_state()
        calls = [mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - 0.9 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - (0.9 + 0.1) % 1.0 + 0.5) % 1.0 - 0.5])]
        self._potential_mock_without_charge.derivative.assert_has_calls(calls)
        random_uniform_mock.assert_not_called()

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [0.2, (0.8 + displacement / 2) % 1.0])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(first_cnode.value.time_stamp, 1.3 + displacement / 1.0)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 2))
        self.assertEqual(first_child.value.position, [0.5, (0.9 + displacement) % 1.0])
        self.assertEqual(first_child.weight, 0.5)
        self.assertEqual(first_child.value.velocity, [0.0, 1.0])
        self.assertEqual(first_child.value.time_stamp, 1.3 + displacement / 1.0)
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

    def test_send_out_state_without_charge_max_displacement_derivative_negative(self, random_expovariate_mock,
                                                                                random_uniform_mock):
        # Event is always rejected, displacement is max_displacement
        self._setUpSendEventTimeMaxDisplacementDerivativeNegative(random_expovariate_mock,
                                                                  self._potential_mock_without_charge)
        displacement = 0.1
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.3), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        out_state = self._event_handler_without_charge.send_out_state()
        calls = [mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - 0.9 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - (0.9 + 0.1) % 1.0 + 0.5) % 1.0 - 0.5])]
        self._potential_mock_without_charge.derivative.assert_has_calls(calls)
        random_uniform_mock.assert_not_called()

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [0.2, (0.8 + displacement / 2) % 1.0])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(first_cnode.value.time_stamp, 1.3 + displacement / 1.0)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 2))
        self.assertEqual(first_child.value.position, [0.5, (0.9 + displacement) % 1.0])
        self.assertEqual(first_child.weight, 0.5)
        self.assertEqual(first_child.value.velocity, [0.0, 1.0])
        self.assertEqual(first_child.value.time_stamp, 1.3 + displacement / 1.0)
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

    def test_send_event_time_with_charge_smaller_max_displacement(self, random_expovariate_mock, _):
        displacement = self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.2, 0.1,
                                                                      self._potential_mock_with_charge)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[2.0, 0.0], time_stamp=1.3,
                                 charge={"charge": 3.4}), weight=1)

        event_time = self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        # First call has the active unit at its initial position, second call moved it by max_displacement
        calls = [mock.call(0, [(0.5 - 0.5 + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4),
                 mock.call(0, [(0.5 - (0.5 + 0.2) % 1.0 + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4)]
        self._potential_mock_with_charge.derivative.assert_has_calls(calls)
        self.assertAlmostEqual(event_time, 1.3 + displacement / 2.0)

    def test_send_event_time_with_charge_max_displacement(self, random_expovariate_mock, _):
        self._setUpSendEventTimeMaxDisplacement(random_expovariate_mock, 0.2, 0.1, self._potential_mock_with_charge)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[2.0, 0.0], time_stamp=1.3,
                                 charge={"charge": 3.4}), weight=1)

        event_time = self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        # First call has the active unit at its initial position, second call moved it by max_displacement
        calls = [mock.call(0, [(0.5 - 0.5 + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4),
                 mock.call(0, [(0.5 - (0.5 + 0.2) % 1.0 + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4)]
        self._potential_mock_with_charge.derivative.assert_has_calls(calls)
        self.assertAlmostEqual(event_time, 1.3 + 0.2 / 2.0)

    def test_send_event_time_with_charge_max_displacement_derivative_negative(self, random_expovariate_mock, _):
        self._setUpSendEventTimeMaxDisplacementDerivativeNegative(random_expovariate_mock,
                                                                  self._potential_mock_with_charge)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[2.0, 0.0], time_stamp=1.3,
                                 charge={"charge": 3.4}), weight=1)

        event_time = self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        # First call has the active unit at its initial position, second call moved it by max_displacement
        calls = [mock.call(0, [(0.5 - 0.5 + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4),
                 mock.call(0, [(0.5 - (0.5 + 0.2) % 1.0 + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4)]
        self._potential_mock_with_charge.derivative.assert_has_calls(calls)
        self.assertAlmostEqual(event_time, 1.3 + 0.2 / 2.0)

    def test_send_out_state_with_charge_smaller_max_displacement_accept(self, random_expovariate_mock,
                                                                        random_uniform_mock):
        displacement = self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.2, 0.1,
                                                                      self._potential_mock_with_charge)
        event_time = 1.3 + displacement / 2.0
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[2.0, 0.0], time_stamp=1.3,
                                 charge={"charge": 3.4}), weight=1)

        self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateAccept(random_uniform_mock, self._potential_mock_with_charge)
        out_state = self._event_handler_with_charge.send_out_state()
        new_position = (0.5 + (event_time - 1.3) * 2.0) % 1.0
        calls = [mock.call(0, [(0.5 - 0.5 + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4),
                 mock.call(0, [(0.5 - (0.5 + 0.2) % 1.0 + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4),
                 mock.call(0, [(0.5 - new_position + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4)]
        self._potential_mock_with_charge.derivative.assert_has_calls(calls)
        random_uniform_mock.assert_called_once_with(0, 0.2 + 0.1)

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.children, [])
        self.assertEqual(first_cnode.value.identifier, (3,))
        self.assertEqual(first_cnode.value.position, [0.5, 0.6])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.charge, {"charge": -1.2})
        self.assertEqual(first_cnode.value.velocity, [2.0, 0.0])
        self.assertEqual(first_cnode.value.time_stamp, event_time)
        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.children, [])
        self.assertEqual(second_cnode.value.identifier, (1,))
        self.assertEqual(second_cnode.value.position, [(0.5 + (event_time - 1.3) * 2.0) % 1.0, 0.9])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.charge, {"charge": 3.4})
        self.assertIsNone(second_cnode.value.velocity)
        self.assertIsNone(second_cnode.value.time_stamp)

    def test_send_out_state_with_charge_smaller_max_displacement_reject(self, random_expovariate_mock,
                                                                        random_uniform_mock):
        displacement = self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.2, 0.1,
                                                                      self._potential_mock_with_charge)
        event_time = 1.3 + displacement / 2.0
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[2.0, 0.0], time_stamp=1.3,
                                 charge={"charge": 3.4}), weight=1)

        self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateReject(random_uniform_mock, 0.1, self._potential_mock_with_charge)
        out_state = self._event_handler_with_charge.send_out_state()
        new_position = (0.5 + (event_time - 1.3) * 2.0) % 1.0
        calls = [mock.call(0, [(0.5 - 0.5 + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4),
                 mock.call(0, [(0.5 - (0.5 + 0.2) % 1.0 + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4),
                 mock.call(0, [(0.5 - new_position + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4)]
        self._potential_mock_with_charge.derivative.assert_has_calls(calls)
        random_uniform_mock.assert_called_once_with(0, 0.2 + 0.1)

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
        self.assertEqual(second_cnode.value.position, [(0.5 + (event_time - 1.3) * 2.0) % 1.0, 0.9])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.charge, {"charge": 3.4})
        self.assertEqual(second_cnode.value.velocity, [2.0, 0.0])
        self.assertEqual(second_cnode.value.time_stamp, event_time)

    def test_send_out_state_with_charge_max_displacement(self, random_expovariate_mock, random_uniform_mock):
        # Event is always rejected, displacement is max_displacement
        self._setUpSendEventTimeMaxDisplacement(random_expovariate_mock, 0.2, 0.1, self._potential_mock_with_charge)
        displacement = 0.2
        event_time = 1.3 + displacement / 2.0
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[2.0, 0.0], time_stamp=1.3,
                                 charge={"charge": 3.4}), weight=1)

        self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])

        out_state = self._event_handler_with_charge.send_out_state()
        calls = [mock.call(0, [(0.5 - 0.5 + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4),
                 mock.call(0, [(0.5 - (0.5 + 0.2) % 1.0 + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4)]
        self._potential_mock_with_charge.derivative.assert_has_calls(calls)
        random_uniform_mock.assert_not_called()

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
        self.assertEqual(second_cnode.value.position, [(0.5 + (event_time - 1.3) * 2.0) % 1.0, 0.9])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.charge, {"charge": 3.4})
        self.assertEqual(second_cnode.value.velocity, [2.0, 0.0])
        self.assertEqual(second_cnode.value.time_stamp, event_time)

    def test_send_out_state_with_charge_max_displacement_derivative_negative(self, random_expovariate_mock,
                                                                             random_uniform_mock):
        # Event is always rejected, displacement is max_displacement
        self._setUpSendEventTimeMaxDisplacementDerivativeNegative(random_expovariate_mock,
                                                                  self._potential_mock_with_charge)
        displacement = 0.2
        event_time = 1.3 + displacement / 2.0
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[2.0, 0.0], time_stamp=1.3,
                                 charge={"charge": 3.4}), weight=1)

        self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])

        out_state = self._event_handler_with_charge.send_out_state()
        calls = [mock.call(0, [(0.5 - 0.5 + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4),
                 mock.call(0, [(0.5 - (0.5 + 0.2) % 1.0 + 0.5) % 1.0 - 0.5, (0.6 - 0.9 + 0.5) % 1.0 - 0.5], -1.2, 3.4)]
        self._potential_mock_with_charge.derivative.assert_has_calls(calls)
        random_uniform_mock.assert_not_called()

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
        self.assertEqual(second_cnode.value.position, [(0.5 + (event_time - 1.3) * 2.0) % 1.0, 0.9])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.charge, {"charge": 3.4})
        self.assertEqual(second_cnode.value.velocity, [2.0, 0.0])
        self.assertEqual(second_cnode.value.time_stamp, event_time)

    def test_send_event_time_leaf_units_in_same_composite_object_smaller_max_displacement(
            self, random_expovariate_mock, _):
        displacement = self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.1, 0.05,
                                                                      self._potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.2), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 0), position=[0.1, 0.3]), weight=0.5))
        event_time = self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        calls = [mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - 0.9 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - (0.9 + 0.1) % 1.0 + 0.5) % 1.0 - 0.5])]
        self._potential_mock_without_charge.derivative.assert_has_calls(calls)
        self.assertAlmostEqual(event_time, 1.2 + displacement / 1.0)

    def test_send_event_time_leaf_units_in_same_composite_object_max_displacement(self, random_expovariate_mock, _):
        self._setUpSendEventTimeMaxDisplacement(random_expovariate_mock, 0.1, 0.05,
                                                self._potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.2), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 0), position=[0.1, 0.3]), weight=0.5))
        event_time = self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        calls = [mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - 0.9 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - (0.9 + 0.1) % 1.0 + 0.5) % 1.0 - 0.5])]
        self._potential_mock_without_charge.derivative.assert_has_calls(calls)
        self.assertAlmostEqual(event_time, 1.2 + 0.1 / 1.0)

    def test_send_event_time_leaf_units_in_same_composite_object_max_displacement_derivative_negative(
            self, random_expovariate_mock, _):
        self._setUpSendEventTimeMaxDisplacementDerivativeNegative(random_expovariate_mock,
                                                                  self._potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.2), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 0), position=[0.1, 0.3]), weight=0.5))
        event_time = self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        calls = [mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - 0.9 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - (0.9 + 0.1) % 1.0 + 0.5) % 1.0 - 0.5])]
        self._potential_mock_without_charge.derivative.assert_has_calls(calls)
        self.assertAlmostEqual(event_time, 1.2 + 0.1 / 1.0)

    def test_send_out_state_leaf_units_in_same_composite_object_accept(self, random_expovariate_mock,
                                                                       random_uniform_mock):
        displacement = self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.1, 0.05,
                                                                      self._potential_mock_without_charge)
        event_time = 1.2 + displacement / 1.0
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.2), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 0), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateAccept(random_uniform_mock, self._potential_mock_without_charge)
        out_state = self._event_handler_without_charge.send_out_state()
        new_position = (0.9 + displacement) % 1.0
        calls = [mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - 0.9 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - (0.9 + 0.1) % 1.0 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - new_position + 0.5) % 1.0 - 0.5])]
        self._potential_mock_without_charge.derivative.assert_has_calls(calls)
        random_uniform_mock.assert_called_once_with(0, 0.2 + 0.05)

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [0.2, (0.8 + displacement / 2) % 1.0])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(first_cnode.value.time_stamp, event_time)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 1))
        self.assertEqual(first_child.value.position, [0.5, (0.9 + displacement) % 1.0])
        self.assertEqual(first_child.weight, 0.5)
        self.assertIsNone(first_child.value.velocity)
        self.assertIsNone(first_child.value.time_stamp)
        self.assertIsNone(first_child.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (0,))
        self.assertEqual(second_cnode.value.position, [0.2, (0.8 + displacement / 2) % 1.0])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(second_cnode.value.time_stamp, event_time)
        self.assertIsNone(second_cnode.value.charge)
        self.assertEqual(len(second_cnode.children), 1)
        second_child = second_cnode.children[0]
        self.assertIs(second_child.parent, second_cnode)
        self.assertEqual(second_child.children, [])
        self.assertEqual(second_child.value.identifier, (0, 0))
        self.assertEqual(second_child.value.position, [0.1, 0.3])
        self.assertEqual(second_child.weight, 0.5)
        self.assertEqual(second_child.value.velocity, [0.0, 1.0])
        self.assertEqual(second_child.value.time_stamp, event_time)
        self.assertIsNone(second_child.value.charge)

    def test_send_out_state_leaf_units_in_same_composite_object_reject(self, random_expovariate_mock,
                                                                       random_uniform_mock):
        displacement = self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.1, 0.05,
                                                                      self._potential_mock_without_charge)
        event_time = 1.2 + displacement / 1.0
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.2), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 0), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        self._setUpSendOutStateReject(random_uniform_mock, 0.05, self._potential_mock_without_charge)
        out_state = self._event_handler_without_charge.send_out_state()
        new_position = (0.9 + displacement) % 1.0
        calls = [mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - 0.9 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - (0.9 + 0.1) % 1.0 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - new_position + 0.5) % 1.0 - 0.5])]
        self._potential_mock_without_charge.derivative.assert_has_calls(calls)
        random_uniform_mock.assert_called_once_with(0, 0.2 + 0.05)

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [0.2, (0.8 + displacement / 2) % 1.0])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(first_cnode.value.time_stamp, event_time)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 1))
        self.assertEqual(first_child.value.position, [0.5, (0.9 + displacement) % 1.0])
        self.assertEqual(first_child.weight, 0.5)
        self.assertEqual(first_child.value.velocity, [0.0, 1.0])
        self.assertEqual(first_child.value.time_stamp, event_time)
        self.assertIsNone(first_child.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (0,))
        self.assertEqual(second_cnode.value.position, [0.2, (0.8 + displacement / 2) % 1.0])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(second_cnode.value.time_stamp, event_time)
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

    def test_send_out_state_leaf_units_in_same_composite_object_max_displacement(self, random_expovariate_mock,
                                                                                 random_uniform_mock):
        # Event is always rejected, displacement is max_displacement
        self._setUpSendEventTimeMaxDisplacement(random_expovariate_mock, 0.1, 0.05,
                                                self._potential_mock_without_charge)
        displacement = 0.1
        event_time = 1.2 + displacement / 1.0
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.2), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 0), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        out_state = self._event_handler_without_charge.send_out_state()
        calls = [mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - 0.9 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - (0.9 + 0.1) % 1.0 + 0.5) % 1.0 - 0.5])]
        self._potential_mock_without_charge.derivative.assert_has_calls(calls)
        random_uniform_mock.assert_not_called()

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [0.2, (0.8 + displacement / 2) % 1.0])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(first_cnode.value.time_stamp, event_time)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 1))
        self.assertEqual(first_child.value.position, [0.5, (0.9 + displacement) % 1.0])
        self.assertEqual(first_child.weight, 0.5)
        self.assertEqual(first_child.value.velocity, [0.0, 1.0])
        self.assertEqual(first_child.value.time_stamp, event_time)
        self.assertIsNone(first_child.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (0,))
        self.assertEqual(second_cnode.value.position, [0.2, (0.8 + displacement / 2) % 1.0])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(second_cnode.value.time_stamp, event_time)
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

    def test_send_out_state_leaf_units_in_same_composite_object_max_displacement_derivative_negative(
            self, random_expovariate_mock, random_uniform_mock):
        # Event is always rejected, displacement is max_displacement
        self._setUpSendEventTimeMaxDisplacementDerivativeNegative(random_expovariate_mock,
                                                                  self._potential_mock_without_charge)
        displacement = 0.1
        event_time = 1.2 + displacement / 1.0
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 1), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.2), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 0), position=[0.1, 0.3]), weight=0.5))
        self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

        out_state = self._event_handler_without_charge.send_out_state()
        calls = [mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - 0.9 + 0.5) % 1.0 - 0.5]),
                 mock.call(1, [(0.1 - 0.5 + 0.5) % 1.0 - 0.5, (0.3 - (0.9 + 0.1) % 1.0 + 0.5) % 1.0 - 0.5])]
        self._potential_mock_without_charge.derivative.assert_has_calls(calls)
        random_uniform_mock.assert_not_called()

        self.assertEqual(len(out_state), 2)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [0.2, (0.8 + displacement / 2) % 1.0])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(first_cnode.value.time_stamp, event_time)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 1))
        self.assertEqual(first_child.value.position, [0.5, (0.9 + displacement) % 1.0])
        self.assertEqual(first_child.weight, 0.5)
        self.assertEqual(first_child.value.velocity, [0.0, 1.0])
        self.assertEqual(first_child.value.time_stamp, event_time)
        self.assertIsNone(first_child.value.charge)

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (0,))
        self.assertEqual(second_cnode.value.position, [0.2, (0.8 + displacement / 2) % 1.0])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [0.0, 0.5])
        self.assertEqual(second_cnode.value.time_stamp, event_time)
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

    def test_potential_with_wrong_number_of_separation_arguments_raises_error(self, _, __):
        potential_mock = mock.MagicMock(spec_set=Potential)
        potential_mock.number_separation_arguments = 2
        potential_mock.number_charge_arguments = 2
        with self.assertRaises(ConfigurationError):
            TwoLeafUnitEventHandlerWithPiecewiseConstantBoundingPotential(potential=potential_mock, offset=0.05,
                                                                          max_displacement=0.1)

    def test_potential_with_wrong_number_of_charge_arguments_raises_error(self, _, __):
        potential_mock = mock.MagicMock(spec_set=Potential)
        potential_mock.number_separation_arguments = 2
        potential_mock.number_charge_arguments = 1
        with self.assertRaises(ConfigurationError):
            TwoLeafUnitEventHandlerWithPiecewiseConstantBoundingPotential(potential=potential_mock, offset=0.05,
                                                                          max_displacement=0.1)

    def test_missing_charges_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.2, 0.1,
                                                       self._potential_mock_with_charge)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6],
                                 charge={"other_charge": -1.2}), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9],
                                 velocity=[2.0, 0.0], time_stamp=1.3,
                                 charge={"other_charge": 3.4}), weight=1)
        with self.assertRaises(KeyError):
            self._event_handler_with_charge.send_event_time([in_state_one, in_state_two])

    def test_in_state_too_short_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.1, 0.05,
                                                       self._potential_mock_without_charge)
        in_state = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[2.0, 0.0],
                             time_stamp=0.7), weight=1)
        in_state.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                     velocity=[2.0, 0.0], time_stamp=0.7), weight=0.5))
        in_state.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.8],
                                     velocity=[2.0, 0.0], time_stamp=0.7), weight=0.5))
        with self.assertRaises(AssertionError):
            self._event_handler_without_charge.send_event_time([in_state])

    def test_in_state_too_long_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.1, 0.05,
                                                       self._potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9],
                                 velocity=[2.0, 0.0], time_stamp=1.3), weight=1)
        in_state_three = Node(Unit(identifier=(4,), position=[0.6, 0.5]), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler_without_charge.send_event_time([in_state_one, in_state_two, in_state_three])

    def test_no_leaf_unit_active_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.1, 0.05,
                                                       self._potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9]), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

    def test_both_leaf_units_active_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.1, 0.05,
                                                       self._potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6], velocity=[1.0, 0.0],
                                 time_stamp=0.0), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[1.0, 0.0],
                                 time_stamp=0.0), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

    def test_more_than_two_leaf_units_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.1, 0.05,
                                                       self._potential_mock_without_charge)
        in_state_one = Node(Unit(identifier=(1,), position=[0.2, 0.4], velocity=[1.0, 0.0],
                                 time_stamp=0.7), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(1, 0), position=[0.5, 0.6],
                                         velocity=[2.0, 0.0], time_stamp=0.7), weight=0.5))
        in_state_one.add_child(Node(Unit(identifier=(1, 1), position=[0.3, 0.8]), weight=0.5))
        in_state_two = Node(Unit(identifier=(2,), position=[0.2, 0.4], velocity=[1.0, 0.0],
                                 time_stamp=0.7), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(2, 0), position=[0.5, 0.6],
                                         velocity=[2.0, 0.0], time_stamp=0.7), weight=0.5))
        in_state_two.add_child(Node(Unit(identifier=(2, 1), position=[0.3, 0.8]), weight=0.5))
        with self.assertRaises(AssertionError):
            self._event_handler_without_charge.send_event_time([in_state_one, in_state_two])

    def test_max_displacement_zero_raises_error(self, _, __):
        with self.assertRaises(ConfigurationError):
            TwoLeafUnitEventHandlerWithPiecewiseConstantBoundingPotential(potential=self._potential_mock_without_charge,
                                                                          offset=0.05, max_displacement=0.0)

    def test_max_displacement_negative_raises_error(self, _, __):
        with self.assertRaises(ConfigurationError):
            TwoLeafUnitEventHandlerWithPiecewiseConstantBoundingPotential(potential=self._potential_mock_without_charge,
                                                                          offset=0.05, max_displacement=-0.1)


if __name__ == '__main__':
    main()
