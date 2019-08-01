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
from event_handler.fixed_separations_event_handler_with_piecewise_constant_bounding_potential import \
    FixedSeparationsEventHandlerWithPiecewiseConstantBoundingPotential
from lifting import Lifting
from potential import Potential
import setting


@mock.patch("event_handler.abstracts.event_handler_with_bounding_potential.random.uniform")
@mock.patch("event_handler.fixed_separations_event_handler_with_piecewise_constant_bounding_potential"
            ".random.expovariate")
class TestFixedSeparationsEventHandlerWithPiecewiseConstantBoundingPotential(TestCase):
    def setUp(self) -> None:
        setting.HypercubicSetting(beta=1.0, dimension=2, system_length=1.0)
        self._lifting_mock = mock.MagicMock(spec_set=Lifting)
        self._potential_mock = mock.MagicMock(spec_set=Potential)
        self._potential_mock.number_separation_arguments = 2
        self._potential_mock.number_charge_arguments = 2
        self._event_handler = FixedSeparationsEventHandlerWithPiecewiseConstantBoundingPotential(
            potential=self._potential_mock, lifting=self._lifting_mock, offset=0.1, max_displacement=0.2,
            separations=[1, 0, 1, 2])

    def tearDown(self) -> None:
        setting.reset()

    def _setUpSendEventTimeSmallerMaxDisplacement(self, random_expovariate_mock, max_displacement, offset,
                                                  active_index):
        # Bending potential returns three derivatives on each call but only the one at the active index is compared
        first_return_list = [0.1, -0.3, 0.2]
        # Rotates the return list so that 0.1 is the value at active index
        first_return_list = first_return_list[-active_index:] + first_return_list[:-active_index]
        second_return_list = [0.2, 0.4, -0.6]
        # Rotate the return list so that 0.2 is the value at active index
        second_return_list = second_return_list[-active_index:] + second_return_list[:-active_index]
        self._potential_mock.derivative.side_effect = [first_return_list, second_return_list]
        # Makes sure that displacement = potential_change / (0.2 + offset) < max_displacement
        # Then the bounding event rate is 0.2 + offset
        potential_change = max_displacement * (0.2 + offset) / 2
        random_expovariate_mock.return_value = potential_change
        displacement = potential_change / (0.2 + offset)
        assert displacement < max_displacement
        return displacement

    def _setUpSendOutStateAccept(self, random_mock_uniform, active_index):
        # bounding event rate is 0.2 + offset
        return_list = [0.2, -0.7, 0.5]
        # Rotates the return list so that 0.2 is the value at active index
        return_list = return_list[-active_index:] + return_list[:-active_index]
        self._potential_mock.derivative.side_effect = [return_list]
        random_mock_uniform.return_value = 0.1

    def _setUpSendOutStateReject(self, random_mock_uniform, offset, active_index):
        # bounding event rate is 0.2 + offset
        return_list = [0.2, -0.7, 0.5]
        # Rotates the return list so that 0.2 is the value at active index
        return_list = return_list[-active_index:] + return_list[:-active_index]
        self._potential_mock.derivative.side_effect = [return_list]
        random_mock_uniform.return_value = 0.2 + offset / 2

    def _setUpSendEventTimeMaxDisplacement(self, random_expovariate_mock, max_displacement, offset, active_index):
        # Bending potential returns three derivatives on each call but only the one at the active index is compared
        first_return_list = [0.1, -0.3, 0.2]
        # Rotates the return list so that 0.1 is the value at active index
        first_return_list = first_return_list[-active_index:] + first_return_list[:-active_index]
        second_return_list = [0.2, 0.4, -0.6]
        # Rotate the return list so that 0.2 is the value at active index
        second_return_list = second_return_list[-active_index:] + second_return_list[:-active_index]
        self._potential_mock.derivative.side_effect = [first_return_list, second_return_list]
        # Makes sure that potential_change / (0.2 + offset) > max_displacement
        # Displacement will then be max_displacement and events will always be rejected
        potential_change = max_displacement * (0.2 + offset) * 2
        random_expovariate_mock.return_value = potential_change

    def _setUpSendEventTimeMaxDisplacementDerivativeNegative(self, random_expovariate_mock, active_index):
        # Bending potential returns three derivatives on each call but only the one at the active index is compared
        first_return_list = [-0.3, -0.3, 0.2]
        # Rotates the return list so that -0.3 is the value at active index
        first_return_list = first_return_list[-active_index:] + first_return_list[:-active_index]
        second_return_list = [-0.2, 0.4, -0.6]
        # Rotate the return list so that -0.2 is the value at active index
        second_return_list = second_return_list[-active_index:] + second_return_list[:-active_index]
        # The maximum of -0.3 and -0.2 plus all offsets above will always be smaller than zero
        # Displacement will then be max_displacement and events will always be rejected
        self._potential_mock.derivative.side_effect = [first_return_list, second_return_list]
        # This value doesn't matter
        random_expovariate_mock.return_value = None

    def test_send_event_time_smaller_max_displacement(self, random_expovariate_mock, _):
        displacement = self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.2, 0.1, 0)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.3), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))

        in_state_three = Node(Unit(identifier=(7,), position=[0.4, 0.2]), weight=1)
        in_state_three.add_child(Node(Unit(identifier=(7, 0), position=[0.8, 0.7]), weight=0.5))

        event_time = self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        # First call has the active unit at its initial position, second call moved it by max_displacement
        # Separations are calculated according to initially set separations array
        calls = [mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                           [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                 mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, ((0.9 + 0.2) % 1.0 - 0.3 + 0.5) % 1.0 - 0.5],
                           [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0)]
        self._potential_mock.derivative.assert_has_calls(calls)
        self.assertAlmostEqual(event_time, 1.3 + displacement / 1.0)

    def test_send_event_time_max_displacement(self, random_expovariate_mock, _):
        self._setUpSendEventTimeMaxDisplacement(random_expovariate_mock, 0.2, 0.1, 0)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.3), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))

        in_state_three = Node(Unit(identifier=(7,), position=[0.4, 0.2]), weight=1)
        in_state_three.add_child(Node(Unit(identifier=(7, 0), position=[0.8, 0.7]), weight=0.5))

        event_time = self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        # First call has the active unit at its initial position, second call moved it by max_displacement
        # Separations are calculated according to initially set separations array
        calls = [mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                           [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                 mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, ((0.9 + 0.2) % 1.0 - 0.3 + 0.5) % 1.0 - 0.5],
                           [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0)]
        self._potential_mock.derivative.assert_has_calls(calls)
        self.assertAlmostEqual(event_time, 1.3 + 0.2 / 1.0)

    def test_send_event_time_max_displacement_derivative_negative(self, random_expovariate_mock, _):
        self._setUpSendEventTimeMaxDisplacementDerivativeNegative(random_expovariate_mock, 0)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.3), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))

        in_state_three = Node(Unit(identifier=(7,), position=[0.4, 0.2]), weight=1)
        in_state_three.add_child(Node(Unit(identifier=(7, 0), position=[0.8, 0.7]), weight=0.5))

        event_time = self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        # First call has the active unit at its initial position, second call moved it by max_displacement
        # Separations are calculated according to initially set separations array
        calls = [mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                           [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                 mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, ((0.9 + 0.2) % 1.0 - 0.3 + 0.5) % 1.0 - 0.5],
                           [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0)]
        self._potential_mock.derivative.assert_has_calls(calls)
        self.assertAlmostEqual(event_time, 1.3 + 0.2 / 1.0)

    def test_send_out_state_smaller_max_displacement_accept(self, random_expovariate_mock, random_uniform_mock):
        displacement = self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.2, 0.1, 0)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.3), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))

        in_state_three = Node(Unit(identifier=(7,), position=[0.4, 0.2]), weight=1)
        in_state_three.add_child(Node(Unit(identifier=(7, 0), position=[0.8, 0.7]), weight=0.5))
        
        self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])

        self._setUpSendOutStateAccept(random_uniform_mock, 0)

        # We want this as the new active leaf unit
        self._lifting_mock.get_active_identifier.return_value = (3, 1)

        out_state = self._event_handler.send_out_state()

        new_position = (0.9 + displacement) % 1.0

        # Potential derivative is called with active leaf unit at initial position,
        # active leaf unit moved by max_displacement and active leaf unit moved by the proposed displacement
        potential_calls = [mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                           mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, ((0.9 + 0.2) % 1.0 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                           mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (new_position - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0)]
        self._potential_mock.derivative.assert_has_calls(potential_calls)
        random_uniform_mock.assert_called_once_with(0, 0.2 + 0.1)
        # lifting.insert is called with the derivatives returned when active leaf unit is moved by proposed displacement
        lifting_calls = [mock.call(0.2, (0, 2), True),
                         mock.call(-0.7, (3, 1), False),
                         mock.call(0.5, (7, 0), False)]
        self._lifting_mock.reset.assert_called_once_with()
        self._lifting_mock.insert.assert_has_calls(lifting_calls)
        self._lifting_mock.get_active_identifier.assert_called_once_with()

        self.assertEqual(len(out_state), 3)
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

        third_cnode = out_state[2]
        self.assertIsNone(third_cnode.parent)
        self.assertEqual(third_cnode.value.identifier, (7,))
        self.assertEqual(third_cnode.value.position, [0.4, 0.2])
        self.assertEqual(third_cnode.weight, 1)
        self.assertIsNone(third_cnode.value.velocity)
        self.assertIsNone(third_cnode.value.time_stamp)
        self.assertIsNone(third_cnode.value.charge)
        self.assertEqual(len(third_cnode.children), 1)
        third_child = third_cnode.children[0]
        self.assertIs(third_child.parent, third_cnode)
        self.assertEqual(third_child.children, [])
        self.assertEqual(third_child.value.identifier, (7, 0))
        self.assertEqual(third_child.value.position, [0.8, 0.7])
        self.assertEqual(third_child.weight, 0.5)
        self.assertIsNone(third_child.value.velocity)
        self.assertIsNone(third_child.value.time_stamp)
        self.assertIsNone(third_child.value.charge)

    def test_send_out_state_smaller_max_displacement_reject(self, random_expovariate_mock, random_uniform_mock):
        displacement = self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.2, 0.1, 0)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.3), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))

        in_state_three = Node(Unit(identifier=(7,), position=[0.4, 0.2]), weight=1)
        in_state_three.add_child(Node(Unit(identifier=(7, 0), position=[0.8, 0.7]), weight=0.5))

        self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])

        self._setUpSendOutStateReject(random_uniform_mock, 0.1, 0)
        out_state = self._event_handler.send_out_state()
        new_position = (0.9 + displacement) % 1.0
        # Potential derivative is called with active leaf unit at initial position,
        # active leaf unit moved by max_displacement and active leaf unit moved by the proposed displacement
        potential_calls = [mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                           mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, ((0.9 + 0.2) % 1.0 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                           mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (new_position - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0)]
        self._potential_mock.derivative.assert_has_calls(potential_calls)
        random_uniform_mock.assert_called_once_with(0, 0.2 + 0.1)

        self._lifting_mock.reset.assert_not_called()
        self._lifting_mock.insert.assert_not_called()
        self._lifting_mock.get_active_identifier.assert_not_called()

        self.assertEqual(len(out_state), 3)
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

        third_cnode = out_state[2]
        self.assertIsNone(third_cnode.parent)
        self.assertEqual(third_cnode.value.identifier, (7,))
        self.assertEqual(third_cnode.value.position, [0.4, 0.2])
        self.assertEqual(third_cnode.weight, 1)
        self.assertIsNone(third_cnode.value.velocity)
        self.assertIsNone(third_cnode.value.time_stamp)
        self.assertIsNone(third_cnode.value.charge)
        self.assertEqual(len(third_cnode.children), 1)
        third_child = third_cnode.children[0]
        self.assertIs(third_child.parent, third_cnode)
        self.assertEqual(third_child.children, [])
        self.assertEqual(third_child.value.identifier, (7, 0))
        self.assertEqual(third_child.value.position, [0.8, 0.7])
        self.assertEqual(third_child.weight, 0.5)
        self.assertIsNone(third_child.value.velocity)
        self.assertIsNone(third_child.value.time_stamp)
        self.assertIsNone(third_child.value.charge)

    def test_send_out_state_max_displacement(self, random_expovariate_mock, random_uniform_mock):
        # Event is always rejected, displacement is max_displacement
        self._setUpSendEventTimeMaxDisplacement(random_expovariate_mock, 0.2, 0.1, 0)
        displacement = 0.2
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.3), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))

        in_state_three = Node(Unit(identifier=(7,), position=[0.4, 0.2]), weight=1)
        in_state_three.add_child(Node(Unit(identifier=(7, 0), position=[0.8, 0.7]), weight=0.5))

        self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])

        out_state = self._event_handler.send_out_state()
        potential_calls = [mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                           mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, ((0.9 + 0.2) % 1.0 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0)]
        self._potential_mock.derivative.assert_has_calls(potential_calls)
        random_uniform_mock.assert_not_called()

        self._lifting_mock.reset.assert_not_called()
        self._lifting_mock.insert.assert_not_called()
        self._lifting_mock.get_active_identifier.assert_not_called()

        self.assertEqual(len(out_state), 3)
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

        third_cnode = out_state[2]
        self.assertIsNone(third_cnode.parent)
        self.assertEqual(third_cnode.value.identifier, (7,))
        self.assertEqual(third_cnode.value.position, [0.4, 0.2])
        self.assertEqual(third_cnode.weight, 1)
        self.assertIsNone(third_cnode.value.velocity)
        self.assertIsNone(third_cnode.value.time_stamp)
        self.assertIsNone(third_cnode.value.charge)
        self.assertEqual(len(third_cnode.children), 1)
        third_child = third_cnode.children[0]
        self.assertIs(third_child.parent, third_cnode)
        self.assertEqual(third_child.children, [])
        self.assertEqual(third_child.value.identifier, (7, 0))
        self.assertEqual(third_child.value.position, [0.8, 0.7])
        self.assertEqual(third_child.weight, 0.5)
        self.assertIsNone(third_child.value.velocity)
        self.assertIsNone(third_child.value.time_stamp)
        self.assertIsNone(third_child.value.charge)

    def test_send_out_state_max_displacement_derivative_negative(self, random_expovariate_mock, random_uniform_mock):
        # Event is always rejected, displacement is max_displacement
        self._setUpSendEventTimeMaxDisplacementDerivativeNegative(random_expovariate_mock, 0)
        displacement = 0.2
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.0, 0.5],
                                 time_stamp=1.3), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 2), position=[0.5, 0.9],
                                         velocity=[0.0, 1.0], time_stamp=1.3), weight=0.5))

        in_state_two = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(3, 1), position=[0.1, 0.3]), weight=0.5))

        in_state_three = Node(Unit(identifier=(7,), position=[0.4, 0.2]), weight=1)
        in_state_three.add_child(Node(Unit(identifier=(7, 0), position=[0.8, 0.7]), weight=0.5))

        self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])
        out_state = self._event_handler.send_out_state()
        potential_calls = [mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                           mock.call(1, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, ((0.9 + 0.2) % 1.0 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.8 - 0.1 + 0.5) % 1.0 - 0.5, (0.7 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0)]
        self._potential_mock.derivative.assert_has_calls(potential_calls)
        random_uniform_mock.assert_not_called()

        self._lifting_mock.reset.assert_not_called()
        self._lifting_mock.insert.assert_not_called()
        self._lifting_mock.get_active_identifier.assert_not_called()

        self.assertEqual(len(out_state), 3)
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

        third_cnode = out_state[2]
        self.assertIsNone(third_cnode.parent)
        self.assertEqual(third_cnode.value.identifier, (7,))
        self.assertEqual(third_cnode.value.position, [0.4, 0.2])
        self.assertEqual(third_cnode.weight, 1)
        self.assertIsNone(third_cnode.value.velocity)
        self.assertIsNone(third_cnode.value.time_stamp)
        self.assertIsNone(third_cnode.value.charge)
        self.assertEqual(len(third_cnode.children), 1)
        third_child = third_cnode.children[0]
        self.assertIs(third_child.parent, third_cnode)
        self.assertEqual(third_child.children, [])
        self.assertEqual(third_child.value.identifier, (7, 0))
        self.assertEqual(third_child.value.position, [0.8, 0.7])
        self.assertEqual(third_child.weight, 0.5)
        self.assertIsNone(third_child.value.velocity)
        self.assertIsNone(third_child.value.time_stamp)
        self.assertIsNone(third_child.value.charge)

    def test_send_event_time_leaf_units_in_same_composite_object_smaller_max_displacement(
            self, random_expovariate_mock, _):
        displacement = self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.2, 0.1, 1)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 0), position=[0.5, 0.9],
                                         charge={"charge": -3.4}), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 1), position=[0.1, 0.3],
                                         velocity=[1.0, 0.0], time_stamp=1.2,
                                         charge={"charge": 1.2}), weight=0.5))

        in_state_three = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                   time_stamp=1.2), weight=1)
        in_state_three.add_child(Node(Unit(identifier=(0, 2), position=[0.7, 0.5],
                                           charge={"charge": -0.7}), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        new_position = (0.1 + 0.2) % 1.0
        potential_calls = [mock.call(0, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - 0.1 + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                           mock.call(0, [(0.5 - new_position + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - new_position + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0)]

        self._potential_mock.derivative.assert_has_calls(potential_calls)
        self.assertAlmostEqual(event_time, 1.2 + displacement / 1.0)

    def test_send_event_time_leaf_units_in_same_composite_object_max_displacement(self, random_expovariate_mock, _):
        self._setUpSendEventTimeMaxDisplacement(random_expovariate_mock, 0.2, 0.1, 1)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 0), position=[0.5, 0.9],
                                         charge={"charge": -3.4}), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 1), position=[0.1, 0.3],
                                         velocity=[1.0, 0.0], time_stamp=1.2,
                                         charge={"charge": 1.2}), weight=0.5))

        in_state_three = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                   time_stamp=1.2), weight=1)
        in_state_three.add_child(Node(Unit(identifier=(0, 2), position=[0.7, 0.5],
                                           charge={"charge": -0.7}), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        new_position = (0.1 + 0.2) % 1.0
        potential_calls = [mock.call(0, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - 0.1 + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                           mock.call(0, [(0.5 - new_position + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - new_position + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0)]
        self._potential_mock.derivative.assert_has_calls(potential_calls)
        self.assertAlmostEqual(event_time, 1.2 + 0.2 / 1.0)

    def test_send_event_time_leaf_units_in_same_composite_object_max_displacement_derivative_negative(
            self, random_expovariate_mock, _):
        self._setUpSendEventTimeMaxDisplacementDerivativeNegative(random_expovariate_mock, 1)
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 0), position=[0.5, 0.9],
                                         charge={"charge": -3.4}), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 1), position=[0.1, 0.3],
                                         velocity=[1.0, 0.0], time_stamp=1.2,
                                         charge={"charge": 1.2}), weight=0.5))

        in_state_three = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                   time_stamp=1.2), weight=1)
        in_state_three.add_child(Node(Unit(identifier=(0, 2), position=[0.7, 0.5],
                                           charge={"charge": -0.7}), weight=0.5))
        event_time = self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])
        # Should be called with beta
        random_expovariate_mock.assert_called_once_with(1)
        new_position = (0.1 + 0.2) % 1.0
        potential_calls = [mock.call(0, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - 0.1 + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                           mock.call(0, [(0.5 - new_position + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - new_position + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0)]
        self._potential_mock.derivative.assert_has_calls(potential_calls)
        self.assertAlmostEqual(event_time, 1.2 + 0.2 / 1.0)

    def test_send_out_state_leaf_units_in_same_composite_object_accept(self, random_expovariate_mock,
                                                                       random_uniform_mock):
        displacement = self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.2, 0.1, 1)
        event_time = 1.2 + displacement / 1.0
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 0), position=[0.5, 0.9],
                                         charge={"charge": -3.4}), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 1), position=[0.1, 0.3],
                                         velocity=[1.0, 0.0], time_stamp=1.2,
                                         charge={"charge": 1.2}), weight=0.5))

        in_state_three = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                   time_stamp=1.2), weight=1)
        in_state_three.add_child(Node(Unit(identifier=(0, 2), position=[0.7, 0.5],
                                           charge={"charge": -0.7}), weight=0.5))
        self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])

        self._setUpSendOutStateAccept(random_uniform_mock, 1)

        # We want this as the new active leaf unit
        self._lifting_mock.get_active_identifier.return_value = (0, 0)

        out_state = self._event_handler.send_out_state()

        new_position = (0.1 + (event_time - 1.2) * 1.0) % 1.0

        # Potential derivative is called with active leaf unit at initial position,
        # active leaf unit moved by max_displacement and active leaf unit moved by the proposed displacement
        potential_calls = [mock.call(0, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - 0.1 + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                           mock.call(0, [(0.5 - 0.1 - 0.2 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - 0.1 - 0.2 + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                           mock.call(0, [(0.5 - new_position + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - new_position + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0)]
        self._potential_mock.derivative.assert_has_calls(potential_calls)
        random_uniform_mock.assert_called_once_with(0, 0.2 + 0.1)
        # lifting.insert is called with the derivatives returned when active leaf unit is moved by proposed displacement
        lifting_calls = [mock.call(0.5, (0, 0), False),
                         mock.call(0.2, (0, 1), True),
                         mock.call(-0.7, (0, 2), False)]
        self._lifting_mock.reset.assert_called_once_with()
        self._lifting_mock.insert.assert_has_calls(lifting_calls)
        self._lifting_mock.get_active_identifier.assert_called_once_with()

        self.assertEqual(len(out_state), 3)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [(0.2 + (event_time - 1.2) * 0.5) % 1.0, 0.8])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.5, 0.0])
        self.assertEqual(first_cnode.value.time_stamp, event_time)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 0))
        self.assertEqual(first_child.value.position, [0.5, 0.9])
        self.assertEqual(first_child.weight, 0.5)
        self.assertEqual(first_child.value.velocity, [1.0, 0.0])
        self.assertEqual(first_child.value.time_stamp, event_time)
        self.assertEqual(first_child.value.charge, {"charge": -3.4})

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (0,))
        self.assertEqual(second_cnode.value.position, [(0.2 + (event_time - 1.2) * 0.5) % 1.0, 0.8])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [0.5, 0.0])
        self.assertEqual(second_cnode.value.time_stamp, event_time)
        self.assertIsNone(second_cnode.value.charge)
        self.assertEqual(len(second_cnode.children), 1)
        second_child = second_cnode.children[0]
        self.assertIs(second_child.parent, second_cnode)
        self.assertEqual(second_child.children, [])
        self.assertEqual(second_child.value.identifier, (0, 1))
        self.assertEqual(second_child.value.position, [(0.1 + (event_time - 1.2) * 1.0) % 1.0, 0.3])
        self.assertEqual(second_child.weight, 0.5)
        self.assertIsNone(second_child.value.velocity)
        self.assertIsNone(second_child.value.time_stamp)
        self.assertEqual(second_child.value.charge, {"charge": 1.2})

        third_cnode = out_state[2]
        self.assertIsNone(third_cnode.parent)
        self.assertEqual(third_cnode.value.identifier, (0,))
        self.assertEqual(third_cnode.value.position, [(0.2 + (event_time - 1.2) * 0.5) % 1.0, 0.8])
        self.assertEqual(third_cnode.weight, 1)
        self.assertEqual(third_cnode.value.velocity, [0.5, 0.0])
        self.assertEqual(third_cnode.value.time_stamp, event_time)
        self.assertIsNone(third_cnode.value.charge)
        self.assertEqual(len(third_cnode.children), 1)
        third_child = third_cnode.children[0]
        self.assertIs(third_child.parent, third_cnode)
        self.assertEqual(third_child.children, [])
        self.assertEqual(third_child.value.identifier, (0, 2))
        self.assertEqual(third_child.value.position, [0.7, 0.5])
        self.assertEqual(third_child.weight, 0.5)
        self.assertIsNone(third_child.value.velocity)
        self.assertIsNone(third_child.value.time_stamp)
        self.assertEqual(third_child.value.charge, {"charge": -0.7})

    def test_send_out_state_leaf_units_in_same_composite_object_reject(self, random_expovariate_mock,
                                                                       random_uniform_mock):
        displacement = self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.2, 0.1, 1)
        event_time = 1.2 + displacement / 1.0
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 0), position=[0.5, 0.9],
                                         charge={"charge": -3.4}), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 1), position=[0.1, 0.3],
                                         velocity=[1.0, 0.0], time_stamp=1.2,
                                         charge={"charge": 1.2}), weight=0.5))

        in_state_three = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                   time_stamp=1.2), weight=1)
        in_state_three.add_child(Node(Unit(identifier=(0, 2), position=[0.7, 0.5],
                                           charge={"charge": -0.7}), weight=0.5))
        self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])

        self._setUpSendOutStateReject(random_uniform_mock, 0.1, 1)
        out_state = self._event_handler.send_out_state()

        new_position = (0.1 + (event_time - 1.2) * 1.0) % 1.0

        # Potential derivative is called with active leaf unit at initial position,
        # active leaf unit moved by max_displacement and active leaf unit moved by the proposed displacement
        potential_calls = [mock.call(0, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - 0.1 + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                           mock.call(0, [(0.5 - 0.1 - 0.2 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - 0.1 - 0.2 + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                           mock.call(0, [(0.5 - new_position + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - new_position + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0)]
        self._potential_mock.derivative.assert_has_calls(potential_calls)
        random_uniform_mock.assert_called_once_with(0, 0.2 + 0.1)
        self._lifting_mock.reset.assert_not_called()
        self._lifting_mock.insert.assert_not_called()
        self._lifting_mock.get_active_identifier.assert_not_called()

        self.assertEqual(len(out_state), 3)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [(0.2 + (event_time - 1.2) * 0.5) % 1.0, 0.8])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.5, 0.0])
        self.assertEqual(first_cnode.value.time_stamp, event_time)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 0))
        self.assertEqual(first_child.value.position, [0.5, 0.9])
        self.assertEqual(first_child.weight, 0.5)
        self.assertIsNone(first_child.value.velocity)
        self.assertIsNone(first_child.value.time_stamp)
        self.assertEqual(first_child.value.charge, {"charge": -3.4})

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (0,))
        self.assertEqual(second_cnode.value.position, [(0.2 + (event_time - 1.2) * 0.5) % 1.0, 0.8])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [0.5, 0.0])
        self.assertEqual(second_cnode.value.time_stamp, event_time)
        self.assertIsNone(second_cnode.value.charge)
        self.assertEqual(len(second_cnode.children), 1)
        second_child = second_cnode.children[0]
        self.assertIs(second_child.parent, second_cnode)
        self.assertEqual(second_child.children, [])
        self.assertEqual(second_child.value.identifier, (0, 1))
        self.assertEqual(second_child.value.position, [(0.1 + (event_time - 1.2) * 1.0) % 1.0, 0.3])
        self.assertEqual(second_child.weight, 0.5)
        self.assertEqual(second_child.value.velocity, [1.0, 0.0])
        self.assertEqual(second_child.value.time_stamp, event_time)
        self.assertEqual(second_child.value.charge, {"charge": 1.2})

        third_cnode = out_state[2]
        self.assertIsNone(third_cnode.parent)
        self.assertEqual(third_cnode.value.identifier, (0,))
        self.assertEqual(third_cnode.value.position, [(0.2 + (event_time - 1.2) * 0.5) % 1.0, 0.8])
        self.assertEqual(third_cnode.weight, 1)
        self.assertEqual(third_cnode.value.velocity, [0.5, 0.0])
        self.assertEqual(third_cnode.value.time_stamp, event_time)
        self.assertIsNone(third_cnode.value.charge)
        self.assertEqual(len(third_cnode.children), 1)
        third_child = third_cnode.children[0]
        self.assertIs(third_child.parent, third_cnode)
        self.assertEqual(third_child.children, [])
        self.assertEqual(third_child.value.identifier, (0, 2))
        self.assertEqual(third_child.value.position, [0.7, 0.5])
        self.assertEqual(third_child.weight, 0.5)
        self.assertIsNone(third_child.value.velocity)
        self.assertIsNone(third_child.value.time_stamp)
        self.assertEqual(third_child.value.charge, {"charge": -0.7})

    def test_send_out_state_leaf_units_in_same_composite_object_max_displacement(self, random_expovariate_mock,
                                                                                 random_uniform_mock):
        # Event is always rejected, displacement is max_displacement
        self._setUpSendEventTimeMaxDisplacement(random_expovariate_mock, 0.2, 0.1, 1)
        displacement = 0.2
        event_time = 1.2 + displacement / 1.0
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 0), position=[0.5, 0.9],
                                         charge={"charge": -3.4}), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 1), position=[0.1, 0.3],
                                         velocity=[1.0, 0.0], time_stamp=1.2,
                                         charge={"charge": 1.2}), weight=0.5))

        in_state_three = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                   time_stamp=1.2), weight=1)
        in_state_three.add_child(Node(Unit(identifier=(0, 2), position=[0.7, 0.5],
                                           charge={"charge": -0.7}), weight=0.5))
        self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])

        out_state = self._event_handler.send_out_state()

        # Potential derivative is called with active leaf unit at initial position,
        # and active leaf unit moved by max_displacement
        potential_calls = [mock.call(0, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - 0.1 + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                           mock.call(0, [(0.5 - 0.1 - 0.2 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - 0.1 - 0.2 + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0)]
        self._potential_mock.derivative.assert_has_calls(potential_calls)
        random_uniform_mock.assert_not_called()
        self._lifting_mock.reset.assert_not_called()
        self._lifting_mock.insert.assert_not_called()
        self._lifting_mock.get_active_identifier.assert_not_called()

        self.assertEqual(len(out_state), 3)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [(0.2 + (event_time - 1.2) * 0.5) % 1.0, 0.8])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.5, 0.0])
        self.assertEqual(first_cnode.value.time_stamp, event_time)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 0))
        self.assertEqual(first_child.value.position, [0.5, 0.9])
        self.assertEqual(first_child.weight, 0.5)
        self.assertIsNone(first_child.value.velocity)
        self.assertIsNone(first_child.value.time_stamp)
        self.assertEqual(first_child.value.charge, {"charge": -3.4})

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (0,))
        self.assertEqual(second_cnode.value.position, [(0.2 + (event_time - 1.2) * 0.5) % 1.0, 0.8])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [0.5, 0.0])
        self.assertEqual(second_cnode.value.time_stamp, event_time)
        self.assertIsNone(second_cnode.value.charge)
        self.assertEqual(len(second_cnode.children), 1)
        second_child = second_cnode.children[0]
        self.assertIs(second_child.parent, second_cnode)
        self.assertEqual(second_child.children, [])
        self.assertEqual(second_child.value.identifier, (0, 1))
        self.assertEqual(second_child.value.position, [(0.1 + (event_time - 1.2) * 1.0) % 1.0, 0.3])
        self.assertEqual(second_child.weight, 0.5)
        self.assertEqual(second_child.value.velocity, [1.0, 0.0])
        self.assertEqual(second_child.value.time_stamp, event_time)
        self.assertEqual(second_child.value.charge, {"charge": 1.2})

        third_cnode = out_state[2]
        self.assertIsNone(third_cnode.parent)
        self.assertEqual(third_cnode.value.identifier, (0,))
        self.assertEqual(third_cnode.value.position, [(0.2 + (event_time - 1.2) * 0.5) % 1.0, 0.8])
        self.assertEqual(third_cnode.weight, 1)
        self.assertEqual(third_cnode.value.velocity, [0.5, 0.0])
        self.assertEqual(third_cnode.value.time_stamp, event_time)
        self.assertIsNone(third_cnode.value.charge)
        self.assertEqual(len(third_cnode.children), 1)
        third_child = third_cnode.children[0]
        self.assertIs(third_child.parent, third_cnode)
        self.assertEqual(third_child.children, [])
        self.assertEqual(third_child.value.identifier, (0, 2))
        self.assertEqual(third_child.value.position, [0.7, 0.5])
        self.assertEqual(third_child.weight, 0.5)
        self.assertIsNone(third_child.value.velocity)
        self.assertIsNone(third_child.value.time_stamp)
        self.assertEqual(third_child.value.charge, {"charge": -0.7})

    def test_send_out_state_leaf_units_in_same_composite_object_max_displacement_derivative_negative(
            self, random_expovariate_mock, random_uniform_mock):
        # Event is always rejected, displacement is max_displacement
        self._setUpSendEventTimeMaxDisplacementDerivativeNegative(random_expovariate_mock, 1)
        displacement = 0.2
        event_time = 1.2 + displacement / 1.0
        in_state_one = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                 time_stamp=1.2), weight=1)
        in_state_one.add_child(Node(Unit(identifier=(0, 0), position=[0.5, 0.9],
                                         charge={"charge": -3.4}), weight=0.5))

        in_state_two = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                 time_stamp=1.2), weight=1)
        in_state_two.add_child(Node(Unit(identifier=(0, 1), position=[0.1, 0.3],
                                         velocity=[1.0, 0.0], time_stamp=1.2,
                                         charge={"charge": 1.2}), weight=0.5))

        in_state_three = Node(Unit(identifier=(0,), position=[0.2, 0.8], velocity=[0.5, 0.0],
                                   time_stamp=1.2), weight=1)
        in_state_three.add_child(Node(Unit(identifier=(0, 2), position=[0.7, 0.5],
                                           charge={"charge": -0.7}), weight=0.5))
        self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])

        out_state = self._event_handler.send_out_state()

        # Potential derivative is called with active leaf unit at initial position,
        # and active leaf unit moved by max_displacement
        potential_calls = [mock.call(0, [(0.5 - 0.1 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - 0.1 + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0),
                           mock.call(0, [(0.5 - 0.1 - 0.2 + 0.5) % 1.0 - 0.5, (0.9 - 0.3 + 0.5) % 1.0 - 0.5],
                                     [(0.7 - 0.1 - 0.2 + 0.5) % 1.0 - 0.5, (0.5 - 0.3 + 0.5) % 1.0 - 0.5], 1.0, 1.0)]
        self._potential_mock.derivative.assert_has_calls(potential_calls)
        random_uniform_mock.assert_not_called()
        self._lifting_mock.reset.assert_not_called()
        self._lifting_mock.insert.assert_not_called()
        self._lifting_mock.get_active_identifier.assert_not_called()

        self.assertEqual(len(out_state), 3)
        first_cnode = out_state[0]
        self.assertIsNone(first_cnode.parent)
        self.assertEqual(first_cnode.value.identifier, (0,))
        self.assertEqual(first_cnode.value.position, [(0.2 + (event_time - 1.2) * 0.5) % 1.0, 0.8])
        self.assertEqual(first_cnode.weight, 1)
        self.assertEqual(first_cnode.value.velocity, [0.5, 0.0])
        self.assertEqual(first_cnode.value.time_stamp, event_time)
        self.assertIsNone(first_cnode.value.charge)
        self.assertEqual(len(first_cnode.children), 1)
        first_child = first_cnode.children[0]
        self.assertIs(first_child.parent, first_cnode)
        self.assertEqual(first_child.children, [])
        self.assertEqual(first_child.value.identifier, (0, 0))
        self.assertEqual(first_child.value.position, [0.5, 0.9])
        self.assertEqual(first_child.weight, 0.5)
        self.assertIsNone(first_child.value.velocity)
        self.assertIsNone(first_child.value.time_stamp)
        self.assertEqual(first_child.value.charge, {"charge": -3.4})

        second_cnode = out_state[1]
        self.assertIsNone(second_cnode.parent)
        self.assertEqual(second_cnode.value.identifier, (0,))
        self.assertEqual(second_cnode.value.position, [(0.2 + (event_time - 1.2) * 0.5) % 1.0, 0.8])
        self.assertEqual(second_cnode.weight, 1)
        self.assertEqual(second_cnode.value.velocity, [0.5, 0.0])
        self.assertEqual(second_cnode.value.time_stamp, event_time)
        self.assertIsNone(second_cnode.value.charge)
        self.assertEqual(len(second_cnode.children), 1)
        second_child = second_cnode.children[0]
        self.assertIs(second_child.parent, second_cnode)
        self.assertEqual(second_child.children, [])
        self.assertEqual(second_child.value.identifier, (0, 1))
        self.assertEqual(second_child.value.position, [(0.1 + (event_time - 1.2) * 1.0) % 1.0, 0.3])
        self.assertEqual(second_child.weight, 0.5)
        self.assertEqual(second_child.value.velocity, [1.0, 0.0])
        self.assertEqual(second_child.value.time_stamp, event_time)
        self.assertEqual(second_child.value.charge, {"charge": 1.2})

        third_cnode = out_state[2]
        self.assertIsNone(third_cnode.parent)
        self.assertEqual(third_cnode.value.identifier, (0,))
        self.assertEqual(third_cnode.value.position, [(0.2 + (event_time - 1.2) * 0.5) % 1.0, 0.8])
        self.assertEqual(third_cnode.weight, 1)
        self.assertEqual(third_cnode.value.velocity, [0.5, 0.0])
        self.assertEqual(third_cnode.value.time_stamp, event_time)
        self.assertIsNone(third_cnode.value.charge)
        self.assertEqual(len(third_cnode.children), 1)
        third_child = third_cnode.children[0]
        self.assertIs(third_child.parent, third_cnode)
        self.assertEqual(third_child.children, [])
        self.assertEqual(third_child.value.identifier, (0, 2))
        self.assertEqual(third_child.value.position, [0.7, 0.5])
        self.assertEqual(third_child.weight, 0.5)
        self.assertIsNone(third_child.value.velocity)
        self.assertIsNone(third_child.value.time_stamp)
        self.assertEqual(third_child.value.charge, {"charge": -0.7})

    def test_no_leaf_unit_active_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.2, 0.1, 1)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6]), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9]), weight=1)
        in_state_three = Node(Unit(identifier=(2,), position=[0.5, 0.7]), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])

    def test_more_than_one_leaf_unit_active_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.1, 0.05, 0)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6], velocity=[1.0, 0.0],
                                 time_stamp=0.0), weight=1)
        in_state_two = Node(Unit(identifier=(1,), position=[0.5, 0.9], velocity=[1.0, 0.0],
                                 time_stamp=0.0), weight=1)
        in_state_three = Node(Unit(identifier=(2,), position=[0.5, 0.7]), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler.send_event_time([in_state_one, in_state_two, in_state_three])

    def test_too_few_leaf_units_raises_error(self, random_expovariate_mock, _):
        self._setUpSendEventTimeSmallerMaxDisplacement(random_expovariate_mock, 0.2, 0.1, 0)
        in_state_one = Node(Unit(identifier=(3,), position=[0.5, 0.6], velocity=[1.0, 0.0],
                                 time_stamp=0.0), weight=1)
        in_state_two = Node(Unit(identifier=(2,), position=[0.5, 0.7]), weight=1)
        with self.assertRaises(AssertionError):
            self._event_handler.send_event_time([in_state_one, in_state_two])

    def test_potential_with_wrong_number_of_separation_arguments_raises_error(self, _, __):
        potential_mock = mock.MagicMock(spec_set=Potential)
        potential_mock.number_separation_arguments = 1
        with self.assertRaises(ConfigurationError):
            FixedSeparationsEventHandlerWithPiecewiseConstantBoundingPotential(
                potential=potential_mock, lifting=self._lifting_mock, offset=0.05, max_displacement=0.1,
                separations=[1, 0, 1, 2])

    def test_max_displacement_zero_raises_error(self, _, __):
        with self.assertRaises(ConfigurationError):
            FixedSeparationsEventHandlerWithPiecewiseConstantBoundingPotential(
                potential=self._potential_mock, lifting=self._lifting_mock, offset=0.05, max_displacement=0.0,
                separations=[1, 0, 1, 2])

    def test_max_displacement_negative_raises_error(self, _, __):
        with self.assertRaises(ConfigurationError):
            FixedSeparationsEventHandlerWithPiecewiseConstantBoundingPotential(
                potential=self._potential_mock, lifting=self._lifting_mock, offset=0.05, max_displacement=-0.1,
                separations=[1, 0, 1, 2])


if __name__ == '__main__':
    main()
