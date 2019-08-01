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
from base.exceptions import ConfigurationError
from event_handler.fixed_interval_dumping_event_handler import FixedIntervalDumpingEventHandler
import setting


class TestFixedIntervalDumpingEventHandler(TestCase):
    def setUp(self) -> None:
        setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        self._event_handler_one = FixedIntervalDumpingEventHandler(dumping_interval=0.5,
                                                                   output_handler="some_output_handler")
        self._event_handler_two = FixedIntervalDumpingEventHandler(dumping_interval=1.3, output_handler="output")

    def tearDown(self) -> None:
        setting.reset()

    def test_number_send_event_time_arguments_one(self):
        self.assertEqual(self._event_handler_one.number_send_event_time_arguments, 0)

    def test_number_send_event_time_arguments_two(self):
        self.assertEqual(self._event_handler_two.number_send_event_time_arguments, 0)

    def test_number_send_out_state_arguments_one(self):
        self.assertEqual(self._event_handler_one.number_send_out_state_arguments, 0)

    def test_number_send_out_state_arguments_two(self):
        self.assertEqual(self._event_handler_one.number_send_out_state_arguments, 0)

    def test_send_event_time_one(self):
        self.assertEqual(self._event_handler_one.send_event_time(), 0.5)
        self.assertEqual(self._event_handler_one.send_event_time(), 1.0)

    def test_send_event_time_two(self):
        self.assertEqual(self._event_handler_two.send_event_time(), 1.3)
        self.assertEqual(self._event_handler_two.send_event_time(), 2.6)

    def test_send_out_state_one(self):
        # Call this to update the event time
        self._event_handler_one.send_event_time()

        out_state = self._event_handler_one.send_out_state()
        self.assertEqual(len(out_state), 0)

    def test_send_out_state_two(self):
        self._event_handler_two.send_event_time()
        out_state = self._event_handler_two.send_out_state()
        self.assertEqual(len(out_state), 0)

    def test_output_handler_one(self):
        self.assertEqual(self._event_handler_one.output_handler, "some_output_handler")

    def test_output_handler_two(self):
        self.assertEqual(self._event_handler_two.output_handler, "output")

    def test_dumping_interval_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            FixedIntervalDumpingEventHandler(dumping_interval=0.0, output_handler="output")

    def test_sampling_interval_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            FixedIntervalDumpingEventHandler(dumping_interval=-0.1, output_handler="output")


if __name__ == '__main__':
    main()
