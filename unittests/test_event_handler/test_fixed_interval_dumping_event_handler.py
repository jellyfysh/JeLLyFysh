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
from unittest import TestCase, main
from jellyfysh.base.exceptions import ConfigurationError
from jellyfysh.base.time import Time
from jellyfysh.event_handler.fixed_interval_dumping_event_handler import FixedIntervalDumpingEventHandler
import jellyfysh.setting as setting
from jellyfysh.setting import hypercubic_setting


class TestFixedIntervalDumpingEventHandler(TestCase):
    def setUp(self) -> None:
        hypercubic_setting.HypercubicSetting(beta=1.0, dimension=3, system_length=1.0)
        self._event_handler_one = FixedIntervalDumpingEventHandler(dumping_interval=0.5,
                                                                   output_handler="some_output_handler")
        self._event_handler_two = FixedIntervalDumpingEventHandler(dumping_interval=1.3, output_handler="output")

    def tearDown(self) -> None:
        setting.reset()

    def test_number_send_event_time_arguments_zero(self):
        self.assertEqual(self._event_handler_one.number_send_event_time_arguments, 0)
        self.assertEqual(self._event_handler_two.number_send_event_time_arguments, 0)

    def test_number_send_out_state_arguments_zero(self):
        self.assertEqual(self._event_handler_one.number_send_out_state_arguments, 0)
        self.assertEqual(self._event_handler_two.number_send_out_state_arguments, 0)

    def test_send_event_time_event_handler_one(self):
        self.assertAlmostEqual(self._event_handler_one.send_event_time(), Time.from_float(0.5), places=13)
        self.assertAlmostEqual(self._event_handler_one.send_event_time(), Time.from_float(1.0), places=13)

    def test_send_event_time_event_handler_two(self):
        self.assertAlmostEqual(self._event_handler_two.send_event_time(), Time.from_float(1.3), places=13)
        self.assertAlmostEqual(self._event_handler_two.send_event_time(), Time.from_float(2.6), places=13)

    def test_send_out_state_event_handler_one(self):
        # Call this to update the event time
        self._event_handler_one.send_event_time()

        out_state = self._event_handler_one.send_out_state()
        self.assertEqual(len(out_state), 0)

    def test_send_out_state_event_handler_two(self):
        self._event_handler_two.send_event_time()
        out_state = self._event_handler_two.send_out_state()
        self.assertEqual(len(out_state), 0)

    def test_output_handler(self):
        self.assertEqual(self._event_handler_one.output_handler, "some_output_handler")
        self.assertEqual(self._event_handler_two.output_handler, "output")

    def test_dumping_interval_zero_raises_error(self):
        with self.assertRaises(ConfigurationError):
            FixedIntervalDumpingEventHandler(dumping_interval=0.0, output_handler="output")

    def test_sampling_interval_negative_raises_error(self):
        with self.assertRaises(ConfigurationError):
            FixedIntervalDumpingEventHandler(dumping_interval=-0.1, output_handler="output")


if __name__ == '__main__':
    main()
