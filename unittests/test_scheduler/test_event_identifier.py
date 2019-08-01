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
from scheduler.event_identifier import EventIdentifier


class TestEventIdentifier(TestCase):
    def setUp(self) -> None:
        self._event_identifier = EventIdentifier()

    def test_identifier_calls_without_delete(self):
        identifier = self._event_identifier.identifier()
        self.assertEqual(identifier, 1)
        identifier = self._event_identifier.identifier()
        self.assertEqual(identifier, 2)
        identifier = self._event_identifier.identifier()
        self.assertEqual(identifier, 3)

    def test_identifier_calls_with_delete(self):
        _ = self._event_identifier.identifier()
        _ = self._event_identifier.identifier()
        _ = self._event_identifier.identifier()
        # identifiers 1, 2, and 3 have now been used
        self._event_identifier.delete_identifier(2)
        # identifier 2 is reusable
        identifier = self._event_identifier.identifier()
        self.assertEqual(identifier, 2)
        identifier = self._event_identifier.identifier()
        self.assertEqual(identifier, 4)
        self._event_identifier.delete_identifier(4)
        self._event_identifier.delete_identifier(1)
        identifier = self._event_identifier.identifier()
        self.assertEqual(identifier, 1)
        identifier = self._event_identifier.identifier()
        self.assertEqual(identifier, 4)
        identifier = self._event_identifier.identifier()
        self.assertEqual(identifier, 5)

    def test_deleting_too_large_identifier_raises_error(self):
        _ = self._event_identifier.identifier()
        _ = self._event_identifier.identifier()
        _ = self._event_identifier.identifier()
        with self.assertRaises(AssertionError):
            self._event_identifier.delete_identifier(4)


if __name__ == '__main__':
    main()
