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
"""Module for the Walker class used in the cell veto event handlers."""
import random
from typing import Any, Sequence


class WalkerItem(object):
    """
    A walker item which associates an arbitrary object with a rate.

    Attributes
    ----------
    item : Any
        The associated object.
    rate : float
        The rate.
    """
    def __init__(self, item: Any, rate: float) -> None:
        """
        The constructor of the WalkerItem class.

        Parameters
        ----------
        item : Any
            The associated object.
        rate : float
            The rate.
        """
        self.item = item
        self.rate = rate


class Walker(object):
    """Class to store Walker's table, the total event rate and to sample an object from the table."""
    def __init__(self, walker_items: Sequence[WalkerItem]) -> None:
        """
        The constructor of the Walker class.

        Parameters
        ----------
        walker_items : Sequence[WalkerItem]
            The sequence of walker items which associate an object with an event rate.

        Raises
        ------
        AssertionError
            If a rate of a walker item is smaller than zero.
        """
        self._total_rate = sum(walker_item.rate for walker_item in walker_items)
        self._mean_rate = self._total_rate / len(walker_items)
        self._table = []
        for walker_item in walker_items:
            assert walker_item.rate >= 0.0
        self._build_table(walker_items)

    def _build_table(self, walker_items: Sequence[WalkerItem]) -> None:
        """Build Walker's table using the given walker items."""
        small_list = []
        large_list = []
        for walker_item in walker_items:
            if walker_item.rate > self._mean_rate:
                large_list.append(walker_item)
            else:
                small_list.append(walker_item)

        while len(small_list) and len(large_list):
            small_item = small_list.pop()
            large_item = large_list.pop()
            self._table.append((small_item, WalkerItem(large_item.item, self._mean_rate - small_item.rate)))
            large_item.rate -= self._mean_rate - small_item.rate

            if large_item.rate < self._mean_rate:
                small_list.append(large_item)
            else:
                large_list.append(large_item)

        while len(small_list):
            assert 1-1e-6 < small_list[-1].rate / self._mean_rate < 1 + 1e-6
            self._table.append((WalkerItem(small_list.pop().item, self._mean_rate),))

        while len(large_list):
            assert 1-1e-6 < large_list[-1].rate / self._mean_rate < 1 + 1e-6
            self._table.append((WalkerItem(large_list.pop().item, self._mean_rate),))

    def sample_cell(self) -> Any:
        """
        Sample a random object out of a walker item with a probability proportional to the rate stored in the item.

        Returns
        -------
        Any
            The sampled object.
        """
        choice_from_table = random.choice(self._table)
        if random.uniform(0.0, self._mean_rate) <= choice_from_table[0].rate:
            return choice_from_table[0].item
        else:
            return choice_from_table[1].item

    @property
    def total_rate(self) -> float:
        """
        Return the sum of all rates of all walker items stored in the Walker table.

        Returns
        -------
        float
            The total rate.
        """
        return self._total_rate
