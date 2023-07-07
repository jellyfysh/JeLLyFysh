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
"""Module for the RatioLifting class."""
import logging
import random
from typing import Any
from jellyfysh.base.logging import log_init_arguments
from .lifting import Lifting


class RatioLifting(Lifting):
    """
    This class implements the ratio lifting scheme.
    """

    def __init__(self) -> None:
        """The constructor of the RatioLifting class."""
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__)
        super().__init__()

    def get_active_identifier(self) -> Any:
        """
        Get the next active global state identifier based on the derivative table.

        Returns
        -------
        Any
            The next active global state identifier.
        """
        super().get_active_identifier()
        random_number = random.uniform(0.0, sum(self._negative_lifting_rates))
        assert abs(self._sum_positive_lifting_rates - sum(self._negative_lifting_rates)) < 1.0e-11
        summed_lifting_rate = 0.
        for index, lifting_rate in enumerate(self._negative_lifting_rates):
            summed_lifting_rate += lifting_rate
            if random_number <= summed_lifting_rate:
                return self._associated_identifiers[index]
        return self._associated_identifiers[-1]
