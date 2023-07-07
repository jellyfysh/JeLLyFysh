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
"""
Module which gives access to hypercubic and general settings.

This module includes the system length of a hypercubic setting. Also, one can access the general settings of the
setting package (see setting.__init__.py).
In this module, the setter class HypercubicSetting inherits from the abstract Setting class. Instantiating this
class will initialize the setting package and this module properly. Also, the hypercuboid_setting module is initialized.
"""
import logging
import random
import sys
from typing import Any, List, MutableMapping, MutableSequence, Sequence
from jellyfysh.base.logging import log_init_arguments
from jellyfysh.setting import hypercuboid_setting
from .periodic_boundaries import PeriodicBoundaries
from .setting import Setting

system_length = None
"""The system length of the hypercube."""

system_length_over_two = None
"""The system length divided by two of the hypercube."""

beta = None
"""The inverse temperature (> 0). Also defined in setting package."""

dimension = None
"""The dimension (> 0). Also defined in setting package."""

number_of_root_nodes = None
"""The number of root nodes in the run (> 0). Also defined in setting package."""

number_of_nodes_per_root_node = None
"""The number of nodes per root node in the run (> 0). Also defined in setting package."""

number_of_node_levels = None
"""The number of node levels in the run (1 or 2). Also defined in setting package."""

random_position = None
"""The function which generates a random position. Also defined in setting package."""

periodic_boundaries = None
"""The instance of a periodic boundaries class. Also defined in setting package."""


def initialized() -> bool:
    """
    Check whether this module is initialized by checking if all attributes are not None.

    Returns
    -------
    bool
        If this module is initialized.
    """
    return all(variable is not None for variable in
               [system_length, system_length_over_two, beta, dimension, number_of_root_nodes,
                number_of_nodes_per_root_node, number_of_node_levels, random_position, periodic_boundaries])


class HypercubicPeriodicBoundaries(PeriodicBoundaries):
    """
    This class implements periodic boundaries for a hypercubic setting.

    An instance of this class is used for the periodic boundaries in the HypercubicSetting class.
    """
    @staticmethod
    def correct_position(position: MutableSequence[float]) -> None:
        """
        Correct the given position vector in place for periodic boundaries.

        Parameters
        ----------
        position : MutableSequence[float]
            The position vector.
        """
        for index, entry in enumerate(position):
            position[index] = HypercubicPeriodicBoundaries.correct_position_entry(entry, index)

    @staticmethod
    def correct_position_entry(position_entry: float, _: int) -> float:
        """
        Return the position entry at the given component corrected for periodic boundaries.

        Parameters
        ----------
        position_entry : float
            The position entry.
        _ : int
            The index of the position entry within the position vector (not used in this class).

        Returns
        -------
        float
            The position entry corrected for periodic boundaries.
        """
        # noinspection PyTypeChecker
        return position_entry % system_length

    @staticmethod
    def separation_vector(reference_position: Sequence[float],
                          target_position: Sequence[float]) -> List[float]:
        """
        Return the shortest separation vector of the target position divided by the reference position.

        Parameters
        ----------
        reference_position : Sequence[float]
            The reference position.
        target_position : Sequence[float]
            The target position.

        Returns
        -------
        List[float]
            The shortest separation vector target_position - reference_position, possibly corrected for periodic
            boundaries.
        """
        # noinspection PyTypeChecker
        separation = [target_position[index] - reference_position[index] for index in range(dimension)]
        HypercubicPeriodicBoundaries.correct_separation(separation)
        return separation

    @staticmethod
    def correct_separation(separation: MutableSequence[float]) -> None:
        """
        Correct the given separation vector in place for periodic boundaries.

        Parameters
        ----------
        separation : MutableSequence[float]
            The separation vector.
        """
        for index, entry in enumerate(separation):
            separation[index] = HypercubicPeriodicBoundaries.correct_separation_entry(entry, index)

    @staticmethod
    def correct_separation_entry(separation_entry: float, _: int) -> float:
        """
        Return the separation entry at the given component corrected for periodic boundaries.

        Parameters
        ----------
        separation_entry : float
            The separation entry.
        _ : int
            The index of the position entry within the position vector (not used in this class).

        Returns
        -------
        float
            The separation entry corrected for periodic boundaries.
        """
        # noinspection PyTypeChecker
        return (separation_entry + system_length_over_two) % system_length - system_length_over_two

    @staticmethod
    def next_image(position_entry: float, _: int) -> float:
        """
        Return the translated position in the next image in the given direction.

        Parameters
        ----------
        position_entry : float
            The starting position entry.
        _ : int
            The direction in which the position entry should be translated (not used in this class).

        Returns
        -------
        float
            The position entry translated in the next image in the given direction.
        """
        # noinspection PyTypeChecker
        return position_entry + system_length


# noinspection PyShadowingNames,PyTypeChecker
class HypercubicSetting(Setting):
    """
    The class which sets up the hypercubic setting by initializing the setting package, this module and the hypercuboid
    setting module as a similar module.
    """
    def __init__(self, beta: float = 1.0, dimension: int = 3, system_length: float = 1.0) -> None:
        """
        The constructor of the HypercubicSetting setter class.

        Besides the arguments of this constructor, it initializes the setting package with an instance of
        HypercubicPeriodicBoundaries as the periodic boundaries and the random position function of this class. The
        setting module .hypercuboid_setting is initialized with the system length in each direction to be the given
        system length.

        Parameters
        ----------
        beta : float, optional
            The module which gets initialized.
        dimension : int, optional
            The dimension.
        system_length : float
            The system length of the hypercube.
        """
        log_init_arguments(logging.getLogger(__name__).debug, self.__class__.__name__, beta=beta, dimension=dimension,
                           system_length=system_length)
        super().__init__(module=sys.modules[__name__], beta=beta, dimension=dimension,
                         periodic_boundaries=HypercubicPeriodicBoundaries(), similar_modules=[hypercuboid_setting])
        _set_system_length(system_length)
        _set_similar_settings(system_length)

    @staticmethod
    def random_position() -> List[float]:
        """
        Return a random position.

        Returns
        -------
        List[float]
            The random position.
        """
        return [random.uniform(0.0, system_length) for _ in range(dimension)]


def _set_system_length(wanted_system_length: float) -> None:
    """
    Setter function which sets the system length and the system length divided by two of the hypercube in this module.

    Parameters
    ----------
    wanted_system_length : float
        The system length.

    Raises
    ------
    AttributeError
        If the system length or the system length divided by two has already been set in this module.
    AttributeError
        If the system length is smaller than or equal zero.
    """
    global system_length
    global system_length_over_two
    if system_length is not None or system_length_over_two is not None:
        raise AttributeError("System length already set!")
    if wanted_system_length <= 0.0:
        raise AttributeError("System length must be greater than 0.0")
    system_length = wanted_system_length
    system_length_over_two = wanted_system_length / 2.0


# noinspection PyTypeChecker
def _set_similar_settings(wanted_system_length: float) -> None:
    """
    Setter function which sets the system lengths and the systems lengths divided by two of the hypercuboid equally in
    each direction in the hypercuboid_setting module.

    Parameters
    ----------
    wanted_system_length : float
        The system length.

    Raises
    ------
    AttributeError
        If the system length has already been set in the hypercuboid_setting module.
    AttributeError
        If the system length is smaller than or equal zero.
    """
    if hypercuboid_setting.system_lengths is not None or hypercuboid_setting.system_lengths_over_two is not None:
        raise AttributeError("System lengths already set!")
    if wanted_system_length <= 0.0:
        raise AttributeError("System length must be greater than 0.0")
    hypercuboid_setting.system_lengths = tuple(wanted_system_length for _ in range(dimension))
    hypercuboid_setting.system_lengths_over_two = tuple(wanted_system_length / 2.0 for _ in range(dimension))


def getstate() -> MutableMapping[str, Any]:
    """
    Return a state of this module that can be pickled.

    This function only stores the global variable system_length that is required for the initialization of the
    HypercubicSetting setter class. The other global variables beta, dimension, number_of_root_nodes,
    number_of_nodes_per_root_node, and number_of_node_levels are only copies from the setting module and are considered
    in its getstate function.

    Returns
    -------
    MutableMapping[str, Any]
        The state that can be pickled.
    """
    return {"system_length": system_length}


def setstate(state: MutableMapping[str, Any]) -> None:
    """
    Use the state dictionary to initialize this module.

    This function calls the HypercubicSetting setter class.

    Parameters
    ----------
    state : MutableMapping[str, Any]
        The state.

    Raises
    ------
    AssertionError
        If the state dictionary misses necessary keys for the initialization of this module.
    """
    assert "beta" in state
    assert "dimension" in state
    assert "system_length" in state
    HypercubicSetting(beta=state["beta"], dimension=state["dimension"], system_length=state["system_length"])


def reset() -> None:
    """
    Reset this setting module.

    This function will set all attributes in this module to None.
    """
    global beta
    beta = None
    global dimension
    dimension = None
    global number_of_root_nodes
    number_of_root_nodes = None
    global number_of_nodes_per_root_node
    number_of_nodes_per_root_node = None
    global number_of_node_levels
    number_of_node_levels = None
    global random_position
    random_position = None
    global periodic_boundaries
    periodic_boundaries = None
    global system_length
    system_length = None
    global system_length_over_two
    system_length_over_two = None
