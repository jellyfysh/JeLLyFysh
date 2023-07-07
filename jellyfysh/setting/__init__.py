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
Package which gives access to general settings.

The general settings are the inverse temperature, the dimension and the number of point masses/composite point objects
in the run. The latter allows all modules to construct all possible global state identifiers. JFV only allows copies
of the same composite point object.
Moreover, this package gives access to a function which generates a random position and an instance of a
PeriodicBoundaries class (see setting.periodic_boundaries.periodic_boundaries.PeriodicBoundaries).

To initialize the attributes of this package, one should use either a setter class (i.e., a class inheriting from
the abstract Setting class in setting.setting.py) or use the public methods. The final more specialized settings (for
example a hypercubic setting) should be defined in their own module with a setter class. Such modules can define
additional attributes, but all the attributes listed here are also accessible in the given module, if such a setter
class was used. A setting module must define a reset, getstate and setstate function.

Internally, this package will store the initialized setting module. Only one module can be initialized, although
the module can give similar modules and a way how these modules should be initialized (for example, a hypercubic
setting can also initialize a hypercuboid setting; by this all modules implemented for a hypercuboid setting will also
work when a hypercubic setting is used).
"""
import logging
from typing import Any, Callable, MutableMapping, Sequence
from types import ModuleType
from .periodic_boundaries import PeriodicBoundaries

beta = None
"""The inverse temperature (> 0)."""

dimension = None
"""The dimension (> 0)."""

number_of_root_nodes = None
"""The number of root nodes in the run (> 0)."""

number_of_nodes_per_root_node = None
"""The number of nodes per root node in the run (> 0)."""

number_of_node_levels = None
"""The number of node levels in the run (1 or 2)."""

random_position = None
"""The function which generates a random position."""

periodic_boundaries = None
"""The instance of a periodic boundaries class."""

_initialized_setting_module = None
_similar_modules = None

_logger = logging.getLogger(__name__)


def getstate() -> MutableMapping[str, Any]:
    """
    Return a state of this module that can be pickled.

    This function stores the global variables beta, dimension, number_of_root_nodes, number_of_nodes_per_root_node, and
    number_of_node_levels. Also, it stores the initialized setting module and its state, so that it can be initialized
    in the setstate method.

    Returns
    -------
    MutableMapping[str, Any]
        The state that can be pickled.
    """
    state = {"beta": beta, "dimension": dimension, "number_of_root_nodes": number_of_root_nodes,
             "number_of_nodes_per_root_node": number_of_nodes_per_root_node,
             "number_of_node_levels": number_of_node_levels, "_initialized_setting_module": _initialized_setting_module}
    # noinspection PyUnresolvedReferences
    state.update(_initialized_setting_module.getstate())
    return state


def setstate(state: MutableMapping[str, Any]) -> None:
    """
    Use the state dictionary to initialize this module.

    This function initializes the setting module that is specified in the state, and sets the number_of_root_nodes,
    number_of_nodes_per_root_node, and number_of_node_levels variables.

    Parameters
    ----------
    state : MutableMapping[str, Any]
        The state.

    Raises
    ------
    AssertionError
        If the state dictionary misses necessary keys for the initialization of this module.
    """
    assert "_initialized_setting_module" in state
    state["_initialized_setting_module"].setstate(state)
    assert beta == state["beta"]
    assert dimension == state["dimension"]
    assert "number_of_root_nodes" in state
    set_number_of_root_nodes(state["number_of_root_nodes"])
    assert "number_of_nodes_per_root_node" in state
    set_number_of_nodes_per_root_node(state["number_of_nodes_per_root_node"])
    assert "number_of_node_levels" in state
    set_number_of_node_levels(state["number_of_node_levels"])


# noinspection PyTypeChecker
def reset() -> None:
    """
    Reset the setting package.

    This function will set all attributes in this package to None and call the reset method of each initialized setting
    module.
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
    global _initialized_setting_module
    if _initialized_setting_module is not None:
        # noinspection PyUnresolvedReferences
        _initialized_setting_module.reset()
        _initialized_setting_module = None
    global _similar_modules
    if _similar_modules is not None:
        for similar_module in _similar_modules:
            similar_module.reset()
        _similar_modules = None


# noinspection PyTypeChecker
def set_number_of_root_nodes(wanted_number_of_root_nodes: int) -> None:
    """
    Setter function which sets the number of root nodes.

    This function sets the number of root nodes in this package, in the initialized module and all similar modules.

    Parameters
    ----------
    wanted_number_of_root_nodes : int
        The number of root nodes.

    Raises
    ------
    AttributeError
        If the number of root nodes has already been set in this package, in the initialized module or similar modules.
    AttributeError
        If the number of root nodes is smaller than or equal zero.
    """
    _logger.debug("Setting number of root nodes to {0}.".format(wanted_number_of_root_nodes))
    global number_of_root_nodes
    if number_of_root_nodes is not None:
        raise AttributeError("Number of root nodes already set in setting package!")
    # noinspection PyUnresolvedReferences
    if _initialized_setting_module.number_of_root_nodes is not None:
        raise AttributeError("Number of root nodes already set in initialized setting module {0}!"
                             .format(_initialized_setting_module.__name__))
    for similar_module in _similar_modules:
        if similar_module.number_of_root_nodes is not None:
            raise AttributeError("Number of root nodes already set in the similar module {0}!"
                                 .format(similar_module.__name__))
    if wanted_number_of_root_nodes <= 0:
        raise AttributeError("Number of root nodes must be greater than 0.")
    number_of_root_nodes = int(wanted_number_of_root_nodes)
    _initialized_setting_module.number_of_root_nodes = int(wanted_number_of_root_nodes)
    for similar_module in _similar_modules:
        similar_module.number_of_root_nodes = int(wanted_number_of_root_nodes)


# noinspection PyTypeChecker
def set_number_of_nodes_per_root_node(wanted_number_of_nodes_per_root_node: int) -> None:
    """
    Setter function which sets the number of nodes per root node.

    This function sets the number of nodes per root node in this package, in the initialized module and all similar
    modules.

    Parameters
    ----------
    wanted_number_of_nodes_per_root_node : int
        The number of nodes per root node.

    Raises
    ------
    AttributeError
        If the number of nodes per root node has already been set in this package, in the initialized module or similar
        modules.
    AttributeError
        If the number of nodes per root node is smaller than or equal zero.
    """
    _logger.debug("Setting number of nodes per root node to {0}.".format(wanted_number_of_nodes_per_root_node))
    global number_of_nodes_per_root_node
    if number_of_nodes_per_root_node is not None:
        raise AttributeError("Number of nodes per root node already set in setting package!")
    # noinspection PyUnresolvedReferences
    if _initialized_setting_module.number_of_nodes_per_root_node is not None:
        raise AttributeError("Number of nodes per root node already set in initialized setting module {0}!"
                             .format(_initialized_setting_module.__name__))
    for similar_module in _similar_modules:
        if similar_module.number_of_nodes_per_root_node is not None:
            raise AttributeError("Number of nodes per root node already set in the similar module {0}!"
                                 .format(similar_module.__name__))
    if wanted_number_of_nodes_per_root_node <= 0:
        raise AttributeError("Number of nodes per root node must be greater than 0.")
    number_of_nodes_per_root_node = int(wanted_number_of_nodes_per_root_node)
    _initialized_setting_module.number_of_nodes_per_root_node = int(wanted_number_of_nodes_per_root_node)
    for similar_module in _similar_modules:
        similar_module.number_of_nodes_per_root_node = int(wanted_number_of_nodes_per_root_node)


# noinspection PyTypeChecker
def set_number_of_node_levels(wanted_number_of_node_levels: int) -> None:
    """
    Setter function which sets the number of node levels.

    This function sets the number of node levels in this package, in the initialized module and all similar modules.

    Parameters
    ----------
    wanted_number_of_node_levels : int
        The number of node levels.

    Raises
    ------
    AttributeError
        If the number of node levels has already been set in this package, in the initialized module or similar modules.
    AttributeError
        If the number of levels is not one or two.
    """
    _logger.debug("Setting number of node levels to {0}.".format(wanted_number_of_node_levels))
    global number_of_node_levels
    if number_of_node_levels is not None:
        raise AttributeError("Number of node levels already set in setting package!")
    # noinspection PyUnresolvedReferences
    if _initialized_setting_module.number_of_node_levels is not None:
        raise AttributeError("Number of node levels is already set in initialized setting module {0}!"
                             .format(_initialized_setting_module.__name__))
    for similar_module in _similar_modules:
        if similar_module.number_of_node_levels is not None:
            raise AttributeError("Number of node levels already set in the similar module {0}!"
                                 .format(similar_module.__name__))
    if not 0 < wanted_number_of_node_levels < 3:
        raise AttributeError("Number of node levels must be 1 or 2.")
    number_of_node_levels = int(wanted_number_of_node_levels)
    _initialized_setting_module.number_of_node_levels = int(wanted_number_of_node_levels)
    for similar_module in _similar_modules:
        similar_module.number_of_node_levels = int(wanted_number_of_node_levels)


# noinspection PyTypeChecker
def _set_beta(wanted_beta: float) -> None:
    """
    Setter function which sets the inverse temperature beta.

    This function sets the inverse temperature beta in this package, in the initialized module and all similar modules.

    Parameters
    ----------
    wanted_beta : int
        The inverse temperature beta.

    Raises
    ------
    AttributeError
        If the inverse temperature beta has already been set in this package, in the initialized module or similar
        modules.
    AttributeError
         If the inverse temperature beta is smaller than or equal zero.
    """
    global beta
    if beta is not None:
        raise AttributeError("Beta already set in setting package!")
    # noinspection PyUnresolvedReferences
    if _initialized_setting_module.beta is not None:
        raise AttributeError("Beta already set in initialized setting module {0}!"
                             .format(_initialized_setting_module.__name__))
    for similar_module in _similar_modules:
        if similar_module.beta is not None:
            raise AttributeError("Beta already set in the similar module {0}!".format(similar_module.__name__))
    if wanted_beta <= 0.0:
        raise AttributeError("Beta must be greater than 0.0")
    beta = wanted_beta
    _initialized_setting_module.beta = wanted_beta
    for similar_module in _similar_modules:
        similar_module.beta = wanted_beta


# noinspection PyTypeChecker
def _set_dimension(wanted_dimension: int) -> None:
    """
    Setter function which sets the dimension.

    This function sets the dimension in this package, in the initialized module and all similar modules.

    Parameters
    ----------
    wanted_dimension : int
        The dimension.

    Raises
    ------
    AttributeError
        If the dimension has already been set in this package, in the initialized module or similar modules.
    AttributeError
         If the dimension is smaller than or equal zero.
    """
    global dimension
    if dimension is not None:
        raise AttributeError("Dimension already set in setting package!")
    # noinspection PyUnresolvedReferences
    if _initialized_setting_module.dimension is not None:
        raise AttributeError("Dimension already set in initialized setting module {0}!"
                             .format(_initialized_setting_module.__name__))
    for similar_module in _similar_modules:
        if similar_module.dimension is not None:
            raise AttributeError("Dimension already set in the similar module {0}!".format(similar_module.__name__))
    if wanted_dimension <= 0:
        raise AttributeError("Dimension must be greater than 0.")
    dimension = int(wanted_dimension)
    _initialized_setting_module.dimension = int(wanted_dimension)
    for similar_module in _similar_modules:
        similar_module.dimension = int(wanted_dimension)


# noinspection PyTypeChecker
def _set_random_position_function(wanted_random_position_function: Callable[[], Sequence[float]]) -> None:
    """
    Setter function which sets the function to generate a random position.

    This function sets the random position function in this package, in the initialized module and all similar modules.

    Parameters
    ----------
    wanted_random_position_function : Callable[[], Sequence[float]]
        The random position function.

    Raises
    ------
    AttributeError
        If the random position function has already been set in this package, in the initialized module or similar
        modules.
    """
    global random_position
    if random_position is not None:
        raise AttributeError("Random position function already set in setting package!")
    # noinspection PyUnresolvedReferences
    if _initialized_setting_module.random_position is not None:
        raise AttributeError("Random position function already set in initialized setting module {0}!"
                             .format(_initialized_setting_module.__name__))
    for similar_module in _similar_modules:
        if similar_module.random_position is not None:
            raise AttributeError("Random position function already set in the similar module {0}!"
                                 .format(similar_module.__name__))
    random_position = wanted_random_position_function
    _initialized_setting_module.random_position = wanted_random_position_function
    for similar_module in _similar_modules:
        similar_module.random_position = wanted_random_position_function


# noinspection PyTypeChecker
def _set_periodic_boundaries(wanted_periodic_boundaries: PeriodicBoundaries) -> None:
    """
    Setter function which sets periodic boundaries.

    This function sets the periodic boundaries in this package, in the initialized module and all similar modules.

    Parameters
    ----------
    wanted_periodic_boundaries : setting.periodic_boundaries.PeriodicBoundaries
        The periodic boundaries.

    Raises
    ------
    AttributeError
        If the periodic boundaries have already been set in this package, in the initialized module or similar
        modules.
    """
    global periodic_boundaries
    if periodic_boundaries is not None:
        raise AttributeError("Periodic boundaries already set in setting package!")
    # noinspection PyUnresolvedReferences
    if _initialized_setting_module.periodic_boundaries is not None:
        raise AttributeError("Periodic boundaries already set in initialized setting module {0}!"
                             .format(_initialized_setting_module.__name__))
    for similar_module in _similar_modules:
        if similar_module.periodic_boundaries is not None:
            raise AttributeError("Periodic boundaries already set in the similar module {0}!"
                                 .format(similar_module.__name__))
    periodic_boundaries = wanted_periodic_boundaries
    _initialized_setting_module.periodic_boundaries = wanted_periodic_boundaries
    for similar_module in _similar_modules:
        similar_module.periodic_boundaries = wanted_periodic_boundaries


def _set_initialized_setting_module(wanted_module: ModuleType) -> None:
    """
    Setter function which sets the initialized setting module.

    Parameters
    ----------
    wanted_module : ModuleType
        The setting module.

    Raises
    ------
    AttributeError
        If the initialized setting module has already been set in this package.
    """
    global _initialized_setting_module
    if _initialized_setting_module is not None:
        raise AttributeError("A setting module has been already initialized!")
    _initialized_setting_module = wanted_module


def _set_similar_modules(wanted_similar_modules: Sequence[ModuleType]) -> None:
    """
    Setter function which sets the similar setting modules.

    Parameters
    ----------
    wanted_similar_modules : Sequence[ModuleType]
        The similar setting modules.

    Raises
    ------
    AttributeError
        If the similar modules have already been set in this package.
    """
    global _similar_modules
    if _similar_modules is not None:
        raise AttributeError("Similar setting modules have been already initialized!")
    _similar_modules = wanted_similar_modules
