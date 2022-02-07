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
"""Module for exceptions and warnings used in the application."""
import logging


class ConfigurationError(Exception):
    """Raised whenever a class is initialized with inappropriate arguments."""
    pass


class EndOfRun(Exception):
    """
    Raised to interrupt the main iteration loop of the mediator.

    Currently only raised within the mediating method of the EndOfRunEventHandler abstract class.
    """
    pass


class FactorSetError(Exception):
    """
    Raised when the parsing of a factor type map from a file fails.

    For more details on the format of the factor type map file, see activator.tagger.factor_type_maps module.
    """
    pass


class InitializerError(Exception):
    """
    Raised when an Initializer class' public method is called before the initialize method has been called.

    For more details on the Initializer abstract class, see base.initializer module.
    """
    pass


class LiftingSchemeError(Exception):
    """
    Raised in Lifting class, when next active identifier is requested although currently active unit was not recorded.

    For more details on the Lifting abstract class, see lifting.lifting module.
    """
    pass


class MediatorError(Exception):
    """
    Raised in the Mediator class, when an event handler has more than one possible mediating or get_arguments method.

    For more details on the Mediator class, see mediator.mediator module.
    """
    pass


class SchedulerError(Exception):
    """
    Raised in the Scheduler class, when the returned smallest candidate event times do not increase.

    For more details on the Scheduler class, see scheduler.scheduler module.
    """
    pass


class TagActivatorError(Exception):
    """
    Raised in the TagActivator class, when it tries to get an event handler to run for a certain tagger but all but
    event handler for this tagger are already running.

    For more details on the TagActivator class, see the activator.tag_activator module.
    """


_logger = logging.getLogger(__name__)


def bounding_potential_warning(event_handler_name: str, bounding_derivative: float, real_derivative: float) -> None:
    """
    Log a warning if the real derivative is greater than zero and larger than the bounding derivative.

    The logging message includes the name of the event handler.

    Parameters
    ----------
    event_handler_name : str
        The event handler name.
    bounding_derivative : float
        The bounding potential derivative.
    real_derivative : float
        The real potential derivative.
    """
    if real_derivative > 0 and bounding_derivative < real_derivative:
        _logger.warning("In the event handler {0} the bounding event rate {1} "
                        "is not bigger than the real event rate {2}"
                        .format(event_handler_name, bounding_derivative, real_derivative))
