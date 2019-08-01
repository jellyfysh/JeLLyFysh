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
"""Module for exceptions used in the application."""


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
    Raised in the Scheduler class, when a non-existing event should be trashed.

    For more details on the Scheduler class, see scheduler.scheduler module.
    """
    pass


class TagActivatorError(Exception):
    """
    Raised in the TagActivator class, when it tries to get an event handler to run for a certain tagger but all but
    event handler for this tagger are already running.

    For more details on the TagActivator class, see the activator.tag_activator module.
    """
