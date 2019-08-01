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
"""Module for the abstract Mediator class."""
from abc import ABCMeta, abstractmethod
import logging
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple
from activator import Activator
from base.exceptions import ConfigurationError, EndOfRun, MediatorError
from base.node import Node
from base import strings
from event_handler import EventHandler
from event_handler.abstracts.abstracts import EventHandlerWithOutputHandler
from input_output_handler import InputOutputHandler
from scheduler import Scheduler
from state_handler import StateHandler


class MediatorAbstractClass(metaclass=ABCMeta):
    """
    The abstract base class of a mediator in the JF application.

    The mediator serves as a central hub for other elements of JF that do not directly connect to each other. In this
    way, interfaces and data exchange are particularly simple.
    The mediator connects to the input-output handler, the state handler, the scheduler, and the activator. The
    activator stores the list of all event handlers (but there shouldn't be a data flow between them, only the mediator
    should call methods of the event handlers) which also gets extracted in this class.
    The mediator also serves as an entry point for the executable run.py and resume.py scripts of this application.
    This class defines the needed methods for this.
    """

    def __init__(self, input_output_handler: InputOutputHandler, state_handler: StateHandler, scheduler: Scheduler,
                 activator: Activator) -> None:
        """
        The constructor of the MediatorAbstractClass class.

        This method checks for all event handlers which are connected to an output handler (by inheriting from the
        class EventHandlerWithOutputHandler) if the output handler exists in the input-output handler.

        Parameters
        ----------
        input_output_handler : input_output_handler.InputOutputHandler
            The input-output handler.
        state_handler : state_handler.StateHandler
            The state handler.
        scheduler : scheduler.Scheduler
            The scheduler.
        activator : activator.Activator
            The activator.

        Raises
        ------
        base.exceptions.ConfigurationError
            If an output handler of an event handler does not exist in the input-output handler.
        """
        activator.initialize(state_handler.extract_global_state())
        self._input_output_handler = input_output_handler
        self._state_handler = state_handler
        self._scheduler = scheduler
        self._activator = activator
        self._event_handlers_list = activator.event_handlers
        used_output_handlers = []
        for event_handler in self._event_handlers_list:
            if isinstance(event_handler, EventHandlerWithOutputHandler):
                if event_handler.output_handler is None:
                    continue
                if event_handler.output_handler not in self._input_output_handler.output_handlers:
                    raise ConfigurationError("The output handler '{0}' in event handler {1} is not existing."
                                             .format(event_handler.output_handler, event_handler.__class__.__name__))
                used_output_handlers.append(event_handler.output_handler)
        for output_handler in self._input_output_handler.output_handlers:
            if output_handler not in used_output_handlers:
                logging.getLogger(__name__).warning("Created output handler {0} was not used in any event handler."
                                                    .format(output_handler))

    @abstractmethod
    def run(self) -> None:
        """
        Loop over the legs of the continuous-time evolution of the event-chain Monte Carlo algorithm.

        This method is called by run.py and resume.py. The loop should only be interrupted when a
        base.exceptions.EndOfRun exception is raised, which is caught in the scripts.
        """
        raise NotImplementedError

    def post_run(self) -> None:
        """
        Call the post_run method of the input-output handler.

        After a base.exceptions.EndOfRun exception was raised in the run method of this class, this method is called by
        run.py and resume.py.
        """
        self._input_output_handler.post_run()

    def deactivate_output(self) -> None:
        """
        Deactivate every output handler in the input-output handler.

        This method is called by resume.py, if there is no need for output.
        """
        self._input_output_handler.deactivate_output()

    @abstractmethod
    def update_logging(self) -> None:
        """
        Update the logging of this class.

        This method is called in resume.py which might be run with a different logging level compared to the run which
        created the dump. This method then ensures that this class logs on the correct level.
        """
        raise NotImplementedError


def _get_bases_names(cls: type) -> Set[str]:
    """Return the names of the base classes of the given class."""
    bases_names = set()
    for subclass in cls.__bases__:
        bases_names.add(subclass.__name__)
        bases_names.update(_get_bases_names(subclass))
    return bases_names


# noinspection PyRedundantParentheses
class Mediator(MediatorAbstractClass, metaclass=ABCMeta):
    """
    The abstract Mediator class which defines mediating methods.

    This class inherits from the MediatorAbstractClass. In this class mediating methods and methods to construct
    arguments for the send_out_state method of event handlers are defined.
    The mediating methods are called when an event of an event handler was committed to the global state. It should
    begin with 'mediate_' and followed by the name of the event handler class. The mediating methods don't have
    any arguments and their return value is None.
    The other method, which constructs arguments for the send_out_state method of an event handler, follows the same
    naming convention but the start is 'get_arguments_'. The objects in the sequence returned besides the
    candidate event time in the send_event_time of an event handler, will be handed to the argument construction
    method (the sequence gets unpacked). The returned object of this method also gets unpacked and is then handed to
    the send_out_state method. Methods which only return a single object, should probably put it into a tuple
    explicitly for this reason.
    Both methods can also be defined for base classes, however only a single base class should then define either of
    these methods.
    """

    def __init__(self, input_output_handler: InputOutputHandler, state_handler: StateHandler, scheduler: Scheduler,
                 activator: Activator) -> None:
        """
        The constructor of the abstract Mediator class.

        Parameters
        ----------
        input_output_handler : input_output_handler.InputOutputHandler
            The input-output handler.
        state_handler : state_handler.StateHandler
            The state handler.
        scheduler : scheduler.Scheduler
            The scheduler.
        activator : activator.Activator
            The activator.

        Raises
        ------
        base.exceptions.MediatorError
            If more than one base class of an event handler defines the mediating or the argument construction method.
        """
        super().__init__(input_output_handler, state_handler, scheduler, activator)
        self._mediating_methods = self._construct_methods_dictionary("mediate_")
        self._out_state_arguments_methods = self._construct_methods_dictionary("get_arguments_")
        self._event_handler_with_shortest_event_time = None
        self._out_state_arguments = {}

    def _construct_methods_dictionary(self, starting_string: str) -> Dict[EventHandler, Callable[..., Any]]:
        """
        Construct a dictionary which maps from the event handlers onto the method defined in this class which starts
        with the given string and ends with the class name of the event handler or any base class of it.
        """
        functions_dictionary = {}
        for event_handler in self._event_handlers_list:
            # We don't have to worry about an alias set by the factory, since the original class is a subclass
            method = getattr(self, starting_string + strings.to_snake_case(event_handler.__class__.__name__), False)
            if not method:
                # No method found, check if the wanted method is defined for a subclass
                possible_methods = [getattr(self, starting_string + strings.to_snake_case(subclass_name), False)
                                    for subclass_name in _get_bases_names(event_handler.__class__)]
                possible_methods = [method for method in possible_methods if method]
                if len(possible_methods) == 0:
                    # No subclass defines a mediating method
                    continue
                if len(possible_methods) > 1:
                    raise MediatorError("More than one possible method for the event handler {0} "
                                        "(read out from the list of base classes): {1}"
                                        .format(event_handler.__class__.__name__, possible_methods))
                method = possible_methods[0]
            functions_dictionary[event_handler] = method
        return functions_dictionary

    def get_arguments_end_of_run_event_handler(self) -> Tuple[Any]:
        """
        Get the arguments for the send_out_state method of an EndOfRunEventHandler.

        The end-of-run event handler needs the extracted active global state in order to time-slice it in its out-state.
        The precise format of the extracted active global state depends on the used state handler.

        Returns
        -------
        (Any, )
            The extracted active global state.
        """
        return (self._state_handler.extract_active_global_state(),)

    def get_arguments_end_of_chain_event_handler(self, new_active_identifiers: Sequence[Any]) -> Tuple[Any, List[Any]]:
        """
        Get the arguments for the send_out_state method of an EndOfChainEventHandler.

        The end-of-chain event handler returned the identifiers of the units which should become active together with
        the candidate event time in the send_event_time method. For the send_out_state method, the event handler needs
        the extracted active global state and the extracted global state of each of the identifiers to activate, in
        order to exchange the velocity.
        The precise format of the extracted active global state and the identifiers depends on the used state handler.

        Parameters
        ----------
        new_active_identifiers : Sequence[Any]
            The sequence of global state identifiers which should become active.

        Returns
        -------
        (Any, List[Any])
            The extracted active global state, the list of extracted global states for the identifiers which should
            become active.
        """
        return (self._state_handler.extract_active_global_state(),
                [self._state_handler.extract_from_global_state(identifier) for identifier in new_active_identifiers])

    def get_arguments_sampling_event_handler(self) -> Tuple[Any]:
        """
        Get the arguments for the send_out_state method of a SamplingEventHandler.

        The sampling event handler needs the extracted active global state in order to time-slice it in its out-state.
        The precise format of the extracted active global state depends on the used state handler.

        Returns
        -------
        (Any, )
            The extracted active global state.
        """
        return (self._state_handler.extract_active_global_state(),)

    def get_arguments_cell_veto_event_handler(self, cell: Any) -> Tuple[Any]:
        """
        Get the arguments for the send_out_state method of a CelLVetoEventHandler.

        The cell-veto event handler returned the sampled cell identifier of a possible target unit together with
        the candidate event time in the send_event_time method. For the send_out_state method, the event handler needs
        the extracted global state of the unit in the target cell, if there is one. If there is no unit in the target
        cell, the extracted global state is set to None.
        The precise format of the extracted global state and the identifiers depends on the used state handler.

        Parameters
        ----------
        cell : Any
            The sampled target cell identifier.

        Returns
        -------
        (Any, )
            The extracted global state of the unit within the sampled target cell.
        """
        identifier = self._activator.get_info_internal_state(self._event_handler_with_shortest_event_time, cell)
        if identifier is None:
            return (None,)
        else:
            return (self._state_handler.extract_from_global_state(identifier),)

    def get_arguments_composite_objects_lifting(self, *identifiers: Any) -> Tuple[List[Any]]:
        """
        Get the arguments for the send_out_state method of a CompositeObjectsLifting event handler.

        The composite-objects-lifting event handler returned the identifiers of the composite objects involved in the
        factor treated by this event handler together with the candidate event time in the send_event_time method. For
        the send_out_state method, the event handler needs the extracted global state of these composite objects.
        The precise format of the extracted global state and the identifiers depends on the used state handler.

        Parameters
        ----------
        identifiers : Any
            The global state identifiers of the composite objects.

        Returns
        -------
        (List[Any], )
            The extracted global state of the composite objects.
        """
        return ([self._state_handler.extract_from_global_state(identifier) for identifier in identifiers],)

    def get_arguments_root_leaf_unit_active_switcher(self) -> Tuple[List[Node]]:
        """
        Get the arguments for the send_out_state method of a RootLeafUnitActiveSwitcher.

        The root-leaf-unit-active switcher needs the active composite objects (with either induced or independent
        velocity) in order to change the independent velocity from the composite object to a leaf unit of the composite
        object or vice versa.
        This mediating method is, as the RootLeafUnitActiveSwitcher, designed to work together with the
        TreeStateHandler. Here the extracted global state is a branch of cnodes containing units. Also, the event
        handlers are responsible for keeping the time-slicing of composite objects and it's point masses consistent.

        Returns
        -------
        (List[base.node.Node], )
            The extracted global state for all active composite objects.
        """
        active_root_unit_identifiers = set()
        for root_cnode in self._state_handler.extract_active_global_state():
            active_root_unit_identifiers.add(root_cnode.value.identifier[:1])
        return ([self._state_handler.extract_from_global_state(identifier)
                 for identifier in active_root_unit_identifiers],)

    def get_arguments_initial_chain_start_of_run_event_handler(self, initial_active_identifier: Any) -> Tuple[Any]:
        """
        Get the arguments for the send_out_state method of an InitialChainStartOfRunEventHandler.

        The initial chain start-of-run event handler returned the identifiers of the units which should initially become
        active at the start of the run together with the candidate event time in the send_event_time method.
        For the send_out_state method, the event handler needs the extracted global state of the identifier to activate,
        in order to set the initial velocity.
        The precise format of the extracted active global state and the identifier depends on the used state handler.

        Parameters
        ----------
        initial_active_identifier : Any
            The global state identifier which should initially become active.

        Returns
        -------
        (Any, )
            The extracted global states for the identifier which should initially become active.
        """
        return (self._state_handler.extract_from_global_state(initial_active_identifier),)

    def mediate_end_of_run_event_handler(self) -> None:
        """
        Mediating method of an EndOfRunEventHandler.

        If the end-of-run event handler is connected to an output handler, it gets the full extracted global state as an
        argument of the write method to start the sampling.
        Also, this method raises an base.exceptions.EndOfRun exception in order to end the run.

        Raises
        ------
        base.exceptions.EndOfRun
            Always.
        """
        if self._event_handler_with_shortest_event_time.output_handler is not None:
            self._input_output_handler.write(self._event_handler_with_shortest_event_time.output_handler,
                                             self._state_handler.extract_global_state())
        raise EndOfRun

    def mediate_sampling_event_handler(self) -> None:
        """
        Mediating method of a SamplingEventHandler.

        The output handler to which the sampling event handler is connected gets the full extracted global state as an
        argument of the write method to start the sampling.
        """
        self._input_output_handler.write(self._event_handler_with_shortest_event_time.output_handler,
                                         self._state_handler.extract_global_state())

    def mediate_dumping_event_handler(self) -> None:
        """
        Mediating method of a DumpingEventHandler.

        The output handler to which the dumping event handler is connected gets the mediator as an argument of the write
        method to dump the full run.
        """
        self._input_output_handler.write(self._event_handler_with_shortest_event_time.output_handler,
                                         self)
