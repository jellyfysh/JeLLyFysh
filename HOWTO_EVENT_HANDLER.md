# How to implement event handlers

The first objects an user might probably want to add to JeLLyFysh are event handlers. These calculate candidate event
times and out-states of events. This file gives some hints on how to add new event-handlers classes when using a tree 
state handler (`state_handler = tree_state_handler` in the section of the mediator) and a tag activator 
(`activator = tag_activator`). 

## The abstract methods of an event handler

The abstract class for an event handler is located in the 
[`src/event_handler/event_handler.py`](src/event_handler/event_handler.py) module. There, you can see the following two
definitions of abstract methods:
```Python3
def send_event_time(self, *in_state: Sequence[Any]) -> Union[float, Tuple[float, Sequence[Any]]]
```
and
```Python3
def send_out_state(self, *args: Any) -> Any
```
These two will be discussed in detail in the following.

### The `send_event_time` method

This method should calculate a candidate event time based on an optional in-state (for a discussion on how to get the 
correct in-state see below). If there is one, the in-state consists of a sequence of cnodes, that are `Node` objects 
(see the [`src/base/node.py`](src/base/node.py) module) containing a `Unit`object (see the 
[`src/base/unit.py`](src/base/unit.py) module). Each cnode in the sequence represents a root node in the tree state 
handler and defines a branch. Use these cnodes to calculate the candidate event time and make sure to store them for the
following call of the `send_out_state` method. 

Take a look at the abstract event handler classes located in the
[`src/event_handler/abstracts`](src/event_handler/abstracts) directory for some useful base classes (which, for 
example, extract a single active leaf unit from the cnodes or offer functions to keep the branches consistent).

It is possible, that the `send_out_state` method relies on additional arguments to construct the out-state. Then you 
need to take care of two things:
1. Implement an argument construction method in the mediator.
2. Return the arguments this argument construction method needs in a sequence together with the candidate event time. 
The sequence will be unpacked before the arguments are given to the argument construction method.

### The argument construction method
In the `Mediator` class in the module [`src/mediator/mediator.py`](src/mediator/mediator.py) you can add your own 
argument construction method for your event handler class. The name of the method should begin with `get_arguments_`
and end with the name of your class in snake_case. The arguments of this method are determined by the `send_event_time`
method.

Within this method you can now use both the global state in the state handler and the internal state in the activator 
to construct the additional arguments of the `send_out_state` method. These should be returned as a sequence, since
the returned values are also unpacked before they are handed to the `send_out_state` method. 

Make sure that you modify neither the global nor the internal state in this method. Just extract information from 
it.

### The `send_out_state` method

Within this method you can use the stored in-state from the `send_event_time` method and the additional arguments 
constructed in the mediator to create the out-state of the event. In order to keep all branches consistent in time, all
units in the in-state should be time-sliced to the candidate event time.

## Creating the in-state

In order for your event handler to receive the correct in-state, you need to take a look at the taggers which are 
located in the [`src/activator/tagger`](src/activator/tagger) directory. These taggers are used in the tag activator for 
several things. 

First of all, they get an event handler and a tag on initialization. Also, the `number_event_handlers` argument of the 
`__init__` method of the tagger specifies how often it should deepcopy the event handler. All the events by event 
handlers from a certain tagger instance will have the tag of that tagger. Each tagger specifies which other taggers 
should be asked to create an event after a preceding event with its tag, and also which events should be trashed in the
scheduler after a preceding event with its tag.

If a tagger should create events, its `yield_identifiers_send_event_time` method is used. This method generates
in-state identifiers. These identifiers are passed to the state handler and then to the `send_event_time` method
of the tagger's event handler. The state handler constructs a branch for each global state identifier, that is it 
copies the information of the node with the identifier together with the information of all its ancestors and 
descendants. It then returns the root cnode of this branch.

The number of event handlers in the tagger should be large enough to treat the number of in-state identifiers generated
by the `yield_identifiers_send_event_time` method (This is only true if the tagger should only create events after all
its event have been trashed. If this is not the case, also the events in the scheduler with the given tag have to be 
considered.) If there are not enough event handlers, the tag activator raises an TagActivatorError.

For your new event handler, either use one of the existing taggers or implement a new one. Then use the `setting` 
package which gives you access to all global state identifiers to create your in-states based on the active global
state. If you want to use an internal state to generate the in-state identifiers, inherit from the
`TaggerWithInternalState` class, which is defined in the 
[`src/activator/tagger/abstracts.py`](src/activator/tagger/abstracts.py) module. 

## The mediating method

After an event from an event handler has been committed to the global state, optionally a mediating method in the 
mediator can be run. This is needed, for example, to trigger sampling in an output handler. if you want exactly this,
just inherit from the `SamplingEventHandler` class which is located in the 
[`src/event_handler/abstracts/sampling_event_handler.py`](src/event_handler/abstracts/sampling_event_handler.py)
module. There, you only need to specify the name of the output handler which should be run. This output handler
then gets the full global state as an argument of its `write` method.

If you really need to add your own mediating method, this can be done again in the `Mediator` class in the module 
[`src/mediator/mediator.py`](src/mediator/mediator.py). The name of the method should begin with `mediate_`
and end with the name of your class in snake_case. This method should neither have arguments nor return anything.

## Concluding remarks

You are of course free to write your own event-handler classes. After all, the project is open-source :). However, a 
fail-safe way of writing your own classes, may be to propose them as JeLLyFysh Issues on GitHub. We will be more than 
happy to assist you with your tasks.
