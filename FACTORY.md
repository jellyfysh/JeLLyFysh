# JeLLyFysh Factory

This file discusses the usage of the function `build_from_config` function within the 
[`jellyfysh.base.factory`](jellyfysh/base/factory.py) module, together with configurations parsed from `.ini` files via 
the `ConfigParser` class out of the Python module [`configparser`](https://docs.python.org/3/library/configparser.html).

The factory function `build_from_config` has the definition (including type hints):

```Python3
def build_from_config(config: configparser.ConfigParser, section: str, package: str, class_name:str = None) -> typing.Any
```

It will instantiate and return the class object specified by the section with the variables set in this section. The 
arguments are described in detail below:

1. `config`: The `ConfigParser` instance, which has parsed the `.ini` file.
2. `section`:  The section of the .ini file where the arguments of the `__init__` method of the class are given. The 
property/value pairs of the section must correspond to the arguments of the `__init__` method. Properties and values 
must be in snake_case, section names in CamelCase.
3. `package`: The package which includes the module of the class. The module and the class must have the same name 
(module in snake_case, class in CamelCase). There is one exception: if there exists a package with the same
name as the module within the given package, this package will be added and entered automatically before the module is 
imported.
4. `class_name`: The name of the class. If it is `None`, `section` is used as the class name.

The parsed values must be converted to the correct types. The factory determines these types based on the **required**
type hints in the `__init__` method. Currently supported types are `bool`, `float`, `int`, and `str`. Similar to the
`configparser` module, this factory recognizes booleans in the section from `yes/no`, `on/off`, `true/false` and
`1/0`. User defined classes are also be possible (the factory is then called recursively and the package is determined 
via the type hint). If the user defined class also needs some `__init__` arguments, these must be defined in a separate
section. Finally, `typing.Sequence` and `typing.Iterable` consisting of one of all these types is also possible. The 
factory creates a list in this case.

It is possible to assign aliases for user defined classes within the configuration file. In the values they should be 
given as `alias (real_class_name)`. The arguments of the `__init__` method for this class are then located in the 
section `[Alias]`. If the factory function is directly called with an alias section, the real class name must be 
specified in the `class_name` argument. For aliased classes, this factory will effectively subclass the real class and 
set `__class__.__name__` to `Alias (RealClassName)`. The alias (or the real class name if there is no alias) gets 
extracted from this attribute via the `get_alias` function of the [`jellyfysh.base.factory`](jellyfysh/base/factory.py) 
module.

As mentioned, user defined classes, which should be constructable by this factory, must be defined in their own module 
with the module name as the class name in snake_case. If the type hint refers to an abstract class, the inheriting class
must be defined in the same package.

The factory function will append the section argument of each call to the `used_sections` list of the 
[`jellyfysh.base.factory`](jellyfysh/base/factory.py) module. This list is used to check if all sections of the 
configuration file were used.

The following sections repeat the most important points when writing a class which should be constructed by the JF 
factory and when writing a configuration file. Also a detailed example is given.

## Writing a class

As discussed, some conventions must be followed to implement a class which can be created by the JF factory:

The values for the options set in the section in the `.ini` file are read in as strings and must be converted to the 
correct types. This is done via reading out the type hints according to 
[PEP484](https://www.python.org/dev/peps/pep-0484/) of the `__init__` method. Currently supported types are `bool`, 
`float`, `int`, and `str`, as well as `typing.Iterable` and `typing.Sequence` objects of these types.

On top of that, all other user defined classes are possible as well, since the function of the factory will be called 
recursively. Note that the user defined class must have the same name as the module in which it is defined.
Moreover if one uses abstract classes as type hints, make sure that the derived classes
and the abstract class are defined within the same package (most probably in different files).
Again, `typing.Iterable` and `typing.Sequence` objects of user defined classes are possible as well.

## Writing the configuration file

The executable `jellyfysh/run.py` script implies one mandatory section in the configuration file with two necessary 
properties:

```INI
[Run]
mediator = some_mediator
setting = some_setting
```

The mediator property specifies the mediator. Possible classes are located in the `jellyfysh.mediator` package. The 
setting property specifies the setting. Possible classes are located in the `jellyfysh.setting` package. All other 
sections in the configuration file are then implied by the `__init__` methods of the mediator and the setting class 
(see example below).

Each section must contain all arguments of the `__init__` methods of the classes. Possible optional keyword arguments 
can be left out if they don't need to be changed. The properties should be named like the arguments of the 
`__init__` method.

If a user defined class should be the value of a property, specify it in snake_case. Then, there are two options. First,
the user defined class is default constructable and no keyword arguments should be changed. For this case, nothing more
must be done in the configuration file. Second, the class needs some arguments. Then, a section with the same name as 
class in CamelCase must exist in the same configuration file. For user defined classes, an alias can be used. The value 
is then `alias (real_class_name)` and the section should be named `[Alias]`.

If the type of an argument in a `__init__` method is `typing.Iterable` or `typing.Sequence`, the values in the 
configuration file should be separated by `, ` (comma followed by a space).

## Example

As an example of a configuration file, consider the following section within the 
[`jellyfysh/config_files/2018_JCP_149_064113/coulomb_atoms/power_bounded.ini`](jellyfysh/config_files/2018_JCP_149_064113/coulomb_atoms/power_bounded.ini) file:

```INI
[SingleProcessMediator]
state_handler = tree_state_handler
scheduler = heap_scheduler
activator = tag_activator
input_output_handler = input_output_handler
```

The class `SingleProcessMediator` is defined in the `jellyfysh.mediator` package. The mediator can therefore be 
constructed with the call (which is done in the [`jellyfysh/run.py`](jellyfysh/run.py) script):

```Python3
mediator = factory.build_from_config(config, "SingleProcessMediator", "mediator")
```

Here, `config` is the `ConfigParser` instance which parsed the relevant configuration file.

The `SingleProcessMediator` class has the following definition of the `__init__` method:

```Python3
class SingleProcessMediator(Mediator):
    def __init__(self, input_output_handler: InputOutputHandler, state_handler: StateHandler, scheduler: Scheduler,
                 activator: Activator) -> None
```

You can see, that the section in the configuration file has all the arguments (except `self`) of this method as the 
properties.

Let's look at the first argument. Via the type hint, the factory can deduce that the package of the input-output handler 
is `input_output_handler`. The next call of the factory function is therefore (which is called recursively in the 
factory):

```Python3
input_output_handler = factory.build_from_config(config, "InputOutputHandler", "input_output_handler")
```

The section of the input-output handler looks like (note that we modified this section here compared to the mentioned 
configuration file for demonstration purposes):

```INI
[InputOutputHandler]
output_handlers = separation_output_handler, another_output_handler (separation_output_handler)
input_handler = random_input_handler
```

The `__init__` method of the `InputOutputHandler` class is:

```Python3
class InputOutputHandler(object):
     def __init__(self, input_handler: InputHandler, output_handlers: Sequence[OutputHandler] = ()) -> None    
```

Three interesting things are happening in this section regarding the output handlers:

1. The `output_handlers` argument is an optional argument. Therefore one could have left out this argument in the 
section and then no output handlers would have been created.
2. The `output_handlers` argument expects a sequence of output handlers. Therefore the corresponding value in the 
section is a comma separated list.
3. The configuration file specifies that two output handlers should be created. These are of the same type, but one of 
them uses an alias. This allows to set different options for these two output handlers.

To construct the two output handlers, the factory will use the following two function calls:

```Python3
first_output_handler = factory.build_from_config(config, "SeparationOutputHandler", 
                                                 "input_output_handler.output_handler")
second_output_handler = factory.build_from_config(config, "AnotherOutputHandler", 
                                                  "input_output_handler.output_handler",
                                                  "SeparationOutputHandler")             
```

The second call includes the `class_name` argument because of the use of the alias.
These two calls imply that two more sections should appear in the configuration file

```INI
[SeparationOutputHandler]
filename = first_file.dat

[AnotherOutputHandler]
filename = second_file.dat
```

The corresponding `__init__` method is

```Python3
class SeparationOutputHandler(OutputHandler):
    def __init__(self, filename: str) -> None
```

Since the argument is of type `str`, the factory now does not have to call itself recursively anymore.

During the creation of the second output handler, the factory subclasses the `SeparationOutputHandler` class and sets
the `__class__.__name__` attribute to `AnotherOutputHandler (SeparationOutputHandler)`. The same attribute of the first
output handler is simply `SeparationOutputHandler`. The `get_alias` function returns `AnotherOutputHandler` for the 
first case and `SeparationOutputHandler` for the second case.
