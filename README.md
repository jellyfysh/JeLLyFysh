[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)

This is the 2.0-alpha version of JeLLyFysh. It implements the generalized Newtonian event-chain Monte Carlo algorithm, 
which was introduced in [\[Hoellmer2023\]](https://doi.org/10.48550/arXiv.2305.02979) (based on 
[\[Klement2019\]](https://doi.org/10.1063/1.5090882) that introduced Newtonian event-chain Monte Carlo for 
hard spheres). 

This is the version of JeLLyFysh that was used for the simulations in 
[\[Hoellmer2023\]](https://doi.org/10.48550/arXiv.2305.02979). The code is not properly documented and organized yet. 
This will be achieved in the final 2.0 version of JeLLyFysh. In contrast to this alpha version that only allows for 
generalized Newtonian event-chain Monte Carlo, the final 2.0 version will also allow to switch between different 
event-chain Monte Carlo variants.

All configuration files for the simulations of [\[Hoellmer2023\]](https://doi.org/10.48550/arXiv.2305.02979) are 
contained in the [jellyfysh/config_files/arXiv_2305_02979](jellyfysh/config_files/arXiv_2305_02979) directory. Besides
the configuration files for this alpha version of JeLLyFysh, this also includes configuration files for 
molecular-dynamics simulations in [Lammps](https://www.lammps.org/#gsc.tab=0), and for Metropolis simulations in 
[DL_MONTE](https://gitlab.com/dl_monte/DL_MONTE-2).

# JeLLyFysh

The JeLLyFysh Python application implements the event-chain Monte Carlo algorithm (ECMC), an event-driven irreversible 
Markov-chain Monte Carlo algorithm for classical *N*-body simulations in statistical mechanics, biophysics and 
electrochemistry. The application's architecture closely mirrors the mathematical formulation of ECMC. For a detailed 
description of the design of this application, see [\[Hoellmer2020\]](https://doi.org/10.1016/j.cpc.2020.107168). For a 
closely connected discussion of ECMC, see [\[Faulkner2018\]](https://doi.org/10.1063/1.5036638). (For more information 
on the references, see the [References.bib](References.bib) file.)

## Installing

The JeLLyFysh Python application can be executed with any Python implementation which supports Python version >= 3.6. It 
is tested with cPython and [PyPy](http://pypy.org). We recommend using the Python 3.8 compatible PyPy3.8 v7.3.7 
(see https://www.pypy.org/download.html), which yields a significant higher performance of the JeLLyFysh application.

The JeLLyFysh application is installed with the help of setuptools (see [setup.py](setup.py)). After you have downloaded 
and uncompressed the JeLLyFysh source, or cloned it from git, first choose a Python interpreter that supports version 
3.6 or greater. We recommend installing the JeLLyFysh application into a virtual environment (e.g., using virtualenv, 
venv, pyenv, ...). This documentation uses the command `pypy3` to call the Python interpreter of the virtual 
environment. If you use a different Python interpreter, change the code accordingly.

1.  Ensure that pip is installed by running

    ```console
    pypy3 -m pip --help
    ```
   
    If this command does not show pip's help message, install it (see, e.g., 
    https://pip.pypa.io/en/stable/installing/). Moreover, the C extensions of the JeLLyFysh application require a C 
    compiler (for example, gcc or clang).


2.  Ensure that pip, setuptools, and wheel are installed and on the latest version by running

    ```console
    pypy3 -m pip install --upgrade pip setuptools wheel
    ```

   
3. You are now ready to install the JeLLyFysh application. To do so, navigate into the directory of this file (and the 
    relevant [setup.py](setup.py) script), and execute

    ```console
    pypy3 -m pip install .
    ```
   
    This will first install the required dependencies of the JeLLyFysh application (see
    [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md)). Afterwards, it compiles the C extensions of the JeLLyFysh 
    application, and exports the [`jellyfysh`](jellyfysh) package to the site-packages directory of your interpreter. 
    Finally, it creates the executables described in the [section "Using JeLLyFysh"](#using-jellyfysh).


4.  A limited amount of modules depends on additional Python packages that are not installed by setuptools. Take a look 
    at the file [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md) to see which parts of the JeLLyFysh application can 
    only be used after additional packages have been installed manually with the help of pip.

### Uninstalling

In order to uninstall the JeLLyFysh application, use

```console
pypy3 -m pip uninstall jellyfysh
```

### Editable mode

Using the command `pypy3 -m pip install .` to install the JeLLyFysh application basically copies and compiles the source 
code into the site-packages directory of your Python interpreter. Therefore, you would need to reinstall the JeLLyFysh 
application anytime source code was changed. Fortunately, setuptools provides an editable mode. This installs the 
JeLLyFysh application without copying any files. Any changes in source code are directly active without reinstallation. 
To install the JeLLyFysh application in editable mode, use

```console
pypy3 -m pip install -e .
```

Of course, a C extension must still be recompiled after any changes to the corresponding C code. This can be achieved by
using

```console
    pypy3 setup.py build_ext -i
```

Uninstalling the JeLLyFysh application in editable mode works as described above.

## Running the tests

To run the tests, simply enter the [`unittests`](unittests) directory and run the 
[`run_tests.py`](unittests/run_tests.py) script with your Python interpreter. These tests may take a while.

## Using JeLLyFysh

The installation process described in the [section "Installing"](#installing) creates three executables that are 
described in the following. Note that you might have to add the directory where pip installs executables to your `PATH` 
environment variable in order to access these executables.

### 1. jellyfysh 

The `jellyfysh` executable runs the JeLLyFysh application and relies on the [`run.py`](jellyfysh/run.py) script, which 
is located in the [`jellyfysh`](jellyfysh) package. (The `jellyfysh` executable is therefore analogous to running the 
[`run.py`](jellyfysh/run.py) script with your Python interpreter.)

The user interface for each run of the JeLLyFysh application consists in a configuration file that is an argument of the 
`jellyfysh` executable. Configuration files should follow the [INI-file format](https://en.wikipedia.org/wiki/INI_file).

The `jellyfysh` executable expects the path to the configuration file as the first positional argument. The executable 
also takes optional arguments. These are:

- `-h`, `--help`: Show the help message and exit.
- `-V`, `--version`: Show program's version number and exit.
- `-v`, `--verbose`: Increase verbosity of logging messages (multiple -v options increase the verbosity, the maximum is 
2).
- `-l LOGFILE`, `--logfile LOGFILE`: Specify the logging file. 

A configuration file is composed of sections that each correspond to a class of the JeLLyFysh application. The only
required section for the run script is

```INI
[Run]
mediator = some_mediator
setting = some_setting
```

`some_mediator` corresponds to the used mediator in the run. The mediator serves as a central hub in the application 
and also hosts the iteration loop over the legs of the continuous-time evolution of ECMC. The two possible mediators are
[`single_process_mediator`](jellyfysh/mediator/single_process_mediator.py), which runs the application in a single process, 
and [`multi_process_mediator`](jellyfysh/mediator/multi_process_mediator/multi_process_mediator.py), which calculates the 
events of ECMC in separate processes. Both are located in the [`jellyfysh.mediator`](jellyfysh/mediator) package. 
Currently, we recommend the single-process version.

`setting` specifies the *NVT* physical parameters of the run. The two possible settings are 
[`hypercubic_setting`](jellyfysh/setting/hypercubic_setting.py) and 
[`hypercuboid_setting`](jellyfysh/setting/hypercuboid_setting.py) (both located in the 
[`jellyfysh.setting`](jellyfysh/setting) package).

The following sections of the configuration file choose the parameters in the `__init__` methods of the mediator and the 
setting. Each section contains pairs of properties and values. The property corresponds to the name of the argument in 
the `__init__` method of the given class, and its value provides the arguments. Properties and values should be given in 
snake_case, sections in CamelCase. For a detailed description of the JeLLyFysh factory, which parses the configuration 
file and constructs the specified classes, see the [FACTORY.md](FACTORY.md) file. There, also a detailed example for the 
interplay between the factory and a configuration file is given. All classes of JeLLyFysh are documented with docstrings 
to clarify their usage. For a general overview of the parts of JeLLyFysh, see 
[\[Hoellmer2020\]](https://doi.org/10.1016/j.cpc.2020.107168).

Instances of the event-handler classes in the JeLLyFysh application compute candidate events. Hints on how to implement 
your own event handlers are given in the [HOWTO_EVENT_HANDLER.md](HOWTO_EVENT_HANDLER.md) file (see 
[\[Hoellmer2020\]](https://doi.org/10.1016/j.cpc.2020.107168) for details).

### 2. jellyfysh-examples

As a starting point into the JeLLyFysh application, we provide exemplary configuration files. The `jellyfysh-examples` 
executable (which relies on the [`create_examples.py`](jellyfysh/create_examples.py) script in the 
[`jellyfysh`](jellyfysh) package) creates a `jellyfysh-examples` directory in the current working directory. This 
directory contains several exemplary configuration files. The configuration files in the 
[`jellyfysh-examples/config_files/2018_JCP_149_064113`](jellyfysh/config_files/2018_JCP_149_064113) directory are 
described in detail in the "Cookbook" section (Section 5) of 
[\[Hoellmer2020\]](https://doi.org/10.1016/j.cpc.2020.107168). They generate output in the 
[`jellyfysh-examples/output/2018_JCP_149_064113`](jellyfysh/output/2018_JCP_149_064113) directory. This output directory 
also contains plotting scripts that compare the results of the JeLLyFysh application with reference data. (The plotting 
scripts, however, rely on additional external dependencies that have to be installed manually, see 
[EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md).) More examples can be found 
in the [`jellyfysh-examples/config_files/hard_disk_dipoles`](jellyfysh/config_files/hard_disk_dipoles) directory that 
generate output in the [`jellyfysh-examples/output/hard_disk_dipoles`](jellyfysh/output/hard_disk_dipoles) directory.

For your first run of the JeLLyFysh application, you may use the following commands:

```console
jellyfysh-examples
cd jellyfysh-examples
jellyfysh config_files/2018_JCP_149_064113/coulomb_atoms/power_bounded.ini
```

This simulates two charged point masses in a three-dimensional cubic simulation box with periodic boundary conditions. 
The generated file `output/2018_JCP_149_064113/coulomb_atoms/SamplesOfSeparation_PowerBounded.dat` contains samples of 
the separation between the point masses. Only if you installed the necessary external dependencies described in 
[EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md), you can plot the data and compare it to reference data from a 
reversible Monte Carlo simulation by running the following commands (again assuming that `pypy3` calls your Python 
interpreter):

```console
cd output/2018_JCP_149_064113/coulomb_atoms
pypy3 plot_histogram_coulomb_atoms.py 
```

This creates a plot and stores it in the `coulomb_atoms.pdf` file.

### 3. jellyfysh-resume

The `jellyfysh-resume` executable (which relies on the [`resume.py`](jellyfysh/resume.py) script in the 
[`jellyfysh`](jellyfysh) package) resumes a dumped run of the JeLLyFysh application. Such a dumped run is generated by 
including the [`DumpingOutputHandler`](jellyfysh/input_output_handler/output_handler/dumping_output_handler.py) in 
a run of the JeLLyFysh application.

We provide an exemplary configuration file that dumps a run of the JeLLyFysh application. To run this example, use the 
following commands:

```console
jellyfysh-examples
cd jellyfysh-examples
jellyfysh config_files/2018_JCP_149_064113/coulomb_atoms/power_bounded_dump.ini
```

The `DumpingOutputHandler` pickles all objects of the running simulation and dumps them into a file whose name depends 
on the used Python interpreter as `dump_PythonImplementation_PythonVersion.dat`. You can resume the run from the moment 
where the dump was created by using

```console
jellyfysh-resume dump_PythonImplementation_PythonVersion.dat
```

The `jellyfysh-resume` executable is mainly used for debugging.

## Contributing

As an open-source project, the JeLLyFysh organization solicits contributions from the community. Please read 
the [contribution guideline](CONTRIBUTING.md) for details.

If you find a bug, please raise an Issue here on GitHub to let us know.

Please note that this project is released with the Contributor Covenant [code of conduct](CODE_OF_CONDUCT.md). By 
participating in this project you agree to abide by its terms. Report unacceptable behavior to 
[werner.krauth@ens.fr](mailto:werner.krauth@ens.fr).

## Versioning

Versioning of the JeLLyFysh project adopts two-to-four-field version numbers defined as Milestone.Feature.AddOn.Patch. 
The current version 1.0 represents the first development milestone which reproduces published data in 
[\[Faulkner2018\]](https://doi.org/10.1063/1.5036638). Patches and bugfixes of this version will be given number 
1.0.0.1, 1.0.0.2 etc. New configuration files and required extensions are expected to lead to versions 1.0.1, 1.0.2 etc. 
In JeLLyFysh development, two-field versions (2.0, 3.0, etc.) may introduce incompatible code, while three- and 
four-field version numbers are intended to be backward compatible.

## Authors

Check the [AUTHORS.md](AUTHORS.md) file to see who participated in this project.

## License

This project is licensed under the GNU General Public License, version 3 (see the [LICENSE](LICENSE) file).

## Contact

If you have questions regarding the JeLLyFysh application, just raise an issue here on GitHub. We are happy to help you!

## Citation

If you use JeLLyFysh in published work, please cite the following reference (see [Hoellmer2020] in 
[References.bib](References.bib)):

Philipp Höllmer, Liang Qin, Michael F. Faulkner, A. C. Maggs, and Werner Krauth,  
JeLLyFysh-Version1.0 -- a Python application for all-atom event-chain Monte Carlo,  
Computer Physics Communications, Volume 253, 107168 (2020), https://doi.org/10.1016/j.cpc.2020.107168.
