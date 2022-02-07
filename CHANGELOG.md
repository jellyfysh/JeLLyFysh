# Changelog

All notable changes to the JeLLyFysh application will be documented in this file.  The latest version comes first, and
the release dates are written in the [ISO 8601 format](https://en.wikipedia.org/wiki/ISO_8601) [YYYY]-[MM]-[DD].

The version numbers of the JeLLyFysh application adopt [semantic versioning](http://semver.org) with two-to-four-field 
version numbers defined as Milestone.Feature.AddOn.Patch. We distinguish between backwards compatible changes, and 
backwards incompatible changes. Here, compatible refers to changes that do not break existing configuration files. 
In JeLLyFysh development, two-field versions (2.0, 3.0, etc.) may introduce incompatible code, while three- and 
four-field version numbers are intended to be backward compatible (or to yield only minor changes in existing 
configuration files). Any version in this changelog explicitly reports which changes in existing configuration files are 
required.

## Version1.1 &ndash; 2022-02-07

### Major Changes

- The arXiv reference [\[Hoellmer2019\]](https://arxiv.org/abs/1907.12502) was replaced by the journal reference 
  [\[Hoellmer2020\]](https://doi.org/10.1016/j.cpc.2020.107168). If you use the JeLLyFysh application in published work,
  please cite the updated reference (see [References.bib](References.bib)).

- New installation process via setuptools (see [`setup.py`](setup.py)). All source code is now
  part of a `jellyfysh` Python package. The new installation process is described in the [README.md](README.md) file. 
  This has the following advantages:
  - Automatically installs all required external dependencies (see 
    [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md)).
  - Simplifies the usage and installation of C extensions.
  - Enables the creation of executables like `jellyfysh` that replace explicitly running Python scripts like 
    [`run.py`](jellyfysh/run.py) with a Python interpreter.
  - Allows for simple access to the `jellyfysh` source code without manipulating `PYTHONPATH` after installation.

- Installation of three executables `jellyfysh`, `jellyfysh-resume`, and `jellyfysh-examples` via setuptools. They are 
  described in detail in the [README.md](README.md) file and simplify using the JeLLyFysh application.

- Some parts of JeLLyFysh are now implemented in C using [CFFI](https://cffi.readthedocs.io/en/latest/) 
  for the interface (which is the [recommended way for PyPy](https://doc.pypy.org/en/latest/extending.html)). This 
  increases the performance of JeLLyFysh.

- The JeLLyFysh application now only supports Python implementations that support Python versions >= 3.6.

- Added support for ECMC simulations of hard spheres, and of hard dipoles consisting of two tethered hard spheres (which 
  is inspired by a recent work of some members of the JeLLyFysh organization, see 
  [\[Hoellmer2021\]](https://arxiv.org/abs/2111.11943) in [References.bib](References.bib)). JeLLyFysh includes new 
  [exemplary configuration files](jellyfysh/config_files/hard_disk_dipoles) for ECMC simulations of two-dimensional 
  tethered hard-disk dipoles together with reference data and plotting scripts.

- Support for general velocities of active units (that are not only aligned with the coordinate axes as in the previous 
  version of JeLLyFysh). This is possible of the following major changes in the [`potential`](jellyfysh/potential) 
  package:
  - The `derivative` method of the abstract [`Potential`](jellyfysh/potential/potential.py) class now receives the full 
    velocity of the active unit as the first argument, and returns the directional time derivative. 
  - The `displacement` method of the abstract [`InvertiblePotential`](jellyfysh/potential/potential.py) now receives the 
    full velocity of the active unit as the first argument, and returns the time displacement.

- Added support for sequential ECMC which rotates the velocity vector of the independent active unit by a constant 
  angle increment in each end-of-chain event (see [\[Hoellmer2021\]](https://arxiv.org/abs/2111.11943)). Some new
  [exemplary configuration files](jellyfysh/config_files/hard_disk_dipoles) for ECMC simulations of two-dimensional 
  tethered hard-disk dipoles use this sequential ECMC.

- Candidate event times are now stored and compared by using a new [`Time`](jellyfysh/base/time.py) class that
  represents a time as the quotient and remainder of an integer division with one.

  This change is necessary because the candidate event times and time stamps of active units always increase in 
  simulations of JeLLyFysh. At the same time, event 
  handlers usually first compute a time displacement and then add this time displacement to a time stamp of an active 
  unit to determine the candidate event time. Here, the time displacements stay in the same order of magnitude during a 
  simulation. If floats are used to store the involved time stamp and resulting candidate event time, events with very 
  close time displacements may yield an equal candidate event time if the time stamp is already rather large. This leads 
  to a decreasing precision during the simulation. This problem is solved by the `Time` class because the precision of 
  the remainder stays constant during a simulation.

###  Required Changes of Configuration Files

- A recent work by some members of the JeLLyFysh organization (see 
  [\[Hoellmer2021\]](https://arxiv.org/abs/2111.11943) in [References.bib](References.bib)) showed that the independent
  active unit should be sampled randomly after an end-of-chain event. Keeping the same unit independent active before 
  and after an end-of-chain event can break irreducibility in simple models. Therefore, the
  `SameActivePeriodicDirectionEndOfChainEventHandler` was removed and should be replaced by the
  [`SingleIndependentActivePeriodicDirectionEndOfChainEventHandler`](jellyfysh/event_handler/single_independent_active_periodic_direction_end_of_chain_event_handler.py).
  This new end-of-chain event handler should be used together with the new 
  [`ActiveGlobalInStateTagger`](jellyfysh/activator/tagger/active_global_state_in_state_tagger.py) because it requires
  the active global state as the in-state in its `send_event_time` method.

  - Because this is a correction of a potential source of error of ECMC itself, we do not increase the Milestone version 
    number although small changes in existing configuration files are necessary.

  - In order to use existing configuration files with the new version 1.1 of the JellyFysh application, the following
    changes are necessary (here for the exemplary configuration file 
    [`jellyfysh/config_files/2018_JCP_149_064113/coulomb_atoms/power_bounded.ini`](jellyfysh/config_files/2018_JCP_149_064113/coulomb_atoms/power_bounded.ini)):
    1. In the `[TagActivator]` section, change the end-of-chain tagger that is connected to the end-of-chain event 
       handler from `no_in_state_tagger` to `active_global_in_state_tagger`. For example, the section 
     
       ```INI
       [TagActivator]
       taggers =
           coulomb (factor_type_map_in_state_tagger),
           sampling (no_in_state_tagger),
           end_of_chain (no_in_state_tagger),
           start_of_run (no_in_state_tagger),
           end_of_run (no_in_state_tagger)
       ```
       
       becomes 

       ```INI
       [TagActivator]
       taggers =
           coulomb (factor_type_map_in_state_tagger),
           sampling (no_in_state_tagger),
           end_of_chain (active_global_state_in_state_tagger),
           start_of_run (no_in_state_tagger),
           end_of_run (no_in_state_tagger)
       ```
       
    2. In the section of the end-of-chain tagger, use the new end-of-chain event handler. For example, replace the 
       section
    
       ```INI
       [EndOfChain]
       create = end_of_chain, coulomb
       trash = end_of_chain, coulomb
       event_handler = same_active_periodic_direction_end_of_chain_event_handler
       ```
       
       by 

       ```INI
       [EndOfChain]
       create = end_of_chain, coulomb
       trash = end_of_chain, coulomb
       event_handler = single_independent_active_periodic_direction_end_of_chain_event_handler
       ```

    3. In the section of the end-of-chain event handler, set the chain time instead of the chain length. For example,
       replace the section
    
       ```INI
       [SameActivePeriodicDirectionEndOfChainEventHandler]
       chain_length = 0.78965
       ```
       
       by 

       ```INI
       [SingleIndependentActivePeriodicDirectionEndOfChainEventHandler]
       chain_time = 0.78965
       ```

        
### Detailed List of Changes

#### Added

- The following changes were necessary in order to allow the installation of the JeLLyFysh application and its 
  executables via setuptools:
  - Added the [`setup.py`](setup.py) file that is used by setuptools for the installation.
  - Replaced the `src` directory by a [`jellyfysh`](jellyfysh) Python package.
  - Changed imports like `from estimator import Estimator` to `from jellyfysh.estimator import Estimator` everywhere.
  - The JeLLyFysh factory in the [`jellyfysh.base.factory`](jellyfysh/base/factory.py) module previously used the 
    `os.isdir` function to test whether certain directories exist in the source code. This only works, however, if the 
    current working directory was the `src` directory. Setuptools offers a resource management API, which stores the 
    location of the [`jellyfysh`](jellyfysh) package once it was installed. The new version uses the `resource_isdir` 
    function, and thus searches for directories in the [`jellyfysh`](jellyfysh) package. This change also made 
    some changes in the corresponding unittest of the factory necessary (because the testing classes that are 
    constructed in the unittest are not a part of the [`jellyfysh`](jellyfysh) package).
  - In the [`jellyfysh.setting.__init__`](jellyfysh/setting/__init__.py) module, the imports of the `HypercubicSetting` 
    and `HypercuboidSetting` classes were removed because this led to circular import problems. This implied changes in 
    all unittests which want to construct such a setting class.
  - In the [`jellyfysh/run.py`](jellyfysh/run.py) script, the arguments of the `main` function were removed so that 
    setuptools can create the `jellyfysh` executable from it (because an executable created by setuptools has to link to 
    a function without any arguments). The same is true for the [`jellyfysh/resume.py`](jellyfysh/resume.py) script that 
    is used for the `jellyfysh-resume` executable.
  - The unittests in [`unittests/test_config_files`](unittests/test_config_files) now use setuptool's resource 
    management API to locate the configuration files that should be tested. 
  - The [`unittests/run_tests.py`](unittests/run_tests.py) script does not set the `PYTHONPATH` to the `src` directory 
    anymore. 
  - The file `unittests/__init__.py` was removed. This is necessary because otherwise setuptools would not only install 
    the [`jellyfysh`](jellyfysh) package but also the [`unittests`](unittests) directory.

- Added the [`jellyfysh/create_examples.py`](jellyfysh/create_examples.py) script that uses the resource management API 
  of setuptools to copy the exemplary configuration files in the [`jellyfysh/config_files`](jellyfysh/config_files) 
  directory, and the plotting scripts and the reference data in the  [`jellyfysh/output`](jellyfysh/output) directory 
  to the current working directory. This script is used by the `jellyfysh-examples` executable.

- Added the [`Time`](jellyfysh/base/time.py) class that stores candidate event times and time stamps in the JeLLyFysh 
  application as the quotient and remainder of an integer division with one.

- Added the [`ListScheduler`](jellyfysh/scheduler/list_scheduler.py) that uses a simple list to store events.

- Added the [`DcdOutputHandler`](jellyfysh/input_output_handler/output_handler/dcd_output_handler.py) that samples the 
  trajectory of a simulation in the binary DCD format used by LAMMPS.

- Added the [`PolarizationOutputHandler`](jellyfysh/input_output_handler/output_handler/polarization_output_handler.py) 
  that samples the polarization of a system made up of charge neutral composite objects.

- Added the [`HardSpherePotential`](jellyfysh/potential/hard_sphere_potential.py). This pair potential is infinite if 
  the distance between two particles is smaller than a given boundary, and zero elsewhere.

- Added the [`HardDipolePotential`](jellyfysh/potential/hard_dipole_potential.py). This pair potential is infinite if
  the distance between two particles lies outside a given range. If the distance lies within the given range, the 
  potential is zero.

- Added the [`SingleIndependentActivePeriodicDirectionEndOfChainEventHandler`](jellyfysh/event_handler/single_independent_active_periodic_direction_end_of_chain_event_handler.py).
  This end-of-chain event handler aligns the velocity of a single independent active (root or leaf) unit periodically 
  with the cartesian axes in the positive direction (e.g., in three dimensions x -> y -> z -> x -> ...), while keeping 
  the entry in the velocity along the relevant axis constant. The independent active unit that is active after an 
  end-of-chain event of this event handler is sampled randomly. This unit is a leaf (root) unit if the independent 
  active unit at the moment of the computation of the candidate event time of this event handler was also a leaf (root) 
  unit. Therefore, this event handler expects the active global in-state in its `send_event_time` method. This can be 
  achieved by using it together with the new 
  [`ActiveGlobalInStateTagger`](jellyfysh/activator/tagger/active_global_state_in_state_tagger.py).

- Added the [`SingleIndependentActiveSequentialDirectionEndOfChainEventHandler`](jellyfysh/event_handler/single_independent_active_sequential_direction_end_of_chain_event_handler.py).
  This end-of-chain event handler can only be used in two dimensions. It rotates the velocity vector of a single
  independent active (root or leaf) unit by an angle around the origin that is set on initialization. The independent 
  active unit that is active after an end-of-chain event of this event handler is sampled randomly. Here, it uses the 
  same procedure as the 
  [`SingleIndependentActivePeriodicDirectionEndOfChainEventHandler`](jellyfysh/event_handler/single_independent_active_periodic_direction_end_of_chain_event_handler.py).

- Added the [`ActiveGlobalInStateTagger`](jellyfysh/activator/tagger/active_global_state_in_state_tagger.py) that is 
  used together with event handlers that expect the active global state for the computation of their candidate event 
  time. One example for such an event handler is the
  [`SingleIndependentActivePeriodicDirectionEndOfChainEventHandler`](jellyfysh/event_handler/single_independent_active_periodic_direction_end_of_chain_event_handler.py).
  
- Added more exemplary configuration files in the 
  [`jellyfysh/config_files/hard_disk_dipoles`](jellyfysh/config_files/hard_disk_dipoles) directory that consider 
  two-dimensional tethered hard-disk dipoles. They generate output in the 
  [`jellyfysh/output/hard_disk_dipoles`](jellyfysh/output/hard_disk_dipoles) directory which also includes plotting
  scripts and reference data.

- Added the [`jellyfysh/output/correct_dcd_file.py`](jellyfysh/output/correct_dcd_file.py) script which corrects a ECMC 
  trajectory stored in the binary DCD format used by LAMMPS for ECMC's non-vanishing average movement.

- Added the [`ExpandedTestCase`](unittests/expanded_test_case.py) class which gives access to an assertion that two 
  sequences are almost equal in the unittests.

- Added more unittests.

#### Changed

- The field that stores the version in the [`jellyfysh.version`](jellyfysh/version.py) module was changed to 
  `__version__` and set to `1.1.0.0`.

- Renamed the `LICENCE` file that contains the GNU General Public License (version 3) of this project to 
  [`LICENSE`](LICENSE).

- The license notice in each file of the JeLLyFysh application was adjusted to include the year 2022 in its copyright.

- The arXiv reference [\[Hoellmer2019\]](https://arxiv.org/abs/1907.12502) was replaced by the journal reference 
  [\[Hoellmer2020\]](https://doi.org/10.1016/j.cpc.2020.107168). The license notice now also uses this updated
  reference.

- Updated the [README.md](README.md) file to describe the new installation process, the executables, and the new 
  [`jellyfysh`](jellyfysh) package. 

- Updated all markdown files to refer to the new [`jellyfysh`](jellyfysh) package.

- Changed the minimum required version of Python from 3.5 to 3.6.

- Several changes to the [`potential`](jellyfysh/potential) package that induced corresponding changes in the event 
  handlers that use the potentials:
  - The `derivative` method of the abstract [`Potential`](jellyfysh/potential/potential.py) class now receives the full 
    velocity of the active unit as the first argument, and returns the directional time derivative.
  - The `displacement` method of the abstract [`InvertiblePotential`](jellyfysh/potential/potential.py) now receives 
    the full velocity of the active unit as the first argument, and returns the time displacement.
  - The abstract [`StandardVelocityPotential`](jellyfysh/potential/abstracts.py) and 
    [`StandardVelocityInvertiblePotential`](jellyfysh/potential/abstracts.py) classes are used for potentials that are 
    still only implemented for velocities that are aligned with the coordinate axes.
  - The abstract [`InvertiblePotential`](jellyfysh/potential/potential.py) now includes a property which signals if a 
    potential change must be sampled for its `displacement` method.

- Time stamps of active units and candidate event times are now instances of the [`Time`](jellyfysh/base/time.py) class.
  This also induced changes in the schedulers which compare candidate event times.

- The derivative of the
  [`MergedImageCoulombPotential`](jellyfysh/potential/merged_image_coulomb_potential/merged_image_coulomb_potential.py) 
  is now implemented in C.

- The displacement and derivative methods of the
  [`InversePowerCoulombBoundingPotential`](jellyfysh/potential/inverse_power_coulomb_bounding_potential/inverse_power_coulomb_bounding_potential.py) 
  are now implemented in C.

- Several changes in the [`HeapScheduler`](jellyfysh/scheduler/heap_scheduler/heap_scheduler.py):
  - The binary min-heap is now implemented in C.
  - It now uses lazy deletion of events. 
  - It now raises an exception if the smallest candidate event time decreases.
  - It optionally logs a warning if the smallest candidate event time is unchanged between two committed events.

- Several changes in the [`PdbInputHandler`](jellyfysh/input_output_handler/input_handler/pdb_input_handler.py):
  - It now corrects the positions from the `.pdb` file for periodic boundaries in order to allow for different system 
    origin choices.
  - It now checks on initialization if the system lengths in the `.pdb` file are the sames as the ones stored in the 
    hypercuboid setting module.
  - If a dimension smaller than three is used, it now checks that superfluous system lengths and position entries are 
    0.0 in the `.pdb` file.
  - It can now be pickled (using the Python [Dill](https://pypi.org/project/dill/) package, see 
    [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md)) and therefore be used together with the 
    [`DumpingOutputHandler`](jellyfysh/input_output_handler/output_handler/dumping_output_handler.py).
  - It now supports the versions >= 2.0 of MDAnalysis.

- Several changes in the [`PdbOutputHandler`](jellyfysh/input_output_handler/output_handler/pdb_output_handler.py):
  - It now corrects the positions of the point masses so that they are the closest to the position of the composite 
    object they belong to. This simplifies the visualization of the topology based on the created `.pdb` file.
  - It now uses the correct alpha, beta, and gamma values of 90.0 for the hypercuboid setting.
  - It can now be pickled (using the Python [Dill](https://pypi.org/project/dill/) package, see 
    [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md)) and therefore be used together with the 
    [`DumpingOutputHandler`](jellyfysh/input_output_handler/output_handler/dumping_output_handler.py).
  - It now supports the versions >= 2.0 of MDAnalysis.
  
- The cell-veto event handlers based on the abstract 
  [`CellVetoEventHandler`](jellyfysh/event_handler/abstracts/cell_veto_event_handler.py) can now use a different 
  potential as their estimator. This allows to use a faster (but probably more imprecise) potential in the estimator.

- The cell-veto event handlers based on the abstract
  [`CellVetoEventHandler`](jellyfysh/event_handler/abstracts/cell_veto_event_handler.py) and the 
  [`CellBoundingPotential`](jellyfysh/potential/cell_bounding_potential.py) now correct the derivative bounds of their 
  estimator for the charges of the involved point masses in the computation of the candidate event time. This is 
  possible because estimators now implement a `charge_correction_factor` method (for the definition of this abstract 
  method, see the abstract [`Estimator`](jellyfysh/estimator/estimator.py) class).

- The [`FinalTimeEndOfRunEventHandler`](jellyfysh/event_handler/final_time_end_of_run_event_handler.py) now only logs a 
  warning (and does not raise an error) if an end-of-run-time equal to 0.0 is used (which may be useful for debugging 
  the initialization stage of the JeLLyFysh application).

- Event handlers with a piecewise-constant bounding potential based on the abstract
  [`EventHandlerWithPiecewiseConstantBoundingPotential`](jellyfysh/event_handler/abstracts/event_handler_with_bounding_potential.py)
  do not use the (slow) `deepcopy` function anymore.

- The [`DipoleRandomNodeCreator`](jellyfysh/input_output_handler/input_handler/random_node_creator/dipole_random_node_creator.py)
  now allows for a minimum initial dipole separation.

- Major refactoring of the [`Cells`](jellyfysh/activator/internal_state/cell_occupancy/cells) package: 
  - A new [`Cell`](jellyfysh/activator/internal_state/cell_occupancy/cells/cells.py) class stores the minimum and 
    maximum position in the respective cell, and its identifier.
  - The methods in the abstract [`Cells`](jellyfysh/activator/internal_state/cell_occupancy/cells/cells.py) class were
    renamed. Where applicable, the new `Cell` class is used in arguments or return values.
  - A new [`CuboidCells`](jellyfysh/activator/internal_state/cell_occupancy/cells/cuboid_cells.py) class construct and 
    stores a cuboid cell system without periodic boundary conditions (in contrast to the 
    [`CuboidPeriodicCells`](jellyfysh/activator/internal_state/cell_occupancy/cells/cuboid_periodic_cells.py) class 
    that already existed in the previous version of JeLLyFysh).

- The abstract [`CellOccupancy`](jellyfysh/activator/internal_state/cell_occupancy/cell_occupancy.py) and its 
  implementation [`SingleActiveCellOccupancy`](jellyfysh/activator/internal_state/single_active_cell_occupancy.py) 
  now allow for more than one occupant per cell. The maximum number of occupants can even be unbounded. This is, 
  for example, useful in hard-disk simulations.

- The unique `uuid` of a run of JeLLyFysh in the [`jellyfysh.base.uuid`](jellyfysh/base/uuid.py) module is now pickled 
  in the [`DumpingOutputHandler`](jellyfysh/input_output_handler/output_handler/dumping_output_handler.py). A resumed 
  run using the executable `jellyfysh-resume` now has the same `uuid` as the original run.

- Various refactorings.

#### Removed

- Removed the `SameActivePeriodicDirectionEndOfChainEventHandler` because keeping the same unit active before and after
  an end-of-chain event can break irreducibility in simple models (see 
  [\[Hoellmer2021\]](https://arxiv.org/abs/2111.11943) in [References.bib](References.bib)).

#### Fixed

- Fixed a bug in the [`PdbInputHandler`](jellyfysh/input_output_handler/input_handler/pdb_input_handler.py) regarding 
  the positions of composite point objects (which are now computed using the closest positions of the point masses that 
  belong to the respective composite point object).

- Fixed a bug in the abstract [`MexicanHatPotential`](jellyfysh/potential/abstracts.py) regarding the placement of a 
  particle outside or inside the potential minimum sphere.

- Fixed description on how to install the MDAnalysis package in PyPy (see 
  [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md)).

- Various fixes of docstrings, type hints, and debug messages.

## Version1.0 &ndash; 2019-08-01

- Initial development milestone which is described in detail in 
  [\[Hoellmer2020\]](https://doi.org/10.1016/j.cpc.2020.107168) (see [References.bib](References.bib)). This version 
  reproduces published data in [\[Faulkner2018\]](https://doi.org/10.1063/1.5036638).
