# External Dependencies

A few of JeLLyFysh's modules depend on non-standard-library Python packages. These additional packages need be installed 
only when when these modules are used.

The plotting scripts located in the [`src/output/2018_JCP_149_064113`](src/output/2018_JCP_149_064113) directory use 
[Matplotlib](https://matplotlib.org) and [NumPy](https://numpy.org). The recommended version of Matplotlib for these 
scripts is 3.1.0.

To read and write Protein Data Bank (pdb) files, the package [MDAnalysis](https://www.mdanalysis.org) is used. This 
concerns the modules 
[`src/input_output_handler/output_handler/pdb_output_handler.py`](src/input_output_handler/output_handler/pdb_output_handler.py) 
and 
[`src/input_output_handler/input_handler/pdb_input_handler.py`](src/input_output_handler/input_handler/pdb_input_handler.py).

To dump entire runs using the module 
[`src/input_output_handler/output_handler/dumping_output_handler.py`](src/input_output_handler/output_handler/dumping_output_handler.py) 
and resume dumped runs using [`src/resume.py`](src/resume.py), the [Dill](https://pypi.org/project/dill/) package is 
used.

## Known bugs

### MDAnalysis for PyPy on MacOs
Some problems exist for MDAnalysis with PyPy (version >= 7) on MacOs 10.14 installed by pip. To fix the, first 
install [SciPy](https://www.scipy.org) (a dependency of MDAnalysis). Here, two things are failing:
1. The installation itself (related to the issue explained 
[here](https://bitbucket.org/pypy/pypy/issues/2942/unable-to-install-numpy-with-pypy3-on)).
2. In version 1.3.0 of SciPy, calling `from scipy.spatial import cKDTree` fails even after a successful installation 
because SciPy does not find some C library.

The second point can be fixed (partially) by installing an older version of SciPy. Then calling twice 
`from scipy.spatial import cKDTree` works. The first point is fixed by setting `MACOSX_DEPLOYMENT_TARGET=10.14`.

Therefore, calling the following should work to install MDAnalysis:
```shell
MACOSX_DEPLOYMENT_TARGET=10.14 pypy3 -m pip install 'scipy<1.3.0'
pypy3 -m pip install MDAnalysis
```

Unfortunately there is still another bug in MDAnalysis when using PyPy. However, this is fixed (together
with the issue that `from scipy.spatial import cKDTree` must be called twice), when importing the package 
in the [`src/input_output_handler/mdanalysis_import.py`](src/input_output_handler/mdanalysis_import.py) module.

### Dill for Python version >= 3.7
In newer versions of Python, abstract classes can no longer be pickled. Therefore, Dill cannot be used
to dump the entire run in these versions (for now).
