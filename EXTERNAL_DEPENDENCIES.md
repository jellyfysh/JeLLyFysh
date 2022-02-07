# External Dependencies

A few of JeLLyFysh's modules depend on non-standard-library Python packages. 

## Required Dependencies

These dependencies are installed by setuptools during the installation of the JeLLyFysh application (see 
[README.md](README.md)).

1. The [Dill](https://pypi.org/project/dill/) package is used in order to dump entire runs using the
   [`DumpingOutputHandler`](jellyfysh/input_output_handler/output_handler/dumping_output_handler.py), and resume dumped 
   runs using the `jellyfysh-resume` executable.


2. The [CFFI](https://cffi.readthedocs.io/en/latest/) package is used to access C code in the JeLLyFysh application. 
   This is the recommended way for PyPy (see https://doc.pypy.org/en/latest/extending.html).

## Optional Dependencies

The following additional packages need be installed only when the described modules are used. They are not installed by 
setuptools during the installation of the JeLLyFysh application (see [README.md](README.md)) but have to be installed 
manually (for example, with the help of pip). 

1. The plotting scripts located in the [`jellyfysh-examples/output`](jellyfysh/output) directory, which is created by 
   the `jellyfysh-examples` executable, use [Matplotlib](https://matplotlib.org) and [NumPy](https://numpy.org). The 
   recommended version of Matplotlib for these scripts is 3.1.0.


2. To interact with Protein Data Bank (pdb) and DCD binary trajectory files, the
    [MDAnalysis](https://www.mdanalysis.org) package is used. This concerns the modules 
    [`jellyfysh.input_output_handler.output_handler.pdb_output_handler`](jellyfysh/input_output_handler/output_handler/pdb_output_handler.py), 
    [`jellyfysh.input_output_handler.input_handler.pdb_input_handler`](jellyfysh/input_output_handler/input_handler/pdb_input_handler.py)
    and
    [`jellyfysh.input_output_handler.output_handler.dcd_output_handler.py`](jellyfysh/input_output_handler/output_handler/dcd_output_handler.py).

## Known bugs

### SciPy/MDAnalysis for PyPy on MacOs

Some problems exist for [SciPy](https://www.scipy.org) (version >= 1.3) with PyPy (version >= 7) on MacOs (SciPy is, 
e.g., installed as a dependency of MDAnalysis) This is also true for SciPy 1.2.3, however the way described in this 
section succeeds in installing it nevertheless.

1. Install openblas using homebrew:

    ```shell
    brew install openblas
    ```
    
    After the installation (or after using `brew info openblas`), homebrew prints openblas' library and include path in 
    a way similar to
    
    ```shell
    For compilers to find openblas you may need to set:
        export LDFLAGS="-L/usr/local/opt/openblas/lib"
        export CPPFLAGS="-I/usr/local/opt/openblas/include"
    ```
    
    These flags are used in the next step.


2. Use openblas' flags from step one to install SciPy with the following command:

    ```shell
    LDFLAGS="-L/usr/local/opt/openblas/lib" CPPFLAGS="-I/usr/local/opt/openblas/include" pypy3 -m pip install 'scipy==1.2.3'
    ``` 

Unfortunately there is still another bug in MDAnalysis when using PyPy. However, this is fixed when importing the 
package in the 
[`jellyfysh.input_output_handler.mdanalysis_import`](jellyfysh/input_output_handler/mdanalysis_import.py) 
module (for a more detailed description of the bug, see the same file).

### Dill for Python version >= 3.7
In newer versions of Python, abstract classes can no longer be pickled. Therefore, Dill cannot be used
to dump the entire run in these versions (for now).
