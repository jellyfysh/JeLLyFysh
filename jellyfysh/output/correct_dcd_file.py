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
"""Executable script which corrects an ECMC trajectory for its non-vanishing average movement."""
from argparse import ArgumentParser, Namespace
from os import path
import sys
from typing import Sequence
import MDAnalysis
import jellyfysh.run as run


def parse_options(args: Sequence[str]) -> Namespace:
    """
    Convert argument strings to objects and assign them as attributes of the argparse namespace. Return the populated
    namespace.

    The argument strings can for example be sys.argv[1:] in order to parse the command line arguments. The argument
    parser parses the following required positional arguments:

    1. topology_file: Specify the path to the topology file of the run (PSF, PDB, CRD, or GRO data format).
    2. trajectory_file: Specify the path to the trajectory file of the run (DCD, XTC, TRR, or XYZ data format).
    Per default, also the following argument is added:
    3. --help, -h: Show the help message and exit.

    Parameters
    ----------
    args : Sequence[str]
        The argument strings.

    Returns
    -------
    argparse.Namespace
        The populated argparse namespace.
    """
    parser = ArgumentParser(description="Correct the trajectory for ECMC's non-vanishing average movement.")
    parser.add_argument("topology_file", help="specify the path to the topology file of the run "
                                              "(PSF, PDB, CRD, or GRO data format)")
    parser.add_argument("trajectory_file", help="specify the path to the trajectory file of the run "
                                                "(DCD, XTC, TRR, or XYZ data format)")
    return parser.parse_args(args)


def main() -> None:
    """
    Correct the trajectory of an ECMC simulation given in the command line arguments for ECMC's non-vanishing average
    movement and store the corrected trajectory in {old_filename}_corrected.{old_file_suffix}.
    """
    run.print_start_message()
    args = parse_options(sys.argv[1:])
    universe = MDAnalysis.Universe(args.topology_file, args.trajectory_file)
    new_trajectory_filename = "{0}_corrected{1}".format(*path.splitext(args.trajectory_file))
    print(f"Writing corrected trajectory to {new_trajectory_filename}.")
    first_center = universe.atoms.center_of_geometry()
    with universe.trajectory.Writer(new_trajectory_filename) as writer:
        for index, _ in enumerate(universe.trajectory):
            difference = universe.atoms.center_of_geometry() - first_center
            for atom in universe.atoms:
                atom.position -= difference
            assert all(abs(d) < 1.0e-6 for d in universe.atoms.center_of_geometry() - first_center)
            writer.write(universe.atoms)
            if index % 10 == 0:
                print(f"Finished {index} timesteps.")


if __name__ == '__main__':
    main()
