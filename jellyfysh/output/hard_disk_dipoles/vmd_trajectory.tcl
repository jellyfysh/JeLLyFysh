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
#
# This script shows the trajectory of a hard dipole simulation (obtained, e.g., by adding a DcdOutputHandler to
# config_files/RotECMCMany/hard_dipoles.ini) in VMD.
# For this, it requires the topology and trajectory of the simulation in the files trajectory.pdb and trajectory.dcd.
# If other filenames are used, change them in the lines 36 and 51.
# The radius of the spheres is set to 0.125 in line 38 and should be changed according to the simulation.
# To use this script, enter VMD's TKConsole ('Extensions -> Tk Console' in the GUI).
# Then, navigate into this directory (using 'cd') and execute this script: 'source vmd_trajectory.tcl'.
# One can further select periodic images to draw in the 'Periodic' tab under 'Graphics -> Representations' in the GUI.
#
#
# Delete all currently loaded molecules.
mol delete all
# Load the topology of the hard dipole system.
# By using 'autobonds off' no bonds between the spheres are guessed by vmd and only those in the .pdb file are used.
mol new trajectory.pdb autobonds off
# Set the radius of the hard spheres.
[atomselect top "all"] set radius 0.476190476190476
# Change the representation and color.
mol delrep 0 top
mol representation VDW 1.0 16
# Another possibility is to use the CPK representation that also draws bonds. However, this representation uses a
# scaled-down radius for the atoms. It seems to be not documented what the scale-down factor is. A comparison with the
# VDW representation seems to indicate that the required scale factor is 4.0 (only tested with radius 0.125).
# mol representation CPK 4.0 0.25 16 16
mol color ResID
mol addrep top
# Draw the periodic box.
pbc box
# Load the trajectory of the hard dipole system.
mol addfile trajectory.dcd