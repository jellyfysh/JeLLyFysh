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
"""Executable script which plots the polarization samples for the hard-disk dipoles."""
import matplotlib
import matplotlib.pyplot as plt
import numpy
import warnings
import jellyfysh.output.plotting_functions as plotting_functions
import jellyfysh.run as run


def plot_setting() -> None:
    """Set up matplotlib."""
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    matplotlib.rcParams["font.family"] = "serif"
    plt.figure(figsize=(200 / 72, 150 / 72))


def axes_setting(axes: plt.Axes) -> None:
    """
    Set up the axes object.

    Parameters
    ----------
    axes : matplotlib.pyplot.Axes
        The axes object.
    """
    axes.tick_params(labelsize=plotting_functions.default['fontsize'],
                     width=plotting_functions.default['borderwidth'], pad=1, length=2)
    for border in axes.spines.values():
        border.set_linewidth(plotting_functions.default['borderwidth'])


def plot_polarization_entries():
    """
    Plot the cumulative distribution function of the polarization entries of the hard-disk dipole run.

    The relevant .ini file are config_files/hard_disk_dipoles/hard_disk_dipoles.ini and
    config_files/hard_disk_dipoles/hard_disk_dipoles_cells.ini. The sampled data is compared reference data from a
    Newtonian ECMC simulation.
    """
    plt.clf()
    plot_setting()
    axes = plt.gca()
    axes_setting(axes)

    number_bins = 1001
    value_range = [-35.0, 35.0]
    bins = numpy.linspace(value_range[0], value_range[1], number_bins)
    line_properties = {'linewidth': 1.0}
    label_properties = {'fontsize': plotting_functions.default['fontsize'], 'labelpad': 2}

    reference_curves = []
    reference_curves += plotting_functions.plot_curve(
        axes, "ReferenceDataPx_81Dipoles_NewtonianECMC.dat", r'Newtonian ECMC $p_x$',
        linestyle='-', color='k', **line_properties)
    reference_curves += plotting_functions.plot_curve(
        axes, "ReferenceDataPy_81Dipoles_NewtonianECMC.dat", r'Newtonian ECMC $p_y$',
        linestyle='-', color='k', **line_properties)

    def px_transformer(line: str) -> float:
        split_line = line.split()
        return float(split_line[0])

    def py_transformer(line: str) -> float:
        split_line = line.split()
        return float(split_line[1])

    data_plots = []
    data_plots += plotting_functions.plot_histogram(
        axes, "Polarization_81Dipoles.dat", 'JellyFysh $p_x$ (without cells)', bins,
        "config_files/hard_disk_dipoles/hard_disk_dipoles.ini",
        line_transformer=px_transformer, linestyle='-', color='C0', **line_properties)
    data_plots += plotting_functions.plot_histogram(
        axes, "Polarization_81Dipoles.dat", 'JellyFysh $p_y$ (without cells)', bins,
        "config_files/hard_disk_dipoles/hard_disk_dipoles.ini",
        line_transformer=py_transformer, linestyle='dotted', color='C0', **line_properties)
    data_plots += plotting_functions.plot_histogram(
        axes, "Polarization_81Dipoles_Cells.dat", 'JellyFysh $p_x$ (with cells)', bins,
        "config_files/hard_disk_dipoles/hard_disk_dipoles_cells.ini",
        line_transformer=px_transformer, linestyle='-', color='C0', **line_properties)
    data_plots += plotting_functions.plot_histogram(
        axes, "Polarization_81Dipoles_Cells.dat", 'JellyFysh $p_y$ (with cells)', bins,
        "config_files/hard_disk_dipoles/hard_disk_dipoles_cells.ini",
        line_transformer=py_transformer, linestyle='dotted', color='C0', **line_properties)

    plotting_functions.add_legend(axes, reference_curves + data_plots, loc='upper left')
    plt.xlabel(r"polarization $p_x$ or $p_y$", **label_properties)
    plt.ylabel(r"$\pi\left(p_x\right)$ or $\pi\left(p_x\right)$ (cumulative)", **label_properties)
    plt.tight_layout(pad=0.1)
    plt.savefig("hard_disk_dipoles_polarization.pdf")


if __name__ == '__main__':
    run.print_start_message()
    if matplotlib.__version__ != '3.1.0':
        warnings.warn("The used version of matplotlib is {0}. Note that the plotting parameters are optimized for "
                      "version 3.1.0. Please check if all elements are displayed.".format(matplotlib.__version__))
    plot_polarization_entries()
