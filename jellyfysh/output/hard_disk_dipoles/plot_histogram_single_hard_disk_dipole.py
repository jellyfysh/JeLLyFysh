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
"""Executable script which plots the polarization samples for the single hard-disk dipole."""
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


def plot_dipole_extension():
    """
    Plot the cumulative distribution function of the dipole extension of the single hard-disk dipole run.

    The relevant .ini file is config_files/hard_disk_dipoles/single_hard_disk_dipole.ini. The sampled data is compared
    to the analytic solution.
    """
    plt.clf()
    plot_setting()
    axes = plt.gca()
    axes_setting(axes)

    minimum_dipole_extension = 2.0 / 3.0
    maximum_dipole_extension = 4.0 / 3.0
    number_bins = 1001
    value_range = [minimum_dipole_extension, maximum_dipole_extension]
    bins = numpy.linspace(value_range[0], value_range[1], number_bins)
    line_properties = {'linewidth': 1.0}
    label_properties = {'fontsize': plotting_functions.default['fontsize'], 'labelpad': 2}

    reference_curve = axes.plot(
        bins,
        (bins ** 2 - minimum_dipole_extension ** 2) / (maximum_dipole_extension ** 2 - minimum_dipole_extension ** 2),
        label=r"analytic", color='k', linestyle='-', **line_properties)

    def rho_line_transformer(line: str) -> float:
        split_line = line.split()
        assert len(split_line) == 2
        return (float(split_line[0]) ** 2 + float(split_line[1]) ** 2) ** 0.5
    data_plot = plotting_functions.plot_histogram(
        axes, "Polarization_SingleHardDiskDipole.dat", 'JellyFysh', bins,
        "config_files/hard_disk_dipoles/single_hard_disk_dipole.ini",
        line_transformer=rho_line_transformer, linestyle='--', **line_properties)

    plotting_functions.add_legend(axes, reference_curve + data_plot, loc='upper left')
    plt.xlabel(r"dipole extension $\rho$", **label_properties)
    plt.ylabel(r"$\pi\left(\rho\right)$ (cumulative)", **label_properties)
    plt.tight_layout(pad=0.1)
    plt.savefig("single_hard_disk_dipole_extension.pdf")


def plot_dipole_angle():
    """
    Plot the cumulative distribution function of the dipole angle of the single hard-disk dipole run.

    The relevant .ini file is config_files/hard_disk_dipoles/single_hard_disk_dipole.ini. The sampled data is compared
    to the analytic solution.
    """
    plt.clf()
    plot_setting()
    axes = plt.gca()
    axes_setting(axes)

    minimum_dipole_angle = -numpy.pi
    maximum_dipole_angle = numpy.pi
    number_bins = 1001
    value_range = [minimum_dipole_angle, maximum_dipole_angle]
    bins = numpy.linspace(value_range[0], value_range[1], number_bins)
    line_properties = {'linewidth': 1.0}
    label_properties = {'fontsize': plotting_functions.default['fontsize'], 'labelpad': 2}

    reference_curve = axes.plot(
        bins, (bins + numpy.pi) / (2.0 * numpy.pi), label=r"analytic", color='k', linestyle='-', **line_properties)

    def theta_line_transformer(line: str) -> float:
        split_line = line.split()
        assert len(split_line) == 2
        return numpy.arctan2(float(split_line[1]), float(split_line[0]))

    data_plot = plotting_functions.plot_histogram(
        axes, "Polarization_SingleHardDiskDipole.dat", 'JellyFysh', bins,
        "config_files/hard_disk_dipoles/single_hard_disk_dipole.ini",
        line_transformer=theta_line_transformer, linestyle='--', **line_properties)

    plotting_functions.add_legend(axes, reference_curve + data_plot, loc='upper left')
    plt.xlabel(r"dipole angle $\theta$", **label_properties)
    plt.ylabel(r"$\pi\left(\theta\right)$ (cumulative)", **label_properties)
    plt.tight_layout(pad=0.1)
    plt.savefig("single_hard_disk_dipole_angle.pdf")


if __name__ == '__main__':
    run.print_start_message()
    if matplotlib.__version__ != '3.1.0':
        warnings.warn("The used version of matplotlib is {0}. Note that the plotting parameters are optimized for "
                      "version 3.1.0. Please check if all elements are displayed.".format(matplotlib.__version__))
    plot_dipole_extension()
    plot_dipole_angle()
