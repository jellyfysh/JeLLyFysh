# JeLLFysh - a Python application for all-atom event-chain Monte Carlo - https://github.com/jellyfysh
# Copyright (C) 2019 The JeLLyFysh organization
# (see the AUTHORS file for the full list of authors)
#
# This file is part of JeLLyFysh.
#
# JeLLyFysh is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either > version 3 of the License, or (at your option) any
# later version.
#
# JeLLyFysh is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with JeLLyFysh in the LICENSE file.
# If not, see <https://www.gnu.org/licenses/>.
#
# If you use JeLLyFysh in published work, please cite the following reference (see [Hoellmer2019] in References.bib):
# Philipp Hoellmer, Liang Qin, Michael F. Faulkner, A. C. Maggs, Werner Krauth
# JeLLyFysh-Version1.0 -- a Python application for all-atom event-chain Monte Carlo,
# arXiv e-prints: 1907.12502 (2019), https://arxiv.org/abs/1907.12502
#
"""Executable script which plots the samples for the coulomb atom runs."""
import importlib
import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy
import warnings


# Add the directory which contains the module plotting_functions to sys.path
this_directory = os.path.dirname(os.path.abspath(__file__))
helper_directory = os.path.abspath(this_directory + "/../../../")
sys.path.insert(0, helper_directory)


plotting_functions_module = importlib.import_module("output.plotting_functions")
run_module = importlib.import_module("run")


def plot_setting() -> None:
    """Set up matplotlib."""
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    matplotlib.rcParams["font.family"] = "serif"
    plt.figure(figsize=(280 / 72, 115 / 72))


def axes_setting(axes: plt.Axes) -> None:
    """
    Set up the axes object.

    Parameters
    ----------
    axes : matplotlib.pyplot.Axes
        The axes object.
    """
    axes.tick_params(labelsize=plotting_functions_module.default['fontsize'],
                     width=plotting_functions_module.default['borderwidth'],
                     pad=1, length=2)
    for border in axes.spines.values():
        border.set_linewidth(plotting_functions_module.default['borderwidth'])


def main() -> None:
    """
    Plot the samples for the coulomb atom runs.

    The relevant .ini files are located in the config_files/2018_JCP_149_064113/coulomb_atoms directory. The sampled
    data is compared to reference data created by reversible Monte Carlo.
    """
    if matplotlib.__version__ != '3.1.0':
        warnings.warn("The used version of matplotlib is {0}. Note that the plotting parameters are optimized for "
                      "version 3.1.0. Please check if all elements are displayed.".format(matplotlib.__version__))

    plot_setting()
    axes = plt.gca()
    axes_setting(axes)

    number_bins = 1001
    value_range = [0.15, 0.85]
    bins = numpy.linspace(value_range[0], value_range[1], number_bins)
    line_properties = {'linewidth': 1.0}
    label_properties = {'fontsize': plotting_functions_module.default['fontsize'], 'labelpad': 2}
    reference_plot, data_plots = [], []

    reference_plot += plotting_functions_module.plot_curve(axes, 'ReferenceDataCoulombAtoms.dat', r"reversible MC",
                                                           mask=value_range, color='k', linestyle='-',
                                                           **line_properties)

    data_plots += plotting_functions_module.plot_histogram(
        axes, "SamplesOfSeparation_PowerBounded.dat", r"power bounded", bins,
        ini_filename="config_files/2018_JCP_149_064113/coulomb_atoms/power_bounded.ini", color='r', linestyle='-.',
        **line_properties)

    data_plots += plotting_functions_module.plot_histogram(
        axes, "SamplesOfSeparation_CellBounded.dat", r"cell bounded", bins,
        ini_filename="config_files/2018_JCP_149_064113/coulomb_atoms/cell_bounded.ini", color='green', linestyle='--',
        **line_properties)

    data_plots += plotting_functions_module.plot_histogram(
        axes, "SamplesOfSeparation_CellVeto.dat", r"cell veto", bins,
        ini_filename="config_files/2018_JCP_149_064113/coulomb_atoms/cell_veto.ini", color='blue', linestyle=':',
        **line_properties)

    plt.xlabel(r"pair separation $\left|\boldsymbol{r}_{12}\right|$", **label_properties)
    plt.ylabel(r"$\pi\left(\left|\boldsymbol{r}_{12}\right|\right)$ (cumulative)", **label_properties)
    plotting_functions_module.add_legend(axes, reference_plot, bbox_to_anchor=(1.04, 1.0), loc='upper left',
                                         title="reference", handlelength=2.5)
    plotting_functions_module.add_legend(axes, data_plots, bbox_to_anchor=(1.04, 0.67), loc='upper left',
                                         title="JeLLyFysh", handlelength=2.5)
    plt.tight_layout(pad=0.3)

    plt.savefig("coulomb_atoms.pdf")


if __name__ == '__main__':
    run_module.print_start_message()
    main()
