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
"""Executable script which plots the samples for the water runs."""
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


def plot_setting():
    """Set up matplotlib."""
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}", r"\usepackage{tikz}"]
    matplotlib.rcParams["font.family"] = "serif"
    plt.figure(figsize=(220 / 72, 120 / 72))


def axes_setting(axes: plt.Axes) -> None:
    """
    Set up the axes object.

    Parameters
    ----------
    axes : matplotlib.pyplot.Axes
        The axes object.
    """
    axes.set_xlim(left=2.3, right=4.0)
    axes.set_xticks([2.5, 3.0, 3.5, 4.0])
    axes.tick_params(labelsize=plotting_functions_module.default['fontsize'],
                     width=plotting_functions_module.default['borderwidth'], pad=1, length=2)
    for border in axes.spines.values():
        border.set_linewidth(plotting_functions_module.default['borderwidth'])


def main():
    """
    Plot the samples for the water runs.

    The relevant .ini files are located in the config_files/2018_JCP_149_064113/water directory. The sampled data is
    compared to reference data created by reversible Monte Carlo.
    """
    if matplotlib.__version__ != '3.1.0':
        warnings.warn("The used version of matplotlib is {0}. Note that the plotting parameters are optimized for "
                      "version 3.1.0. Please check if all elements are displayed.".format(matplotlib.__version__))

    plot_setting()
    axes = plt.gca()
    axes_setting(axes)

    number_bins = 1001
    value_range = [2, 7]
    bins = numpy.linspace(value_range[0], value_range[1], number_bins)
    curve_properties = {'linewidth': 1.0}
    label_properties = {'fontsize': plotting_functions_module.default['fontsize'], 'labelpad': 2}

    reference_plot, data_plots = [], []
    reference_plot += plotting_functions_module.plot_curve(axes, 'ReferenceOOSeparation.dat', r"1",
                                                           color='k', linestyle='-', **curve_properties)
    data_plots += plotting_functions_module.plot_histogram(
        axes, "SamplesOfOOSeparation_CoulombPowerBounded_LJInverted.dat", r"2", bins,
        ini_filename="config_files/2018_JCP_149_064113/water/coulomb_power_bounded_lj_inverted.ini",
        dashes=[2, 2], **curve_properties)
    data_plots += plotting_functions_module.plot_histogram(
        axes, "SamplesOfOOSeparation_CoulombPowerBounded_LJCellBounded.dat", r"3", bins,
        ini_filename="config_files/2018_JCP_149_064113/water/coulomb_power_bounded_lj_cell_bounded.ini",
        linestyle='--', **curve_properties)
    data_plots += plotting_functions_module.plot_histogram(
        axes, "SamplesOfOOSeparation_CoulombCellVeto_LJInverted.dat", r"4", bins,
        ini_filename="config_files/2018_JCP_149_064113/water/coulomb_cell_veto_lj_inverted.ini",
        linestyle='-.', **curve_properties)
    data_plots += plotting_functions_module.plot_histogram(
        axes, "SamplesOfOOSeparation_CoulombCellVeto_LJCellVeto.dat", r"5", bins,
        ini_filename="config_files/2018_JCP_149_064113/water/coulomb_cell_veto_lj_cell_veto.ini",
        linestyle=':', **curve_properties)

    plt.xlabel(r"pair separation $\left|\boldsymbol{r}_{\mathrm{OO}}\right|$", **label_properties)
    plt.ylabel(r"$\pi\left(\left|\boldsymbol{r}_{\mathrm{OO}}\right|\right)$ (cumulative)", **label_properties)
    plotting_functions_module.add_legend(axes, reference_plot, bbox_to_anchor=(1.04, 1.0), loc='upper left',
                                         title="reference", handlelength=4)
    plotting_functions_module.add_legend(axes, data_plots, bbox_to_anchor=(1.04, 0.77), loc='upper left',
                                         title="JeLLyFysh", handlelength=4)
    plt.tight_layout(pad=0.2)

    plt.savefig("water.pdf")


if __name__ == '__main__':
    run_module.print_start_message()
    main()
