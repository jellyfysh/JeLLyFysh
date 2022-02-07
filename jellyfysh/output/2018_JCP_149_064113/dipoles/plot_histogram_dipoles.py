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
"""Executable script which plots the samples for the dipole runs."""
from typing import Any, List
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy
import jellyfysh.output.plotting_functions as plotting_functions
import jellyfysh.run as run


_string_r13 = r"$\left|\boldsymbol{r}_{13}\right|$ "
_string_r14 = r"$\left|\boldsymbol{r}_{14}\right|$ "


def plot_two_histograms(axes_one: plt.Axes, axes_two: plt.Axes, filename: str, label: str, bins: numpy.ndarray,
                        ini_filename: str = None, **kwargs: Any) -> List[lines.Line2D]:
    """
    Plot two histograms in the two axes objects with the values in the given files.

    The filename is split before the file format. Then '_13' and '_14' are included before the file format. These two
    files are then used to plot the two histograms.
    The files can include single line comments which start with #. Each line should contain a single float.

    Parameters
    ----------
    axes_one : matplotlib.pyplot.Axes
        The first axes object.
    axes_two : matplotlib.pyplot.Axes
        The second axes object.
    filename : str
        The filename.
    label : str
        The label of the plot.
    bins : numpy.ndarray
        The bins.
    ini_filename : str
        The name of the .ini file which created the file.
    kwargs : Any
        Additional kwargs which will be passed to the axes.plot method.

    Returns
    -------
    matplotlib.lines.Line2D
        A list of objects representing the plotted data.
    """
    split_filename = filename.split(".")
    if len(split_filename) != 2:
        warnings.warn("Given filename {0} could not be processed to produce figure 11.".format(filename))
        return []
    filename_13 = split_filename[0] + "_13." + split_filename[1]
    filename_14 = split_filename[0] + "_14." + split_filename[1]
    plot1 = plotting_functions.plot_histogram(axes_one, filename_13, _string_r13 + label, bins,
                                              ini_filename=ini_filename, linestyle='--', dashes=[2, 2], **kwargs)
    plot2 = plotting_functions.plot_histogram(axes_two, filename_14, _string_r14 + label, bins,
                                              ini_filename=ini_filename, linestyle='-.', dashes=[3, 2, 1, 2],
                                              color=plot1[0].get_color() if plot1 else None, **kwargs)
    return plot1 + plot2


def plot_setting() -> None:
    """Set up matplotlib."""
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    matplotlib.rcParams["font.family"] = "serif"
    plt.figure(figsize=(250 / 72, 140 / 72))


def axes_setting(axes: matplotlib.pyplot.Axes) -> None:
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


def main() -> None:
    """
    Plot the samples for the dipole runs.

    The relevant .ini files are located in the config_files/2018_JCP_149_064113/dipoles directory. The sampled data is
    compared to reference data created by reversible Monte Carlo.
    """
    if matplotlib.__version__ != '3.1.0':
        warnings.warn("The used version of matplotlib is {0}. Note that the plotting parameters are optimized for "
                      "version 3.1.0. Please check if all elements are displayed.".format(matplotlib.__version__))

    plot_setting()

    axes_13 = plt.gca()
    axes_14 = axes_13.twiny()
    axes_setting(axes_13)
    axes_setting(axes_14)

    axes_14.set_xlim([-0.05, 1.0])
    axes_13.set_xlim([-0.15, 0.9])
    axes_13.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axes_14.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])

    number_bins = 1000
    value_range = [0.0, 3 ** 0.5 / 2]
    bins = numpy.linspace(value_range[0], value_range[1], number_bins)
    curve_properties = {'linewidth': 1.0}
    label_properties = {'fontsize': plotting_functions.default['fontsize'], 'labelpad': 2}

    reference_plots, data_plots = [], []
    reference_plots += plotting_functions.plot_curve(axes_13, 'ReferenceDataDipoles_13.dat',
                                                     label=_string_r13 + r"reversible MC",
                                                     color='k', linestyle='-', **curve_properties)
    reference_plots += plotting_functions.plot_curve(axes_14, 'ReferenceDataDipoles_14.dat',
                                                     label=_string_r14 + r"reversible MC",
                                                     color='k', linestyle='--', **curve_properties)

    data_plots += plot_two_histograms(axes_13, axes_14, "SamplesOfSeparation_AtomFactors.dat",
                                      r"atom mode", bins,
                                      "config_files/2018_JCP_149_064113/dipoles/atom_factors.ini",
                                      **curve_properties)
    data_plots += plot_two_histograms(axes_13, axes_14, "SamplesOfSeparation_DipoleMotion.dat",
                                      r"atom/dipole mode", bins[10:],
                                      "config_files/2018_JCP_149_064113/dipoles/dipole_motion.ini",
                                      **curve_properties)
    data_plots += plot_two_histograms(axes_13, axes_14, "SamplesOfSeparation_CellBounded.dat",
                                      r"cell bounded", bins[20:],
                                      "config_files/2018_JCP_149_064113/dipoles/cell_bounded.ini",
                                      **curve_properties)
    data_plots += plot_two_histograms(axes_13, axes_14, "SamplesOfSeparation_CellVeto.dat",
                                      r"cell-veto", bins[30:],
                                      "config_files/2018_JCP_149_064113/dipoles/cell_veto.ini",
                                      **curve_properties)
    # data_plots += plot_two_histograms(axes_13, axes_14, "SamplesOfSeparation_DipoleFactors_InsideFirst.dat",
    #                                   r"dipole factors, inside first", bins[30:],
    #                                   "config_files/2018_JCP_149_064113/dipoles/dipole_factors_inside_first.ini",
    #                                   **curve_properties)
    # data_plots += plot_two_histograms(axes_13, axes_14, "SamplesOfSeparation_DipoleFactors_OutsideFirst.dat",
    #                                   r"dipole factors, outside first", bins[30:],
    #                                   "config_files/2018_JCP_149_064113/dipoles/dipole_factors_outside_first.ini",
    #                                   **curve_properties)
    # data_plots += plot_two_histograms(axes_13, axes_14, "SamplesOfSeparation_DipoleFactors_Ratio.dat",
    #                                   r"dipole factors, ratio", bins[30:],
    #                                   "config_files/2018_JCP_149_064113/dipoles/dipole_factors_ratio.ini",
    #                                   **curve_properties)

    plotting_functions.add_legend(axes_13, reference_plots, bbox_to_anchor=(1.04, 1.20), loc="upper left")
    plotting_functions.add_legend(axes_13, data_plots, bbox_to_anchor=(1.04, 0.9), loc="upper left",
                                  title='JeLLyFysh')
    axes_13.set_xlabel(r'pair separation $|\mathbf{r}_{13}|$', **label_properties)
    axes_14.set_xlabel(r'pair separation $|\mathbf{r}_{14}|$', **label_properties)
    axes_13.set_ylabel(r'$ \pi \left( |\mathbf{r}_{ij}| \right)$ (cumulative)', **label_properties)
    plt.tight_layout(pad=0.5)
    plt.savefig("dipoles.pdf")


if __name__ == '__main__':
    run.print_start_message()
    main()
