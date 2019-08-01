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
"""Module for useful plotting functions used in this directory."""
from typing import Any, List, Sequence
import warnings
import numpy
import matplotlib.pyplot as plt
import matplotlib.lines as lines


default = {'fontsize': 7, 'borderwidth': 0.2}
"""Default values for the plots."""


def plot_histogram(axes: plt.Axes, filename: str, label: str, bins: numpy.ndarray, ini_filename: str = None,
                   **kwargs: Any) -> List[lines.Line2D]:
    """
    Plot a histogram in the axes object with the values in the given file.

    The file can include single line comments which start with #. Each line should contain a single float.

    Parameters
    ----------
    axes : matplotlib.pyplot.Axes
        The axes object.
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
    try:
        distances = []
        with open(filename) as file:
            for line in file:
                if line.startswith("#"):
                    continue
                distances.append(float(line))
        hists, edges = numpy.histogram(distances, bins=bins, density=False)
        total = sum(hists)
        x_values = edges
        y_values = [0] + list(numpy.cumsum(hists / total))
        plot = axes.plot(x_values, y_values, label=label, **kwargs)
        return plot
    except FileNotFoundError:
        if ini_filename is not None:
            warnings.warn("Could not open the file {0}. Please run the .ini file {1}.".format(filename, ini_filename))
        else:
            warnings.warn("Could not open the file {0}.".format(filename))
        return []


def plot_curve(axes: plt.Axes, filename: str, label: str, mask: Sequence[float] = None,
               **kwargs: Any) -> List[lines.Line2D]:
    """
    Plot a curve in the axes object with the values in the given file.

    The file can include single line comments which start with #. Each line should contain two floats which are
    the x and the y value, respectively.

    Parameters
    ----------
    axes : matplotlib.pyplot.Axes
        The axes object.
    filename : str
        The filename.
    label : str
        The label of the plot.
    mask : Sequence[float]
        The range of the x-axis within data is plotted.
    kwargs : Any
        Additional kwargs which will be passed to the axes.plot method.

    Returns
    -------
    matplotlib.lines.Line2D
        A list of objects representing the plotted data.
    """
    try:
        if mask is None:
            mask = [-float('inf'), float('inf')]
        data = []
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                split_str = line.split()
                x_value, y_value = float(split_str[0]), float(split_str[1])
                if mask[0] <= x_value <= mask[1]:
                    data.append((x_value, y_value))
        sorted(data)
        x_values = [a[0] for a in data]
        y_values = [a[1] for a in data]
        return axes.plot(x_values, y_values, label=label, **kwargs)
    except FileNotFoundError:
        warnings.warn("Could not open the file {}.".format(filename))
        return []


def add_legend(axes: plt.Axes, plots: Sequence[lines.Line2D], **kwargs: Any) -> None:
    """
    Add the legend to the axes object.

    Parameters
    ----------
    axes : matplotlib.pyplot.Axes
        The axes object.
    plots : Sequence[matplotlib.lines.Line2D]
        The plotted data containing the legends.
    kwargs : Any
        Additional kwargs which will be passed to the axes.legend method.

    Returns
    -------
    matplotlib.lines.Line2D
        A list of objects representing the plotted data.
    """
    labels = [plot.get_label() for plot in plots]
    kwargs['fontsize'] = kwargs.get('fontsize', default['fontsize'])
    kwargs['title_fontsize'] = kwargs.get('title_fontsize', default['fontsize'])
    legend = axes.legend(plots, labels, edgecolor='k', **kwargs)
    legend.get_frame().set_linewidth(default['borderwidth'])
    legend.get_frame().set_edgecolor('k')
    axes.add_artist(legend)


def set_default(name: str, value: Any) -> None:
    """
    Add a default value to the global default dictionary.

    Parameters
    ----------
    name : str
        The key.
    value : Any
        The value.
    """
    global default
    default[name] = value
