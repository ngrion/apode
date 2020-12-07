#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Apode Project (https://github.com/ngrion/apode).
# Copyright (c) 2020, Néstor Grión and Sofía Sappia
# License: MIT
#   Full Text: https://github.com/ngrion/apode/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Plots for Apode."""

# =============================================================================
# IMPORTS
# =============================================================================
import attr

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd


# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_HEIGHT = 4
DEFAULT_WIDTH = 5


# =============================================================================
# CLASSES
# =============================================================================


@attr.s(frozen=True)
class PlotAccsessor:
    """Plots for Apode.

    The following plots are implemented:

    - hist : Histogram (default)
    - lorenz : Lorenz curve (relative, generalized, absolute)
    - pen : Pen Parade
    - tip : Tip curve

    Parameters
    ----------
    method : String
        Plot type.
    **kwargs
        Arbitrary keyword arguments.

    """

    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        """Return the ApodeData object."""
        method = "hist" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    # ver n=0,1
    def _lorenz_data(self, alpha="r"):
        """Lorenz Curve data."""
        y = self.idf.data[self.idf.income_column].values
        y = np.sort(y)
        n = len(y)
        z = np.cumsum(y) / y.sum()
        q = np.arange(0, n + 1) / n
        qd = q
        if alpha == "r":
            pass
        elif alpha == "g":
            mu = np.mean(y)
            z = z * mu
            qd = q * mu
        elif alpha == "a":
            mu = np.mean(y)
            qd = q * 0
            z = np.cumsum(y - mu)
        z = np.insert(z, 0, 0)
        return pd.DataFrame({"population": q, "variable": z, "line": qd})

    # ver n=0,1
    def _pen_data(self, pline=None):
        """Pen Parade Curve data."""
        y = self.idf.data[self.idf.income_column].values
        y = np.sort(y)
        n = len(y)
        me = np.median(y)
        q = np.arange(0, n + 1) / n
        mu = np.mean(y)
        qd = np.ones(n + 1) * mu / me
        z = np.copy(y) / me
        z = np.insert(z, 0, 0)
        return pd.DataFrame({"population": q, "variable": z, "line": qd}), me

    # ver n=0,1
    def _tip_data(self, pline):
        """TIP Curve data."""
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        y = self.idf.data[self.idf.income_column].values
        ys = np.sort(y)
        n = len(ys)
        q = sum(ys < pline)
        ygap = np.zeros(n)
        ygap[0:q] = (pline - ys[0:q]) / pline

        z = np.cumsum(ygap) / n
        z = np.insert(z, 0, 0)
        p = np.arange(0, n + 1) / n
        return pd.DataFrame({"population": p, "variable": z})

    def lorenz(self, alpha="r", ax=None, **kwargs):
        """Lorenz Curve.

        A Lorenz curve is a graphical representation of the distribution
        of income or wealth within a population. Lorenz curves graph
        percentiles of the population against cumulative income or wealth
        of people at or below that percentile. [13]_

        Parameters
        ----------
        alpha: string, optional(default='r')
            Options are r: relative, 'g': generalized, 'a': absolut.
        ax: axes object, optional

        Return
        ------
        out: plot
            Matplotlib plot

        References
        ----------
        .. [13] Lorenz, M. O. (1905). Methods for measuring concentration
           of wealth. Journal of the American Statistical Association
           9, 209-219.

        """
        df = self._lorenz_data(alpha)
        q = df.population
        z = df.variable
        qd = df.line
        if ax is None:
            ax = plt.gca()
            fig = plt.gcf()
            fig.set_size_inches(h=DEFAULT_HEIGHT, w=DEFAULT_WIDTH)
        ax.plot(q, z, **kwargs)
        ax.plot(q, qd, **kwargs)
        ax.set_xlabel("Cumulative % of population")
        if alpha == "r":
            ax.set_ylabel("Cumulative % of variable")
            ax.set_title("Lorenz Curve")
        elif alpha == "g":
            ax.set_ylabel("Scaled Cumulative % of variable")
            ax.set_title("Generalized Lorenz Curve")
        elif alpha == "a":
            ax.set_ylabel("Cumulative deviation")
            ax.set_title("Absolut Lorenz Curve")
        else:
            raise ValueError(
                f"'alpha' must be either 'r', 'g' or 'a'. Found '{alpha}'"
            )
        return ax

    def pen(self, pline=None, ax=None, **kwargs):
        """Pen Parade Curve.

        Pen's Parade or The Income Parade is a concept described in a 1971 book
        published by Dutch economist Jan Pen describing income distribution.
        The parade is defined as a succession of every person in the economy,
        with their height proportional to their income, and ordered from lowest
        to greatest. [14]_

        Parameters
        ----------
        pline: float, optional
        ax: axes object, optional

        Return
        ------
        out: plot
            Matplotlib plot

        References
        ----------
        .. [14] Pen, J. (1971). Income Distribution. London: Allen Lane,
           The Penguin Press.

        """
        df, me = self._pen_data(pline=None)
        q = df.population
        z = df.variable
        qd = df.line
        if ax is None:
            ax = plt.gca()
            fig = plt.gcf()
            fig.set_size_inches(h=DEFAULT_HEIGHT, w=DEFAULT_WIDTH)
        ax.plot(q, z, **kwargs)
        ax.plot(q, qd, label="Mean", **kwargs)
        if not (pline is None):
            qpl = np.ones(len(z)) * pline / me
            ax.plot(q, qpl, label="Poverty line")
        ax.set_xlabel("Cumulative % of population")
        ax.set_ylabel("Medianized variable")
        ax.set_title("Pen's Parade")
        ax.legend()
        return ax

    def tip(self, pline, ax=None, **kwargs):
        """TIP Curve.

        Three 'I's of Poverty (TIP) curves, based on distributions
        of poverty gaps, provide evocative graphical summaries of
        the incidence, intensity, and inequality dimensions of
        poverty, and a means for checking for unanimous poverty
        orderings according to a wide class of poverty indices. [15]_

        Parameters
        ----------
        pline: float, optional
        ax: axes object, optional

        Return
        ------
        out: plot
            Matplotlib plot

        References
        ----------
        .. [15] Jenkins S. P., Lambert P., 1997. Three “I’s of Poverty”
           Curves, with an Analysis of UK Poverty Trends, Oxford
           Economic Papers, 49, pp. 317-327.

        """
        df = self._tip_data(pline)
        p = df.population
        z = df.variable
        if ax is None:
            ax = plt.gca()
            fig = plt.gcf()
            fig.set_size_inches(h=DEFAULT_HEIGHT, w=DEFAULT_WIDTH)
        ax.plot(p, z, **kwargs)
        ax.set_title("TIP Curve")
        ax.set_ylabel("Cumulated poverty gaps")
        ax.set_xlabel("Cumulative % of population")
        return ax

    def __getattr__(self, aname):
        """Apply Plot method."""
        return getattr(self.idf.data.plot, aname)
