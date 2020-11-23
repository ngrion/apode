#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Apode Project (https://github.com/mchalela/apode).
# Copyright (c) 2020, Néstor Grión and Sofía Sappia
# License: MIT
#   Full Text: https://github.com/ngrion/apode/blob/master/LICENSE.txt

from unittest import mock

from apode import datasets
from apode import plots

from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import pytest


# =============================================================================
# TESTS COMMON
# =============================================================================


@check_figures_equal()
def test_default_call(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    test_ax = fig_test.subplots()
    data.plot.lorenz(ax=test_ax, alpha="r")

    exp_ax = fig_ref.subplots()
    data.plot(method="lorenz", ax=exp_ax, alpha="r")


def test_invalid():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(AttributeError):
        data.plot("foo")


# =============================================================================
# TESTS LORENZ
# =============================================================================
@check_figures_equal()
def test_plot_relative_lorenz(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)

    test_ax = fig_test.subplots()
    data.plot.lorenz(ax=test_ax, alpha="r")

    exp_ax = fig_ref.subplots()
    df = data.plot._lorenz_data(alpha="r")
    exp_ax.plot(df.population, df.variable)
    exp_ax.plot(df.population, df.line)
    exp_ax.set_xlabel("Cumulative % of population")
    exp_ax.set_ylabel("Cumulative % of variable")
    exp_ax.set_title("Lorenz Curve")


@check_figures_equal()
def test_plot_generalized_lorenz(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)

    test_ax = fig_test.subplots()
    data.plot.lorenz(ax=test_ax, alpha="g")

    exp_ax = fig_ref.subplots()
    df = data.plot._lorenz_data(alpha="g")
    exp_ax.plot(df.population, df.variable)
    exp_ax.plot(df.population, df.line)
    exp_ax.set_xlabel("Cumulative % of population")
    exp_ax.set_ylabel("Scaled Cumulative % of variable")
    exp_ax.set_title("Generalized Lorenz Curve")


@check_figures_equal()
def test_plot_absolute_lorenz(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)

    test_ax = fig_test.subplots()
    data.plot.lorenz(ax=test_ax, alpha="a")

    exp_ax = fig_ref.subplots()
    df = data.plot._lorenz_data(alpha="a")
    exp_ax.plot(df.population, df.variable)
    exp_ax.plot(df.population, df.line)
    exp_ax.set_xlabel("Cumulative % of population")
    exp_ax.set_ylabel("Cumulative deviation")
    exp_ax.set_title("Absolut Lorenz Curve")


def test_lorenz_invalid_alpha():
    data = datasets.make_uniform(seed=42, size=300)
    with pytest.raises(ValueError):
        data.plot.lorenz("j")
        data.plot.lorenz("j")
        data.plot.lorenz(2)
        data.plot.lorenz(0)


@check_figures_equal()
def test_plot_lorenz_axes_None(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)

    # expected
    exp_ax = fig_ref.subplots()
    fig_ref.set_size_inches(w=plots.DEFAULT_WIDTH, h=plots.DEFAULT_HEIGHT)
    df = data.plot._lorenz_data(alpha="g")
    exp_ax.plot(df.population, df.variable)
    exp_ax.plot(df.population, df.line)
    exp_ax.set_xlabel("Cumulative % of population")
    exp_ax.set_ylabel("Scaled Cumulative % of variable")
    exp_ax.set_title("Generalized Lorenz Curve")

    # test
    test_ax = fig_test.subplots()
    with mock.patch("matplotlib.pyplot.gcf", return_value=fig_test):
        with mock.patch("matplotlib.pyplot.gca", return_value=test_ax):
            data.plot.lorenz(alpha="g")


# =============================================================================
# TESTS TIP
# =============================================================================


@check_figures_equal()
def test_plot_tip(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)
    pline = 3

    # expected
    exp_ax = fig_ref.subplots()
    df = data.plot._tip_data(pline=pline)
    exp_ax.plot(df.population, df.variable)
    exp_ax.set_title("TIP Curve")
    exp_ax.set_ylabel("Cumulated poverty gaps")
    exp_ax.set_xlabel("Cumulative % of population")
    exp_ax.legend()

    # test
    test_ax = fig_test.subplots()
    data.plot.tip(ax=test_ax, pline=pline)


def test_tip_invalid_alpha():
    data = datasets.make_uniform(seed=42, size=300)
    with pytest.raises(ValueError):
        data.plot.tip(pline=-2)
        data.plot.tip(pline=-0.001)


@check_figures_equal()
def test_plot_tip_axes_None(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)
    pline = 3
    # expected
    exp_ax = fig_ref.subplots()
    fig_ref.set_size_inches(w=plots.DEFAULT_WIDTH, h=plots.DEFAULT_HEIGHT)
    df = data.plot._tip_data(pline=pline)
    exp_ax.plot(df.population, df.variable)
    exp_ax.set_title("TIP Curve")
    exp_ax.set_ylabel("Cumulated poverty gaps")
    exp_ax.set_xlabel("Cumulative % of population")
    exp_ax.legend()

    # test
    test_ax = fig_test.subplots()
    with mock.patch("matplotlib.pyplot.gcf", return_value=fig_test):
        with mock.patch("matplotlib.pyplot.gca", return_value=test_ax):
            data.plot.tip(pline=pline)


# =============================================================================
# TESTS PEN
# =============================================================================
@check_figures_equal()
def test_plot_pen(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)
    pline = 3

    # expected
    exp_ax = fig_test.subplots()
    df, me = data.plot._pen_data(pline=pline)
    exp_ax.plot(df.population, df.variable)
    exp_ax.plot(df.population, df.line, label="Mean")
    qpl = np.ones(len(df.variable)) * pline / me
    exp_ax.plot(df.population, qpl, label="Poverty line")
    exp_ax.set_xlabel("Cumulative % of population")
    exp_ax.set_ylabel("Medianized variable")
    exp_ax.set_title("Pen's Parade")
    exp_ax.legend()

    # test
    test_ax = fig_ref.subplots()
    data.plot.pen(ax=test_ax, pline=pline)


@check_figures_equal()
def test_plot_pen_pline_None(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)

    # expected
    exp_ax = fig_ref.subplots()
    df, me = data.plot._pen_data(pline=None)
    exp_ax.plot(df.population, df.variable)
    exp_ax.plot(df.population, df.line, label="Mean")
    exp_ax.set_xlabel("Cumulative % of population")
    exp_ax.set_ylabel("Medianized variable")
    exp_ax.set_title("Pen's Parade")
    exp_ax.legend()

    # test
    test_ax = fig_test.subplots()
    data.plot.pen(ax=test_ax, pline=None)


@check_figures_equal()
def test_plot_pen_axes_None(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)
    pline = 3
    # expected
    exp_ax = fig_ref.subplots()
    fig_ref.set_size_inches(w=plots.DEFAULT_WIDTH, h=plots.DEFAULT_HEIGHT)
    df, me = data.plot._pen_data(pline=pline)
    exp_ax.plot(df.population, df.variable)
    exp_ax.plot(df.population, df.line, label="Mean")
    qpl = np.ones(len(df.variable)) * pline / me
    exp_ax.plot(df.population, qpl, label="Poverty line")
    exp_ax.set_xlabel("Cumulative % of population")
    exp_ax.set_ylabel("Medianized variable")
    exp_ax.set_title("Pen's Parade")
    exp_ax.legend()

    # test
    test_ax = fig_test.subplots()
    with mock.patch("matplotlib.pyplot.gcf", return_value=fig_test):
        with mock.patch("matplotlib.pyplot.gca", return_value=test_ax):
            data.plot.pen(pline=pline)


@check_figures_equal()
def test_plot_pen_axes_None_pline_None(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)
    # expected
    exp_ax = fig_ref.subplots()
    fig_ref.set_size_inches(w=plots.DEFAULT_WIDTH, h=plots.DEFAULT_HEIGHT)
    df, me = data.plot._pen_data(pline=None)
    exp_ax.plot(df.population, df.variable)
    exp_ax.plot(df.population, df.line, label="Mean")
    exp_ax.set_xlabel("Cumulative % of population")
    exp_ax.set_ylabel("Medianized variable")
    exp_ax.set_title("Pen's Parade")
    exp_ax.legend()

    # test
    test_ax = fig_test.subplots()
    with mock.patch("matplotlib.pyplot.gcf", return_value=fig_test):
        with mock.patch("matplotlib.pyplot.gca", return_value=test_ax):
            data.plot.pen(pline=None)


# =============================================================================
# TESTS HIST
# =============================================================================
@check_figures_equal()
def test_plot_hist(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)

    test_ax = fig_test.subplots()
    data.plot.hist(ax=test_ax)

    exp_ax = fig_ref.subplots()
    data.data.plot.hist(ax=exp_ax)


@check_figures_equal()
def test_plot_hist_ax_None(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)

    # expected
    exp_ax = fig_ref.subplots()
    with mock.patch("matplotlib.pyplot.gcf", return_value=fig_ref):
        with mock.patch("matplotlib.pyplot.gca", return_value=exp_ax):
            data.plot.hist(ax=None)

    # test
    test_ax = fig_test.subplots()
    with mock.patch("matplotlib.pyplot.gcf", return_value=fig_test):
        with mock.patch("matplotlib.pyplot.gca", return_value=test_ax):
            data.data.plot.hist(ax=None)


@pytest.mark.xfail
def test_hist_isequal():
    data = datasets.make_uniform(seed=42, size=300)
    assert data.plot.hist is data.plot.hist
