#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Apode Project (https://github.com/mchalela/apode).
# Copyright (c) 2020, Néstor Grión and Sofía Sappia
# License: MIT
#   Full Text: https://github.com/ngrion/apode/blob/master/LICENSE.txt

import pytest
from matplotlib.testing.decorators import check_figures_equal

from apode import datasets
from apode import plots

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
    data.plot.lorenz(ax=exp_ax, alpha="r")


@check_figures_equal()
def test_plot_generalized_lorenz(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)

    test_ax = fig_test.subplots()
    data.plot.lorenz(ax=test_ax, alpha="g")

    exp_ax = fig_ref.subplots()
    data.plot.lorenz(ax=exp_ax, alpha="g")


@check_figures_equal()
def test_plot_absolute_lorenz(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)

    test_ax = fig_test.subplots()
    data.plot.lorenz(ax=test_ax, alpha="a")

    exp_ax = fig_ref.subplots()
    data.plot.lorenz(ax=exp_ax, alpha="a")


def test_lorenz_invalid_alpha():
    data = datasets.make_uniform(seed=42, size=300)
    with pytest.raises(ValueError):
        data.plot.lorenz("j")


@check_figures_equal()
def test_plot_lorenz_axes_None(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)
    # expected
    width = plots.DEFAULT_WIDTH
    height = plots.DEFAULT_HEIGHT

    test_ax = fig_test.subplots()
    fig_test.set_size_inches(w=width, h=height)
    data.plot.lorenz(ax=test_ax, alpha="g")

    # test
    data.plot.lorenz(alpha="g")


# =============================================================================
# TESTS TIP
# =============================================================================


@check_figures_equal()
def test_plot_tip(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)

    test_ax = fig_test.subplots()
    data.plot.tip(ax=test_ax, pline=3)

    exp_ax = fig_ref.subplots()
    data.plot.tip(ax=exp_ax, pline=3)


def test_tip_invalid_alpha():
    data = datasets.make_uniform(seed=42, size=300)
    with pytest.raises(ValueError):
        data.plot.tip(pline=-2)
        data.plot.tip(pline=-0.001)


@check_figures_equal()
def test_plot_tip_axes_None(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)
    # expected
    width = plots.DEFAULT_WIDTH
    height = plots.DEFAULT_HEIGHT

    test_ax = fig_test.subplots()
    fig_test.set_size_inches(w=width, h=height)
    data.plot.tip(ax=test_ax, pline=10)

    # test
    data.plot.tip(pline=10)


# =============================================================================
# TESTS PEN
# =============================================================================
@check_figures_equal()
def test_plot_pen(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)

    test_ax = fig_test.subplots()
    data.plot.pen(ax=test_ax, pline=3)

    exp_ax = fig_ref.subplots()
    data.plot.pen(ax=exp_ax, pline=3)


@check_figures_equal()
def test_plot_pen_axes_None(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)
    # expected
    width = plots.DEFAULT_WIDTH
    height = plots.DEFAULT_HEIGHT

    test_ax = fig_test.subplots()
    fig_test.set_size_inches(w=width, h=height)
    data.plot.pen(ax=test_ax)

    # test
    data.plot.pen()


# =============================================================================
# TESTS HIST
# =============================================================================
@check_figures_equal()
def test_plot_hist(fig_test, fig_ref):
    data = datasets.make_uniform(seed=42, size=300)

    test_ax = fig_test.subplots()
    data.plot.hist(ax=test_ax)

    exp_ax = fig_ref.subplots()
    data.plot.hist(ax=exp_ax)
