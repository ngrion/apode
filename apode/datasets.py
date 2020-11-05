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

"""Data simulation tools for Apode."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import pandas as pd

from apode.basic import ApodeData


# =============================================================================
# FUNCTIONS
# =============================================================================


def make_pareto(seed=None, a=5, size=100, c=200, nbin=None):
    """Pareto Distribution.

    Parameters
    ----------
    seed: int, optional(default=None)

    a: float, optional(default=5)

    size: int, optional(default=100)

    c: int, optional(default=200)

    nbin: int, optional(default=None)

    Return
    ------
    out: float array
        Array of random numbers.

    """
    random = np.random.RandomState(seed=seed)
    y = c * random.pareto(a=a, size=size)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_uniform(seed=None, size=100, mu=100, nbin=None):
    """Uniform Distribution.

    Parameters
    ----------
    seed: int, optional(default=None)

    size: int, optional(default=100)

    mu: float, optional(default=100)

    nbin: int, optional(default=None)

    Return
    ------
    out: float array
        Array of random numbers.

    """
    random = np.random.RandomState(seed=seed)
    y = random.uniform(size=size) * mu
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_lognormal(seed=None, size=100, sigma=1.0, nbin=None):
    """Lognormal Distribution.

    Parameters
    ----------
    seed: int, optional(default=None)

    size: int, optional(default=100)

    sigma: float, optional(default=1.0)

    nbin: int, optional(default=None)

    Return
    ------
    out: float array
        Array of random numbers.

    """
    random = np.random.RandomState(seed=seed)
    y = random.lognormal(mean=3.3, sigma=sigma, size=size)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_chisquare(seed=None, size=100, df=5, c=10, nbin=None):
    """Chisquare Distribution.

    Parameters
    ----------
    seed: int, optional(default=None)

    size: int, optional(default=100)

    df: float, optional(default=5)

    c: float, optional(default=10)

    nbin: int, optional(default=None)

    Return
    ------
    out: float array
        Array of random numbers.

    """
    random = np.random.RandomState(seed=seed)
    y = c * random.chisquare(df=df, size=size)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_gamma(seed=None, size=100, shape=1, scale=50.0, nbin=None):
    """Gamma Distribution.

    Parameters
    ----------
    seed: int, optional(default=None)

    size: int, optional(default=100)

    shape: float, optional(default=1.0)

    scale: float, optional(default=50.0)

    nbin: int, optional(default=None)

    Return
    ------
    out: float array
        Array of random numbers.

    """
    random = np.random.RandomState(seed=seed)
    y = random.gamma(shape=shape, scale=scale, size=size)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_weibull(seed=None, size=100, a=1.5, c=50, nbin=None):
    """Weibull Distribution.

    Parameters
    ----------
    seed: int, optional(default=None)

    size: int, optional(default=100)

    a: float, optional(default=1.5)

    c: float, optional(default=50)

    nbin: int, optional(default=None)

    Return
    ------
    out: float array
        Array of random numbers.

    """
    random = np.random.RandomState(seed=seed)
    y = c * random.weibull(a=a, size=size)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_exponential(seed=None, size=100, scale=1, c=50, nbin=None):
    """Exponential Distribution.

    Parameters
    ----------
    seed: int, optional(default=None)

    size: int, optional(default=100)

    scale: float, optional(default=1.0)

    c: float, optional(default=50)

    nbin: int, optional(default=None)

    Return
    ------
    out: float array
        Array of random numbers.

    """
    random = np.random.RandomState(seed=seed)
    y = c * random.exponential(scale=scale, size=size)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_constant(size=100, nbin=None):
    """Constant value Distribution.

    Parameters
    ----------
    size: int, optional(default=100)

    nbin: int, optional(default=None)

    Return
    ------
    out: float array

    """
    y = np.ones(size) * 10
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_linear(size=100, nbin=None):
    """Linear value Distribution.

    Parameters
    ----------
    size: int, optional(default=100)

    nbin: int, optional(default=None)

    Return
    ------
    out: float array

    """
    y = np.arange(1, size + 1)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_squared(size=100, nbin=None):
    """Squared value Distribution.

    Parameters
    ----------
    size: int, optional(default=100)

    nbin: int, optional(default=None)

    Return
    ------
    out: float array

    """
    y = np.power(np.arange(1, size + 1), 2)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_extreme(size=100, nbin=None):
    """Extreme value Distribution.

    Parameters
    ----------
    size: int, optional(default=100)

    nbin: int, optional(default=None)

    Return
    ------
    out: float array

    """
    y = np.concatenate((np.zeros(size - 1), [10]))
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_unimodal(size=100, nbin=None):
    """Unimodal Distribution.

    Parameters
    ----------
    size: int, optional(default=100)

    nbin: int, optional(default=None)

    Return
    ------
    out: float array

    """
    n1 = size // 2
    a = np.power(np.arange(1, n1 + 1), 1.2)
    b = n1 ^ 2
    y = np.concatenate((b - a, b + a))
    y = np.sort(y - np.min(y) + 1)
    y = np.concatenate((np.zeros(size - 1), [10]))
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_bimodal(size=100, nbin=None):
    """Bimodal Distribution.

    Parameters
    ----------
    size: int, optional(default=100)

    nbin: int, optional(default=None)

    Return
    ------
    out: float array

    """
    n1 = size // 2
    a = np.power(np.arange(1, n1 + 1), 0.5)
    b = n1 ^ 2
    y = np.concatenate((b - a, b + a))
    y = np.sort(y - np.min(y) + 1)
    y = np.concatenate((np.zeros(size - 1), [10]))
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


# generalizar columnanme?
def binning(df, pos=0, nbin=None):
    """Binning function.

    Agrupa valores de un dataframe en nbin categorías.

    Parameters
    ----------
    df: DataFrame

    pos: int, optional(default=0)
        Options are r: relative, 'g': generalized, 'a': absolut.

    nbin: int, optional(default=None)

    Return
    ------
    out: DataFrame
        Grouped data

    """
    if nbin is None:
        nbin = np.trunc(np.sqrt(df.shape[0]))
    s1 = df.groupby(pd.cut(df.iloc[:, pos], nbin)).count()
    s2 = df.groupby(pd.cut(df.iloc[:, pos], nbin)).mean()
    dfb = pd.concat([s1, s2], axis=1).dropna()
    dfb.columns = ["weight", "x"]
    dfb.reset_index(drop=True, inplace=True)
    return dfb
