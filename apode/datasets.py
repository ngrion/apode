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

def make_pareto(seed=None, a=5, size=100, c=200, nbin=None)
    random = np.random.RandomState(seed=seed)
    y = c * random.pareto(a=a, size=size)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_uniform(seed=None, size=100, mu=100, nbin=None)
    random = np.random.RandomState(seed=seed)
    y = random.uniform(size=size) * mu
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_lognormal(seed=None, size=100, sigma=1.0, nbin=None):
    random = np.random.RandomState(seed=seed)
    y = random.lognormal(mean=3.3, sigma=sigma, size=size)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_chisquare(seed=None, size=100, df=5, c=10, nbin=None):
    random = np.random.RandomState(seed=seed)
    y = c * random.chisquare(df=df, size=size)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_gamma(seed=None, size=100, shape=1, scale=50.0, nbin=None):
    random = np.random.RandomState(seed=seed)
    y = c * random.gamma(shape=shape, scale=scale, size=size)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_weibull(seed=None, size=100, a=1.5, c=50, nbin=None):
    random = np.random.RandomState(seed=seed)
    y = c * random.weibull(a=a, size=size)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def make_exponential(seed=None, size=100, scale=1, c=50, nbin=None):
    random = np.random.RandomState(seed=seed)
    y = c * random.exponential(scale=scale, size=size)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return ApodeData(df, varx="x")
    else:
        df = binning(df, nbin=nbin)
        return ApodeData(df, varx="x")


def distribution_examples(rg, dist, n, nbin=None):
    y = distribution_examples_aux(rg, dist, n)
    df = pd.DataFrame({"x": y})
    if nbin is None:
        return df
    else:
        return binning(df, nbin=nbin)


# generalizar columnanme
# puede remover nan
def binning(df, pos=0, nbin=None):
    if nbin is None:
        nbin = int(np.sqrt(df.shape[0]))
    s1 = df.groupby(pd.cut(df.iloc[:, pos], nbin)).count()
    s2 = df.groupby(pd.cut(df.iloc[:, pos], nbin)).mean()
    dfb = pd.concat([s1, s2], axis=1).dropna()
    dfb.columns = ["weight", "x"]
    dfb.reset_index(drop=True, inplace=True)
    return dfb
