#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Apode Project (https://github.com/mchalela/apode).
# Copyright (c) 2020, Néstor Grión and Sofía Sappia
# License: MIT
#   Full Text: https://github.com/ngrion/apode/blob/master/LICENSE.txt

from apode import ApodeData
from apode import datasets
from apode.concentration import ConcentrationMeasures
from apode.inequality import InequalityMeasures
from apode.polarization import PolarizationMeasures
from apode.poverty import PovertyMeasures


import numpy as np

import pandas as pd

import pytest


def test_df_converter():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert isinstance(data.data, pd.DataFrame)


def test_invalid():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(AttributeError):
        data.poverty("foo")


def test_income_column_validator():
    random = np.random.RandomState(seed=42)
    x = random.uniform(size=300)
    df1 = pd.DataFrame({"x": x})
    with pytest.raises(ValueError):
        ApodeData(df1, income_column="y")


def test_call_poverty():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0
    pov = PovertyMeasures(data)
    assert pov.idf.data.equals(data.data)
    assert pov.headcount(pline) == data.poverty.headcount(pline)


def test_call_inequality():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    ineq = InequalityMeasures(data)
    assert ineq.idf.data.equals(data.data)
    assert ineq.gini() == data.inequality.gini()


def test_call_polarization():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pol = PolarizationMeasures(data)
    assert pol.idf.data.equals(data.data)
    assert pol.wolfson() == data.polarization.wolfson()


def test_call_concentration():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    conc = ConcentrationMeasures(data)
    assert conc.idf.data.equals(data.data)
    assert conc.rosenbluth() == data.concentration.rosenbluth()


def test_getitem_numeric_slices():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    expected1 = data.data[2:5]
    result1 = data[2:5].data
    expected2 = data.data[:-1]
    result2 = data[:-1].data
    expected3 = data.data[:]
    result3 = data[:].data
    assert expected1.equals(result1)
    assert expected2.equals(result2)
    assert expected3.equals(result3)


def test_getitem_column_slice():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(AttributeError):
        data["x"]
        data["y"]
        data["income_column"]
