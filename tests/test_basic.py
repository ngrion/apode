#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Apode Project (https://github.com/mchalela/apode).
# Copyright (c) 2020, Néstor Grión and Sofía Sappia
# License: MIT
#   Full Text: https://github.com/ngrion/apode/blob/master/LICENSE.txt

import pytest
import pandas as pd
import numpy as np

from apode.poverty import PovertyMeasures
from apode.inequality import InequalityMeasures
from apode.concentration import ConcentrationMeasures
from apode.polarization import PolarizationMeasures
from apode.basic import  ApodeData


def test_df_converter(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert isinstance(data.data, pd.DataFrame)

def test_invalid(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    with pytest.raises(AttributeError):
        data.poverty("foo")

def test_income_column_validator():
    random = np.random.RandomState(seed=42)
    x = random.uniform(size=300)
    df1 = pd.DataFrame({"x": x})
    with pytest.raises(ValueError):
        ApodeData(df1, income_column="y")


def test_call_poverty(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    pline = 0
    pov = PovertyMeasures(data)
    assert pov.idf.data.equals(data.data)
    assert pov.headcount(pline) == data.poverty.headcount(pline)


def test_call_inequality(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    ineq = InequalityMeasures(data)
    assert ineq.idf.data.equals(data.data)
    assert ineq.gini() == data.inequality.gini()

def test_call_polarization(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    pol = PolarizationMeasures(data)
    assert pol.idf.data.equals(data.data)
    assert pol.wolfson() == data.polarization.wolfson()

def test_call_concentration(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    conc = ConcentrationMeasures(data)
    assert conc.idf.data.equals(data.data)
    assert conc.rosenbluth() == data.concentration.rosenbluth()