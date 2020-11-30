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
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert isinstance(ad.data, pd.DataFrame)


def test_invalid():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(AttributeError):
        ad.poverty("foo")


def test_empty_data():
    df = pd.DataFrame({'x'', []'})
    with pytest.raises(ValueError):
        ApodeData(df, 'x')


def test_income_column_validator():
    random = np.random.RandomState(seed=42)
    x = random.uniform(size=300)
    df1 = pd.DataFrame({"x": x})
    with pytest.raises(ValueError):
        ApodeData(df1, income_column="y")


def test_call_poverty():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.3
    pov = PovertyMeasures(ad)
    assert pov.idf.data.equals(ad.data)
    assert pov.headcount(pline) == ad.poverty.headcount(pline)


def test_call_inequality():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    ineq = InequalityMeasures(ad)
    assert ineq.idf.data.equals(ad.data)
    assert ineq.gini() == ad.inequality.gini()


def test_call_polarization():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pol = PolarizationMeasures(ad)
    assert pol.idf.data.equals(ad.data)
    assert pol.wolfson() == ad.polarization.wolfson()


def test_call_concentration():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    conc = ConcentrationMeasures(ad)
    assert conc.idf.data.equals(ad.data)
    assert conc.rosenbluth() == ad.concentration.rosenbluth()


def test_getitem_numeric_slices():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    expected1 = ad.data[2:5]
    result1 = ad[2:5].data
    expected2 = ad.data[:-1]
    result2 = ad[:-1].data
    expected3 = ad.data[:]
    result3 = ad[:].data
    assert expected1.equals(result1)
    assert expected2.equals(result2)
    assert expected3.equals(result3)


def test_getitem_column_slice():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(AttributeError):
        ad["x"]
    with pytest.raises(KeyError):
        ad["y"]
    with pytest.raises(KeyError):
        ad["income_column"]


def test_getattr():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.shape == ad.data.shape
    np.testing.assert_array_equal(ad.sum(), ad.data.sum())


def test_repr():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pd.option_context("display.show_dimensions", False):
        df_body = repr(ad.data).splitlines()
    footer = ad._get_footer()
    expected = "\n".join(df_body + [footer])
    assert repr(ad) == expected


def test_repr_html():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pd.option_context("display.show_dimensions", False):
        df_html = ad.data._repr_html_()
    ad_id = id(ad)
    footer = ad._get_footer(html=True)
    parts = [
        f'<div class="apode-data-container" id={ad_id}>',
        df_html,
        footer,
        "</div>",
    ]
    expected = "".join(parts)
    assert ad._repr_html_() == expected


def test_dir():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    for a in dir(ad):
        assert hasattr(ad, a)
