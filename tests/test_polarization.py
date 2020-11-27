#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Apode Project (https://github.com/mchalela/apode).
# Copyright (c) 2020, Néstor Grión and Sofía Sappia
# License: MIT
#   Full Text: https://github.com/ngrion/apode/blob/master/LICENSE.txt

from apode import datasets
from apode.basic import ApodeData

import numpy as np

import pandas as pd

import pytest


# =============================================================================
# TESTS COMMON
# =============================================================================


def test_default_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.polarization("ray")
    method_result = ad.polarization.ray()
    assert call_result == method_result


def test_invalid():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(AttributeError):
        ad.polarization("foo")


# =============================================================================
# TESTS RAY
# =============================================================================
def test_ray_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.polarization.ray() == 0.001130140610524159


def test_ray_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.polarization("ray") == 0.001130140610524159


def test_ray_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.polarization("ray")
    method_result = ad.polarization.ray()
    assert call_result == method_result


# =============================================================================
# TESTS WOLFOSN
# =============================================================================
def test_wolfson_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(ad.polarization.wolfson(), 0.3422542089105541)


def test_wolfson_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(ad.polarization("wolfson"), 0.3422542089105541)


def test_wolfson_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.polarization("wolfson")
    method_result = ad.polarization.wolfson()
    assert call_result == method_result


def test_wolfson_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    ad2 = ApodeData(df2, income_column="x")
    assert ad.polarization(method="wolfson") == ad2.polarization(
        method="wolfson"
    )
