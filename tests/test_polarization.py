#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Apode Project (https://github.com/mchalela/apode).
# Copyright (c) 2020, Néstor Grión and Sofía Sappia
# License: MIT
#   Full Text: https://github.com/ngrion/apode/blob/master/LICENSE.txt

import pytest
import numpy as np
import pandas as pd

from apode.basic import ApodeData

from apode import datasets


# =============================================================================
# TESTS COMMON
# =============================================================================


def test_default_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.polarization("ray")
    method_result = data.polarization.ray()
    assert call_result == method_result


def test_invalid():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(AttributeError):
        data.polarization("foo")


# =============================================================================
# TESTS RAY
# =============================================================================
def test_ray_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.polarization.ray() == 0.001130140610524159


def test_ray_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.polarization("ray") == 0.001130140610524159


def test_ray_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.polarization("ray")
    method_result = data.polarization.ray()
    assert call_result == method_result


# =============================================================================
# TESTS WOLFOSN
# =============================================================================
def test_wolfson_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(data.polarization.wolfson(), 0.3422542089105541)


def test_wolfson_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(
        data.polarization("wolfson"), 0.3422542089105541
    )


def test_wolfson_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.polarization("wolfson")
    method_result = data.polarization.wolfson()
    assert call_result == method_result


def test_wolfson_symmetry():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.polarization(method="wolfson") == dr2.polarization(
        method="wolfson"
    )
