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


# =============================================================================
# TESTS COMMON
# =============================================================================
def test_default_call(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    call_result = data.concentration("herfindahl")
    method_result = data.concentration.herfindahl()
    assert call_result == method_result


def test_invalid(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    with pytest.raises(AttributeError):
        data.concentration("foo")


# =============================================================================
# TESTS HERFINDAHL
# =============================================================================
def test_herfindahl_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.concentration.herfindahl(normalized=True) == 0.0011776319218515382
    assert data.concentration.herfindahl(normalized=False) == 0.004507039815445367


def test_herfindahl_call(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.concentration("herfindahl", normalized=True) == 0.0011776319218515382
    assert data.concentration("herfindahl", normalized=False) == 0.004507039815445367


def test_herfindahl_call_equal_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    call_result = data.concentration("herfindahl")
    method_result = data.concentration.herfindahl()
    assert call_result == method_result


def test_herfindahl_empty_array():
    y = []
    df = pd.DataFrame({"x": y})
    data = ApodeData(df, varx="x")
    assert data.concentration.herfindahl() == 0


# =============================================================================
# TESTS ROSENBLUTH
# =============================================================================
def test_rosenbluth_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.concentration.rosenbluth() == 0.00506836225627098


def test_rosenbluth_call(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.concentration("rosenbluth") == 0.00506836225627098


def test_rosenbluth_call_equal_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    call_result = data.concentration("rosenbluth")
    method_result = data.concentration.rosenbluth()
    assert call_result == method_result


def test_rosenbluth_symmetry(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.concentration(method="rosenbluth") == \
           dr2.concentration(method="rosenbluth")


# =============================================================================
# TESTS CONCENTRATION_RATIO
# =============================================================================
def test_concentration_ratio_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.concentration.concentration_ratio(k=20) == 0.12913322818634668


def test_concentration_ratio_call(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.concentration("concentration_ratio", k=20) == 0.12913322818634668


def test_concentration_ratio_call_equal_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    call_result = data.concentration("concentration_ratio", k=20)
    method_result = data.concentration.concentration_ratio(k=20)
    assert call_result == method_result


def test_concentration_ratio_symmetry(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.concentration(method="concentration_ratio", k=20) == \
           dr2.concentration(method="concentration_ratio", k=20)


def test_concentrarion_k_range(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    n = len(data.data.values)
    with pytest.raises(ValueError):
        data.concentration(method="concentration_ratio", k=n + 1)
        data.concentration(method="concentration_ratio", k=0)
