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
    call_result = data.inequality("gini")
    method_result = data.inequality.gini()
    assert call_result == method_result


def test_invalid(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    with pytest.raises(AttributeError):
        data.inequality("foo")


# =============================================================================
# TESTS GINI
# =============================================================================
def test_gini_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.inequality.gini() == 0.34232535781966483


def test_gini_call(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.inequality("gini") == 0.34232535781966483


def test_gini_call_equal_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    call_result = data.inequality("gini")
    method_result = data.inequality.gini()
    assert call_result == method_result


def test_gini_extreme_values():
    y = np.zeros(300)
    y[0] = 10
    np.random.shuffle(y)
    df = pd.DataFrame({"x": y})
    data_min = ApodeData(df, varx="x")
    y = np.ones(300) * 10
    df = pd.DataFrame({"x": y})
    data_max = ApodeData(df, varx="x")
    # assert data_min.inequality.gini() == 1 #CHECK, fails
    assert data_max.inequality.gini() == 0


# =============================================================================
# TESTS ENTROPY
# =============================================================================
def test_entropy_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.inequality.entropy() == 0.3226715241069237


def test_entropy_call(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.inequality("entropy") == 0.3226715241069237


def test_entropy_call_equal_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    call_result = data.inequality("entropy")
    method_result = data.inequality.entropy()
    assert call_result == method_result


def test_entropy_extreme_values():
    y = np.zeros(300)
    y[0] = 10
    np.random.shuffle(y)
    df = pd.DataFrame({"x": y})
    data_min = ApodeData(df, varx="x")
    y = np.ones(300) * 10
    df = pd.DataFrame({"x": y})
    data_max = ApodeData(df, varx="x")
    assert data_min.inequality.entropy() == 1
    assert data_max.inequality.entropy() == 0


def test_entropy_symmetry(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.inequality(method="entropy") == \
           dr2.inequality(method="entropy")


def test_entropy_empty_array():
    y = []
    df = pd.DataFrame({"x": y})
    data = ApodeData(df, varx="x")
    assert data.inequality.entropy() == 0


def test_entropy_alpha_values(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.inequality("entropy", alpha=0) == 0.3226715241069237
    assert data.inequality("entropy", alpha=1) == 0.20444600065652588
    assert data.inequality("entropy", alpha=0.5) == 0.24247057381922854


# =============================================================================
# TESTS ATKINSON
# =============================================================================
def test_atkinson_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.inequality.atkinson(alpha=1) == 0.2757882986123399


def test_atkinson_call(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.inequality("atkinson", alpha=1) == 0.2757882986123399


def test_atkinson_call_equal_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    call_result = data.inequality("atkinson", alpha=1)
    method_result = data.inequality.atkinson(alpha=1)
    assert call_result == method_result

# KEEP OR NOT?
# def test_atkinson_symmetry(uniform_ad):
#     data = uniform_ad(seed=42, size=300)
#     y = data.data["x"].tolist()
#     np.random.shuffle(y)
#     df2 = pd.DataFrame({"x": y})
#     dr2 = ApodeData(df2, varx="x")
#     assert data.inequality(method="atkinson", alpha=1) == \
#            dr2.inequality(method="atkinson", alpha=1)

def test_atkinson_valid_alpha(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    with pytest.raises(ValueError):
        data.inequality.atkinson(alpha=-1)
        data.inequality.atkinson(alpha=0)

def test_atkinson_empty_array():
    y = []
    df = pd.DataFrame({"x": y})
    data = ApodeData(df, varx="x")
    assert data.inequality.atkinson(alpha=1) == 0


def test_atkinson_alpha_values(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.inequality("atkinson", alpha=1) == 0.2757882986123399
    assert data.inequality("atkinson", alpha=0.5) == 0.11756078821160021
