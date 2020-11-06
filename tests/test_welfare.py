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
    call_result = data.welfare("utilitarian")
    method_result = data.welfare.utilitarian()
    assert call_result == method_result


def test_invalid(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    with pytest.raises(AttributeError):
        data.welfare("foo")


# =============================================================================
# TESTS UTILITARIAN
# =============================================================================
def test_utilitarian_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.welfare.utilitarian() == 0.4952045990934922


def test_utilitarian_call(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.welfare("utilitarian") == 0.4952045990934922


def test_utilitarian_call_equal_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    call_result = data.welfare("utilitarian")
    method_result = data.welfare.utilitarian()
    assert call_result == method_result


def test_utilitarian_symmetry(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.welfare(method="utilitarian") == \
           dr2.welfare(method="utilitarian")


def test_utilitarian_replication(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    k = 2  # factor
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.welfare("utilitarian") == \
           dr2.welfare("utilitarian")


def test_utilitarian_homogeneity(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    k = 2  # factor
    y = data.data["x"].tolist()
    y = k * y
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.welfare("utilitarian") == \
           dr2.welfare("utilitarian")


# =============================================================================
# TESTS RAWLSIAN
# =============================================================================
def test_rawlsian_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.welfare.rawlsian() == 0.005061583846218687


def test_rawlsian_call(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.welfare("rawlsian") == 0.005061583846218687


def test_rawlsian_call_equal_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    call_result = data.welfare("rawlsian")
    method_result = data.welfare.rawlsian()
    assert call_result == method_result


def test_rawlsian_symmetry(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.welfare(method="rawlsian") == \
           dr2.welfare(method="rawlsian")


def test_rawlsian_replication(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    k = 2  # factor
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.welfare("rawlsian") == \
           dr2.welfare("rawlsian")


def test_rawlsian_homogeneity(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    k = 2  # factor
    y = data.data["x"].tolist()
    y = k * y
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.welfare("rawlsian") == \
           dr2.welfare("rawlsian")


# =============================================================================
# TESTS ISOELASTIC
# =============================================================================
def test_isoelastic_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.welfare.isoelastic(alpha=0) == 0.4952045990934922


def test_isoelastic_call(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.welfare("isoelastic", alpha=0) == 0.4952045990934922


def test_isoelastic_call_equal_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    call_result = data.welfare("isoelastic", alpha=0)
    method_result = data.welfare.isoelastic(alpha=0)
    assert call_result == method_result


def test_isoelastic_alpha_cases(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    alpha_null = data.welfare("isoelastic", alpha=0)
    alpha_inf = data.welfare("isoelastic", alpha=1e2000)
    alpha_one = data.welfare("isoelastic", alpha=1)
    alpha_other = data.welfare("isoelastic", alpha=10)

    assert alpha_null == 0.4952045990934922
    assert alpha_inf == 0.005061583846218687
    assert alpha_one == -1.0254557944163005
    assert alpha_other == -2.579772844232791e+17


def test_isoelastic_symmetry(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.welfare(method="isoelastic", alpha=0) == \
           dr2.welfare(method="isoelastic", alpha=0)


def test_isoelastic_replication(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    k = 2  # factor
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.welfare("isoelastic", alpha=0) == \
           dr2.welfare("isoelastic", alpha=0)


def test_isoelastic_homogeneity(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    k = 2  # factor
    y = data.data["x"].tolist()
    y = k * y
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.welfare("isoelastic", alpha=0) == \
           dr2.welfare("isoelastic", alpha=0)


# =============================================================================
# TESTS SEN
# =============================================================================
def test_sen_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.welfare.sen() == 0.32568350751486885


def test_sen_call(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.welfare("sen") == 0.32568350751486885


def test_sen_call_equal_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    call_result = data.welfare("sen")
    method_result = data.welfare.sen()
    assert call_result == method_result


def test_sen_symmetry(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.welfare(method="sen") == \
           dr2.welfare(method="sen")


# =============================================================================
# TESTS THEILL
# =============================================================================
def test_theill_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.welfare.theill() == 0.35863296524449223


def test_theill_call(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.welfare("theill") == 0.35863296524449223


def test_theill_call_equal_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    call_result = data.welfare("theill")
    method_result = data.welfare.theill()
    assert call_result == method_result


def test_theill_symmetry(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.welfare(method="theill") == \
           dr2.welfare(method="theill")


def test_theill_replication(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    k = 2  # factor
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.welfare("theill") == \
           dr2.welfare("theill")


def test_theill_homogeneity(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    k = 2  # factor
    y = data.data["x"].tolist()
    y = k * y
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.welfare("theill") == \
           dr2.welfare("theill")


# =============================================================================
# TESTS theilt
# =============================================================================
def test_theilt_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.welfare.theilt() == 0.4036406524522584


def test_theilt_call(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    assert data.welfare("theilt") == 0.4036406524522584


def test_theilt_call_equal_method(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    call_result = data.welfare("theilt")
    method_result = data.welfare.theilt()
    assert call_result == method_result


def test_theilt_symmetry(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(data.welfare(method="theilt"), \
           dr2.welfare(method="theilt"))


def test_theilt_replication(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    k = 2  # factor
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.welfare("theilt") == \
           dr2.welfare("theilt")


def test_theilt_homogeneity(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    k = 2  # factor
    y = data.data["x"].tolist()
    y = k * y
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.welfare("theilt") == \
           dr2.welfare("theilt")
