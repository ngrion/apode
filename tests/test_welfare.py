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
    call_result = ad.welfare("utilitarian")
    method_result = ad.welfare.utilitarian()
    assert call_result == method_result


def test_invalid():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(AttributeError):
        ad.welfare("foo")


# =============================================================================
# TESTS UTILITARIAN
# =============================================================================
def test_utilitarian_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.welfare.utilitarian() == 0.4952045990934922


def test_utilitarian_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.welfare("utilitarian") == 0.4952045990934922


def test_utilitarian_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.welfare("utilitarian")
    method_result = ad.welfare.utilitarian()
    assert call_result == method_result


def test_utilitarian_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(
        ad.welfare(method="utilitarian"), dr2.welfare(method="utilitarian")
    )


def test_utilitarian_replication():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    y = k * ad.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.welfare("utilitarian") == dr2.welfare("utilitarian")


def test_utilitarian_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    y = ad.data["x"].tolist()
    y = k * y
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.welfare("utilitarian") == dr2.welfare("utilitarian")


# =============================================================================
# TESTS RAWLSIAN
# =============================================================================
def test_rawlsian_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.welfare.rawlsian() == 0.005061583846218687


def test_rawlsian_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.welfare("rawlsian") == 0.005061583846218687


def test_rawlsian_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.welfare("rawlsian")
    method_result = ad.welfare.rawlsian()
    assert call_result == method_result


def test_rawlsian_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.welfare(method="rawlsian") == dr2.welfare(method="rawlsian")


def test_rawlsian_replication():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    y = k * ad.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.welfare("rawlsian") == dr2.welfare("rawlsian")


def test_rawlsian_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    y = ad.data["x"].tolist()
    y = k * y
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.welfare("rawlsian") == dr2.welfare("rawlsian")


# =============================================================================
# TESTS ISOELASTIC
# =============================================================================
def test_isoelastic_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.welfare.isoelastic(alpha=0) == 0.4952045990934922


def test_isoelastic_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.welfare("isoelastic", alpha=0) == 0.4952045990934922


def test_isoelastic_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.welfare("isoelastic", alpha=0)
    method_result = ad.welfare.isoelastic(alpha=0)
    assert call_result == method_result


def test_isoelastic_alpha_cases():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    alpha_null = ad.welfare("isoelastic", alpha=0)
    alpha_inf = ad.welfare("isoelastic", alpha=1e2000)
    alpha_one = ad.welfare("isoelastic", alpha=1)
    alpha_other = ad.welfare("isoelastic", alpha=10)

    assert alpha_null == 0.4952045990934922
    assert alpha_inf == 0.005061583846218687
    assert alpha_one == -1.0254557944163005
    assert alpha_other == -2.579772844232791e17


def test_isoelastic_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(
        ad.welfare(method="isoelastic", alpha=0),
        dr2.welfare(method="isoelastic", alpha=0),
    )


def test_isoelastic_replication():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    y = k * ad.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.welfare("isoelastic", alpha=0) == dr2.welfare(
        "isoelastic", alpha=0
    )


def test_isoelastic_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    y = ad.data["x"].tolist()
    y = k * y
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.welfare("isoelastic", alpha=0) == dr2.welfare(
        "isoelastic", alpha=0
    )


# =============================================================================
# TESTS SEN
# =============================================================================
def test_sen_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(ad.welfare.sen(), 0.32568350751486885)


def test_sen_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(ad.welfare("sen"), 0.32568350751486885)


def test_sen_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.welfare("sen")
    method_result = ad.welfare.sen()
    assert call_result == method_result


def test_sen_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(
        ad.welfare(method="sen"), dr2.welfare(method="sen")
    )


# =============================================================================
# TESTS THEILL
# =============================================================================
def test_theill_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.welfare.theill() == 0.35863296524449223


def test_theill_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.welfare("theill") == 0.35863296524449223


def test_theill_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.welfare("theill")
    method_result = ad.welfare.theill()
    assert call_result == method_result


def test_theill_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(
        ad.welfare(method="theill"), dr2.welfare(method="theill")
    )


def test_theill_replication():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    y = k * ad.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.welfare("theill") == dr2.welfare("theill")


def test_theill_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    y = ad.data["x"].tolist()
    y = k * y
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.welfare("theill") == dr2.welfare("theill")


# =============================================================================
# TESTS theilt
# =============================================================================
def test_theilt_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.welfare.theilt() == 0.4036406524522584


def test_theilt_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.welfare("theilt") == 0.4036406524522584


def test_theilt_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.welfare("theilt")
    method_result = ad.welfare.theilt()
    assert call_result == method_result


def test_theilt_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(
        ad.welfare(method="theilt"), dr2.welfare(method="theilt")
    )


def test_theilt_replication():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    y = k * ad.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.welfare("theilt") == dr2.welfare("theilt")


def test_theilt_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    y = ad.data["x"].tolist()
    y = k * y
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.welfare("theilt") == dr2.welfare("theilt")
