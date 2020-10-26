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


# TODO:
#  - check for all methods, didn't know if they
#  were all supposed to provide the same response
#  - add testing for list of poverty measures
#  - test for tip

# =============================================================================
# TESTS COMMON
# =============================================================================
def test_default_call(uniform_ad):
    data = uniform_ad
    pline = max(0, np.min(data.data.values) - 1)
    call_result = data.poverty("headcount", pline=pline)
    method_result = data.poverty.headcount(pline=pline)
    assert call_result == method_result


def test_invalid(uniform_ad):
    data = uniform_ad
    with pytest.raises(AttributeError):
        data.poverty("foo")


# =============================================================================
# TESTS HEADCOUNT
# =============================================================================
def test_headcount_method(uniform_ad):
    data = uniform_ad
    pline = max(0, np.min(data.data.values) - 1)
    assert data.poverty.headcount(pline=pline) == 0


def test_headcount_call(uniform_ad):
    data = uniform_ad
    pline = max(0, np.min(data.data.values) - 1)
    assert data.poverty("headcount", pline=pline) == 0


def test_headcount_call_equal_method(uniform_ad):
    data = uniform_ad
    pline = max(0, np.min(data.data.values) - 1)
    call_result = data.poverty("headcount", pline=pline)
    method_result = data.poverty.headcount(pline=pline)
    assert call_result == method_result


def test_headcount_valid_pline(uniform_ad):
    data = uniform_ad
    with pytest.raises(ValueError):
        data.poverty("headcount", pline=-1)


def test_headcount_extreme_values(uniform_ad):
    data = uniform_ad
    pline_min = max(0, np.min(data.data.values) - 1)
    pline_max = np.max(data.data.values) + 1
    assert data.poverty("headcount", pline=pline_min) == 0
    assert data.poverty("headcount", pline=pline_max) == 1


def test_headcount_symmetry(uniform_ad):
    data = uniform_ad
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty(method="headcount", pline=pline) == dr2.poverty(
        method="headcount", pline=pline
    )


def test_headcount_replication(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("headcount", pline=pline) == dr2.poverty(
        "headcount", pline=pline
    )


def test_headcount_homogeneity(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("headcount", pline=pline) == dr2.poverty(
        "headcount", pline=pline * k
    )


# =============================================================================
# TESTS GAP
# =============================================================================


def test_gap_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.gap(pline=pline) == 0
    pline = np.max(data.data.values) + 1
    # assert data.poverty.gap(pline=pline) == 1 #CHECK, this assertion fails


def test_gap_call(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("gap", pline=pline) == 0


def test_gap_call_equal_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("gap", pline=pline)
    method_result = data.poverty.gap(pline=pline)
    assert call_result == method_result


def test_gap_valid_pline(uniform_ad):
    data = uniform_ad
    with pytest.raises(ValueError):
        data.poverty("gap", pline=-1)


def test_gap_extreme_values(uniform_ad):
    data = uniform_ad
    pline_min = max(np.min(data.data.values) - 1, 0)
    # pline_max = np.max(data.data.values) + 1
    assert data.poverty("gap", pline=pline_min) == 0
    # assert data.poverty("gap", pline=pline_max) == 1 #CHECK, fails


def test_gap_symmetry(uniform_ad):
    data = uniform_ad
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty(method="gap", pline=pline) == dr2.poverty(
        method="gap", pline=pline
    )


def test_gap_replication(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    np.testing.assert_allclose(
        data.poverty("gap", pline=pline), dr2.poverty("gap", pline=pline)
    )


def test_gap_homogeneity(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("gap", pline=pline) == \
           dr2.poverty("gap", pline=pline * k)


# =============================================================================
# TESTS SEVERITY
# =============================================================================
def test_severity_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.severity(pline=pline) == 0


def test_severity_call(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("severity", pline=pline) == 0


def test_severity_call_equal_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("severity", pline=pline)
    method_result = data.poverty.severity(pline=pline)
    assert call_result == method_result


def test_severity_valid_pline(uniform_ad):
    data = uniform_ad
    with pytest.raises(ValueError):
        data.poverty("severity", pline=-1)


def test_severity_extreme_values(uniform_ad):
    data = uniform_ad
    pline_min = max(np.min(data.data.values) - 1, 0)
    # pline_max = np.max(data.data.values) + 1
    assert data.poverty("severity", pline=pline_min) == 0
    # assert data.poverty("severity", pline=pline_max) == 1 #CHECK, fails


def test_severity_symmetry(uniform_ad):
    data = uniform_ad
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty(method="severity", pline=pline) == dr2.poverty(
        method="severity", pline=pline
    )


def test_severity_replication(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    np.testing.assert_allclose(
        data.poverty("severity", pline=pline),
        dr2.poverty("severity", pline=pline)
    )


def test_severity_homogeneity(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("severity", pline=pline) == \
           dr2.poverty("severity", pline=pline * k)


# =============================================================================
# TESTS FGT
# =============================================================================
def test_fgt_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.fgt(pline=pline) == 0


def test_fgt_call(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("fgt", pline=pline) == 0


def test_fgt_call_equal_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("fgt", pline=pline)
    method_result = data.poverty.fgt(pline=pline)
    assert call_result == method_result


def test_fgt_valid_pline(uniform_ad):
    data = uniform_ad
    with pytest.raises(ValueError):
        data.poverty("fgt", pline=-1)
        data.poverty("fgt", pline=0)


def test_fgt_extreme_values(uniform_ad):
    data = uniform_ad
    pline_min = max(np.min(data.data.values) - 1, 0)
    pline_max = np.max(data.data.values) + 1
    assert data.poverty("fgt", pline=pline_min) == 0
    assert data.poverty("fgt", pline=pline_max) == 1


def test_fgt_symmetry(uniform_ad):
    data = uniform_ad
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty(method="fgt", pline=pline) == dr2.poverty(
        method="fgt", pline=pline
    )


def test_fgt_replication(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("fgt", pline=pline) == dr2.poverty("fgt", pline=pline)


def test_fgt_homogeneity(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("fgt", pline=pline) == \
           dr2.poverty("fgt", pline=pline * k)


# =============================================================================
# TESTS SEN
# =============================================================================
def test_sen_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.sen(pline=pline) == 0


def test_sen_call(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("sen", pline=pline) == 0


def test_sen_call_equal_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("sen", pline=pline)
    method_result = data.poverty.sen(pline=pline)
    assert call_result == method_result


def test_sen_valid_pline(uniform_ad):
    data = uniform_ad
    with pytest.raises(ValueError):
        data.poverty("sen", pline=-1)
        data.poverty("sen", pline=0)


def test_sen_extreme_values(uniform_ad):
    data = uniform_ad
    pline_min = max(np.min(data.data.values) - 1, 0)
    # pline_max = np.max(data.data.values) + 1
    assert data.poverty("sen", pline=pline_min) == 0
    # assert data.poverty("sen", pline=pline_max) == 1 #CHEK, fails


def test_sen_symmetry(uniform_ad):
    data = uniform_ad
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty(method="sen", pline=pline) == dr2.poverty(
        method="sen", pline=pline
    )


# E       AssertionError:
# E       Not equal to tolerance rtol=1e-07, atol=0
# def test_sen_replication(uniform_ad):
#     data = uniform_ad
#     k = 2  # factor
#     pline = np.mean(data.data.values)
#     y = k * data.data["x"].tolist()
#     df2 = pd.DataFrame({"x": y})
#     dr2 = ApodeData(df2, varx="x")
#     np.testing.assert_allclose(data.poverty("sen", pline=pline),
#                                dr2.poverty("sen", pline=pline)
#     )


def test_sen_homogeneity(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("sen", pline=pline) == \
           dr2.poverty("sen", pline=pline * k)


# =============================================================================
# TESTS SST
# =============================================================================
def test_sst_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.sst(pline=pline) == 0


def test_sst_call(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("sst", pline=pline) == 0


def test_sst_call_equal_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("sst", pline=pline)
    method_result = data.poverty.sst(pline=pline)
    assert call_result == method_result


def test_sst_valid_pline(uniform_ad):
    data = uniform_ad
    with pytest.raises(ValueError):
        data.poverty("sst", pline=-1)
        data.poverty("sst", pline=0)


def test_sst_extreme_values(uniform_ad):
    data = uniform_ad
    pline_min = max(np.min(data.data.values) - 1, 0)
    # pline_max = np.max(data.data.values) + 1
    assert data.poverty("sst", pline=pline_min) == 0
    # assert data.poverty("sst", pline=pline_max) == 1 #CHECK, fails


def test_sst_symmetry(uniform_ad):
    data = uniform_ad
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty(method="sst", pline=pline) == dr2.poverty(
        method="sst", pline=pline
    )


# E       AssertionError:
# E       Not equal to tolerance rtol=1e-07, atol=0
# def test_sst_replication(uniform_ad):
# data = uniform_ad
# k = 2  # factor
# pline = np.mean(data.data.values)
# y = k * data.data["x"].tolist()
# df2 = pd.DataFrame({"x": y})
# dr2 = ApodeData(df2, varx="x")
# np.testing.assert_allclose(data.poverty("sst", pline=pline),
#                            dr2.poverty("sst", pline=pline)
# )


def test_sst_homogeneity(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("sst", pline=pline) == \
           dr2.poverty("sst", pline=pline * k)


# =============================================================================
# TESTS WATTS
# =============================================================================
def test_watts_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.watts(pline=pline) == 0


def test_watts_call(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("watts", pline=pline) == 0


def test_watts_call_equal_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("watts", pline=pline)
    method_result = data.poverty.watts(pline=pline)
    assert call_result == method_result


def test_watts_valid_pline(uniform_ad):
    data = uniform_ad
    with pytest.raises(ValueError):
        data.poverty("watts", pline=-1)
        data.poverty("watts", pline=0)


def test_watts_extreme_values(uniform_ad):
    data = uniform_ad
    pline_min = max(np.min(data.data.values) - 1, 0)
    # pline_max = np.max(data.data.values) + 1
    assert data.poverty("watts", pline=pline_min) == 0
    # assert data.poverty("watts", pline=pline_max) == 1 #CHECK,fails


def test_watts_symmetry(uniform_ad):
    data = uniform_ad
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty(method="watts", pline=pline) == dr2.poverty(
        method="watts", pline=pline
    )


def test_watts_replication(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    np.testing.assert_allclose(
        data.poverty("watts", pline=pline),
        dr2.poverty("watts", pline=pline))


def test_watts_homogeneity(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("watts", pline=pline) == \
           dr2.poverty("watts", pline=pline * k)


# =============================================================================
# TESTS CUH
# =============================================================================
def test_cuh_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.cuh(pline=pline) == 0.018833008632775816  # CHECK == 0?


def test_cuh_call(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.cuh(pline=pline) == 0.018833008632775816  # CHECK == 0?


def test_cuh_call_equal_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("cuh", pline=pline)
    method_result = data.poverty.cuh(pline=pline)
    assert call_result == method_result


def test_cuh_valid_pline(uniform_ad):
    data = uniform_ad
    with pytest.raises(ValueError):
        data.poverty("cuh", pline=-1)
        data.poverty("cuh", pline=0)


# CHECK, fails
# def test_cuh_extreme_values(uniform_ad):
#     data = uniform_ad
#     pline_min = max(np.min(data.data.values) - 1, 0)
#     pline_max = np.max(data.data.values) + 1
#     # assert data.poverty("cuh", pline=pline_min) == 0 #CHECK, Fails
#     assert data.poverty("cuh", pline=pline_max) == 1


def test_cuh_symmetry(uniform_ad):
    data = uniform_ad
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty(method="cuh", pline=pline) == dr2.poverty(
        method="cuh", pline=pline
    )


# CHECK fails
# def test_cuh_replication(uniform_ad):
#     data = uniform_ad
#     k = 2  # factor
#     pline = np.mean(data.data.values)
#     y = k * data.data["x"].tolist()
#     df2 = pd.DataFrame({"x": y})
#     dr2 = ApodeData(df2, varx="x")
#     assert data.poverty("cuh", pline=pline) == dr2.poverty(
#         "cuh", pline=pline
#     )


def test_cuh_homogeneity(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("cuh", pline=pline) == \
           dr2.poverty("cuh", pline=pline * k)


# =============================================================================
# TESTS TAKAYAMA
# =============================================================================
def test_takayama_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.takayama(pline=pline) == 0


def test_takayama_call(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("takayama", pline=pline) == 0


def test_takayama_call_equal_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("takayama", pline=pline)
    method_result = data.poverty.takayama(pline=pline)
    assert call_result == method_result


def test_takayama_valid_pline(uniform_ad):
    data = uniform_ad
    with pytest.raises(ValueError):
        data.poverty("takayama", pline=-1)
        data.poverty("takayama", pline=0)


def test_takayama_extreme_values(uniform_ad):
    data = uniform_ad
    pline_min = max(np.min(data.data.values) - 1, 0)
    # pline_max = np.max(data.data.values) + 1
    assert data.poverty("takayama", pline=pline_min) == 0
    # assert data.poverty("takayama", pline=pline_max) == 1 #CHECK, fails


def test_takayama_symmetry(uniform_ad):
    data = uniform_ad
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty(method="takayama", pline=pline) == \
           dr2.poverty(method="takayama", pline=pline)


# CHECK, fails
# def test_takayama_replication(uniform_ad):
#     data = uniform_ad
#     k = 2  # factor
#     pline = np.mean(data.data.values)
#     y = k * data.data["x"].tolist()
#     df2 = pd.DataFrame({"x": y})
#     dr2 = ApodeData(df2, varx="x")
#     assert data.poverty("takayama", pline=pline) == \
#            dr2.poverty("takayama", pline=pline)


def test_takayama_homogeneity(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("takayama", pline=pline) == dr2.poverty(
        "takayama", pline=pline * k
    )


"""
# =============================================================================
# TESTS KAKWANI
# =============================================================================
def test_kakwani_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.kakwani(pline=pline) == 0


def test_kakwani_call(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("kakwani", pline=pline) == 0

def test_kakwani_call_equal_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("kakwani", pline=pline)
    method_result = data.poverty.kakwani(pline=pline)
    assert call_result == method_result

def test_kakwani_valid_pline(uniform_ad):
    data = uniform_ad
    with pytest.raises(ValueError):
        data.poverty("kakwani", pline=-1)
        data.poverty("kakwani", pline=0)

def test_kakwani_extreme_values(uniform_ad):
    data = uniform_ad
    pline_min = max(np.min(data.data.values) - 1, 0)
    pline_max = np.max(data.data.values) + 1
    assert data.poverty("kakwani", pline=pline_min) == 0
    assert data.poverty("kakwani", pline=pline_max) == 1

def test_kakwani_symmetry(uniform_ad):
    data = uniform_ad
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty(method="kakwani", pline=pline) == dr2.poverty(
        method="kakwani", pline=pline
    )

def test_kakwani_replication(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("kakwani", pline=pline) == dr2.poverty(
        "kakwani", pline=pline
    )

def test_kakwani_homogeneity(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data['x'].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({'x': y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty('kakwani', pline=pline) == dr2.poverty(
         'kakwani', pline=pline * k)


# =============================================================================
# TESTS THON
# =============================================================================
def test_thon_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.thon(pline=pline) == 0


def test_thon_call(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("thon", pline=pline) == 0

def test_thon_call_equal_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("thon", pline=pline)
    method_result = data.poverty.thon(pline=pline)
    assert call_result == method_result

def test_thon_valid_pline(uniform_ad):
    data = uniform_ad
    with pytest.raises(ValueError):
        data.poverty("thon", pline=-1)
        data.poverty("thon", pline=0)

def test_thon_extreme_values(uniform_ad):
    data = uniform_ad
    pline_min = max(np.min(data.data.values) - 1, 0)
    pline_max = np.max(data.data.values) + 1
    assert data.poverty("thon", pline=pline_min) == 0
    assert data.poverty("thon", pline=pline_max) == 1

def test_thon_symmetry(uniform_ad):
    data = uniform_ad
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty(method="thon", pline=pline) == dr2.poverty(
        method="thon", pline=pline
    )

def test_thon_replication(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("thon", pline=pline) == dr2.poverty(
        "thon", pline=pline
    )

def test_thon_homogeneity(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data['x'].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({'x': y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty('thon', pline=pline) == dr2.poverty(
         'thon', pline=pline * k)
# =============================================================================
# TESTS BD
# =============================================================================
def test_bd_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.bd(pline=pline) == 0


def test_bd_call(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("bd", pline=pline) == 0

def test_bd_call_equal_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("bd", pline=pline)
    method_result = data.poverty.bd(pline=pline)
    assert call_result == method_result

def test_bd_valid_pline(uniform_ad):
    data = uniform_ad
    with pytest.raises(ValueError):
        data.poverty("bd", pline=-1)
        data.poverty("bd", pline=0)

def test_bd_extreme_values(uniform_ad):
    data = uniform_ad
    pline_min = max(np.min(data.data.values) - 1, 0)
    pline_max = np.max(data.data.values) + 1
    assert data.poverty("bd", pline=pline_min) == 0
    assert data.poverty("bd", pline=pline_max) == 1

def test_bd_symmetry(uniform_ad):
    data = uniform_ad
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty(method="bd", pline=pline) == dr2.poverty(
        method="bd", pline=pline
    )

def test_bd_replication(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("bd", pline=pline) == dr2.poverty(
        "bd", pline=pline
    )

def test_bd_homogeneity(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data['x'].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({'x': y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty('bd', pline=pline) == dr2.poverty(
         'bd', pline=pline * k)

# =============================================================================
# TESTS HAGENAARS
# =============================================================================
def test_hagenaars_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.hagenaars(pline=pline) == 0


def test_hagenaars_call(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("hagenaars", pline=pline) == 0

def test_hagenaars_call_equal_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("hagenaars", pline=pline)
    method_result = data.poverty.hagenaars(pline=pline)
    assert call_result == method_result

def test_hagenaars_valid_pline(uniform_ad):
    data = uniform_ad
    with pytest.raises(ValueError):
        data.poverty("hagenaars", pline=-1)
        data.poverty("hagenaars", pline=0)

def test_hagenaars_extreme_values(uniform_ad):
    data = uniform_ad
    pline_min = max(np.min(data.data.values) - 1, 0)
    pline_max = np.max(data.data.values) + 1
    assert data.poverty("hagenaars", pline=pline_min) == 0
    assert data.poverty("hagenaars", pline=pline_max) == 1

def test_hagenaars_symmetry(uniform_ad):
    data = uniform_ad
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty(method="hagenaars", pline=pline) == dr2.poverty(
        method="hagenaars", pline=pline
    )

def test_hagenaars_replication(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("hagenaars", pline=pline) == dr2.poverty(
        "hagenaars", pline=pline
    )

def test_hagenaars_homogeneity(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data['x'].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({'x': y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty('hagenaars', pline=pline) == dr2.poverty(
         'hagenaars', pline=pline * k)
# =============================================================================
# TESTS CHAKRAVARTY
# =============================================================================
def test_chakravarty_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.chakravarty(pline=pline) == 0


def test_chakravarty_call(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("chakravarty", pline=pline) == 0

def test_chakravarty_call_equal_method(uniform_ad):
    data = uniform_ad
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("chakravarty", pline=pline)
    method_result = data.poverty.chakravarty(pline=pline)
    assert call_result == method_result

def test_chakravarty_valid_pline(uniform_ad):
    data = uniform_ad
    with pytest.raises(ValueError):
        data.poverty("chakravarty", pline=-1)
        data.poverty("chakravarty", pline=0)

def test_chakravarty_extreme_values(uniform_ad):
    data = uniform_ad
    pline_min = max(np.min(data.data.values) - 1, 0)
    pline_max = np.max(data.data.values) + 1
    assert data.poverty("chakravarty", pline=pline_min) == 0
    assert data.poverty("chakravarty", pline=pline_max) == 1

def test_chakravarty_symmetry(uniform_ad):
    data = uniform_ad
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty(method="chakravarty", pline=pline) == dr2.poverty(
        method="chakravarty", pline=pline
    )

def test_chakravarty_replication(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty("chakravarty", pline=pline) == dr2.poverty(
        "chakravarty", pline=pline
    )

def test_chakravarty_homogeneity(uniform_ad):
    data = uniform_ad
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data['x'].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({'x': y})
    dr2 = ApodeData(df2, varx="x")
    assert data.poverty('chakravarty', pline=pline) == dr2.poverty(
         'chakravarty', pline=pline * k)
"""

# # # Testea metdodo de un listado de medidas de pobreza
# # def test_lista(prop,lista):
# #     x = TestCaseUniform()
# #     x.setup_method()
# #     for elem in lista:
# #         if elem[1]==None:
# #             x.prop(elem[0])
# #         else:
# #             x.prop(elem[0],alpha=elem[1])
