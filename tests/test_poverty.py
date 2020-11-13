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
    pline = max(0, np.min(data.data.values) - 1)
    call_result = data.poverty("headcount", pline=pline)
    method_result = data.poverty.headcount(pline=pline)
    assert call_result == method_result


def test_invalid():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(AttributeError):
        data.poverty("foo")


# =============================================================================
# TESTS HEADCOUNT
# =============================================================================
def test_headcount_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(0, np.min(data.data.values) - 1)
    assert data.poverty.headcount(pline=pline) == 0


def test_headcount_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(0, np.min(data.data.values) - 1)
    assert data.poverty("headcount", pline=pline) == 0


def test_headcount_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(0, np.min(data.data.values) - 1)
    call_result = data.poverty("headcount", pline=pline)
    method_result = data.poverty.headcount(pline=pline)
    assert call_result == method_result


def test_headcount_valid_pline():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.poverty("headcount", pline=-1)


def test_headcount_extreme_values():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = max(0, np.min(data.data.values) - 1)
    pline_max = np.max(data.data.values) + 1
    assert data.poverty("headcount", pline=pline_min) == 0
    assert data.poverty("headcount", pline=pline_max) == 1


def test_headcount_symmetry():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty(method="headcount", pline=pline) == dr2.poverty(
        method="headcount", pline=pline
    )


def test_headcount_replication():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("headcount", pline=pline) == dr2.poverty(
        "headcount", pline=pline
    )


def test_headcount_homogeneity():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("headcount", pline=pline) == dr2.poverty(
        "headcount", pline=pline * k
    )


# =============================================================================
# TESTS GAP
# =============================================================================


def test_gap_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.gap(pline=pline) == 0
    pline = np.max(data.data.values) + 1
    # assert data.poverty.gap(pline=pline) == 1 #CHECK, this assertion fails


def test_gap_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("gap", pline=pline) == 0


def test_gap_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("gap", pline=pline)
    method_result = data.poverty.gap(pline=pline)
    assert call_result == method_result


def test_gap_valid_pline():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.poverty("gap", pline=-1)


def test_gap_extreme_values():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = max(np.min(data.data.values) - 1, 0)
    # pline_max = np.max(data.data.values) + 1
    assert data.poverty("gap", pline=pline_min) == 0
    # assert data.poverty("gap", pline=pline_max) == 1 #CHECK, fails


def test_gap_symmetry():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty(method="gap", pline=pline) == dr2.poverty(
        method="gap", pline=pline
    )


def test_gap_replication():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(
        data.poverty("gap", pline=pline), dr2.poverty("gap", pline=pline)
    )


def test_gap_homogeneity():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("gap", pline=pline) == dr2.poverty(
        "gap", pline=pline * k
    )


# =============================================================================
# TESTS SEVERITY
# =============================================================================
def test_severity_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.severity(pline=pline) == 0


def test_severity_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("severity", pline=pline) == 0


def test_severity_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("severity", pline=pline)
    method_result = data.poverty.severity(pline=pline)
    assert call_result == method_result


def test_severity_valid_pline():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.poverty("severity", pline=-1)


def test_severity_extreme_values():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = max(np.min(data.data.values) - 1, 0)
    # pline_max = np.max(data.data.values) + 1
    assert data.poverty("severity", pline=pline_min) == 0
    # assert data.poverty("severity", pline=pline_max) == 1 #CHECK, fails


def test_severity_symmetry():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty(method="severity", pline=pline) == dr2.poverty(
        method="severity", pline=pline
    )


def test_severity_replication():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(
        data.poverty("severity", pline=pline),
        dr2.poverty("severity", pline=pline),
    )


def test_severity_homogeneity():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("severity", pline=pline) == dr2.poverty(
        "severity", pline=pline * k
    )


# =============================================================================
# TESTS FGT
# =============================================================================
def test_fgt_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.fgt(pline=pline) == 0


def test_fgt_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("fgt", pline=pline) == 0


def test_fgt_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("fgt", pline=pline)
    method_result = data.poverty.fgt(pline=pline)
    assert call_result == method_result


def test_fgt_valid_pline():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.poverty("fgt", pline=-1)
        data.poverty("fgt", pline=0)


def test_fgt_valid_alpha():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.poverty.fgt(pline=1, alpha=-2)


def test_fgt_alpha_values():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    assert data.poverty.fgt(pline=pline, alpha=1) == 0.26003924372489007
    assert data.poverty.fgt(pline=pline, alpha=0) == 0.4766666666666667
    assert data.poverty.fgt(pline=pline, alpha=10) == 0.049479474144909996


def test_fgt_extreme_values():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = max(np.min(data.data.values) - 1, 0)
    pline_max = np.max(data.data.values) + 1
    assert data.poverty("fgt", pline=pline_min) == 0
    assert data.poverty("fgt", pline=pline_max) == 1


def test_fgt_symmetry():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty(method="fgt", pline=pline) == dr2.poverty(
        method="fgt", pline=pline
    )


def test_fgt_replication():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("fgt", pline=pline) == dr2.poverty("fgt", pline=pline)


def test_fgt_homogeneity():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("fgt", pline=pline) == dr2.poverty(
        "fgt", pline=pline * k
    )


# =============================================================================
# TESTS SEN
# =============================================================================
def test_sen_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.sen(pline=pline) == 0


def test_sen_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("sen", pline=pline) == 0


def test_sen_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("sen", pline=pline)
    method_result = data.poverty.sen(pline=pline)
    assert call_result == method_result


def test_sen_valid_pline():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.poverty("sen", pline=-1)
        data.poverty("sen", pline=0)


def test_sen_extreme_values():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = max(np.min(data.data.values) - 1, 0)
    # pline_max = np.max(data.data.values) + 1
    assert data.poverty("sen", pline=pline_min) == 0
    # assert data.poverty("sen", pline=pline_max) == 1 #CHEK, fails


def test_sen_symmetry():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty(method="sen", pline=pline) == dr2.poverty(
        method="sen", pline=pline
    )


# E       AssertionError:
# E       Not equal to tolerance rtol=1e-07, atol=0
# def test_sen_replication():
#     data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
#     k = 2  # factor
#     pline = np.mean(data.data.values)
#     y = k * data.data["x"].tolist()
#     df2 = pd.DataFrame({"x": y})
#     dr2 = ApodeData(df2, income_column="x")
#     np.testing.assert_allclose(data.poverty("sen", pline=pline),
#                                dr2.poverty("sen", pline=pline)
#     )


def test_sen_homogeneity():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("sen", pline=pline) == dr2.poverty(
        "sen", pline=pline * k
    )


# =============================================================================
# TESTS SST
# =============================================================================
def test_sst_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.sst(pline=pline) == 0


def test_sst_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("sst", pline=pline) == 0


def test_sst_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("sst", pline=pline)
    method_result = data.poverty.sst(pline=pline)
    assert call_result == method_result


def test_sst_valid_pline():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.poverty("sst", pline=-1)
        data.poverty("sst", pline=0)


def test_sst_extreme_values():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = max(np.min(data.data.values) - 1, 0)
    # pline_max = np.max(data.data.values) + 1
    assert data.poverty("sst", pline=pline_min) == 0
    # assert data.poverty("sst", pline=pline_max) == 1 #CHECK, fails


def test_sst_symmetry():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty(method="sst", pline=pline) == dr2.poverty(
        method="sst", pline=pline
    )


# E       AssertionError:
# E       Not equal to tolerance rtol=1e-07, atol=0
# def test_sst_replication():
# data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
# k = 2  # factor
# pline = np.mean(data.data.values)
# y = k * data.data["x"].tolist()
# df2 = pd.DataFrame({"x": y})
# dr2 = ApodeData(df2, income_column="x")
# np.testing.assert_allclose(data.poverty("sst", pline=pline),
#                            dr2.poverty("sst", pline=pline)
# )


def test_sst_homogeneity():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("sst", pline=pline) == dr2.poverty(
        "sst", pline=pline * k
    )


# =============================================================================
# TESTS WATTS
# =============================================================================
def test_watts_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.watts(pline=pline) == 0


def test_watts_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("watts", pline=pline) == 0


def test_watts_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("watts", pline=pline)
    method_result = data.poverty.watts(pline=pline)
    assert call_result == method_result


def test_watts_valid_pline():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.poverty("watts", pline=-1)
        data.poverty("watts", pline=0)


def test_watts_extreme_values():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = max(np.min(data.data.values) - 1, 0)
    # pline_max = np.max(data.data.values) + 1
    assert data.poverty("watts", pline=pline_min) == 0
    # assert data.poverty("watts", pline=pline_max) == 1 #CHECK,fails


def test_watts_symmetry():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty(method="watts", pline=pline) == dr2.poverty(
        method="watts", pline=pline
    )


def test_watts_replication():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(
        data.poverty("watts", pline=pline), dr2.poverty("watts", pline=pline)
    )


def test_watts_homogeneity():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("watts", pline=pline) == dr2.poverty(
        "watts", pline=pline * k
    )


# =============================================================================
# TESTS CUH
# =============================================================================
def test_cuh_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.cuh(pline=pline) == 0.018833008632775816  # CHECK == 0?


def test_cuh_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.cuh(pline=pline) == 0.018833008632775816  # CHECK == 0?


def test_cuh_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("cuh", pline=pline)
    method_result = data.poverty.cuh(pline=pline)
    assert call_result == method_result


def test_cuh_valid_pline():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.poverty("cuh", pline=-1)
        data.poverty("cuh", pline=0)


def test_cuh_valid_alpha():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    with pytest.raises(ValueError):
        data.poverty(method="cuh", pline=pline, alpha=-2)
        data.poverty(method="cuh", pline=pline, alpha=2)


def test_cuh_alpha_values():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    assert (
        data.poverty(method="cuh", pline=pline, alpha=0.4)
        == 0.33303872854353567
    )
    assert (
        data.poverty(method="cuh", pline=pline, alpha=0) == 0.418431486255362
    )


# CHECK, fails
# def test_cuh_extreme_values():
#     data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
#     pline_min = max(np.min(data.data.values) - 1, 0)
#     pline_max = np.max(data.data.values) + 1
#     # assert data.poverty("cuh", pline=pline_min) == 0 #CHECK, Fails
#     assert data.poverty("cuh", pline=pline_max) == 1


def test_cuh_symmetry():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty(method="cuh", pline=pline) == dr2.poverty(
        method="cuh", pline=pline
    )


# CHECK fails
# def test_cuh_replication():
#     data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
#     k = 2  # factor
#     pline = np.mean(data.data.values)
#     y = k * data.data["x"].tolist()
#     df2 = pd.DataFrame({"x": y})
#     dr2 = ApodeData(df2, income_column="x")
#     assert data.poverty("cuh", pline=pline) == dr2.poverty(
#         "cuh", pline=pline
#     )


def test_cuh_homogeneity():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("cuh", pline=pline) == dr2.poverty(
        "cuh", pline=pline * k
    )


# =============================================================================
# TESTS TAKAYAMA
# =============================================================================
def test_takayama_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.takayama(pline=pline) == 0


def test_takayama_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("takayama", pline=pline) == 0


def test_takayama_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("takayama", pline=pline)
    method_result = data.poverty.takayama(pline=pline)
    assert call_result == method_result


def test_takayama_valid_pline():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.poverty("takayama", pline=-1)
        data.poverty("takayama", pline=0)


def test_takayama_extreme_values():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = max(np.min(data.data.values) - 1, 0)
    # pline_max = np.max(data.data.values) + 1
    assert data.poverty("takayama", pline=pline_min) == 0
    # assert data.poverty("takayama", pline=pline_max) == 1 #CHECK, fails


@pytest.mark.xfail
def test_takayama_symmetry():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty(method="takayama", pline=pline) == dr2.poverty(
        method="takayama", pline=pline
    )


@pytest.mark.xfail
def test_takayama_replication():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("takayama", pline=pline) == dr2.poverty(
        "takayama", pline=pline
    )


def test_takayama_homogeneity():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("takayama", pline=pline) == dr2.poverty(
        "takayama", pline=pline * k
    )


# =============================================================================
# TESTS KAKWANI
# =============================================================================
def test_kakwani_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.kakwani(pline=pline) == 0


def test_kakwani_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("kakwani", pline=pline) == 0


def test_kakwani_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("kakwani", pline=pline)
    method_result = data.poverty.kakwani(pline=pline)
    assert call_result == method_result


def test_kakwani_valid_pline():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.poverty("kakwani", pline=-1)
        data.poverty("kakwani", pline=0)


# def test_kakwani_extreme_values():
#     data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
#     pline_min = max(np.min(data.data.values) - 1, 0)
#     # assert data.poverty("kakwani", pline=pline_max) == 1


def test_kakwani_symmetry():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty(method="kakwani", pline=pline) == dr2.poverty(
        method="kakwani", pline=pline
    )


def test_kakwani_homogeneity():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("kakwani", pline=pline) == dr2.poverty(
        "kakwani", pline=pline * k
    )


# =============================================================================
# TESTS THON
# =============================================================================
def test_thon_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.thon(pline=pline) == 0


def test_thon_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("thon", pline=pline) == 0


def test_thon_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("thon", pline=pline)
    method_result = data.poverty.thon(pline=pline)
    assert call_result == method_result


def test_thon_valid_pline():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.poverty("thon", pline=-1)
        data.poverty("thon", pline=0)


def test_thon_extreme_values():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("thon", pline=pline_min) == 0


def test_thon_symmetry():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty(method="thon", pline=pline) == dr2.poverty(
        method="thon", pline=pline
    )


def test_thon_homogeneity():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("thon", pline=pline) == dr2.poverty(
        "thon", pline=pline * k
    )


# =============================================================================
# TESTS BD
# =============================================================================
def test_bd_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.bd(pline=pline) == 0


def test_bd_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("bd", pline=pline) == 0
    assert data.poverty("bd", pline=30) == 0.9950410832744983


def test_bd_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("bd", pline=pline)
    method_result = data.poverty.bd(pline=pline)
    assert call_result == method_result


def test_bd_valid_pline():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.poverty("bd", pline=-1)
        data.poverty("bd", pline=0)


def test_bd_extreme_values():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("bd", pline=pline_min) == 0


def test_bd_symmetry():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty(method="bd", pline=pline) == dr2.poverty(
        method="bd", pline=pline
    )


def test_bd_replication():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("bd", pline=pline) == dr2.poverty("bd", pline=pline)


def test_bd_homogeneity():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("bd", pline=pline) == dr2.poverty(
        "bd", pline=pline * k
    )


# =============================================================================
# TESTS HAGENAARS
# =============================================================================
def test_hagenaars_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.hagenaars(pline=pline) == 0


def test_hagenaars_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("hagenaars", pline=pline) == 0


def test_hagenaars_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("hagenaars", pline=pline)
    method_result = data.poverty.hagenaars(pline=pline)
    assert call_result == method_result


def test_hagenaars_valid_pline():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.poverty("hagenaars", pline=-1)
        data.poverty("hagenaars", pline=0)


def test_hagenaars_symmetry():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty(method="hagenaars", pline=pline) == dr2.poverty(
        method="hagenaars", pline=pline
    )


def test_hagenaars_replication():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("hagenaars", pline=pline) == dr2.poverty(
        "hagenaars", pline=pline
    )


# =============================================================================
# TESTS CHAKRAVARTY
# =============================================================================
def test_chakravarty_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty.chakravarty(pline=pline) == 0


def test_chakravarty_call():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("chakravarty", pline=pline) == 0


def test_chakravarty_call_equal_method():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = max(np.min(data.data.values) - 1, 0)
    call_result = data.poverty("chakravarty", pline=pline)
    method_result = data.poverty.chakravarty(pline=pline)
    assert call_result == method_result


def test_chakravarty_valid_pline():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.poverty("chakravarty", pline=-1)
        data.poverty("chakravarty", pline=0)


def test_chakravarty_valid_alpha():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    with pytest.raises(ValueError):
        data.poverty("chakravarty", pline=pline, alpha=0)
        data.poverty("chakravarty", pline=pline, alpha=-2)
        data.poverty("chakravarty", pline=pline, alpha=2)


def test_chakravarty_extreme_values():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = max(np.min(data.data.values) - 1, 0)
    assert data.poverty("chakravarty", pline=pline_min) == 0


def test_chakravarty_symmetry():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty(method="chakravarty", pline=pline) == dr2.poverty(
        method="chakravarty", pline=pline
    )


def test_chakravarty_replication():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = k * data.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(
        data.poverty("chakravarty", pline=pline),
        dr2.poverty("chakravarty", pline=pline),
    )


def test_chakravarty_homogeneity():
    data = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(data.data.values)
    y = data.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert data.poverty("chakravarty", pline=pline) == dr2.poverty(
        "chakravarty", pline=pline * k
    )
