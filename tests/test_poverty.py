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
    pline = 0.5 * np.median(ad.data.values)
    call_result = ad.poverty("headcount", pline=pline)
    method_result = ad.poverty.headcount(pline=pline)
    assert call_result == method_result


def test_invalid():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(AttributeError):
        ad.poverty("foo")


def test_get_pline_none():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    # pline is None
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty.headcount(pline=None) == ad.poverty.headcount(
        pline=pline
    )


def test_get_pline_factor():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    # factor < 0:
    with pytest.raises(ValueError):
        ad.poverty.hagenaars(pline=pline, factor=-3)
    with pytest.raises(ValueError):
        ad.poverty.chakravarty(pline=pline, factor=-3)
    with pytest.raises(ValueError):
        ad.poverty.hagenaars(pline=None, factor=-3)
    with pytest.raises(ValueError):
        ad.poverty.chakravarty(pline=None, factor=-3)


def test_get_pline_median():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    factor = 0.3
    pline = factor * np.median(ad.data.values)
    assert ad.poverty.headcount(
        pline="median", factor=factor
    ) == ad.poverty.headcount(pline=pline)


def test_get_pline_mean():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    factor = 0.3
    pline = factor * np.mean(ad.data.values)
    assert ad.poverty.headcount(
        pline="mean", factor=factor
    ) == ad.poverty.headcount(pline=pline)


def test_get_pline_quantile():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    # pline = "quantile"
    q = 0.3
    factor = 0.3
    pline = factor * np.quantile(ad.data.values, q)
    assert ad.poverty.chakravarty(
        pline="quantile", factor=factor, q=q
    ) == ad.poverty.chakravarty(pline=pline)
    assert ad.poverty.hagenaars(
        pline="quantile", factor=factor, q=q
    ) == ad.poverty.hagenaars(pline=pline)
    # pline = "quantile", q out of range
    with pytest.raises(ValueError):
        ad.poverty.hagenaars(pline="quantile", q=1.2)
    with pytest.raises(ValueError):
        ad.poverty.hagenaars(pline="quantile", q=-0.2)


# =============================================================================
# TESTS HEADCOUNT
# =============================================================================
def test_headcount_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty.headcount(pline=pline) == 0.27


def test_headcount_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty("headcount", pline=pline) == 0.27


def test_headcount_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    call_result = ad.poverty("headcount", pline=pline)
    method_result = ad.poverty.headcount(pline=pline)
    assert call_result == method_result


def test_headcount_valid_pline():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.poverty("headcount", pline=-1)
    with pytest.raises(ValueError):
        ad.poverty("headcount", pline=0)


def test_headcount_extreme_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=100, nbin=None)
    pline_min = np.min(ad.data.values) / 2
    pline_max = np.max(ad.data.values) + 1
    assert ad.poverty("headcount", pline=pline_min) == 0
    assert ad.poverty("headcount", pline=pline_max) == 1


def test_headcount_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty(method="headcount", pline=pline) == dr2.poverty(
        method="headcount", pline=pline
    )


def test_headcount_replication():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = k * ad.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("headcount", pline=pline) == dr2.poverty(
        "headcount", pline=pline
    )


def test_headcount_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("headcount", pline=pline) == dr2.poverty(
        "headcount", pline=pline * k
    )


# =============================================================================
# TESTS GAP
# =============================================================================


def test_gap_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty.gap(pline=pline) == 0.13715275200855706


def test_gap_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty("gap", pline=pline) == 0.13715275200855706


def test_gap_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    call_result = ad.poverty("gap", pline=pline)
    method_result = ad.poverty.gap(pline=pline)
    assert call_result == method_result


def test_gap_valid_pline():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.poverty("gap", pline=-1)


def test_gap_extreme_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = np.min(ad.data.values) / 2
    pline_max = np.max(ad.data.values) + 1
    assert ad.poverty("gap", pline=pline_min) == 0
    assert ad.poverty("gap", pline=pline_max) <= 1


def test_gap_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty(method="gap", pline=pline) == dr2.poverty(
        method="gap", pline=pline
    )


def test_gap_replication():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = k * ad.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(
        ad.poverty("gap", pline=pline), dr2.poverty("gap", pline=pline)
    )


def test_gap_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("gap", pline=pline) == dr2.poverty(
        "gap", pline=pline * k
    )


# =============================================================================
# TESTS SEVERITY
# =============================================================================
def test_severity_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty.severity(pline=pline) == 0.0925444945807559


def test_severity_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty("severity", pline=pline) == 0.0925444945807559


def test_severity_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    call_result = ad.poverty("severity", pline=pline)
    method_result = ad.poverty.severity(pline=pline)
    assert call_result == method_result


def test_severity_valid_pline():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.poverty("severity", pline=-1)


def test_severity_extreme_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = np.min(ad.data.values) / 2
    pline_max = np.max(ad.data.values) + 1
    assert ad.poverty("severity", pline=pline_min) == 0
    assert ad.poverty("severity", pline=pline_max) <= 1


def test_severity_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty(method="severity", pline=pline) == dr2.poverty(
        method="severity", pline=pline
    )


def test_severity_replication():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = k * ad.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(
        ad.poverty("severity", pline=pline),
        dr2.poverty("severity", pline=pline),
    )


def test_severity_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("severity", pline=pline) == dr2.poverty(
        "severity", pline=pline * k
    )


# =============================================================================
# TESTS FGT
# =============================================================================
def test_fgt_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty.fgt(pline=pline) == 0.27


def test_fgt_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty("fgt", pline=pline) == 0.27


def test_fgt_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    call_result = ad.poverty("fgt", pline=pline)
    method_result = ad.poverty.fgt(pline=pline)
    assert call_result == method_result


def test_fgt_valid_pline():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.poverty("fgt", pline=-1)
    with pytest.raises(ValueError):
        ad.poverty("fgt", pline=0)


def test_fgt_valid_alpha():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.poverty.fgt(pline=1, alpha=-2)


def test_fgt_alpha_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    assert ad.poverty.fgt(pline=pline, alpha=1) == 0.26003924372489007
    assert ad.poverty.fgt(pline=pline, alpha=0) == 0.4766666666666667
    assert ad.poverty.fgt(pline=pline, alpha=10) == 0.049479474144909996


def test_fgt_extreme_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = np.min(ad.data.values) / 2
    pline_max = np.max(ad.data.values) + 1
    assert ad.poverty("fgt", pline=pline_min) == 0
    assert ad.poverty("fgt", pline=pline_max) <= 1


def test_fgt_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty(method="fgt", pline=pline) == dr2.poverty(
        method="fgt", pline=pline
    )


def test_fgt_replication():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = k * ad.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("fgt", pline=pline) == dr2.poverty("fgt", pline=pline)


def test_fgt_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("fgt", pline=pline) == dr2.poverty(
        "fgt", pline=pline * k
    )


# =============================================================================
# TESTS SEN
# =============================================================================
def test_sen_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty.sen(pline=pline) == 0.1826297337125855


def test_sen_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty("sen", pline=pline) == 0.1826297337125855


def test_sen_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    call_result = ad.poverty("sen", pline=pline)
    method_result = ad.poverty.sen(pline=pline)
    assert call_result == method_result


def test_sen_valid_pline():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.poverty("sen", pline=-1)
    with pytest.raises(ValueError):
        ad.poverty("sen", pline=0)


def test_sen_extreme_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = np.min(ad.data.values) / 2
    pline_max = np.max(ad.data.values) + 1
    assert ad.poverty("sen", pline=pline_min) == 0
    assert ad.poverty("sen", pline=pline_max) <= 1


def test_sen_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty(method="sen", pline=pline) == dr2.poverty(
        method="sen", pline=pline
    )


def test_sen_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("sen", pline=pline) == dr2.poverty(
        "sen", pline=pline * k
    )


# =============================================================================
# TESTS SST
# =============================================================================
def test_sst_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty.sst(pline=pline) == 0.24950968072455512


def test_sst_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty("sst", pline=pline) == 0.24950968072455512


def test_sst_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    call_result = ad.poverty("sst", pline=pline)
    method_result = ad.poverty.sst(pline=pline)
    assert call_result == method_result


def test_sst_valid_pline():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.poverty("sst", pline=-1)
    with pytest.raises(ValueError):
        ad.poverty("sst", pline=0)


# @pytest.mark.xfail
def test_sst_extreme_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = np.min(ad.data.values) / 2
    pline_max = np.max(ad.data.values) + 1
    assert ad.poverty("sst", pline=pline_min) == 0
    assert ad.poverty("sst", pline=pline_max) <= 1  # CHECK, fails


def test_sst_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty(method="sst", pline=pline) == dr2.poverty(
        method="sst", pline=pline
    )


def test_sst_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("sst", pline=pline) == dr2.poverty(
        "sst", pline=pline * k
    )


# =============================================================================
# TESTS WATTS
# =============================================================================
def test_watts_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty.watts(pline=pline) == 0.2724322042654472


def test_watts_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty("watts", pline=pline) == 0.2724322042654472


def test_watts_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    call_result = ad.poverty("watts", pline=pline)
    method_result = ad.poverty.watts(pline=pline)
    assert call_result == method_result


def test_watts_valid_pline():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.poverty("watts", pline=-1)
    with pytest.raises(ValueError):
        ad.poverty("watts", pline=0)


def test_watts_extreme_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = np.min(ad.data.values) / 2
    assert ad.poverty("watts", pline=pline_min) == 0


def test_watts_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty(method="watts", pline=pline) == dr2.poverty(
        method="watts", pline=pline
    )


def test_watts_replication():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = k * ad.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(
        ad.poverty("watts", pline=pline), dr2.poverty("watts", pline=pline)
    )


def test_watts_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("watts", pline=pline) == dr2.poverty(
        "watts", pline=pline * k
    )


# =============================================================================
# TESTS CUH
# =============================================================================
def test_cuh_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty.cuh(pline=pline) == 0.18341653809400216


def test_cuh_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty.cuh(pline=pline) == 0.18341653809400216


def test_cuh_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    call_result = ad.poverty("cuh", pline=pline)
    method_result = ad.poverty.cuh(pline=pline)
    assert call_result == method_result


def test_cuh_valid_pline():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.poverty("cuh", pline=-1)
    with pytest.raises(ValueError):
        ad.poverty("cuh", pline=0)


def test_cuh_valid_alpha():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    with pytest.raises(ValueError):
        ad.poverty(method="cuh", pline=pline, alpha=-2)
    with pytest.raises(ValueError):
        ad.poverty(method="cuh", pline=pline, alpha=2)


def test_cuh_alpha_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    assert (
        ad.poverty(method="cuh", pline=pline, alpha=0.4) == 0.3739168025918481
    )
    assert (
        ad.poverty(method="cuh", pline=pline, alpha=0) == 0.14377616581364483
    )


def test_cuh_extreme_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=100, nbin=None)
    pline_min = np.min(ad.data.values) / 2
    pline_max = np.max(ad.data.values) + 1
    assert ad.poverty("cuh", pline=pline_min) == 0  # CHECK, Fails
    assert ad.poverty("cuh", pline=pline_max) <= 1


def test_cuh_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty(method="cuh", pline=pline) == dr2.poverty(
        method="cuh", pline=pline
    )


def test_cuh_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("cuh", pline=pline) == dr2.poverty(
        "cuh", pline=pline * k
    )


# =============================================================================
# TESTS TAKAYAMA
# =============================================================================
def test_takayama_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    #   assert ad.poverty.takayama(pline=pline) == 0.13021647687646376
    np.testing.assert_allclose(
        ad.poverty.takayama(pline=pline),
        0.13021647687646376,
    )


def test_takayama_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    #   assert ad.poverty("takayama", pline=pline) == 0.13021647687646376
    np.testing.assert_allclose(
        ad.poverty("takayama", pline=pline),
        0.13021647687646376,
    )


def test_takayama_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    call_result = ad.poverty("takayama", pline=pline)
    method_result = ad.poverty.takayama(pline=pline)
    assert call_result == method_result


def test_takayama_valid_pline():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.poverty("takayama", pline=-1)
    with pytest.raises(ValueError):
        ad.poverty("takayama", pline=0)


def test_takayama_extreme_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = np.min(ad.data.values) / 2
    pline_max = np.max(ad.data.values) + 1
    assert ad.poverty("takayama", pline=pline_min) == 0
    assert ad.poverty("takayama", pline=pline_max) <= 1  # CHE¶CK, fails


def test_takayama_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty(method="takayama", pline=pline) == dr2.poverty(
        method="takayama", pline=pline
    )


def test_takayama_replication():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = k * ad.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(
        ad.poverty("takayama", pline=pline),
        dr2.poverty("takayama", pline=pline),
    )


def test_takayama_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("takayama", pline=pline) == dr2.poverty(
        "takayama", pline=pline * k
    )


def test_takayama_avoid_zero_div_error():
    # u = 0
    df = pd.DataFrame({"x": np.zeros(10)})
    ad = ApodeData(df, income_column="x")
    pline = 0.2
    assert ad.poverty.takayama(pline=pline) == 0
    # n = 0
    df = pd.DataFrame({"x": []})
    ad = ApodeData(df, income_column="x")
    assert ad.poverty.takayama(pline=pline) == 0


# =============================================================================
# TESTS KAKWANI
# =============================================================================
def test_kakwani_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    # assert ad.poverty.kakwani(pline=pline) == 0.2027705302170293
    np.testing.assert_allclose(
        ad.poverty.kakwani(pline=pline), 0.2027705302170293
    )


def test_kakwani_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    # assert ad.poverty("kakwani", pline=pline) == 0.2027705302170293
    np.testing.assert_allclose(
        ad.poverty("kakwani", pline=pline), 0.2027705302170293
    )


def test_kakwani_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    call_result = ad.poverty("kakwani", pline=pline)
    method_result = ad.poverty.kakwani(pline=pline)
    assert call_result == method_result


def test_kakwani_valid_pline():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.poverty("kakwani", pline=-1)
    with pytest.raises(ValueError):
        ad.poverty("kakwani", pline=0)


def test_kakwani_extreme_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = np.min(ad.data.values) / 2
    pline_max = np.max(ad.data.values) + 1
    assert ad.poverty("kakwani", pline=pline_min) == 0
    assert ad.poverty("kakwani", pline=pline_max) <= 1


def test_kakwani_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty(method="kakwani", pline=pline) == dr2.poverty(
        method="kakwani", pline=pline
    )


def test_kakwani_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("kakwani", pline=pline) == dr2.poverty(
        "kakwani", pline=pline * k
    )


# =============================================================================
# TESTS THON
# =============================================================================
def test_thon_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty.thon(pline=pline) == 0.24913640189161163


def test_thon_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty("thon", pline=pline) == 0.24913640189161163


def test_thon_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    call_result = ad.poverty("thon", pline=pline)
    method_result = ad.poverty.thon(pline=pline)
    assert call_result == method_result


def test_thon_valid_pline():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.poverty("thon", pline=-1)
    with pytest.raises(ValueError):
        ad.poverty("thon", pline=0)


def test_thon_extreme_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = np.min(ad.data.values) / 2
    pline_max = np.max(ad.data.values) + 1
    assert ad.poverty("thon", pline=pline_min) == 0
    assert ad.poverty("thon", pline=pline_max) <= 1


def test_thon_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty(method="thon", pline=pline) == dr2.poverty(
        method="thon", pline=pline
    )


def test_thon_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("thon", pline=pline) == dr2.poverty(
        "thon", pline=pline * k
    )


# =============================================================================
# TESTS BD
# =============================================================================
def test_bd_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty.bd(pline=pline) == 0.2170854187584956


def test_bd_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty("bd", pline=pline) == 0.2170854187584956
    assert ad.poverty("bd", pline=30) == 0.9950410832744983


def test_bd_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    call_result = ad.poverty("bd", pline=pline)
    method_result = ad.poverty.bd(pline=pline)
    assert call_result == method_result


def test_bd_valid_pline():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.poverty("bd", pline=-1)
    with pytest.raises(ValueError):
        ad.poverty("bd", pline=0)


def test_bd_extreme_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = np.min(ad.data.values) / 2
    pline_max = np.max(ad.data.values) + 1
    assert ad.poverty("bd", pline=pline_min) == 0
    assert ad.poverty("bd", pline=pline_max) <= 1


def test_bd_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty(method="bd", pline=pline) == dr2.poverty(
        method="bd", pline=pline
    )


def test_bd_replication():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = k * ad.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("bd", pline=pline) == dr2.poverty("bd", pline=pline)


def test_bd_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("bd", pline=pline) == dr2.poverty("bd", pline=pline * k)


# =============================================================================
# TESTS HAGENAARS
# =============================================================================
# range not in [0,1]
def test_hagenaars_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty.hagenaars(pline=pline) == -0.19985793327576523


def test_hagenaars_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty("hagenaars", pline=pline) == -0.19985793327576523


def test_hagenaars_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    call_result = ad.poverty("hagenaars", pline=pline)
    method_result = ad.poverty.hagenaars(pline=pline)
    assert call_result == method_result


def test_hagenaars_valid_pline():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.poverty("hagenaars", pline=-1)
    with pytest.raises(ValueError):
        ad.poverty("hagenaars", pline=0)


def test_hagenaars_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty(method="hagenaars", pline=pline) == dr2.poverty(
        method="hagenaars", pline=pline
    )


def test_hagenaars_replication():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = k * ad.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("hagenaars", pline=pline) == dr2.poverty(
        "hagenaars", pline=pline
    )


def test_hagenaars_q_zero():
    df = pd.DataFrame({"x": np.arange(1, 12)})
    ad = ApodeData(df, income_column="x")
    pline = min(ad.data.values) - 0.1
    assert ad.poverty.hagenaars(pline=pline) == 0


# =============================================================================
# TESTS CHAKRAVARTY
# =============================================================================
def test_chakravarty_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty.chakravarty(pline=pline) == 0.09170826904700106


def test_chakravarty_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    assert ad.poverty("chakravarty", pline=pline) == 0.09170826904700106


def test_chakravarty_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = 0.5 * np.median(ad.data.values)
    call_result = ad.poverty("chakravarty", pline=pline)
    method_result = ad.poverty.chakravarty(pline=pline)
    assert call_result == method_result


def test_chakravarty_valid_pline():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.poverty("chakravarty", pline=-1)
    with pytest.raises(ValueError):
        ad.poverty("chakravarty", pline=0)


def test_chakravarty_valid_alpha():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    with pytest.raises(ValueError):
        ad.poverty("chakravarty", pline=pline, alpha=0)
    with pytest.raises(ValueError):
        ad.poverty("chakravarty", pline=pline, alpha=-2)
    with pytest.raises(ValueError):
        ad.poverty("chakravarty", pline=pline, alpha=2)


def test_chakravarty_extreme_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline_min = np.min(ad.data.values) / 2
    pline_max = np.max(ad.data.values) + 1
    assert ad.poverty("chakravarty", pline=pline_min) == 0
    assert ad.poverty("chakravarty", pline=pline_max) <= 1


def test_chakravarty_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty(method="chakravarty", pline=pline) == dr2.poverty(
        method="chakravarty", pline=pline
    )


def test_chakravarty_replication():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = k * ad.data["x"].tolist()
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(
        ad.poverty("chakravarty", pline=pline),
        dr2.poverty("chakravarty", pline=pline),
    )


def test_chakravarty_homogeneity():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    k = 2  # factor
    pline = np.mean(ad.data.values)
    y = ad.data["x"].tolist()
    y = [yi * k for yi in y]
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    assert ad.poverty("chakravarty", pline=pline) == dr2.poverty(
        "chakravarty", pline=pline * k
    )
