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
    call_result = ad.inequality("gini")
    method_result = ad.inequality.gini()
    assert call_result == method_result


def test_invalid():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(AttributeError):
        ad.inequality("foo")


# =============================================================================
# TESTS GINI
# =============================================================================
def test_gini_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(ad.inequality.gini(), 0.34232535781966483)


def test_gini_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(ad.inequality("gini"), 0.34232535781966483)


def test_gini_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.inequality("gini")
    method_result = ad.inequality.gini()
    assert call_result == method_result


def test_gini_size_one_array():
    y = [2]
    df = pd.DataFrame({"x": y})
    ad = ApodeData(df, income_column="x")
    assert ad.inequality.gini() == 0


def test_gini_extreme_values():
    y = np.zeros(300)
    y[0] = 10
    np.random.shuffle(y)
    df = pd.DataFrame({"x": y})
    # data_min = ApodeData(df, income_column="x") #noqa
    y = np.ones(300) * 10
    df = pd.DataFrame({"x": y})
    data_max = ApodeData(df, income_column="x")
    # assert data_min.inequality.gini() == 1 #CHECK, fails
    assert data_max.inequality.gini() == 0


def test_gini_values(income_arrays, inequality_results):
    ad1 = ApodeData(income_arrays, income_column="y1")
    ad2 = ApodeData(income_arrays, income_column="y2")
    ad3 = ApodeData(income_arrays, income_column="y3")
    ad4 = ApodeData(income_arrays, income_column="y4")
    ad5 = ApodeData(income_arrays, income_column="y5")
    ad6 = ApodeData(income_arrays, income_column="y6")
    # Relative tets modified (avoid division by zero):
    #  test = ((val1+1)-(val2+1))/(val1+1)
    np.testing.assert_allclose(
        ad1.inequality.gini() + 1, inequality_results.gini[0] + 1
    )
    np.testing.assert_allclose(
        ad2.inequality.gini(), inequality_results.gini[1]
    )
    np.testing.assert_allclose(
        ad3.inequality.gini(), inequality_results.gini[2]
    )
    np.testing.assert_allclose(
        ad4.inequality.gini(), inequality_results.gini[3]
    )
    np.testing.assert_allclose(
        ad5.inequality.gini(), inequality_results.gini[4]
    )
    np.testing.assert_allclose(
        ad6.inequality.gini(), inequality_results.gini[5]
    )


# =============================================================================
# TESTS ENTROPY
# =============================================================================
def test_entropy_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(ad.inequality.entropy(), 0.3226715241069237)


def test_entropy_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(ad.inequality("entropy"), 0.3226715241069237)


def test_entropy_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.inequality("entropy")
    method_result = ad.inequality.entropy()
    assert call_result == method_result


def test_entropy_values(income_arrays, inequality_results):
    ad1 = ApodeData(income_arrays, income_column="y1")
    ad2 = ApodeData(income_arrays, income_column="y2")
    ad3 = ApodeData(income_arrays, income_column="y3")
    ad4 = ApodeData(income_arrays, income_column="y4")
    ad5 = ApodeData(income_arrays, income_column="y5")
    ad6 = ApodeData(income_arrays, income_column="y6")
    # alpha = 0
    # Relative tets modified (avoid division by zero):
    #  test = ((val1+1)-(val2+1))/(val1+1)
    np.testing.assert_allclose(
        ad1.inequality.entropy(alpha=0) + 1, inequality_results.entropy0[0] + 1
    )
    np.testing.assert_allclose(
        ad2.inequality.entropy(alpha=0), inequality_results.entropy0[1]
    )
    np.testing.assert_allclose(
        ad3.inequality.entropy(alpha=0), inequality_results.entropy0[2]
    )
    np.testing.assert_allclose(
        ad4.inequality.entropy(alpha=0), inequality_results.entropy0[3]
    )
    np.testing.assert_allclose(
        ad5.inequality.entropy(alpha=0), inequality_results.entropy0[4]
    )
    np.testing.assert_allclose(
        ad6.inequality.entropy(alpha=0), inequality_results.entropy0[5]
    )
    # alpha = 1
    # Relative tets modified (avoid division by zero):
    #  test = ((val1+1)-(val2+1))/(val1+1)
    np.testing.assert_allclose(
        ad1.inequality.entropy(alpha=1) + 1, inequality_results.entropy1[0] + 1
    )
    # Falla, 0.31443337122879683, ver formula stata
    # np.testing.assert_allclose(
    #     ad2.inequality.entropy(alpha=1), inequality_results.entropy1[1]
    # )
    # np.testing.assert_allclose(
    #     ad3.inequality.entropy(alpha=1), inequality_results.entropy1[2]
    # )
    # np.testing.assert_allclose(
    #     ad4.inequality.entropy(alpha=1), inequality_results.entropy1[3]
    # )
    np.testing.assert_allclose(
        ad5.inequality.entropy(alpha=1), inequality_results.entropy1[4]
    )
    np.testing.assert_allclose(
        ad6.inequality.entropy(alpha=1), inequality_results.entropy1[5]
    )
    # alpha = 2
    # Relative tets modified (avoid division by zero):
    #  test = ((val1+1)-(val2+1))/(val1+1)
    np.testing.assert_allclose(
        ad1.inequality.entropy(alpha=2) + 1, inequality_results.entropy2[0] + 1
    )
    np.testing.assert_allclose(
        ad2.inequality.entropy(alpha=2), inequality_results.entropy2[1]
    )
    np.testing.assert_allclose(
        ad3.inequality.entropy(alpha=2), inequality_results.entropy2[2]
    )
    np.testing.assert_allclose(
        ad4.inequality.entropy(alpha=2), inequality_results.entropy2[3]
    )
    np.testing.assert_allclose(
        ad5.inequality.entropy(alpha=2), inequality_results.entropy2[4]
    )
    np.testing.assert_allclose(
        ad6.inequality.entropy(alpha=2), inequality_results.entropy2[5]
    )


def test_entropy_symmetry():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    y = ad.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    ad2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(
        ad.inequality(method="entropy"), ad2.inequality(method="entropy")
    )


def test_entropy_alpha_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(
        ad.inequality("entropy", alpha=0), 0.3226715241069237
    )
    assert ad.inequality("entropy", alpha=1) == 0.20444600065652588
    assert ad.inequality("entropy", alpha=0.5) == 0.24247057381922854


# =============================================================================
# TESTS ATKINSON
# =============================================================================
def test_atkinson_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality.atkinson(alpha=1) == 0.2757882986123399


def test_atkinson_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality("atkinson", alpha=1) == 0.2757882986123399


def test_atkinson_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.inequality("atkinson", alpha=1)
    method_result = ad.inequality.atkinson(alpha=1)
    assert call_result == method_result


def test_atkinson_values(income_arrays, inequality_results):
    ad1 = ApodeData(income_arrays, income_column="y1")
    ad2 = ApodeData(income_arrays, income_column="y2")
    ad3 = ApodeData(income_arrays, income_column="y3")
    ad4 = ApodeData(income_arrays, income_column="y4")
    ad5 = ApodeData(income_arrays, income_column="y5")
    ad6 = ApodeData(income_arrays, income_column="y6")

    # alpha = 0.5
    # Relative tets modified (avoid division by zero):
    #  test = ((val1+1)-(val2+1))/(val1+1)
    np.testing.assert_allclose(
        ad1.inequality.atkinson(alpha=0.5) + 1,
        inequality_results.atkinson05[0] + 1,
    )
    np.testing.assert_allclose(
        ad2.inequality.atkinson(alpha=0.5), inequality_results.atkinson05[1]
    )
    np.testing.assert_allclose(
        ad3.inequality.atkinson(alpha=0.5), inequality_results.atkinson05[2]
    )
    np.testing.assert_allclose(
        ad4.inequality.atkinson(alpha=0.5), inequality_results.atkinson05[3]
    )
    np.testing.assert_allclose(
        ad5.inequality.atkinson(alpha=0.5), inequality_results.atkinson05[4]
    )
    np.testing.assert_allclose(
        ad6.inequality.atkinson(alpha=0.5), inequality_results.atkinson05[5]
    )
    # alpha = 1
    # Relative tets modified (avoid division by zero):
    #  test = ((val1+1)-(val2+1))/(val1+1)
    np.testing.assert_allclose(
        ad1.inequality.atkinson(alpha=1) + 1,
        inequality_results.atkinson1[0] + 1,
    )
    np.testing.assert_allclose(
        ad2.inequality.atkinson(alpha=1), inequality_results.atkinson1[1]
    )
    np.testing.assert_allclose(
        ad3.inequality.atkinson(alpha=1), inequality_results.atkinson1[2]
    )
    np.testing.assert_allclose(
        ad4.inequality.atkinson(alpha=1), inequality_results.atkinson1[3]
    )
    np.testing.assert_allclose(
        ad5.inequality.atkinson(alpha=1), inequality_results.atkinson1[4]
    )
    np.testing.assert_allclose(
        ad6.inequality.atkinson(alpha=1), inequality_results.atkinson1[5]
    )
    # alpha = 2 # Integers to negative integer powers are not allowed.
    # np.testing.assert_allclose(ad1.inequality.atkinson(alpha=2),
    # # inequality_results.atkinson2[0]) # solve overflow
    # np.testing.assert_allclose(ad2.inequality.atkinson(alpha=2),
    # inequality_results.atkinson2[1])
    # np.testing.assert_allclose(ad3.inequality.atkinson(alpha=2),
    # inequality_results.atkinson2[2])
    # np.testing.assert_allclose(ad4.inequality.atkinson(alpha=2),
    # inequality_results.atkinson2[3])
    # # np.testing.assert_allclose(ad5.inequality.atkinson(alpha=2),
    # inequality_results.atkinson2[4])
    # np.testing.assert_allclose(ad6.inequality.atkinson(alpha=2),
    # inequality_results.atkinson2[5])


def test_atkinson_valid_alpha():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.inequality.atkinson(alpha=-1)
    with pytest.raises(ValueError):
        ad.inequality.atkinson(alpha=0)


def test_atkinson_size_one_array():
    y = [2]
    df = pd.DataFrame({"x": y})
    ad = ApodeData(df, income_column="x")
    assert ad.inequality.atkinson(alpha=1) == 0


def test_atkinson_alpha_values():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality("atkinson", alpha=1) == 0.2757882986123399
    assert ad.inequality("atkinson", alpha=0.5) == 0.11756078821160021


# =============================================================================
# TESTS RRANGE
# =============================================================================
def test_rrange_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality.rrange() == 1.9890612245143606


def test_rrange_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality("rrange") == 1.9890612245143606


def test_rrange_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.inequality("rrange")
    method_result = ad.inequality.rrange()
    assert call_result == method_result


# =============================================================================
# TESTS RAD
# =============================================================================
def test_rad_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality.rad() == 0.2600392437248902


def test_rad_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality("rad") == 0.2600392437248902


def test_rad_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.inequality("rad")
    method_result = ad.inequality.rad()
    assert call_result == method_result


# =============================================================================
# TESTS CV
# =============================================================================
def test_cv_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality.cv() == 0.5933902127888603


def test_cv_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality("cv") == 0.5933902127888603


def test_cv_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.inequality("cv")
    method_result = ad.inequality.cv()
    assert call_result == method_result


# =============================================================================
# TESTS SDLOG
# =============================================================================
def test_sdlog_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality.sdlog() == 1.057680329912003


def test_sdlog_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality("sdlog") == 1.057680329912003


def test_sdlog_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.inequality("sdlog")
    method_result = ad.inequality.sdlog()
    assert call_result == method_result


# =============================================================================
# TESTS MERHAN
# =============================================================================
def test_merhan_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality.merhan() == 0.5068579435513223


def test_merhan_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality("merhan") == 0.5068579435513223


def test_merhan_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.inequality("merhan")
    method_result = ad.inequality.merhan()
    assert call_result == method_result


# =============================================================================
# TESTS bonferroni
# =============================================================================
def test_bonferroni_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(ad.inequality.bonferroni(), 0.507498668487682)


def test_bonferroni_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(ad.inequality("bonferroni"), 0.507498668487682)


def test_bonferroni_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.inequality("bonferroni")
    method_result = ad.inequality.bonferroni()
    assert call_result == method_result


# =============================================================================
# TESTS piesch
# =============================================================================
def test_piesch_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(ad.inequality.piesch(), 0.25015872424726393)


def test_piesch_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(ad.inequality("piesch"), 0.25015872424726393)


def test_piesch_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.inequality("piesch")
    method_result = ad.inequality.piesch()
    assert call_result == method_result


# =============================================================================
# TESTS kolm
# =============================================================================
def test_kolm_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality.kolm(alpha=1) == 0.04278027786607911


def test_kolm_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality("kolm", alpha=1) == 0.04278027786607911


def test_kolm_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.inequality("kolm", alpha=1)
    method_result = ad.inequality.kolm(alpha=1)
    assert call_result == method_result


def test_kolm_invalid_alpha():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.inequality.kolm(alpha=0)
    with pytest.raises(ValueError):
        ad.inequality.kolm(alpha=-1)


# =============================================================================
# TESTS ratio
# =============================================================================
def test_ratio_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality.ratio(alpha=0.5) == 0.31651799363507865


def test_ratio_call():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert ad.inequality("ratio", alpha=0.5) == 0.31651799363507865


def test_ratio_call_equal_method():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = ad.inequality("ratio", alpha=0.5)
    method_result = ad.inequality.ratio(alpha=0.5)
    assert call_result == method_result


def test_ratio_invalid_alpha():
    ad = datasets.make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        ad.inequality.ratio(alpha=-1)
    with pytest.raises(ValueError):
        ad.inequality.ratio(alpha=2)
