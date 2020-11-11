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

from apode.datasets import make_uniform


# =============================================================================
# TESTS COMMON
# =============================================================================


def test_default_call():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.inequality("gini")
    method_result = data.inequality.gini()
    assert call_result == method_result


def test_invalid():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(AttributeError):
        data.inequality("foo")


# =============================================================================
# TESTS GINI
# =============================================================================
# @pytest.mark.xfail
def test_gini_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(data.inequality.gini(), 0.34232535781966483)


def test_gini_call():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(data.inequality("gini"), 0.34232535781966483)


def test_gini_call_equal_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.inequality("gini")
    method_result = data.inequality.gini()
    assert call_result == method_result


def test_gini_empty_array():
    y = []
    df = pd.DataFrame({"x": y})
    data = ApodeData(df, income_column="x")
    assert data.inequality.gini() == 0


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


@pytest.mark.xfail
def test_gini_values(income_arrays, inequality_results):
    df_income = income_arrays
    df_ineq = inequality_results
    dat1 = ApodeData(df_income, income_column="y1")
    dat2 = ApodeData(df_income, income_column="y2")
    dat3 = ApodeData(df_income, income_column="y3")
    dat4 = ApodeData(df_income, income_column="y4")
    dat5 = ApodeData(df_income, income_column="y5")
    dat6 = ApodeData(df_income, income_column="y6")
    np.testing.assert_allclose(dat1.inequality.gini(),
                               df_ineq.gini[0])  # solve overflow
    np.testing.assert_allclose(dat2.inequality.gini(), df_ineq.gini[1])
    np.testing.assert_allclose(dat3.inequality.gini(), df_ineq.gini[2])
    np.testing.assert_allclose(dat4.inequality.gini(), df_ineq.gini[3])
    np.testing.assert_allclose(dat5.inequality.gini(), df_ineq.gini[4])
    np.testing.assert_allclose(dat6.inequality.gini(), df_ineq.gini[5])


# =============================================================================
# TESTS ENTROPY
# =============================================================================
def test_entropy_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(data.inequality.entropy(), 0.3226715241069237)


def test_entropy_call():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(data.inequality("entropy"), 0.3226715241069237)


def test_entropy_call_equal_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.inequality("entropy")
    method_result = data.inequality.entropy()
    assert call_result == method_result


@pytest.mark.xfail
def test_entropy_values(income_arrays, inequality_results):
    df_income = income_arrays
    df_ineq = inequality_results
    dat1 = ApodeData(df_income, income_column="y1")
    dat2 = ApodeData(df_income, income_column="y2")
    dat3 = ApodeData(df_income, income_column="y3")
    dat4 = ApodeData(df_income, income_column="y4")
    dat5 = ApodeData(df_income, income_column="y5")
    dat6 = ApodeData(df_income, income_column="y6")
    # alpha = 0
    np.testing.assert_allclose(dat1.inequality.entropy(alpha=0),
                               df_ineq.entropy0[0])  # solve overflow
    np.testing.assert_allclose(
        dat2.inequality.entropy(alpha=0), df_ineq.entropy0[1]
    )
    np.testing.assert_allclose(
        dat3.inequality.entropy(alpha=0), df_ineq.entropy0[2]
    )
    np.testing.assert_allclose(
        dat4.inequality.entropy(alpha=0), df_ineq.entropy0[3]
    )
    np.testing.assert_allclose(
        dat5.inequality.entropy(alpha=0), df_ineq.entropy0[4]
    )
    np.testing.assert_allclose(
        dat6.inequality.entropy(alpha=0), df_ineq.entropy0[5]
    )
    # alpha = 1
    np.testing.assert_allclose(dat1.inequality.entropy(alpha=1),
                               df_ineq.entropy1[0])  # solve overflow
    np.testing.assert_allclose(
        dat2.inequality.entropy(alpha=1), df_ineq.entropy1[1]
    )
    np.testing.assert_allclose(
        dat3.inequality.entropy(alpha=1), df_ineq.entropy1[2]
    )
    np.testing.assert_allclose(dat4.inequality.entropy(alpha=1),
                               df_ineq.entropy1[3])  # returns NaN, solve
    np.testing.assert_allclose(
        dat5.inequality.entropy(alpha=1), df_ineq.entropy1[4]
    )
    np.testing.assert_allclose(
        dat6.inequality.entropy(alpha=1), df_ineq.entropy1[5]
    )
    # alpha = 2
    np.testing.assert_allclose(dat1.inequality.entropy(alpha=2),
                               df_ineq.entropy2[0])  # solve overflow
    np.testing.assert_allclose(
        dat2.inequality.entropy(alpha=2), df_ineq.entropy2[1]
    )
    np.testing.assert_allclose(
        dat3.inequality.entropy(alpha=2), df_ineq.entropy2[2]
    )
    np.testing.assert_allclose(
        dat4.inequality.entropy(alpha=2), df_ineq.entropy2[3]
    )
    np.testing.assert_allclose(
        dat5.inequality.entropy(alpha=2), df_ineq.entropy2[4]
    )
    np.testing.assert_allclose(
        dat6.inequality.entropy(alpha=2), df_ineq.entropy2[5]
    )


def test_entropy_symmetry():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    y = data.data["x"].tolist()
    np.random.shuffle(y)
    df2 = pd.DataFrame({"x": y})
    dr2 = ApodeData(df2, income_column="x")
    np.testing.assert_allclose(data.inequality(method="entropy"),
                               dr2.inequality(method="entropy"))


def test_entropy_empty_array():
    y = []
    df = pd.DataFrame({"x": y})
    data = ApodeData(df, income_column="x")
    assert data.inequality.entropy() == 0


def test_entropy_alpha_values():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(data.inequality("entropy", alpha=0), 0.3226715241069237)
    assert data.inequality("entropy", alpha=1) == 0.20444600065652588
    assert data.inequality("entropy", alpha=0.5) == 0.24247057381922854


# =============================================================================
# TESTS ATKINSON
# =============================================================================
def test_atkinson_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality.atkinson(alpha=1) == 0.2757882986123399


def test_atkinson_call():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality("atkinson", alpha=1) == 0.2757882986123399


def test_atkinson_call_equal_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.inequality("atkinson", alpha=1)
    method_result = data.inequality.atkinson(alpha=1)
    assert call_result == method_result

@pytest.mark.xfail
def test_atkinson_values(income_arrays, inequality_results):
    df_income = income_arrays
    df_ineq = inequality_results
    dat1 = ApodeData(df_income, income_column="y1")
    dat2 = ApodeData(df_income, income_column="y2")
    dat3 = ApodeData(df_income, income_column="y3")
    dat4 = ApodeData(df_income, income_column="y4")
    dat5 = ApodeData(df_income, income_column="y5")
    dat6 = ApodeData(df_income, income_column="y6")

    # alpha = 0.5
    np.testing.assert_allclose(dat1.inequality.atkinson(alpha=0.5),
                               df_ineq.atkinson05[0])  # solve overflow
    np.testing.assert_allclose(
        dat2.inequality.atkinson(alpha=0.5), df_ineq.atkinson05[1]
    )
    np.testing.assert_allclose(
        dat3.inequality.atkinson(alpha=0.5), df_ineq.atkinson05[2]
    )
    np.testing.assert_allclose(
        dat4.inequality.atkinson(alpha=0.5), df_ineq.atkinson05[3]
    )
    np.testing.assert_allclose(
        dat5.inequality.atkinson(alpha=0.5), df_ineq.atkinson05[4]
    )
    np.testing.assert_allclose(
        dat6.inequality.atkinson(alpha=0.5), df_ineq.atkinson05[5]
    )
    # alpha = 1
    np.testing.assert_allclose(dat1.inequality.atkinson(alpha=1),
                                                        df_ineq.atkinson1[0])  # solve overflow
    np.testing.assert_allclose(
        dat2.inequality.atkinson(alpha=1), df_ineq.atkinson1[1]
    )
    np.testing.assert_allclose(
        dat3.inequality.atkinson(alpha=1), df_ineq.atkinson1[2]
    )
    np.testing.assert_allclose(dat4.inequality.atkinson(alpha=1),
                               df_ineq.atkinson1[3])  # FAILS
    np.testing.assert_allclose(
        dat5.inequality.atkinson(alpha=1), df_ineq.atkinson1[4]
    )
    np.testing.assert_allclose(
        dat6.inequality.atkinson(alpha=1), df_ineq.atkinson1[5]
    )
    # alpha = 2 # Integers to negative integer powers are not allowed.
    # np.testing.assert_allclose(dat1.inequality.atkinson(alpha=2),
    # # df_ineq.atkinson2[0]) # solve overflow
    # np.testing.assert_allclose(dat2.inequality.atkinson(alpha=2),
    # df_ineq.atkinson2[1])
    # np.testing.assert_allclose(dat3.inequality.atkinson(alpha=2),
    # df_ineq.atkinson2[2])
    # np.testing.assert_allclose(dat4.inequality.atkinson(alpha=2),
    # df_ineq.atkinson2[3])
    # # np.testing.assert_allclose(dat5.inequality.atkinson(alpha=2),
    # df_ineq.atkinson2[4])
    # np.testing.assert_allclose(dat6.inequality.atkinson(alpha=2),
    # df_ineq.atkinson2[5])


# KEEP OR NOT?
# def test_atkinson_symmetry():
#     data = make_uniform(seed=42, size=300, mu=1, nbin=None)
#     y = data.data["x"].tolist()
#     np.random.shuffle(y)
#     df2 = pd.DataFrame({"x": y})
#     dr2 = ApodeData(df2, income_column="x")
#     assert data.inequality(method="atkinson", alpha=1) == \
#            dr2.inequality(method="atkinson", alpha=1)


def test_atkinson_valid_alpha():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.inequality.atkinson(alpha=-1)
        data.inequality.atkinson(alpha=0)


def test_atkinson_empty_array():
    y = []
    df = pd.DataFrame({"x": y})
    data = ApodeData(df, income_column="x")
    assert data.inequality.atkinson(alpha=1) == 0


def test_atkinson_alpha_values():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality("atkinson", alpha=1) == 0.2757882986123399
    assert data.inequality("atkinson", alpha=0.5) == 0.11756078821160021


# =============================================================================
# TESTS RRANGE
# =============================================================================
def test_rrange_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality.rrange() == 1.9890612245143606


def test_rrange_call():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality("rrange") == 1.9890612245143606


def test_rrange_empty_array():
    y = []
    df = pd.DataFrame({"x": y})
    data = ApodeData(df, income_column="x")
    assert data.inequality.rrange() == 0


def test_rrange_call_equal_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.inequality("rrange")
    method_result = data.inequality.rrange()
    assert call_result == method_result


# =============================================================================
# TESTS RAD
# =============================================================================
def test_rad_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality.rad() == 0.2600392437248902


def test_rad_call():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality("rad") == 0.2600392437248902


def test_rad_call_equal_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.inequality("rad")
    method_result = data.inequality.rad()
    assert call_result == method_result


def test_rad_empty_array():
    y = []
    df = pd.DataFrame({"x": y})
    data = ApodeData(df, income_column="x")
    assert data.inequality.rad() == 0


# =============================================================================
# TESTS CV
# =============================================================================
def test_cv_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality.cv() == 0.5933902127888603


def test_cv_call():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality("cv") == 0.5933902127888603


def test_cv_call_equal_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.inequality("cv")
    method_result = data.inequality.cv()
    assert call_result == method_result


def test_cv_empty_array():
    y = []
    df = pd.DataFrame({"x": y})
    data = ApodeData(df, income_column="x")
    assert data.inequality.cv() == 0


# =============================================================================
# TESTS SDLOG
# =============================================================================
def test_sdlog_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality.sdlog() == 1.057680329912003


def test_sdlog_call():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality("sdlog") == 1.057680329912003


def test_sdlog_call_equal_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.inequality("sdlog")
    method_result = data.inequality.sdlog()
    assert call_result == method_result


def test_sdlog_empty_array():
    y = []
    df = pd.DataFrame({"x": y})
    data = ApodeData(df, income_column="x")
    assert data.inequality.sdlog() == 0


# =============================================================================
# TESTS MERHAN
# =============================================================================
def test_merhan_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality.merhan() == 0.5068579435513223


def test_merhan_call():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality("merhan") == 0.5068579435513223


def test_merhan_call_equal_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.inequality("merhan")
    method_result = data.inequality.merhan()
    assert call_result == method_result


def test_merhan_empty_array():
    y = []
    df = pd.DataFrame({"x": y})
    data = ApodeData(df, income_column="x")
    assert data.inequality.merhan() == 0


# =============================================================================
# TESTS bonferroni
# =============================================================================
def test_bonferroni_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(data.inequality.bonferroni(), 0.507498668487682)


def test_bonferroni_call():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(data.inequality("bonferroni"), 0.507498668487682)


def test_bonferroni_call_equal_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.inequality("bonferroni")
    method_result = data.inequality.bonferroni()
    assert call_result == method_result


def test_bonferroni_empty_array():
    y = []
    df = pd.DataFrame({"x": y})
    data = ApodeData(df, income_column="x")
    assert data.inequality.bonferroni() == 0


# =============================================================================
# TESTS piesch
# =============================================================================
def test_piesch_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(data.inequality.piesch(), 0.25015872424726393)


def test_piesch_call():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    np.testing.assert_allclose(data.inequality("piesch"), 0.25015872424726393)


def test_piesch_call_equal_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.inequality("piesch")
    method_result = data.inequality.piesch()
    assert call_result == method_result


def test_piesch_empty_array():
    y = []
    df = pd.DataFrame({"x": y})
    data = ApodeData(df, income_column="x")
    assert data.inequality.piesch() == 0


# =============================================================================
# TESTS kolm
# =============================================================================
def test_kolm_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality.kolm(alpha=1) == 0.04278027786607911


def test_kolm_call():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality("kolm", alpha=1) == 0.04278027786607911


def test_kolm_call_equal_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.inequality("kolm", alpha=1)
    method_result = data.inequality.kolm(alpha=1)
    assert call_result == method_result


def test_kolm_empty_array():
    y = []
    df = pd.DataFrame({"x": y})
    data = ApodeData(df, income_column="x")
    assert data.inequality.kolm(alpha=1) == 0


def test_kolm_invalid_alpha():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.inequality.kolm(alpha=0)
        data.inequality.kolm(alpha=-1)


# =============================================================================
# TESTS ratio
# =============================================================================
def test_ratio_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality.ratio(alpha=0.5) == 0.31651799363507865


def test_ratio_call():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    assert data.inequality("ratio", alpha=0.5) == 0.31651799363507865


def test_ratio_call_equal_method():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    call_result = data.inequality("ratio", alpha=0.5)
    method_result = data.inequality.ratio(alpha=0.5)
    assert call_result == method_result


def test_ratio_empty_array():
    y = []
    df = pd.DataFrame({"x": y})
    data = ApodeData(df, income_column="x")
    assert data.inequality.ratio(alpha=0.5) == 0


def test_ratio_invalid_alpha():
    data = make_uniform(seed=42, size=300, mu=1, nbin=None)
    with pytest.raises(ValueError):
        data.inequality.ratio(alpha=1)
        data.inequality.ratio(alpha=-1)
        data.inequality.ratio(alpha=2)
