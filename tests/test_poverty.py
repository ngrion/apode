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


# # # Testea metdodo de un listado de medidas de pobreza
# # def test_lista(prop,lista):
# #     x = TestCaseUniform()
# #     x.setup_method()
# #     for elem in lista:
# #         if elem[1]==None:
# #             x.prop(elem[0])
# #         else:
# #             x.prop(elem[0],alpha=elem[1])
