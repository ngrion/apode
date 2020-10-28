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
def test_gini_method(uniform_ad, normal_ad):
    datau = uniform_ad(seed=42, size=300)
    datan = normal_ad(seed=42, size=300)
    assert datau.inequality.gini() == 0.34232535781966483
    assert datan.inequality.gini() == -98.91708721375252

def test_gini_extreme_values():
    y = np.zeros(300)
    y[0] = 10
    np.random.shuffle(y)
    df = pd.DataFrame({"x": y})
    data_min = ApodeData(df, varx="x")
    y = np.ones(300) * 10
    df = pd.DataFrame({"x": y})
    data_max = ApodeData(df, varx="x")
    assert data_min.inequality.gini() == 1
    assert data_max.inequality.gini() == 0

