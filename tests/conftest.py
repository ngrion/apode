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


@pytest.fixture(scope="session")
def uniform_ad():
    def make(*, seed, **kwargs):
        random = np.random.RandomState(seed=seed)
        x = random.uniform(**kwargs)
        df1 = pd.DataFrame({"x": x})
        return ApodeData(df1, varx="x")

    return make


@pytest.fixture(scope="session")
def normal_ad():
    def make(*, seed, **kwargs):
        random = np.random.RandomState(seed=seed)
        x = random.normal(**kwargs)
        df1 = pd.DataFrame({"x": x})
        return ApodeData(df1, varx="x")

    return make


@pytest.fixture(scope="session")
def income_arrays():
    return pd.read_csv("income.csv")


@pytest.fixture(scope="session")
def inequality_results():
    return pd.read_csv("test_ineq.csv")
