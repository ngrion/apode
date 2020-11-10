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

import pathlib
import os

from apode.basic import ApodeData

PATH = pathlib.Path(os.path.abspath(os.path.dirname((__file__))))

TEST_DATA_PATH = PATH / "test_data"


@pytest.fixture(scope="session")
def uniform_ad():
    def make(*, seed, **kwargs):
        random = np.random.RandomState(seed=seed)
        x = random.uniform(**kwargs)
        df1 = pd.DataFrame({"x": x})
        return ApodeData(df1, income_column="x")

    return make


@pytest.fixture(scope="session")
def normal_ad():
    def make(*, seed, **kwargs):
        random = np.random.RandomState(seed=seed)
        x = random.normal(**kwargs)
        df1 = pd.DataFrame({"x": x})
        return ApodeData(df1, income_column="x")

    return make


@pytest.fixture(scope="session")
def income_arrays():
    return pd.read_csv(TEST_DATA_PATH / "income.csv")


@pytest.fixture(scope="session")
def inequality_results():
    return pd.read_csv(TEST_DATA_PATH / "test_ineq.csv")
