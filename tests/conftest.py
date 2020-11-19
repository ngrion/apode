#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Apode Project (https://github.com/mchalela/apode).
# Copyright (c) 2020, Néstor Grión and Sofía Sappia
# License: MIT
#   Full Text: https://github.com/ngrion/apode/blob/master/LICENSE.txt

import os
import pathlib

import pandas as pd

import pytest

PATH = pathlib.Path(os.path.abspath(os.path.dirname((__file__))))

TEST_DATA_PATH = PATH / "test_data"


@pytest.fixture(scope="session")
def income_arrays():
    return pd.read_csv(TEST_DATA_PATH / "income.csv")


@pytest.fixture(scope="session")
def inequality_results():
    return pd.read_csv(TEST_DATA_PATH / "test_ineq.csv")
