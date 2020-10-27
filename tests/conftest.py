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

random = np.random.RandomState(seed=42)


@pytest.fixture
def uniform_ad():
    x = random.uniform(size=300)
    df1 = pd.DataFrame({"x": x})
    return ApodeData(df1, varx="x")


@pytest.fixture
def normal_ad():
    x = random.normal(size=300)
    df1 = pd.DataFrame({"x": x})
    return ApodeData(df1, varx="x")
