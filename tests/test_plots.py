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
from unittest.mock import patch
from apode import plots

# =============================================================================
# TESTS COMMON
# =============================================================================
def test_default_call(uniform_ad):
    data = uniform_ad(seed=42, size=300)
    # call_result = data.plot("hist")
    # method_result = data.plot.hist()
    # assert call_result == method_result
    with patch("plots.PlotAccessor.plot.hist()") as show_patch:
        # code_in_my_module_that_plots()
        data.plot.hist()
        assert show_patch.called

# def test_invalid(uniform_ad):
#     data = uniform_ad(seed=42, size=300)
#     with pytest.raises(AttributeError):
#         data.welfare("foo")
