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

from apode.inequality import InequalityMeasures

# def test_entropy(uniform_ad):
#     data = uniform_ad
#     y = data.data["x"].values.sort()
#     ineq = InequalityMeasures(pd.DataFrame({"x": y}))
#     np.testing.assert_allclose(ineq.entropy(alpha=0,sort=False), 0.3226715241069236)
#     np.testing.assert_allclose(data.inequality('entropy', alpha=0, sort=False), 0.3226715241069236)
#     np.testing.assert_allclose(data.inequality('entropy', alpha=0, sort=True), 0.3226715241069236)
#
# #
# # Entropia general  (necesita sort?) (ver rango a)
# def entropy(self, alpha=0, sort=False):
#     a = alpha
#     y = self.idf.data[self.idf.varx].values
#     if not sort:
#         y = np.sort(y)
#     n = len(y)
#     if n == 0:
#         return 0
#     u = np.mean(y)
#     if a == 0.0:
#         return np.sum(np.log(u / y)) / n
#     elif a == 1.0:
#         return np.sum((y / u) * np.log(y / u)) / n
#     return (1 / (a * (a - 1))) * (np.sum(pow(y / u, a)) / n - 1)
