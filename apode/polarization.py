#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Apode Project (https://github.com/ngrion/apode).
# Copyright (c) 2020, Néstor Grión and Sofía Sappia
# License: MIT
#   Full Text: https://github.com/ngrion/apode/blob/master/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Polarization measures for Apode."""


# =============================================================================
# IMPORTS
# =============================================================================


import numpy as np
import attr


# =============================================================================
# FUNCTIONS
# =============================================================================


@attr.s(frozen=True)
class PolarizationMeasures:
    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        method = "esteban_ray" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    # Esteban and Ray index of polarization
    # generalizar parametro
    def ray(self):
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        alpha = 1  # (0,1.6]
        p_er = 0
        for i in range(len(y)):
            for j in range(len(y)):
                pi = 1 / n
                pj = 1 / n
                p_er += np.power(pi, 1 + alpha) * pj * abs(y[i] - y[j])
        return p_er

    # Wolfson index of bipolarization (normalizado)
    # ver que n> sea grande
    def wolfson(self):
        ys = np.sort(self.idf.data[self.idf.varx].values)
        ysa = np.cumsum(ys) / np.sum(ys)
        n = len(ys)
        # if (n % 2) == 0:
        #     i = int(n / 2)
        #     L = (ysa[i - 1] + ysa[i]) / 2
        # else:
        #     i = int((n + 1) / 2)
        #     L = ysa[i - 1]
        i = int(n/2)     # criterio de R
        L = ysa[i - 1]
        g = self.idf.inequality.gini()
        # p_w = (np.mean(ys) / np.median(ys)) * (0.5 - L - g)
        p_w = 4 * (0.5 - L - g/2) * (np.mean(ys) / np.median(ys))
        return p_w
