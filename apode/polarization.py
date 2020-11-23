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
import attr

import numpy as np


# =============================================================================
# FUNCTIONS
# =============================================================================


@attr.s(frozen=True)
class PolarizationMeasures:
    """Polarization Measures.

    The following welfare measures are implemented:

    - ray : Esteban and Ray index
    - wolfson : Wolfson index

    Parameters
    ----------
    method : String
        Polarization measure.

    """

    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        """Return the ApodeData object."""
        method = "ray" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    # generalizar parametro
    def ray(self):
        """Esteban and Ray index of polarization.

        Esteban and Ray index of polarization.

        Return
        ------
        out: float
            Polarization measure.

        References
        ----------
        .. Esteban, J.M. y D. Ray (1994), “On the Measurement of
           Polarization”, Econometrica, vol. 62, N. 4, julio, pp.
           819-851.

        """
        y = self.idf.data[self.idf.income_column].values
        pij = 1 / len(y)
        alpha = 1  # (0,1.6]
        p_er = 0
        for yi in y:
            for yj in y:
                p_er += np.power(pij, 1 + alpha) * pij * abs(yi - yj)
        return p_er

    def wolfson(self):
        """Wolfson index of bipolarization.

        Wolfson index of bipolarization (normalized).

        Return
        ------
        out: float
            Polarization measure.

        References
        ----------
        .. Wolfson, Michael C. 1994. “When Inequalities Diverge.”
           The American Economic Review 84 (2): 353–58.

        """
        ys = np.sort(self.idf.data[self.idf.income_column].values)
        ysa = np.cumsum(ys) / np.sum(ys)
        n = len(ys)
        # if (n % 2) == 0:
        #     i = int(n / 2)
        #     L = (ysa[i - 1] + ysa[i]) / 2
        # else:
        #     i = int((n + 1) / 2)
        #     L = ysa[i - 1]
        i = int(n / 2)  # criterio de R
        L = ysa[i - 1]
        g = self.idf.inequality.gini()
        # p_w = (np.mean(ys) / np.median(ys)) * (0.5 - L - g)
        p_w = 4 * (0.5 - L - g / 2) * (np.mean(ys) / np.median(ys))
        return p_w
