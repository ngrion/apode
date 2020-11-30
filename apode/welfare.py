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

"""Welfare measures for Apode."""

# =============================================================================
# IMPORTS
# =============================================================================
import attr

import numpy as np


# =============================================================================
# FUNCTIONS
# =============================================================================


@attr.s(frozen=True)
class WelfareMeasures:
    """Welfare Measures.

    The following welfare measures are implemented:

    - utilitarian : Utilitarian utility function
    - rawlsian : Rawlsian utility function
    - isoelastic : Isoelastic utility function
    - sen : Sen utility function
    - theill : Theill utility function
    - theilt : Theilt utility function

    Parameters
    ----------
    method : String
        Welfare measure.
    **kwargs
        Arbitrary keyword arguments.

    """

    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        """Return the ApodeData object."""
        method = "utilitarian" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    def utilitarian(self):
        """Utilitarian utility function.

        The utilitarian utility function.

        Return
        ------
        out: float
            Utility value.

        """
        y = self.idf.data[self.idf.income_column].values
        return np.mean(y)

    def rawlsian(self):
        """Rawlsian utility function.

        The rawlsian utility function.

        Return
        ------
        out: float
            Utility value.

        """
        y = self.idf.data[self.idf.income_column].values
        return np.min(y)

    def isoelastic(self, alpha):
        """Isoelastic utility function.

        The isoelastic utility function.

        Return
        ------
        out: float
            Utility value.

        """
        y = self.idf.data[self.idf.income_column].values
        if alpha == 0:
            return np.mean(y)
        elif alpha == np.Inf:
            return np.min(y)
        elif alpha == 1:
            return (1 / len(y)) * np.sum(np.log(y))
        return (1 / len(y)) * np.sum(np.power(y, 1 - alpha)) / (1 - alpha)

    def sen(self):
        """Sen utility function.

        The Sen utility function.

        Return
        ------
        out: float
            Utility value.

        """
        y = self.idf.data[self.idf.income_column].values
        u = np.mean(y)
        g = self.idf.inequality.gini()
        return u * (1 - g)

    def theill(self):
        """Theil L utility function.

        The Theil L utility function.

        Return
        ------
        out: float
            Utility value.

        """
        y = self.idf.data[self.idf.income_column].values
        u = np.mean(y)
        tl = self.idf.inequality.entropy(alpha=0)
        return u * np.exp(-tl)

    def theilt(self):
        """Theil T utility function.

        The Theil T utility function.

        Return
        ------
        out: float
            Utility value.

        """
        y = self.idf.data[self.idf.income_column].values
        u = np.mean(y)
        tt = self.idf.inequality.entropy(alpha=1)
        return u * np.exp(-tt)
