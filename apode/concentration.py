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

"""Concentration measures for Apode."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import attr


# =============================================================================
# FUNCTIONS
# =============================================================================


@attr.s(frozen=True)
class ConcentrationMeasures:
    """Concentration Measures.

    The following concentration measures are implemented:
    - herfindahl : Herfindahl-Hirschman Index
    - rosenbluth : Rosenbluth Index
    - concentration_ratio : Concentration Ratio Index

    Parameters
    ----------
    method : String
        Concentration measure.
    **kwargs
        Arbitrary keyword arguments.

    """

    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        """Return the ApodeData object."""
        method = "herfindahl" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    def herfindahl(self, normalized=True):  # cambiar normalize
        """Herfindahl-Hirschman index.

        The Herfindahl-Hirschman index measures the proportion of
        counted as poor.

        Parameters
        ----------
        normalized: bool(default=true)
            Fraction considered.

        Return
        ------
        out: float
            Concentration measures.

        """
        y = self.idf.data[self.idf.varx].values
        w = y / sum(y)
        n = len(y)
        if n == 0:
            return 0
        else:
            h = np.square(w).sum()
            if normalized:
                return (h - 1.0 / n) / (1.0 - 1.0 / n)
            else:
                return h

    def rosenbluth(self):
        """Rosenbluth index.

        The Rosenbluth index measures the proportion of the population that
        counted as poor.

        Return
        ------
        out: float
            Concentration measures.

        """
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        g = self.idf.inequality.gini()
        return 1 / (n * (1 - g))

    def concentration_ratio(self, k):
        """Concentration Ratio index.

        The Concentration Ratio index measures the proportion of
        counted as poor.

        Parameters
        ----------
        k: int
            Fraction considered.

        Return
        ------
        out: float
            Concentration measures.

        """
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        if k < 0 or k > n:
            raise ValueError(
                "n must be an positive integer " "smaller than the data size"
            )
        else:
            ys = np.sort(y)[::-1]
            return ys[:k].sum() / ys.sum()
