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
import attr

import numpy as np


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

        The Herfindahl-Hirschman index It is defined as the sum of
        the squares of the market shares of the firms within the
        industry.

        Parameters
        ----------
        normalized: bool(default=true)
            The normalized index ranges from 0 to 1.

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. [1] Hirschman, A.O (1964), “The Paternity of an Index”,
               American Economic Review, 54 (5), 761.

        """
        y = self.idf.data[self.idf.income_column].values
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
            Index measure.

        References
        ----------
        .. [1] Rosenbluth, G. (1955). Measures of concentration, Business
               Concentration and Price Policy. National Bureau of Economic
               Research. Special Conference Series No. 5. Princeton, 57–89.

        """
        y = self.idf.data[self.idf.income_column].values
        n = len(y)
        g = self.idf.inequality.gini()
        return 1 / (n * (1 - g))

    def concentration_ratio(self, k):
        """Concentration Ratio index.

        The concentration ratio is calculated as the sum of the market share
        percentage held by the largest specified number of firms in an
        industry.

        Parameters
        ----------
        k: int
            The number of firms included in the concentration ratio
            calculation.

        Return
        ------
        out: float
            Index measure.

        """
        y = self.idf.data[self.idf.income_column].values
        n = len(y)
        if k < 0 or k > n:
            raise ValueError(
                "n must be an positive integer " "smaller than the data size"
            )
        else:
            ys = np.sort(y)[::-1]
            return ys[:k].sum() / ys.sum()
