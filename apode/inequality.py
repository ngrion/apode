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

"""Inequality measures for Apode."""

# =============================================================================
# IMPORTS
# =============================================================================

import attr

import numpy as np


# =============================================================================
# FUNCTIONS
# =============================================================================


@attr.s(frozen=True)
class InequalityMeasures:
    """Inequality measures for Apode.

    The following inequality measures are implemented:

    - gini: Gini Index
    - entropy: Generalized Entropy Index
    - atkinson: Atkinson Index
    - rrange: Relative Range
    - rad: Relative average deviation
    - cv: Coefficient of variation
    - sdlog: Standard deviation of log
    - merhan: Merhan index
    - piesch: Piesch Index
    - bonferroni: Bonferroni Indices
    - kolm: Kolm Index

    Parameters
    ----------
    method : String
        Inequality measure.
    **kwargs
        Arbitrary keyword arguments.

    """

    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        """Return the ApodeData object."""
        method = "gini" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    def rrange(self):
        """Relative range.

        This measure divides the difference between the highest
        and lowest income by the mean income.

        Return
        ------
        out: float
            Index measure.

        """
        y = self.idf.data[self.idf.income_column].values
        u = np.mean(y)
        return (np.max(y) - np.min(y)) / u

    def rad(self):
        """Relative average deviation.

        Ratio of the sum of the absolute value of the distance between
        each income in the distribution and the mean income, to total
        income. [11]_

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. [11] Atkinson, AB (1970) On the measurement of inequality.
           Journal of Economic Theory, 2 (3), pp. 244–263.

        """
        y = self.idf.data[self.idf.income_column].values
        n = len(y)
        u = np.mean(y)
        return sum(abs(y - u)) / (2 * n * u)

    def cv(self):
        """Coefficient of variation.

        It is the quotient between the standard deviation and the mean. [21]_

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. [21] Atkinson, AB (1970) On the measurement of inequality.
           Journal of Economic Theory, 2 (3), pp. 244–263.

        """
        y = self.idf.data[self.idf.income_column].values
        u = np.mean(y)
        return np.std(y) / u

    def sdlog(self):
        """Calculate Standard deviation of logarithms.

        Attach great importance to income transfers at the lower end. [3]_

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. [3] Atkinson, AB (1970) On the measurement of inequality.
           Journal of Economic Theory, 2 (3), pp. 244–263.

        """
        y = self.idf.data[self.idf.income_column].values
        n = len(y)
        u = np.mean(y)
        return np.sqrt(sum(pow((np.log(u) - np.log(y)), 2)) / n)

    def ratio(self, alpha):
        """Dispersion Ratio (Kuznets Ratio).

        This measure presents the ratio of the average income of the richest
        alpha percent of the population to the average income of the poorest
        alpha percent.

        Parameters
        ----------
        alpha: float

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Haughton, J., and Khandker, S. R. (2009). Handbook on poverty
           and inequality. Washington, DC: World Bank.

        """
        y = self.idf.data[self.idf.income_column].values
        if (alpha < 0) or (alpha > 1):
            raise ValueError(f"'alpha' must be in [0,1]. Found '{alpha}'")
        y = np.sort(y)
        n = len(y)
        k = int(np.floor(alpha * n))
        return np.mean(y[:k]) / np.mean(y[n - k :])  # noqa

    def gini(self):
        """Gini Coefficient.

        The Gini Coefficient

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Gini, C. (1914), 'Sulla misura della concentrazione e della
           variabilità dei caratteri', Atti del Reale Istituto Veneto di
           Scienze, Lettere ed Arti 73, 1203-1248.

        """
        y = self.idf.data[self.idf.income_column].values
        y = np.sort(y)
        n = len(y)
        if n == 1:
            return 0.0
        if y[0] == y[n - 1]:
            return 0.0
        u = np.mean(y)
        ii = np.arange(n)
        a = np.sum(np.dot(n - ii, y))
        g = (n + 1) / (n - 1) - 2 / (n * (n - 1) * u) * a
        return g * (n - 1) / n

    def merhan(self):
        """Merhan Coefficient.

        The Merhan Coefficient

        Return
        ------
        out: float
            Merhan Coefficient.

        References
        ----------
        .. Mehran, Farhad, 1976. "Linear Measures of Income Inequality,"
           Econometrica, Econometric Society, vol. 44(4), pages 805-809,
           July.

        """
        y = self.idf.data[self.idf.income_column].values
        y = np.sort(y)
        n = len(y)
        f = 1.0 / (n * np.mean(y))
        pi = np.arange(n - 1) / n
        pi[0] = 1 / n
        qi = f * np.cumsum(y[:-1])
        p_q = pi - qi
        pi[0] = 0
        # print(np.sum(np.dot(1 - pi, p_q)) * 6 / n)
        return np.sum(np.dot(1 - pi, p_q)) * 6 / n

    def piesch(self):
        """Piesch Coefficient.

        The Piesch Coefficient

        Return
        ------
        out: float
            Index measure.


        References
        ----------
        .. Piesch, W. (1975). Statistische Konzentrationsmasse. Mohr
           (Paul Siebeck), Tübingen.

        """
        y = self.idf.data[self.idf.income_column].values
        y = np.sort(y)
        n = len(y)
        f = 1.0 / (n * np.mean(y))
        pi = np.arange(n - 1) / n
        pi[0] = 1 / n
        qi = f * np.cumsum(y[:-1])
        p_q = pi - qi
        pi[0] = 1
        return np.sum(np.dot(pi, p_q)) * 3 / n

    def bonferroni(self):
        """Bonferroni Coefficient.

        The Bonferroni Coefficient

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Bonferroni, C.E. (1930), Elementi di Statistica Generale,
           Seeber, Firenze.

        """
        y = self.idf.data[self.idf.income_column].values
        y = np.sort(y)
        n = len(y)
        ii = np.arange(n - 1)
        ii[0] = 1
        x = np.cumsum(y[:-1])
        s = np.sum(x / ii)
        u = (x[-1] + y[-1]) / n
        return 1 - (1 / ((n - 1) * u)) * s

    def kolm(self, alpha):
        """Kolm Coefficient.

        The Kolm Coefficient

        Parameters
        ----------
        alpha: float

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Kolm, S.-Ch. (1 976a). ‘Unequal Inequalitites 1’, Journal of
           Economic Theory.

        """
        y = self.idf.data[self.idf.income_column].values
        if alpha <= 0:
            raise ValueError("Alpha must be strictly positive (>0.0)")
        n = len(y)
        u = np.mean(y)
        return (1 / alpha) * (
            np.log((1.0 / n) * np.sum(np.exp(alpha * (u - y))))
        )

    # ver testear rango a
    def entropy(self, alpha=0):
        """General Entropy index.

        The entropy index measures the proportion of the population that
        counted as poor.

        Parameters
        ----------
        alpha: float, optional(default=0)

        Return
        ------
        out: float
            Index measure.

        """
        a = alpha
        y = self.idf.data[self.idf.income_column].values
        n = len(y)
        u = np.mean(y)
        if a == 0.0:
            return np.sum(np.log(u / y)) / n
        elif a == 1.0:
            # return np.sum((y / u) * np.log(y / u)) / n
            # return np.dot(np.log(y / u), y / u) / n
            y2 = np.log(y / u) * (y / u)
            return np.sum(y2[y > 0]) / n
        return (1 / (a * (a - 1))) * (np.sum(pow(y / u, a)) / n - 1)

    def atkinson(self, alpha=2):
        """Atkinson index.

        The Atkinson index measures the proportion of the population that
        counted as poor.

        Parameters
        ----------
        alpha: float, optional(default=2)

        Return
        ------
        out: float
            Index measure.

        References
        ----------
        .. Atkinson, AB (1970) On the measurement of inequality.
           Journal of Economic Theory, 2 (3), pp. 244–263.

        """
        y = self.idf.data[self.idf.income_column].values + 0.0
        if alpha <= 0:
            raise ValueError("Alpha must be strictly positive (>0.0)")
        n = len(y)
        if n == 1:
            return 0.0
        ymu = np.mean(y)
        if alpha == 1:
            if np.min(y) == 0:
                yede = 0.0
            else:
                # yede = np.power(np.product(y), 1 / n)
                yede = np.exp(np.mean(np.log(y)))
            # y_nz = y[y != 0]
            # ylog = np.log(y_nz)
            # h = np.mean(ylog)
            # return 1 - np.exp(h) / np.mean(y_nz)
        else:
            yede = np.power(np.mean(np.power(y, 1 - alpha)), 1 / (1 - alpha))
            # with np.errstate(divide="ignore"):
            #     a1 = np.sum(np.power(y, 1 - alpha)) / n
            #     return 1 - np.power(a1, 1 / (1 - alpha)) / np.mean(y)
        return 1 - yede / ymu
