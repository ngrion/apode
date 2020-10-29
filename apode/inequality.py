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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import attr


# =============================================================================
# FUNCTIONS
# =============================================================================


@attr.s(frozen=True)
class InequalityMeasures:
    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        method = "gini" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    # Entropia general  (necesita sort?) (ver rango a)
    def entropy(self, alpha=0, sort=False):
        a = alpha
        y = self.idf.data[self.idf.varx].values
        if not sort:
            y = np.sort(y)
        n = len(y)
        if n == 0:
            return 0
        u = np.mean(y)
        if a == 0.0:
            return np.sum(np.log(u / y)) / n
        elif a == 1.0:
            return np.sum((y / u) * np.log(y / u)) / n
        return (1 / (a * (a - 1))) * (np.sum(pow(y / u, a)) / n - 1)

    # Atkinson inequality index
    def atkinson(self, alpha, sort=False):  # poner alpha=2?
        y = self.idf.data[self.idf.varx].values
        if alpha <= 0:
            raise TypeError("Alpha must be strictly positive (>0.0)")
        n = len(y)
        if n == 0:
            return 0
        # if not sort:
        #     y = np.sort(y)
        if alpha == 1:
            y_nz = y[y != 0]
            ylog = np.log(y_nz)
            h = np.mean(ylog)
            # return 1 - n * np.exp(h)
            return 1 - np.exp(h) / np.mean(y_nz)
        else:
            a1 = np.sum(np.power(y, 1 - alpha)) / n
            return 1 - np.power(a1, 1 / (1 - alpha)) / np.mean(y)
            # n2 = np.power(n, alpha / (alpha - 1.0))
            # h1 = np.power(y, 1.0 - alpha).sum()
            # h2 = np.power(h1, 1.0 / (1.0 - alpha))
            # return 1 - n2 * h2

    # Relative range
    def rrange(self, sort=False):
        y = self.idf.data[self.idf.varx].values
        # if not sort:
        #     y = np.sort(y)
        n = len(y)
        if n == 0:
            return 0
        u = np.mean(y)
        return (max(y) - min(y)) / u

    # relative average deviation
    def rad(self, sort=False):
        y = self.idf.data[self.idf.varx].values
        # if not sort:
        #     y = np.sort(y)
        n = len(y)
        if n == 0:
            return 0
        u = np.mean(y)
        return sum(abs(y - u)) / (2 * n * u)

    # Coeficiente de variacion
    def cv(self, sort=False):
        y = self.idf.data[self.idf.varx].values
        # if not sort:
        #     y = np.sort(y)
        n = len(y)
        if n == 0:
            return 0
        u = np.mean(y)
        return np.std(y) / u

    # Desv Est de los logaritmos
    def sdlog(self, sort=False):
        y = self.idf.data[self.idf.varx].values
        # if not sort:
        #     y = np.sort(y)
        n = len(y)
        if n == 0:
            return 0
        u = np.mean(y)
        return np.sqrt(sum(pow((np.log(u) - np.log(y)), 2)) / n)

    # ---- Lineales ---

    # Gini
    def gini(self, sort=True):
        y = self.idf.data[self.idf.varx].values
        if sort:
            y = np.sort(y)
        n = len(y)
        if n == 0:
            return 0
        u = np.mean(y)
        a = 0
        for i in range(0, n):
            a = a + (n - i) * y[i]
        g = (n + 1) / (n - 1) - 2 / (n * (n - 1) * u) * a
        g = g * (n - 1) / n
        return g

    # Merhan
    def merhan(self, sort=True):
        y = self.idf.data[self.idf.varx].values
        if sort:
            y = np.sort(y)
        n = len(y)
        if n == 0:
            return 0
        u = np.mean(y)
        f = 1.0 / (n * u)
        syi = y[0]
        pi = 1.0 / n
        qi = f * y[0]
        m = pi - qi
        for i in range(1, n - 1):  # do i=2,n-1
            pi = i / n
            syi = syi + y[i]
            qi = f * syi
            m = m + (1 - pi) * (pi - qi)
        return m * 6 / n

    # Piesch
    def piesch(self, sort=True):
        y = self.idf.data[self.idf.varx].values
        if sort:
            y = np.sort(y)
        n = len(y)
        if n == 0:
            return 0
        u = np.mean(y)
        f = 1.0 / (n * u)
        syi = y[0]
        pi = 1.0 / n
        qi = f * y[0]
        m = pi - qi
        for i in range(1, n - 1):  # 2,n-1
            pi = i / n
            syi = syi + y[i]
            qi = f * syi
            m = m + pi * (pi - qi)
        return m * 3 / n

    # --- otros ---

    # Bonferroni
    def bonferroni(self, sort=True):
        y = self.idf.data[self.idf.varx].values
        if sort:
            y = np.sort(y)
        n = len(y)
        if n == 0:
            return 0
        s = y[0]
        x = y[0]
        for i in range(1, n - 1):  # i=2,n-1
            x = x + y[i]
            s = s + x / i
        u = (x + y[n - 1]) / n
        return 1 - (1 / ((n - 1) * u)) * s

    # Kolm
    def kolm(self, alpha, sort=False):
        y = self.idf.data[self.idf.varx].values
        if alpha <= 0:
            raise TypeError("Alpha must be strictly positive (>0.0)")
        # if not sort:
        #     y = np.sort(y)
        n = len(y)
        if n == 0:
            return 0
        u = np.mean(y)
        return (1 / alpha) * (np.log((1.0 / n) *
                                     np.sum(np.exp(alpha * (u - y)))))

    # ratio
    def ratio(self, alpha, sort=True):
        y = self.idf.data[self.idf.varx].values
        if (alpha < 0) or (alpha > 1):
            raise ValueError(f"'alpha' must be in [0,1]. Found '{alpha}'")
        if sort:
            y = np.sort(y)
        n = len(y)
        if n == 0:
            return 0
        k = int(np.floor(alpha * n))
        return np.mean(y[:k]) / np.mean(y[n - k:])

    # tipo = 'r' o None  simple
    # tipo = 'g'  generalizada
    #  tipo = 'a' absoluta
    # ver n=0,1
    def lorenz(self, alpha="r", plot=True, sort=True):
        y = self.idf.data[self.idf.varx].values
        if sort:
            y = np.sort(y)
        n = len(y)
        z = np.cumsum(y) / y.sum()
        q = np.arange(0, n + 1) / n
        qd = q
        if alpha == "r":
            pass
        elif alpha == "g":
            mu = np.mean(y)
            z = z * mu
            qd = q * mu
        elif alpha == "a":
            mu = np.mean(y)
            qd = q * 0
            z = np.cumsum(y - mu)

        z = np.insert(z, 0, 0)

        df = pd.DataFrame({"population": q, "variable": z})

        if plot:
            plt.plot(q, z)
            plt.plot(q, qd)
            plt.xlabel("Cumulative % of population")
            if alpha == "r":
                plt.ylabel("Cumulative % of variable")
                plt.title("Lorenz Curve")
            elif alpha == "g":
                plt.ylabel("Scaled Cumulative % of variable")
                plt.title("Generalized Lorenz Curve")
            elif alpha == "a":
                plt.ylabel("Cumulative deviaton")
                plt.title("Absolut Lorenz Curve")
            plt.show()

        return df

    # Pen Parade
    # ver n=0,1
    def pen(self, plot=True, pline=None, sort=True):
        y = self.idf.data[self.idf.varx].values
        if sort:
            y = np.sort(y)
        n = len(y)
        me = np.median(y)
        q = np.arange(0, n + 1) / n
        mu = np.mean(y)
        qd = np.ones(n + 1) * mu / me
        z = np.copy(y) / me
        z = np.insert(z, 0, 0)
        df = pd.DataFrame({"population": q, "variable": z})
        if plot:
            plt.plot(q, z)
            plt.plot(q, qd, label="Mean")
            if not (pline is None):
                qpl = np.ones(n + 1) * pline / me
                plt.plot(q, qpl, label="Poverty line")
            plt.xlabel("Cumulative % of population")
            plt.ylabel("Medianized variable")
            plt.title("Pen's Parade")
            plt.legend()
            plt.show()
        return df
