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

"""Poverty measures for Apode."""

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
class PovertyMeasures:
    """Poverty Measures.
    The following poverty measures are implemented:
    - Headcount index
    - Poverty gap index
    - Squared Poverty Gap (Poverty Severity) Index
    - Foster–Greer–Thorbecke Indices
    - Sen Index
    - Sen-Shorrocks-Thon Index
    - Watts index
    - Clark, Ulph and Hemming index
    - Takayama Index
    - Kakwani Indices
    - Thon Index
    - Blackorby and Donaldson Indices
    - Hagenaars Index
    - Chakravarty Indices
    - TIP curve
    Parameters
    ----------
    data: ApodeData object
        Income data.
    pline: number, scalar
        Poverty line
    alpha: parameter, optional
    Attributes
    ----------
    ninguno: int
        Ninguno?.
    """

    idf = attr.ib()

    def __call__(self, method=None, **kwargs):
        method = "headcount" if method is None else method
        method_func = getattr(self, method)
        return method_func(**kwargs)

    def headcount(self, pline):
        """Headcount index.
        The headcount index measures the proportion of the population that
        counted as poor. More info:
        https://en.wikipedia.org/wiki/Head_count_ratio
        """
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        return q / n

    def gap(self, pline):
        """Poverty gap index.
        The poverty gap index adds up the extent to which individuals
        on average fall below the poverty line, and expresses it as
        a percentage of the poverty line. More info:
        https://en.wikipedia.org/wiki/Poverty_gap_index
        """
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        br = (pline - yp) / pline
        return np.sum(br) / n

    def severity(self, pline):
        """Squared Poverty Gap (Poverty Severity) Index.
        To construct a measure of poverty that takes into account inequality
        among the poor, some researchers use the squared poverty gap index.
        This is simply a weighted sum of poverty gaps (as a proportion of the
        poverty line), where the weights are the proportionate poverty gaps
        themselves More info:
        https://www.unescwa.org/squared-poverty-gap-index
        """
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        br = np.power((pline - yp) / pline, 2)
        return np.sum(br) / n

    def fgt(self, pline, alpha=0):
        """Foster–Greer–Thorbecke Indices.
        When parameter α = 0, P0 is simply the headcount index. When α = 1,
        the index is the poverty gap index P1, and when α is set equal
        to 2, P2 is the poverty severity index.
        A α se le conoce con el nombre de parámetro de aversión a la pobreza y,
        por tanto, cuanto mayor sea α, más énfasis se le da al más pobre de
        los pobres. More info:
        https://en.wikipedia.org/wiki/Foster%E2%80%93Greer%E2%80%93Thorbecke_indices
        """
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        if alpha < 0:
            raise ValueError(f"'alpha' must be >= 0. Found '{alpha}'")
        elif alpha == 0:
            return q / n
        elif alpha == 1:
            br = (pline - yp) / pline
            return np.sum(br) / n
        br = np.power((pline - yp) / pline, 2)
        return np.sum(br) / n

    def sen(self, pline):
        """Sen Index.
        Sen (1976) proposed an index that seeks to combine the effects of the
        number of poor, the depth of their poverty, and the distribution
        poverty within the group. More info:
        ...
        """
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        p0 = self.headcount(pline=pline)
        p1 = self.gap(pline=pline)
        gp = self.idf.inequality.gini()
        return p0 * gp + p1 * (1 - gp)

    def sst(self, pline):
        """Sen-Shorrocks-Thon Index.
        The Sen index has been modified by others, and one of the
        most attractive versions is the Sen-Shorrocks-Thon (SST)
        index. One strength of the SST index is that it can help
        give a good sense of the sources of change in poverty over
        time. This is because the index can be decomposed.
        More info:
        ...
        """
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        p0 = self.headcount(pline=pline)
        p1 = self.gap(pline=pline)
        gp = self.idf.inequality.gini()
        return p0 * p1 * (1 + gp)

    def watts(self, pline):
        """Watts index.
        Harold Watts (1968) propuso la siguiente medida de
        pobreza sensible a la distribución de rentas.
        More info:
        ...
        """
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        return sum(np.log(pline / yp)) / n

    def cuh(self, pline, alpha=0):
        """Clark, Ulph and Hemming index.
        Clark, Hemming y Ulph (1981) proponen utilizar en la medida
        de pobreza de Sen, la medida de Atkinson en lugar del índice
        de Gini de los pobres.
        More info:
        ...
        """
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        if (alpha < 0) or (alpha > 1):
            raise ValueError(f"'alpha' must be in [0,1]. Found '{alpha}'")
        if alpha == 0:
            return 1 - np.power(np.product(yp / pline) / n, 1 / n)
        else:
            return 1 - np.power(
                (sum(np.power(yp / pline, alpha)) + (n - q)) / n, 1 / alpha
            )

    def takayama(self, pline):
        """Takayama Index.
        Takayama (1979) define su medida de pobreza calculando el índice
        de Gini de la distribución censurada por la línea de pobreza.
        More info:
        ...
        """
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        u = (yp.sum() + (n - q) * pline) / n
        a = 0
        for i in range(0, q):
            a = a + (n - i + 1) * y[i]
        for i in range(q, n):
            a = a + (n - i + 1) * pline
        if u * n * n == 0:
            return 0  # to avoid NaNs for zero division error
        return 1 + 1 / n - (2 / (u * n * n)) * a

    # Kakwani Index
    def kakwani(self, pline, alpha=2):
        """Kakwani Indices.
        La familia de Kakwani (1980) que pondera los déficit mediante
        una potencia del número de orden que ocupa cada individuo
        dentro del subgrupo de pobres. El parámetro α identifica una
        cierta “aversión” al lugar ocupado en la sociedad.
        More info:
        ...
        """
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        # yp = ys[0:q] # unused, commenting it out (delete?)
        # alpha = 2.0  # elegible
        a = 0.0
        u = 0.0
        for i in range(0, q):
            f = np.power(q - i + 2, alpha)  # ver +2
            a = a + f
            u = u + f * (pline - ys[i])
        if u == 0:
            return 0  # to avoid NaNs for zero division error
        return (q / (n * pline * a)) * u

    def thon(self, pline):
        """Thon Index.
        La diferencia entre esta medida (Thon,1979) y la de Sen radica
        en la función de ponderación. Aquí se pondera el individuo pobre
        por el lugar que ocupa dentro de toda la población, y no solo
        respecto a los pobres.
        More info:
        ...
        """
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        # yp = ys[0:q] # unused, commenting it out (delete?)
        u = 0
        for i in range(0, q):
            u = u + (n - i + 1) * (pline - ys[i])
        return (2 / (n * (n + 1) * pline)) * u

    def bd(self, pline, alpha=2):
        """Blackorby and Donaldson Indices.
        Blackorby y Donaldson (1980) proponen una medida de pobreza de tipo
        normativo.
        More info:
        ...
        """
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        u = yp.sum() / q
        # atkp = atkinson(yp, alpha)
        # gp = self.idf.inequality.gini()
        atkp = self.idf.inequality.atkinson(alpha=alpha)
        yedep = u * (1 - atkp)
        return (q / n) * (pline - yedep) / pline

    def hagenaars(self, pline):
        """Hagenaars Index.
        Hagenaars (1984) para obtener la medida de pobreza considera
        la función de evaluación social de la renta como V(x) = ln(x).
        More info:
        ...
        """
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        y = self.idf.data[self.idf.varx].values
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        ug = np.exp(sum(np.log(yp)) / q)  # o normalizar con el maximo
        return (q / n) * ((np.log(pline) - np.log(ug)) / np.log(pline))

    def chakravarty(self, pline, alpha=0.5):
        """Chakravarty Indices.
        Chakravarty (1983) es una medida ética de pobreza. El índice de pobreza
        se obtiene, entonces, como la suma normalizada de las carencias de
        utilidad de los pobres.
        More info:
        ...
        """
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        y = self.idf.data[self.idf.varx].values
        if (alpha <= 0) or (alpha >= 1):
            raise ValueError(f"'alpha' must be in (0,1). Found '{alpha}'")
        n = len(y)
        ys = np.sort(y)
        q = np.sum(ys < pline)
        yp = ys[0:q]
        return sum(1 - np.power(yp / pline, alpha)) / n

    # TIP Curve
    def tip(self, pline, plot=True):
        """TIP curve.
        Three 'I's of Poverty (TIP) curves, based on distributions
        of poverty gaps, provide evocative graphical summaries of
        the incidence, intensity, and inequality dimensions of
        poverty, and a means for checking for unanimous poverty
        orderings according to a wide class of poverty indices.
        More info: Jenkins and Lambert (1997)
        """
        if pline < 0:
            raise ValueError(f"'pline' must be >= 0. Found '{pline}'")
        y = self.idf.data[self.idf.varx].values
        ys = np.sort(y)
        n = len(ys)
        q = sum(ys < pline)
        ygap = np.zeros(n)
        ygap[0:q] = (pline - ys[0:q]) / pline

        z = np.cumsum(ygap) / n
        z = np.insert(z, 0, 0)
        p = np.arange(0, n + 1) / n
        df = pd.DataFrame({"population": p, "variable": z})
        if plot:
            plt.plot(p, z)
            plt.title("TIP Curve")
            plt.ylabel("Cumulated poverty gaps")
            plt.xlabel("Cumulative % of population")
            plt.show()
        return df
